"""Generic evaluation: N registry models x one frozen dataset x the chosen metrics.

    bash run.sh run --spec ../../artifacts/eval/<id>/spec.json        # the API writes the spec

spec.json: {"id", "dataset": <dataset version folder>, "splits": ["test"], "limit": 0, "seed": 7,
            "models": [<registry entries>], "metrics": [...], "judge_model": "qwen3:14b", "batch": 8}

Every model gets the same items, the same task prompt (with the schema or other context the
dataset carries) and greedy decoding. Answers are cached per model and prompt, so re-running an
evaluation, or adding a model to one, only generates what is missing (and never pays twice).

Writes artifacts/eval/{id}/:
- events.jsonl: progress (stage, model, done/total) for the Studio
- items.jsonl: the evaluated items (inputs, reference, slice labels)
- results.jsonl: one row per item x model: answer, extracted SQL, latency, tokens, scores, details
- report.json: scores per model x metric x slice with 95% bootstrap intervals, paired comparisons
  of every model pair, failure breakdowns, token usage and cost
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from agritune_core.sqlexec import execute
from agritune_core.tasks import task_of

from . import ARTIFACTS, REPO
from .metrics import METRICS, preview, score_sql, score_text
from .stats import mean_ci, paired_delta
from .structured import STRUCTURED, baselines, score_structured, weighted_kappa
from .targets import Answer, Gen

EVALS = ARTIFACTS / "eval"
GEN_CACHE = EVALS / "gen_cache"
REGISTRY_URL = "http://127.0.0.1:8104"
OLLAMA_URL = "http://127.0.0.1:11434"
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Events:
    def __init__(self, path: Path, eval_id: str):
        self.path, self.eval_id = path, eval_id

    def emit(self, type_: str, **kw) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"job_id": self.eval_id, "service": "eval", "type": type_, "ts": now()} | kw,
                               ensure_ascii=False) + "\n")


# ---------- items ----------

def load_items(ds_dir: Path, splits: list[str], limit: int, seed: int) -> tuple[list[dict], dict]:
    ref = json.loads((ds_dir / "dataset_ref.json").read_text(encoding="utf-8"))
    task = task_of(ref)
    rows = []
    for split in splits:
        if split not in ref["splits"]:
            raise SystemExit(f"{ref['id']}@{ref['version']} has no split {split} (has {', '.join(ref['splits'])})")
        rows += [r | {"split": split} for r in pq.read_table(REPO / ref["splits"][split]).to_pylist() if task.keep(r)]
    rows.sort(key=lambda r: (r["split"], r["id"]))
    if limit and len(rows) > limit:  # the same sample for every model, and for every re-run
        rows = sorted(random.Random(seed).sample(rows, limit), key=lambda r: (r["split"], r["id"]))
    return rows, ref


SLICE_META = ("difficulty", "category")


def slice_labels(task, row: dict, db, tables_json, sql: bool) -> dict:
    labels = {"split": row["split"]}
    if row.get("lang"):
        labels["lang"] = row["lang"]
    meta = row.get("meta")
    meta = json.loads(meta) if isinstance(meta, str) and meta else meta or {}
    for k in dict.fromkeys([*SLICE_META, *task.slices]):  # e.g. BIRD's difficulty, a task's expected label
        if meta.get(k) not in (None, ""):
            labels[k] = str(meta[k])
    if sql:
        from .spider_match import hardness

        h = hardness(str(row[task.target]), db, tables_json)
        if h:
            labels["hardness"] = h
    return labels


# ---------- generation ----------

def fingerprint(m: dict) -> str:
    return json.dumps([m["kind"], m.get("hf_id"), m.get("adapter_uri"), m.get("ollama_tag"), m.get("api_model"),
                       m.get("options")], sort_keys=True)


class GenCache:
    def __init__(self, model: dict):
        GEN_CACHE.mkdir(parents=True, exist_ok=True)
        self.path = GEN_CACHE / f"{model['id']}.jsonl"
        self.fp = fingerprint(model)
        self.rows = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.rows[r["key"]] = r["answer"]

    def key(self, messages: list[dict], max_new_tokens: int, gen: Gen = Gen()) -> str:
        parts = [self.fp, messages, max_new_tokens, *gen.cache_extra()]
        return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()

    def get(self, key: str) -> Answer | None:
        a = self.rows.get(key)
        return Answer(**a) if a else None

    def put(self, key: str, a: Answer) -> None:
        if a.error:  # errors are retried next time
            return
        self.rows[key] = a.__dict__
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "answer": a.__dict__}, ensure_ascii=False) + "\n")


CHUNK = 64  # answers generated between cache writes


def ollama_unload_all() -> None:
    """Frees the GPU before transformers loads weights (Ollama keeps models resident for minutes)."""
    import httpx

    try:
        for m in httpx.get(f"{OLLAMA_URL}/api/ps", timeout=5).json().get("models", []):
            httpx.post(f"{OLLAMA_URL}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=30)
    except httpx.HTTPError:
        pass


def generate_all(models: list[dict], convs: list[list[dict]], max_new: int, batch: int, ev: Events,
                 gen: Gen = Gen(), tier_convs: dict[str, list[list[dict]]] | None = None) -> dict[str, list]:
    """Answers per model id. HF models sharing a base load it once; base answers use the adapter off.

    `tier_convs`: the conversations for models whose registry entry has that `prompt_tier`, when the
    dataset carries a variant for it (e.g. Strong Hire's prompts for small models); others get `convs`."""
    out: dict[str, list[Answer | None]] = {}
    todo: dict[str, list[int]] = {}
    caches = {m["id"]: GenCache(m) for m in models}
    conv_of = {m["id"]: (tier_convs or {}).get(m.get("prompt_tier") or "", convs) for m in models}
    keys = {m["id"]: [caches[m["id"]].key(c, max_new, gen) for c in conv_of[m["id"]]] for m in models}
    for m in models:
        out[m["id"]] = [caches[m["id"]].get(k) for k in keys[m["id"]]]
        todo[m["id"]] = [i for i, a in enumerate(out[m["id"]]) if a is None]
        if len(todo[m["id"]]) < len(convs):
            ev.emit("log", message=f"{m['id']}: {len(convs) - len(todo[m['id']])} answers from cache")

    def run(model, target, **kw):
        idx = todo[model["id"]]
        if not idx:
            return
        total = len(idx)
        ev.emit("progress", stage="generate", model=model["id"], done=0, total=total)
        t0, errors = time.time(), 0
        for s in range(0, total, CHUNK):  # cache each chunk, so an interrupted evaluation resumes from here
            part = idx[s:s + CHUNK]
            prog = lambda n, s=s: ev.emit("progress", stage="generate", model=model["id"], done=s + n, total=total)  # noqa: E731
            answers = target.chat([conv_of[model["id"]][i] for i in part], max_new_tokens=max_new, progress=prog,
                                  gen=gen, **kw)
            for i, a in zip(part, answers):
                out[model["id"]][i] = a
                caches[model["id"]].put(keys[model["id"]][i], a)
            errors += sum(1 for a in answers if a.error)
            ev.emit("progress", stage="generate", model=model["id"], done=s + len(part), total=total)
        ev.emit("progress", stage="generate", model=model["id"], done=total, total=total,
                message=f"{total} answers in {time.time() - t0:.0f}s" + (f", {errors} errors" if errors else ""))

    hf = [m for m in models if m["kind"] in ("hf_base", "hf_adapter") and todo[m["id"]]]
    if hf:
        ollama_unload_all()
        import torch

        from .targets import HFTarget

        by_base = defaultdict(list)
        for m in hf:
            by_base[m["hf_id"]].append(m)
        for base, group in by_base.items():
            adapters = [m for m in group if m["kind"] == "hf_adapter"]
            bases = [m for m in group if m["kind"] == "hf_base"]
            loads = [(a, REPO / a["adapter_uri"]) for a in adapters] or [(None, None)]
            for k, (adapter_model, adapter_path) in enumerate(loads):
                ev.emit("log", message=f"loading {base}" + (f" + {adapter_model['id']}" if adapter_model else ""))
                target = HFTarget(base, str(adapter_path) if adapter_path else None)
                if k == 0:
                    for b in bases:  # the base with the adapter switched off: the same weights
                        with target.use(False):
                            run(b, target, batch=batch)
                if adapter_model:
                    with target.use(True):
                        run(adapter_model, target, batch=batch)
                del target
                gc.collect()
                torch.cuda.empty_cache()

    for m in models:
        if not todo[m["id"]] or m["kind"] in ("hf_base", "hf_adapter"):
            continue
        if m["kind"] == "ollama":
            from .targets import OllamaTarget

            t = OllamaTarget(m["ollama_tag"], OLLAMA_URL)
            run(m, t)
            t.unload()
        elif m["kind"] == "api":
            from .targets import AnthropicTarget, GeminiTarget

            cls = AnthropicTarget if m["provider"] == "anthropic" else GeminiTarget
            try:
                t = cls(m["api_model"], m.get("options"))
            except RuntimeError as e:  # no API key: every answer is an error, the rest of the run goes on
                ev.emit("log", message=f"{m['id']}: {e}")
                for i in todo[m["id"]]:
                    out[m["id"]][i] = Answer("", 0.0, error=str(e))
                continue
            run(m, t)
    return out  # type: ignore[return-value]


# ---------- scoring ----------

def score_all(task, items: list[dict], models: list[dict], answers: dict, metrics: list[str], root, tables_json,
              ev: Events) -> dict[tuple[str, str], dict]:
    """(model, item) -> {"scores": {...}, "details": {...}} for the non-judge metrics."""
    sql = task.attachments and task.attachments["type"] == "sqlite_dbs"
    want = set(metrics)
    res: dict[tuple[str, str], dict] = {}
    gold = {}
    if sql:
        ev.emit("progress", stage="score", message="running gold queries", done=0, total=len(items))
        with ThreadPoolExecutor(4) as pool:
            for it, g in zip(items, pool.map(lambda it: execute(task.attachment(root, it), str(it[task.target]), 30), items)):
                gold[it["id"]] = g

    def one(args):
        m, i, it = args
        a = answers[m["id"]][i]
        scores, details = {}, {}
        if sql:
            scores, details = score_sql(a.text, str(it[task.target]), task.attachment(root, it), gold[it["id"]],
                                        want, tables_json)
        else:
            scores = score_text(a.text, str(it[task.target]), want)
            try:
                more, details = score_structured(task, a.text, it[task.target], want, m.get("prompt_tier"))
            except Exception as e:  # one odd answer must not fail the evaluation: it scores 0 where it could not be read
                more = {k: 0.0 for k in want & STRUCTURED if METRICS[k].kind in ("rate", "score")}
                details = {"scoring_error": f"{type(e).__name__}: {e}"[:300]}
            scores |= more
        if "latency" in want and not a.error:
            scores["latency"] = round(a.latency_s, 3)
        if "ttft" in want and not a.error and a.ttft_s is not None:
            scores["ttft"] = round(a.ttft_s, 3)
        return (m["id"], it["id"]), {"scores": scores, "details": details}

    jobs = [(m, i, it) for m in models for i, it in enumerate(items)]
    with ThreadPoolExecutor(4) as pool:
        for n, (k, v) in enumerate(pool.map(one, jobs), 1):
            res[k] = v
            if n % 200 == 0:
                ev.emit("progress", stage="score", done=n, total=len(jobs))
    for it in items:
        if sql:
            it["_gold_result"] = preview(gold[it["id"]])
    return res


def judge_all(task, items, models, answers, res, metrics, judge_model, ev: Events) -> dict:
    """Adds llm_judge scores and sql_failure_judge categories to `res`; returns judge info for the report."""
    from .judges import OllamaJudge

    want = set(metrics)
    if not want & {"llm_judge", "sql_failure_judge"}:
        return {}
    ollama_unload_all()
    j = OllamaJudge(judge_model, OLLAMA_URL)
    info = {"model": judge_model}
    idx = {it["id"]: i for i, it in enumerate(items)}
    if "llm_judge" in want:
        if task.rubric == "mango_advice_v1":  # the mango judge keeps its 5 criteria, averaged
            import asyncio

            from .judge import CRITERIA, Judge

            mj = Judge(model=judge_model)
            jobs = [(m["id"], it) for m in models for it in items]

            async def go():
                return await asyncio.gather(*(mj.grade(it["question"], str(it[task.target]), answers[mid][idx[it["id"]]].text)
                                              for mid, it in jobs))

            for (mid, it), v in zip(jobs, asyncio.run(go())):
                res[(mid, it["id"])]["scores"]["llm_judge"] = sum(v[c] for c in CRITERIA) / len(CRITERIA)
                res[(mid, it["id"])]["details"]["llm_judge"] = v
        else:
            jobs = [(m["id"], it) for m in models for it in items]
            ev.emit("progress", stage="judge", message="llm_judge", done=0, total=len(jobs))
            verdicts = j.many(lambda mid, it: j.rubric(task.rubric or "qa_reference_v1", task.user(it),
                                                       str(it[task.target]), answers[mid][idx[it["id"]]].text),
                              jobs, lambda n: ev.emit("progress", stage="judge", message="llm_judge", done=n, total=len(jobs)))
            for (mid, it), v in zip(jobs, verdicts):
                res[(mid, it["id"])]["details"]["llm_judge"] = v
                if "score" in v:
                    res[(mid, it["id"])]["scores"]["llm_judge"] = float(v["score"])
        info["rubric"] = task.rubric or "qa_reference_v1"
    if "sql_failure_judge" in want:
        jobs = [(m["id"], it) for m in models for it in items if res[(m["id"], it["id"])]["scores"].get("sql_exec") == 0]
        ev.emit("progress", stage="judge", message="failure analysis", done=0, total=len(jobs))

        def fail(mid, it):
            r = res[(mid, it["id"])]
            return j.failure(it["question"], it.get("schema", ""), str(it[task.target]), it["_gold_result"],
                             r["details"].get("sql", ""), r["details"].get("result", {}))

        verdicts = j.many(fail, jobs, lambda n: ev.emit("progress", stage="judge", message="failure analysis",
                                                         done=n, total=len(jobs)))
        for (mid, it), v in zip(jobs, verdicts):
            res[(mid, it["id"])]["details"]["sql_failure_judge"] = v
            if "category" in v:
                res[(mid, it["id"])]["scores"]["sql_failure_judge"] = v["category"]
        info["failure_categories"] = True
    j.http.post("/api/generate", json={"model": judge_model, "keep_alive": 0})
    return info


# ---------- report ----------

def kappa_ci(pairs: list[list[int]], seed: int = 0) -> tuple[float | None, list[float | None]]:
    """Weighted kappa over (model, human, categories) pairs, with a 95% bootstrap interval over the pairs."""
    k = pairs[0][2] if pairs else 2
    value = weighted_kappa([(a, b) for a, b, _ in pairs], k)
    if value is None:
        return None, [None, None]
    rng = np.random.default_rng(seed)
    boots = [weighted_kappa([(pairs[i][0], pairs[i][1]) for i in rng.integers(0, len(pairs), len(pairs))], k)
             for _ in range(1000)]
    boots = [b for b in boots if b is not None]
    return value, ([float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else [None, None])


def aggregate(items, models, res, metrics):
    numeric = [m for m in metrics if METRICS[m].kind not in ("category", "kappa")]
    slices = [("all", "all")] + sorted({(k, v) for it in items for k, v in it["_slices"].items()})
    in_slice = lambda it, s: s[0] == "all" or it["_slices"].get(s[0]) == s[1]  # noqa: E731
    scores, comps, breakdowns = [], [], {}
    for metric in numeric:
        for s in slices:
            ids = [it["id"] for it in items if in_slice(it, s)]
            vals = {}
            for m in models:
                v = {i: res[(m["id"], i)]["scores"][metric] for i in ids if metric in res[(m["id"], i)]["scores"]}
                if not v:
                    continue
                vals[m["id"]] = v
                value, ci = mean_ci(list(v.values()))
                row = {"model": m["id"], "metric": metric, "slice": {"key": s[0], "value": s[1]},
                       "value": round(value, 4), "ci95": [round(x, 4) for x in ci], "n": len(v)}
                if METRICS[metric].kind == "seconds":
                    arr = np.asarray(list(v.values()))
                    row |= {"p50": round(float(np.percentile(arr, 50)), 3), "p90": round(float(np.percentile(arr, 90)), 3)}
                scores.append(row)
            if not METRICS[metric].compare:  # a neutral value (e.g. a match score): read it, do not rank it
                continue
            for a, b in combinations([m["id"] for m in models], 2):
                if a not in vals or b not in vals:
                    continue
                common = sorted(set(vals[a]) & set(vals[b]))
                if len(common) < 2:
                    continue
                d = paired_delta([vals[a][i] for i in common], [vals[b][i] for i in common])
                verdict = d["verdict"]
                if not METRICS[metric].higher_is_better:  # lower latency is better
                    verdict = {"b_better": "a_better", "a_better": "b_better"}.get(verdict, verdict)
                comps.append({"a": a, "b": b, "metric": metric, "slice": {"key": s[0], "value": s[1]},
                              "delta": round(d["delta"], 4), "ci95": [round(x, 4) for x in d["ci95"]],
                              "verdict": verdict, "n": len(common)})
    for metric in [m for m in metrics if METRICS[m].kind == "kappa"]:  # over the item set, from pairs
        for s in slices:
            for m in models:
                pairs = [p for it in items if in_slice(it, s)
                         for p in (res[(m["id"], it["id"])]["details"].get("pairs") or {}).get(metric, [])]
                value, ci = kappa_ci(pairs)
                if value is None:
                    continue
                scores.append({"model": m["id"], "metric": metric, "slice": {"key": s[0], "value": s[1]},
                               "value": round(value, 4), "ci95": [None if x is None else round(x, 4) for x in ci],
                               "n": len(pairs)})
    for metric in [m for m in metrics if METRICS[m].kind == "category"]:
        breakdowns[metric] = {m["id"]: dict(Counter(res[(m["id"], it["id"])]["scores"][metric] for it in items
                                                    if metric in res[(m["id"], it["id"])]["scores"]).most_common())
                              for m in models}
    return scores, comps, breakdowns


def usage(models, answers) -> dict:
    out = {}
    for m in models:
        ans = answers[m["id"]]
        tin = sum(a.input_tokens or 0 for a in ans)
        tout = sum(a.output_tokens or 0 for a in ans)
        price = PRICES.get(m.get("api_model") or "")
        out[m["id"]] = {"input_tokens": tin, "output_tokens": tout, "errors": sum(1 for a in ans if a.error),
                        "usd": round(tin / 1e6 * price[0] + tout / 1e6 * price[1], 4) if price else None}
    return out


def post_registry(report: dict) -> None:
    import httpx

    for m in report["models"]:
        metrics = {s["metric"]: s["value"] for s in report["scores"] if s["model"] == m["id"] and s["slice"]["key"] == "all"}
        try:
            httpx.post(f"{REGISTRY_URL}/models/{m['id']}/evaluations", timeout=5,
                       json={"eval_id": report["id"], "dataset": report["dataset"], "task": report["task"],
                             "split": ",".join(report["splits"]), "n": report["n_items"], "metrics": metrics,
                             "created_at": report["created_at"]})
        except httpx.HTTPError:
            pass  # the registry may be down; the report on disk is the record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    out = Path(a.spec).parent
    ev = Events(out / "events.jsonl", spec["id"])
    ev.emit("started", message=f"{len(spec['models'])} models on {spec['dataset']}")
    try:
        ds_dir = REPO / spec["dataset"] if not Path(spec["dataset"]).is_absolute() else Path(spec["dataset"])
        items, ref = load_items(ds_dir, spec.get("splits", ["test"]), spec.get("limit", 0), spec.get("seed", 7))
        task = task_of(ref)
        metrics = list(dict.fromkeys(spec["metrics"]))
        if "sql_failure_judge" in metrics and "sql_exec" not in metrics:
            metrics.insert(0, "sql_exec")  # failures are defined by execution accuracy
        root = REPO / ref["attachments"] if ref.get("attachments") else None
        tables_json = str(root / "tables.json") if root and (root / "tables.json").exists() else None
        sql = bool(task.attachments and task.attachments["type"] == "sqlite_dbs")
        models = spec["models"]
        ev.emit("log", message=f"{len(items)} items x {len(models)} models, task {task.ref}, metrics {', '.join(metrics)}")

        convs, rows = [], []
        for it in items:
            row = task.with_context(it, root)
            it.update({k: row[k] for k in task.context})
            convs.append(task.messages(row))
            rows.append(row)
            it["_slices"] = slice_labels(task, it, task.attachment(root, it) if sql else None, tables_json, sql)
        tiers = {m.get("prompt_tier") for m in models} - {None, ""}
        tier_convs = {t: [task.messages(r, t) for r in rows] for t in tiers if any(t in task.tiers(r) for r in rows)}
        for t in tier_convs:
            ev.emit("log", message=f"models with prompt_tier {t} get the dataset's {t} prompts")
        max_new = int(spec.get("max_new_tokens") or task.decoding.get("max_new_tokens", 400))
        gen = Gen(json_schema=task.json_schema, stream=bool(task.decoding.get("stream")))
        answers = generate_all(models, convs, max_new, int(spec.get("batch", 8)), ev, gen=gen, tier_convs=tier_convs)

        res = score_all(task, items, models, answers, metrics, root, tables_json, ev)
        judge = judge_all(task, items, models, answers, res, metrics, spec.get("judge_model", "qwen3:14b"), ev)

        with (out / "items.jsonl").open("w", encoding="utf-8") as f:
            for n, (it, conv) in enumerate(zip(items, convs)):
                variants = {t: c[n] for t, c in tier_convs.items() if c[n] != conv}
                f.write(json.dumps({"id": it["id"], "split": it["split"], "slices": it["_slices"],
                                    "inputs": {k: it.get(k) for k in task.inputs}, "reference": it[task.target],
                                    "gold_result": it.get("_gold_result"), "messages": conv,
                                    **({"tier_messages": variants} if variants else {}),
                                    "max_new_tokens": max_new}, ensure_ascii=False, default=str) + "\n")
        with (out / "results.jsonl").open("w", encoding="utf-8") as f:
            for m in models:
                for i, it in enumerate(items):
                    ans = answers[m["id"]][i]
                    r = res[(m["id"], it["id"])]
                    f.write(json.dumps({"model": m["id"], "item": it["id"], "output": ans.text, "error": ans.error,
                                        "latency_s": round(ans.latency_s, 3), "input_tokens": ans.input_tokens,
                                        "output_tokens": ans.output_tokens,
                                        "ttft_s": None if ans.ttft_s is None else round(ans.ttft_s, 3)} | r,
                                       ensure_ascii=False, default=str) + "\n")

        scores, comps, breakdowns = aggregate(items, models, res, metrics)
        report = {"id": spec["id"], "version": 2, "task": task.ref, "dataset": f"{ref['id']}@{ref['version']}",
                  "dataset_path": spec["dataset"], "splits": spec.get("splits", ["test"]), "n_items": len(items),
                  "limit": spec.get("limit", 0), "seed": spec.get("seed", 7),
                  "models": models, "metrics": metrics,
                  "metric_info": {m: METRICS[m].__dict__ for m in metrics},
                  "decoding": {"do_sample": False, "max_new_tokens": max_new, "thinking": "off where the model allows",
                               "json_schema": bool(gen.json_schema), "stream": gen.stream,
                               "prompt_tiers": sorted(tier_convs)},
                  "scores": scores, "comparisons": comps, "breakdowns": breakdowns, "usage": usage(models, answers),
                  "judge": judge, "baselines": baselines(task, [it[task.target] for it in items], set(metrics)),
                  "created_at": now(), "workspace_id": "default"}
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        post_registry(report)
        head = [f"{s['model']}: {s['metric']} {s['value']:.3f}" for s in scores if s["slice"]["key"] == "all"]
        ev.emit("succeeded", message="; ".join(head)[:900])
        print("\n".join(head))
    except Exception as e:
        ev.emit("failed", message=f"{type(e).__name__}: {e}"[:500])
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
