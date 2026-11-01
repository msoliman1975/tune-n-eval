"""Eval service REST API.

    bash run.sh api            # http://127.0.0.1:8103, docs at /docs

- Generic evaluations (run.py): N registry models x a frozen dataset x chosen metrics.
- Mango evaluations (evaluate.py): a fine-tuned run vs its base; requests that carry `model_ref`.
- Playground: POST /generate answers one question with several models and runs the SQL.

Evaluations run one at a time as subprocesses under the machine-wide GPU lock that training also
takes. State lives on disk: artifacts/eval/{id}/job.json, events.jsonl, then report.json.
"""

from __future__ import annotations

import fcntl
import json
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from agritune_core.sqlexec import execute, extract_sql
from agritune_core.tasks import task_of
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import ARTIFACTS, REPO, SERVICE_DIR
from .metrics import APPROACHES, METRICS, catalog, preview
from .run import PRICES, REGISTRY_URL

EVALS = ARTIFACTS / "eval"
GPU_LOCK = Path("/tmp/agritune-gpu.lock")
TERMINAL = {"succeeded", "failed", "cancelled"}

app = FastAPI(title="agritune-eval", version="0.2.0")
jobs: "queue.Queue[tuple[str, str, list[str]]]" = queue.Queue()
current: dict[str, subprocess.Popen] = {}


def set_job(eval_id: str, **kw) -> None:
    path = EVALS / eval_id / "job.json"
    job = json.loads(path.read_text()) if path.exists() else {"id": eval_id}
    path.write_text(json.dumps(job | kw | {"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}))


def job(eval_id: str) -> dict:
    p = EVALS / eval_id / "job.json"
    return json.loads(p.read_text()) if p.exists() else {}


def worker() -> None:
    while True:
        eval_id, module, argv = jobs.get()
        if job(eval_id).get("status") == "cancelled":
            continue
        with GPU_LOCK.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)  # waits while a training run holds the GPU
            set_job(eval_id, status="running")
            with (EVALS / eval_id / "stdout.log").open("w") as out:
                proc = subprocess.Popen([sys.executable, "-m", module, *argv], cwd=SERVICE_DIR, stdout=out,
                                        stderr=subprocess.STDOUT)
                current[eval_id] = proc
                code = proc.wait()
                current.pop(eval_id, None)
        if job(eval_id).get("status") != "cancelled":
            set_job(eval_id, status="succeeded" if code == 0 else "failed", exit_code=code)


def recover() -> None:
    """Evaluations a restart of this service interrupted (the queue lives in memory, and the running job
    dies with the service). Generic ones go back in the queue and resume from their cached answers;
    mango ones have no answer cache and are marked failed."""
    for p in sorted(EVALS.glob("eval-*/job.json")) if EVALS.exists() else []:
        j = json.loads(p.read_text())
        d = p.parent
        if j.get("status") not in ("queued", "running") or (d / "report.json").exists():
            continue
        if j.get("kind") == "generic" and (d / "spec.json").exists():
            set_job(d.name, status="queued", resumed=j.get("resumed", 0) + 1)
            jobs.put((d.name, "agritune_eval.run", ["--spec", str(d / "spec.json")]))
        else:
            set_job(d.name, status="failed", error="interrupted by a restart of the eval service; run it again")


threading.Thread(target=worker, daemon=True).start()


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO / path


def registry_model(model_id: str) -> dict:
    try:
        r = httpx.get(f"{REGISTRY_URL}/models/{model_id}", timeout=5)
    except httpx.HTTPError:
        raise HTTPException(503, "the model registry is not running")
    if r.status_code == 404:
        raise HTTPException(404, f"no model {model_id} in the registry")
    r.raise_for_status()
    return r.json()


def dataset_ref(path: str) -> dict:
    p = resolve(path) / "dataset_ref.json"
    if not p.exists():
        raise HTTPException(404, f"no dataset_ref.json in {path}")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------- catalog ----------

@app.get("/metrics")
def metrics(task: str | None = None) -> dict:
    from agritune_core.tasks import load_task

    spec = load_task(task).metrics if task else None
    return {"approaches": APPROACHES, "metrics": catalog(spec)}


@app.get("/suites")
def suites() -> list[dict]:
    """The mango suites (evaluate.py)."""
    return [{"id": "mango_mcq@v1", "metric": "accuracy", "scale": [0, 1]},
            {"id": "mango_ref@v1", "metric": "chrf", "scale": [0, 1]},
            {"id": "mango_advice@v1", "metric": "judge_mean", "scale": [1, 5], "rubric": "mango_advice_v1"}]


# ---------- generic evaluations ----------

class EvalSpec(BaseModel):
    models: list[str] = Field(min_length=1)
    dataset: str  # dataset version folder, relative to the repo
    splits: list[str] = ["test"]
    limit: int = Field(0, ge=0)  # items to sample (0 = all); the same sample for every model
    seed: int = 7
    metrics: list[str] = []  # empty = the task's defaults
    judge_model: str = "qwen3:14b"
    batch: int = 4  # Gemma 4 E4B keeps its per-layer embeddings in bf16: ~10 GB at batch 3-4
    max_new_tokens: int | None = None
    confirm_paid: bool = False  # required when a paid API model is included


def count_items(ref: dict, splits: list[str], limit: int) -> int:
    import pyarrow.parquet as pq

    task = task_of(ref)
    n = 0
    for s in splits:
        if s not in ref["splits"]:
            raise HTTPException(400, f"{ref['id']}@{ref['version']} has no split {s}")
        rows = pq.read_table(REPO / ref["splits"][s]).to_pylist() if task.row_filter else None
        n += sum(task.keep(r) for r in rows) if rows is not None else ref["counts"].get(s, 0)
    return min(n, limit) if limit else n


def plan(spec: EvalSpec) -> tuple[dict, list[dict], list[str], dict]:
    ref = dataset_ref(spec.dataset)
    task = task_of(ref)
    allowed = task.metrics.get("available", list(METRICS))
    chosen = spec.metrics or task.metrics.get("default", [])
    bad = [m for m in chosen if m not in allowed]
    if bad:
        raise HTTPException(400, f"{', '.join(bad)} do not apply to {task.ref} (choose from {', '.join(allowed)})")
    models = [registry_model(m) for m in dict.fromkeys(spec.models)]
    n = count_items(ref, spec.splits, spec.limit)
    est = {"n_items": n, "task": task.ref, "models": {}}
    chars = 6000 if task.attachments else 800  # rough prompt size: schema-heavy SQL prompts vs plain questions
    for m in models:
        e = {"kind": m["kind"], "paid": m["paid"]}
        if m["paid"]:
            tin, tout = n * chars / 3.2, n * 1200  # output includes default thinking
            price = PRICES.get(m.get("api_model") or "")
            e |= {"est_input_tokens": int(tin), "est_output_tokens": int(tout),
                  "est_usd": round(tin / 1e6 * price[0] + tout / 1e6 * price[1], 2) if price else None}
        est["models"][m["id"]] = e
    return ref, models, chosen, est


@app.post("/evaluations/estimate")
def estimate(spec: EvalSpec) -> dict:
    return plan(spec)[3]


@app.post("/evaluations", status_code=202)
def create(body: dict) -> dict:
    if "model_ref" in body:
        return create_mango(MangoRequest(**body))
    spec = EvalSpec(**body)
    ref, models, chosen, est = plan(spec)
    paid = [m["id"] for m in models if m["paid"]]
    if paid and not spec.confirm_paid:
        cost = sum(e.get("est_usd") or 0 for e in est["models"].values())
        raise HTTPException(402, f"{', '.join(paid)} cost money (about ${cost:.2f} for {est['n_items']} items); "
                                 "send confirm_paid: true to run them")
    eval_id = f"eval-{time.strftime('%Y%m%d-%H%M%S')}"
    out = EVALS / eval_id
    out.mkdir(parents=True)
    full = spec.model_dump() | {"id": eval_id, "models": models, "metrics": chosen}
    (out / "spec.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    set_job(eval_id, status="queued", kind="generic", request=spec.model_dump(), estimate=est)
    jobs.put((eval_id, "agritune_eval.run", ["--spec", str(out / "spec.json")]))
    return {"id": eval_id, "status": "queued", "estimate": est}


@app.post("/evaluations/{eval_id}/cancel")
def cancel(eval_id: str) -> dict:
    if not (EVALS / eval_id).exists():
        raise HTTPException(404, "no such evaluation")
    set_job(eval_id, status="cancelled")
    if proc := current.get(eval_id):
        proc.terminate()
    return {"id": eval_id, "status": "cancelled"}


def events(eval_id: str) -> list[dict]:
    p = EVALS / eval_id / "events.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def progress(eval_id: str) -> dict:
    ev = events(eval_id)
    latest = {}
    for e in ev:
        if e["type"] == "progress":
            latest[(e.get("stage"), e.get("model") or e.get("message"))] = e
    logs = [e["message"] for e in ev if e["type"] in ("log", "started", "failed") and e.get("message")]
    return {"stages": list(latest.values()), "log": logs[-12:]}


@app.get("/evaluations")
def list_evals() -> list[dict]:
    if not EVALS.exists():
        return []
    out = []
    for d in sorted((p for p in EVALS.iterdir() if p.is_dir() and p.name.startswith("eval-")), reverse=True):
        j = job(d.name)
        rep = json.loads((d / "report.json").read_text(encoding="utf-8")) if (d / "report.json").exists() else None
        spec = json.loads((d / "spec.json").read_text()) if (d / "spec.json").exists() else None
        status = j.get("status", "succeeded" if rep else "unknown")
        row = {"id": d.name, "status": "succeeded" if rep and status != "cancelled" else status,
               "kind": "generic" if (spec or (rep and rep.get("version") == 2)) else "mango",
               "created_at": (rep or {}).get("created_at") or j.get("updated_at")}
        if rep and rep.get("version") == 2:
            primary = rep["metrics"][0]
            row |= {"dataset": rep["dataset"], "task": rep["task"], "models": [m["id"] for m in rep["models"]],
                    "n_items": rep["n_items"], "primary_metric": primary,
                    "headline": {s["model"]: s["value"] for s in rep["scores"]
                                 if s["metric"] == primary and s["slice"]["key"] == "all"}}
        elif rep:
            row |= {"dataset": rep["dataset"], "models": [m["id"] for m in rep["models"]]}
        elif spec:
            row |= {"dataset": spec["dataset"], "models": [m["id"] for m in spec["models"]]}
        out.append(row)
    return out


@app.get("/evaluations/{eval_id}")
def get_eval(eval_id: str) -> dict:
    d = EVALS / eval_id
    if not d.exists():
        raise HTTPException(404, "no such evaluation")
    if (d / "report.json").exists():
        return json.loads((d / "report.json").read_text(encoding="utf-8")) | {"status": "succeeded"}
    j = job(eval_id)
    tail = (d / "stdout.log").read_text(errors="replace").splitlines()[-8:] if (d / "stdout.log").exists() else []
    spec = json.loads((d / "spec.json").read_text()) if (d / "spec.json").exists() else None
    return {"id": eval_id, "status": j.get("status", "unknown"), "log_tail": tail, "estimate": j.get("estimate"),
            "spec": spec, "progress": progress(eval_id)}


def load_results(eval_id: str) -> tuple[dict, list[dict], dict]:
    d = EVALS / eval_id
    if not (d / "report.json").exists():
        raise HTTPException(409, "the evaluation has not finished")
    rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
    if rep.get("version") != 2:
        raise HTTPException(400, "a mango evaluation: use /examples")
    items = [json.loads(l) for l in (d / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    res: dict[str, dict] = {}
    for line in (d / "results.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        res.setdefault(r["item"], {})[r["model"]] = r
    return rep, items, res


PASS_AT = {"rate": 1.0, "score": 0.5, "scale": 4.0}  # what counts as a pass when browsing answers


def passed(metric: str, v) -> bool | None:
    kind = METRICS[metric].kind
    if not isinstance(v, (int, float)) or kind not in PASS_AT:
        return None
    return v >= PASS_AT[kind]


@app.get("/evaluations/{eval_id}/items")
def eval_items(eval_id: str, model: str | None = None, metric: str | None = None, outcome: str = "all",
               slice: str | None = None, q: str | None = None, offset: int = 0, limit: int = 25) -> dict:
    """Every item with every model's answer and scores. outcome: all | fail | pass | disagree, judged on
    `metric` (default: the first metric) for `model` (fail/pass) or across models (disagree)."""
    rep, items, res = load_results(eval_id)
    metric = metric or rep["metrics"][0]
    rows = []
    for it in items:
        per = res.get(it["id"], {})
        if slice:
            k, _, v = slice.partition(":")
            if it["slices"].get(k) != v:
                continue
        if q and q.lower() not in json.dumps([it["inputs"], it["reference"]] + [r["output"] for r in per.values()],
                                             ensure_ascii=False, default=str).lower():
            continue
        vals = {m: r["scores"].get(metric) for m, r in per.items()}
        if outcome in ("fail", "pass"):
            ok = passed(metric, vals.get(model or rep["models"][0]["id"]))
            if ok is None or ok != (outcome == "pass"):
                continue
        elif outcome == "disagree" and len({v for v in vals.values() if v is not None}) < 2:
            continue
        rows.append({k: v for k, v in it.items() if k != "messages"} | {"results": per})  # prompts: per item
    return {"total": len(rows), "metric": metric, "items": rows[offset:offset + min(limit, 200)]}


@app.get("/evaluations/{eval_id}/items/{item_id}")
def eval_item(eval_id: str, item_id: str) -> dict:
    """One item in full: the exact prompt every model saw, the reference and every result.

    `messages` are the chat messages the evaluation sent (stored since 2026-10-26; older evaluations
    re-render them from the task and dataset, flagged `prompt_source: "re-rendered"`). `as_sent` is
    each model's form of them: the chat-templated string for local models, the request body for
    Ollama and the APIs."""
    rep, items, res = load_results(eval_id)
    it = next((x for x in items if x["id"] == item_id), None)
    if not it:
        raise HTTPException(404, "no such item")
    messages, source = it.get("messages"), "stored"
    if messages is None:
        ref = dataset_ref(rep["dataset_path"])
        task = task_of(ref)
        root = REPO / ref["attachments"] if ref.get("attachments") else None
        import pyarrow.parquet as pq

        row = next((r for r in pq.read_table(REPO / ref["splits"][it["split"]]).to_pylist() if r["id"] == item_id), None)
        messages, source = (task.messages(task.with_context(row, root)) if row else None), "re-rendered"
    as_sent = {}
    if messages:
        from .targets import Gen
        from .targets import as_sent as render_for

        task = task_of(dataset_ref(rep["dataset_path"]))
        max_new = it.get("max_new_tokens") or task.decoding.get("max_new_tokens", 400)
        gen = Gen(json_schema=task.json_schema, stream=bool(task.decoding.get("stream")))
        for m in rep["models"]:
            try:  # a model with a prompt_tier saw that variant of the prompt, when the item has one
                own = (it.get("tier_messages") or {}).get(m.get("prompt_tier") or "", messages)
                as_sent[m["id"]] = render_for(m, own, max_new, REPO, gen)
            except Exception as e:  # e.g. a tokenizer no longer on disk
                as_sent[m["id"]] = {"format": "error", "label": f"{type(e).__name__}: {e}"[:300]}
    return it | {"messages": messages, "prompt_source": source, "as_sent": as_sent, "results": res.get(item_id, {})}


# ---------- playground ----------

class GenerateRequest(BaseModel):
    models: list[str] = Field(min_length=1, max_length=6)
    dataset: str  # supplies the task and the attachments (databases)
    inputs: dict  # e.g. {"question": "...", "db_id": "concert_singer"}
    confirm_paid: bool = False


def gpu_busy() -> bool:
    with GPU_LOCK.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    """Answers one input with each model, using the task prompt exactly as in training and evaluation.
    Local models answer through Ollama, so an HF model needs an Ollama export first."""
    from .targets import AnthropicTarget, GeminiTarget, OllamaTarget

    ref = dataset_ref(req.dataset)
    task = task_of(ref)
    missing = [k for k in task.required if not req.inputs.get(k)]
    if missing:
        raise HTTPException(400, f"missing {', '.join(missing)}")
    root = REPO / ref["attachments"] if ref.get("attachments") else None
    try:
        row = task.with_context(dict(req.inputs), root)
    except FileNotFoundError:
        raise HTTPException(404, "no such database in this dataset")
    messages = task.messages(row)
    models = [registry_model(m) for m in req.models]
    if any(m["paid"] for m in models) and not req.confirm_paid:
        raise HTTPException(402, "a paid API model is selected; confirm to call it")
    local = [m for m in models if m["kind"] != "api"]
    if local and gpu_busy():
        raise HTTPException(409, "the GPU is busy with a training run or an evaluation; try again when it finishes")
    sql = bool(task.attachments and task.attachments["type"] == "sqlite_dbs")
    max_new = task.decoding.get("max_new_tokens", 400)
    from .targets import Gen

    gen = Gen(json_schema=task.json_schema, stream=bool(task.decoding.get("stream")))
    out = []
    for m in models:
        r = {"model": m["id"], "name": m["name"]}
        try:
            own = task.messages(row, m.get("prompt_tier")) if task.messages_field else messages
            if m["kind"] == "api":
                t = (AnthropicTarget if m["provider"] == "anthropic" else GeminiTarget)(m["api_model"], m.get("options"))
            elif m.get("ollama_tag"):
                t = OllamaTarget(m["ollama_tag"])
            else:
                raise RuntimeError("not servable: export this model to Ollama first (Training -> Export)")
            a = t.one(own, max_new, gen)
            r |= {"output": a.text, "latency_s": round(a.latency_s, 2), "input_tokens": a.input_tokens,
                  "output_tokens": a.output_tokens, "error": a.error,
                  "ttft_s": None if a.ttft_s is None else round(a.ttft_s, 2)}
            if sql and not a.error:
                q = extract_sql(a.text)
                r |= {"sql": q, "result": preview(execute(task.attachment(root, row), q, timeout_s=10), 50)}
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"[:300]
        out.append(r)
    return {"task": task.ref, "messages": messages, "answers": out}


# ---------- mango evaluations (evaluate.py) ----------

class MangoRequest(BaseModel):
    model_ref: str  # path to a run's model_ref.json, relative to the repo or absolute
    dataset: str  # dataset version folder
    judge: bool = False  # grade open answers with the local judge; MCQ and chrF++ run either way
    open_limit: int = 200


def create_mango(req: MangoRequest) -> dict:
    if not resolve(req.model_ref).exists():
        raise HTTPException(404, f"no model ref at {req.model_ref}")
    if not (resolve(req.dataset) / "dataset_ref.json").exists():
        raise HTTPException(404, f"no dataset_ref.json in {req.dataset}")
    eval_id = f"eval-{time.strftime('%Y%m%d-%H%M%S')}"
    (EVALS / eval_id).mkdir(parents=True)
    set_job(eval_id, status="queued", kind="mango", request=req.model_dump())
    argv = ["--eval-id", eval_id, "--model-ref", str(resolve(req.model_ref)), "--dataset", str(resolve(req.dataset)),
            "--open-limit", str(req.open_limit)] + (["--yes"] if req.judge else [])
    jobs.put((eval_id, "agritune_eval.evaluate", argv))
    return {"id": eval_id, "status": "queued"}


@app.get("/evaluations/{eval_id}/examples")
def examples(eval_id: str, limit: int = 50) -> list[dict]:
    """Mango evaluations: open items with both models' answers and verdicts, biggest changes first."""
    d = EVALS / eval_id
    if not (d / "judgments.jsonl").exists():
        return []
    rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
    base, tuned = rep["models"][0]["id"], rep["models"][-1]["id"]
    ans = {(r["model"], r["item"]): r for r in map(json.loads, (d / "answers.jsonl").read_text(encoding="utf-8").splitlines())}
    ver = {(r["model"], r["item"]): r for r in map(json.loads, (d / "judgments.jsonl").read_text(encoding="utf-8").splitlines())}
    score = lambda v: sum(v[c] for c in ("correctness", "completeness", "actionability", "safety", "language")) / 5  # noqa: E731
    rows = []
    for (m, item), v in ver.items():
        if m != base or (tuned, item) not in ver:
            continue
        vt = ver[(tuned, item)]
        rows.append({"item": item, "delta": round(score(vt) - score(v), 2),
                     "base": {"answer": ans[(base, item)]["answer"], "verdict": v},
                     "tuned": {"answer": ans[(tuned, item)]["answer"], "verdict": vt}})
    rows.sort(key=lambda r: -abs(r["delta"]))
    return rows[:limit]


def main() -> None:
    import uvicorn

    recover()  # here, not on import: tests import the app against the real artifacts folder
    uvicorn.run(app, host="127.0.0.1", port=8103)


if __name__ == "__main__":
    main()
