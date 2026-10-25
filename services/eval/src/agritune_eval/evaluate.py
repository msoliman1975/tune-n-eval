"""Compares a fine-tuned model with its base on a dataset's heldout splits; writes an EvalReport.

    bash run.sh evaluate --model-ref ../../artifacts/train/<run>/model_ref.json \
                         --dataset ../../artifacts/data/datasets/mango-sft/v0.1          # MCQ only, judge estimate
    bash run.sh evaluate ... --yes                                                     # also judge open answers

Suites:
- mango_mcq@v1: accuracy, letter with the highest next-token probability, averaged over the 4
  rotations of the choices so position bias cancels out.
- mango_advice@v1: open answers graded by Claude against the reference (rubrics/mango_advice_v1.md).
Every score has a 95% bootstrap interval; fine-tuned vs base is a paired difference on the same items.
Writes artifacts/eval/{eval_id}/: answers.jsonl, judgments.jsonl, report.json.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from . import ARTIFACTS
from .judge import CRITERIA, Judge
from .stats import mean_ci, paired_delta

SUITE_MCQ, SUITE_ADVICE = "mango_mcq@v1", "mango_advice@v1"
DECODING = {"do_sample": False, "max_new_tokens": 400, "thinking": False}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_items(ds_dir: Path, splits: list[str], open_limit: int, mcq_limit: int) -> tuple[list[dict], dict]:
    ref = json.loads((ds_dir / "dataset_ref.json").read_text(encoding="utf-8"))
    rng = random.Random(7)
    items = []
    for split in splits:
        if split not in ref["splits"]:
            continue
        rows = pq.read_table(ARTIFACTS.parent / ref["splits"][split]).to_pylist()
        for typ, limit in (("open_qa", open_limit), ("mcq", mcq_limit)):
            group = sorted((r for r in rows if r["type"] == typ), key=lambda r: r["id"])
            group = rng.sample(group, limit) if limit and len(group) > limit else group
            items += [r | {"split": split} for r in group]
    return items, ref


def score_rows(models: list[str], items: list[dict], per_model: dict) -> tuple[list, list]:
    """Scores per model x suite x slice, and paired comparisons of the last model vs the first."""
    slices = defaultdict(list)  # (suite, split, lang) -> item ids
    for it in items:
        suite = SUITE_MCQ if it["type"] == "mcq" else SUITE_ADVICE
        for lang in (it["lang"], "all"):
            slices[(suite, it["split"], lang)].append(it["id"])
    scores, comps = [], []
    for (suite, split, lang), ids in sorted(slices.items()):
        metric = "accuracy" if suite == SUITE_MCQ else "judge_mean"
        vals = {}
        for m in models:
            v = [per_model[m][i] for i in ids if i in per_model[m]]
            if not v:
                continue
            vals[m] = v
            value, ci = mean_ci(v)
            scores.append({"model": m, "suite": suite, "metric": metric, "slice": {"split": split, "lang": lang},
                           "value": round(value, 4), "ci95": [round(x, 4) for x in ci], "n": len(v)})
        a, b = models[0], models[-1]
        if a in vals and b in vals and len(vals[a]) == len(vals[b]):
            d = paired_delta(vals[a], vals[b])
            comps.append({"a": a, "b": b, "suite": suite, "slice": {"split": split, "lang": lang},
                          "delta": round(d["delta"], 4), "ci95": [round(x, 4) for x in d["ci95"]],
                          "verdict": d["verdict"]})
    return scores, comps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-ref", required=True, help="model_ref.json of a fine-tuned run (hf_adapter)")
    ap.add_argument("--dataset", required=True, help="dataset version folder holding dataset_ref.json")
    ap.add_argument("--splits", nargs="+", default=["heldout", "heldout_unseen"])
    ap.add_argument("--open-limit", type=int, default=200, help="open questions per split (0 = all)")
    ap.add_argument("--mcq-limit", type=int, default=0, help="MCQ per split (0 = all)")
    ap.add_argument("--judge-model", default="claude-opus-5-5")
    ap.add_argument("--yes", action="store_true", help="call the Claude judge")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-id", help="set by the API, which creates the folder first")
    a = ap.parse_args()

    ft = json.loads(Path(a.model_ref).read_text(encoding="utf-8"))
    base = {"id": f"base:{ft['base']}", "kind": "hf_base", "base": ft["base"], "chat_template": ft["chat_template"]}
    models = [base["id"], ft["id"]]
    items, ref = load_items(Path(a.dataset).resolve(), a.splits, a.open_limit, a.mcq_limit)
    open_items = [it for it in items if it["type"] == "open_qa"]
    mcq_items = [it for it in items if it["type"] == "mcq"]
    eval_id = a.eval_id or f"eval-{time.strftime('%Y%m%d-%H%M%S')}"
    out = ARTIFACTS / "eval" / eval_id
    out.mkdir(parents=True, exist_ok=bool(a.eval_id))
    print(f"{eval_id}: {len(mcq_items)} MCQ + {len(open_items)} open items x 2 models "
          f"({ref['id']}@{ref['version']}, splits {', '.join(a.splits)})")

    from .targets import HFTarget

    target = HFTarget(ft["base"], str(ARTIFACTS.parent / ft["uri"]))
    answers: dict[str, dict[str, str]] = {}
    mcq_ok: dict[str, dict[str, float]] = {}
    for m, adapter in ((base["id"], False), (ft["id"], True)):
        t0 = time.time()
        with target.use(adapter):
            picks = target.mcq(mcq_items, batch=a.batch)
            gens = target.generate([it["question"] for it in open_items], DECODING["max_new_tokens"], a.batch)
        # credit = share of the 4 choice rotations answered correctly
        mcq_ok[m] = {it["id"]: sum(x == it["answer"] for x in p) / 4 for it, p in zip(mcq_items, picks)}
        answers[m] = {it["id"]: g for it, g in zip(open_items, gens)}
        with (out / "answers.jsonl").open("a", encoding="utf-8") as f:
            for it, p in zip(mcq_items, picks):
                f.write(json.dumps({"model": m, "item": it["id"], "type": "mcq", "picks": p, "correct": it["answer"]}) + "\n")
            for it, g in zip(open_items, gens):
                f.write(json.dumps({"model": m, "item": it["id"], "type": "open_qa", "answer": g}, ensure_ascii=False) + "\n")
        acc = sum(mcq_ok[m].values()) / max(len(mcq_ok[m]), 1)
        print(f"{m}: MCQ accuracy {acc:.3f}, {len(gens)} open answers in {time.time() - t0:.0f}s", flush=True)
    del target
    gc.collect()

    judge = Judge(model=a.judge_model)
    triples = [(it["question"], it["answer"], answers[m][it["id"]]) for m in models for it in open_items]
    per_model: dict[str, dict[str, float]] = {m: dict(mcq_ok[m]) for m in models}
    judged = False
    if open_items and not a.yes:
        print(judge.estimate(judge.pending(triples)) + ". Re-run with --yes to judge; this report has MCQ only.")
    elif open_items:
        async def run() -> list[dict]:
            return await asyncio.gather(*(judge.grade(*t) for t in triples))

        verdicts = asyncio.run(run())
        judged = True
        k = 0
        with (out / "judgments.jsonl").open("w", encoding="utf-8") as f:
            for m in models:
                for it in open_items:
                    v = verdicts[k]; k += 1
                    per_model[m][it["id"]] = sum(v[c] for c in CRITERIA) / len(CRITERIA)
                    f.write(json.dumps({"model": m, "item": it["id"]} | v, ensure_ascii=False) + "\n")
        print(f"judge spent about ${judge.cost():.2f}")

    scores, comps = score_rows(models, items, per_model)
    report = {
        "id": eval_id, "models": [base, ft], "suites": [SUITE_MCQ] + ([SUITE_ADVICE] if judged else []),
        "dataset": f"{ref['id']}@{ref['version']}", "decoding": DECODING, "scores": scores, "comparisons": comps,
        "contamination": {"status": "not_checked"},
        "judge": {"model": a.judge_model, "rubric": "mango_advice_v1"} if judged else {},
        "created_at": now(), "workspace_id": "default",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'suite':16} {'split':15} {'lang':4} {'base':>7} {'tuned':>7} {'delta [95% CI]':>24}  verdict")
    by = {(s["model"], s["suite"], s["slice"]["split"], s["slice"]["lang"]): s for s in scores}
    for c in comps:
        key = (c["suite"], c["slice"]["split"], c["slice"]["lang"])
        va, vb = by[(c["a"], *key)]["value"], by[(c["b"], *key)]["value"]
        print(f"{key[0]:16} {key[1]:15} {key[2]:4} {va:7.3f} {vb:7.3f} "
              f"{c['delta']:+7.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  {c['verdict']}")
    print(f"\nreport -> {out / 'report.json'}")


if __name__ == "__main__":
    main()
