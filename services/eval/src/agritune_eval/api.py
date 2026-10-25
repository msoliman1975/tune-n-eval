"""Eval service REST API.

    bash run.sh api            # http://127.0.0.1:8103, docs at /docs

Evaluations run one at a time as subprocesses under the same machine-wide GPU lock as training.
State lives on disk: artifacts/eval/{id}/job.json while it runs, report.json when it is done.
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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import ARTIFACTS, SERVICE_DIR

EVALS = ARTIFACTS / "eval"
GPU_LOCK = Path("/tmp/agritune-gpu.lock")

app = FastAPI(title="agritune-eval", version="0.1.0")
jobs: "queue.Queue[tuple[str, list[str]]]" = queue.Queue()


def set_job(eval_id: str, **kw) -> None:
    path = EVALS / eval_id / "job.json"
    job = json.loads(path.read_text()) if path.exists() else {"id": eval_id}
    path.write_text(json.dumps(job | kw | {"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}))


def worker() -> None:
    while True:
        eval_id, argv = jobs.get()
        with GPU_LOCK.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)  # waits while a training run holds the GPU
            set_job(eval_id, status="running")
            with (EVALS / eval_id / "stdout.log").open("w") as out:
                code = subprocess.call([sys.executable, "-m", "agritune_eval.evaluate", *argv],
                                       cwd=SERVICE_DIR, stdout=out, stderr=subprocess.STDOUT)
        set_job(eval_id, status="succeeded" if code == 0 else "failed", exit_code=code)


threading.Thread(target=worker, daemon=True).start()


class EvalRequest(BaseModel):
    model_ref: str  # path to a run's model_ref.json, relative to the repo or absolute
    dataset: str  # dataset version folder
    judge: bool = False  # call the Claude judge (costs money); MCQ runs either way
    open_limit: int = 200


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ARTIFACTS.parent / path


@app.get("/suites")
def suites() -> list[dict]:
    return [{"id": "mango_mcq@v1", "metric": "accuracy", "scale": [0, 1]},
            {"id": "mango_advice@v1", "metric": "judge_mean", "scale": [1, 5], "rubric": "mango_advice_v1"}]


@app.post("/evaluations", status_code=202)
def create(req: EvalRequest) -> dict:
    if not resolve(req.model_ref).exists():
        raise HTTPException(404, f"no model ref at {req.model_ref}")
    if not (resolve(req.dataset) / "dataset_ref.json").exists():
        raise HTTPException(404, f"no dataset_ref.json in {req.dataset}")
    eval_id = f"eval-{time.strftime('%Y%m%d-%H%M%S')}"
    (EVALS / eval_id).mkdir(parents=True)
    set_job(eval_id, status="queued", request=req.model_dump())
    argv = ["--eval-id", eval_id, "--model-ref", str(resolve(req.model_ref)), "--dataset", str(resolve(req.dataset)),
            "--open-limit", str(req.open_limit)] + (["--yes"] if req.judge else [])
    jobs.put((eval_id, argv))
    return {"id": eval_id, "status": "queued"}


@app.get("/evaluations")
def list_evals() -> list[dict]:
    if not EVALS.exists():
        return []
    out = []
    for d in sorted((p for p in EVALS.iterdir() if p.is_dir()), reverse=True):
        job = json.loads((d / "job.json").read_text()) if (d / "job.json").exists() else {}
        rep = json.loads((d / "report.json").read_text()) if (d / "report.json").exists() else None
        out.append({"id": d.name, "status": job.get("status", "succeeded" if rep else "unknown"),
                    "dataset": rep["dataset"] if rep else None, "suites": rep["suites"] if rep else None,
                    "models": [m["id"] for m in rep["models"]] if rep else None,
                    "created_at": rep["created_at"] if rep else job.get("updated_at")})
    return out


@app.get("/evaluations/{eval_id}")
def get_eval(eval_id: str) -> dict:
    d = EVALS / eval_id
    if not d.exists():
        raise HTTPException(404, "no such evaluation")
    if (d / "report.json").exists():
        return json.loads((d / "report.json").read_text(encoding="utf-8")) | {"status": "succeeded"}
    job = json.loads((d / "job.json").read_text()) if (d / "job.json").exists() else {}
    tail = (d / "stdout.log").read_text(errors="replace").splitlines()[-5:] if (d / "stdout.log").exists() else []
    return {"id": eval_id, "status": job.get("status", "unknown"), "log_tail": tail}


@app.get("/evaluations/{eval_id}/examples")
def examples(eval_id: str, limit: int = 50) -> list[dict]:
    """Open items with both models' answers and verdicts, biggest score changes first."""
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

    uvicorn.run(app, host="127.0.0.1", port=8103)


if __name__ == "__main__":
    main()
