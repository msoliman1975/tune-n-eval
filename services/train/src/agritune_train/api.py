"""Train service REST API.

    bash run.sh api            # http://127.0.0.1:8102, docs at /docs

Jobs run one at a time as subprocesses (`python -m agritune_train.sft ...`) under a machine-wide
GPU lock that the Eval service also takes, so training and evaluation never share the GPU.
Run state lives on disk (artifacts/train/{run}/events.jsonl), so it survives an API restart.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .presets import PRESETS
from .sft import ARTIFACTS, RUNS, EventLog

GPU_LOCK = Path("/tmp/agritune-gpu.lock")
SERVICE_DIR = Path(__file__).resolve().parents[2]
TERMINAL = {"succeeded", "failed", "cancelled"}

app = FastAPI(title="agritune-train", version="0.1.0")
jobs: "queue.Queue[tuple[str, list[str]]]" = queue.Queue()
current: dict[str, subprocess.Popen] = {}


def worker() -> None:
    while True:
        run_id, argv = jobs.get()
        log = EventLog(RUNS / run_id / "events.jsonl", run_id)
        if (RUNS / run_id / "cancel").exists():
            log.emit("cancelled", message="cancelled before start")
            continue
        with GPU_LOCK.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)  # waits while an evaluation holds the GPU
            with (RUNS / run_id / "stdout.log").open("w") as out:
                proc = subprocess.Popen([sys.executable, "-m", "agritune_train.sft", *argv],
                                        cwd=SERVICE_DIR, stdout=out, stderr=subprocess.STDOUT)
                current[run_id] = proc
                code = proc.wait()
                current.pop(run_id, None)
        if code != 0 and last_event(run_id).get("type") not in TERMINAL:
            cancelled = (RUNS / run_id / "cancel").exists()
            log.emit("cancelled" if cancelled else "failed", message=f"exit code {code}; see stdout.log")


threading.Thread(target=worker, daemon=True).start()


def events(run_id: str) -> list[dict]:
    path = RUNS / run_id / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def last_event(run_id: str) -> dict:
    ev = events(run_id)
    return ev[-1] if ev else {}


def summary(run_id: str) -> dict:
    ev = events(run_id)
    status = next((e["type"] for e in reversed(ev) if e["type"] in TERMINAL | {"started", "queued"}), "unknown")
    metrics = [e for e in ev if e["type"] == "metric"]
    done = next((e for e in ev if e["type"] == "succeeded"), None)
    ref_path = RUNS / run_id / "model_ref.json"
    return {
        "id": run_id, "status": status, "created_at": ev[0]["ts"] if ev else None,
        "message": ev[0].get("message") if ev else None,
        "step": metrics[-1].get("step") if metrics else 0, "total": metrics[-1].get("total") if metrics else None,
        "result": done.get("metrics") if done else None,
        "model_ref": json.loads(ref_path.read_text()) if ref_path.exists() else None,
    }


class RunRequest(BaseModel):
    preset: str
    dataset: str  # dataset version folder, relative to the repo or absolute
    epochs: float = 2
    lr: float = 2e-4
    rank: int = 16
    max_steps: int = -1


@app.get("/presets")
def get_presets() -> dict:
    return PRESETS


@app.post("/runs/estimate")
def estimate(req: RunRequest) -> dict:
    """VRAM and duration from the M0 measurements and the dataset size."""
    if req.preset not in PRESETS:
        raise HTTPException(404, f"unknown preset {req.preset}")
    p = PRESETS[req.preset]
    ref = json.loads((resolve(req.dataset) / "dataset_ref.json").read_text())
    rows = ref["counts"].get("train", 0)
    secs = rows * 350 * req.epochs / p["tokens_per_s"]  # ~350 tokens per Q&A including the template
    if req.max_steps > 0:  # a packed step is seq tokens per micro-batch x grad accumulation (4)
        secs = min(secs, req.max_steps * p["seq"] * 4 / p["tokens_per_s"])
    return {"preset": req.preset, "vram_gb": p["measured_vram_gb"], "vram_total_gb": 11.9,
            "fits": p["measured_vram_gb"] < 11.0, "train_rows": rows, "est_minutes": max(1, round(secs / 60))}


def resolve(dataset: str) -> Path:
    path = Path(dataset)
    path = path if path.is_absolute() else ARTIFACTS.parent / path
    if not (path / "dataset_ref.json").exists():
        raise HTTPException(404, f"no dataset_ref.json in {dataset}")
    return path


@app.post("/runs", status_code=202)
def create_run(req: RunRequest) -> dict:
    if req.preset not in PRESETS:
        raise HTTPException(404, f"unknown preset {req.preset}")
    ds = resolve(req.dataset)
    run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{req.preset}"
    (RUNS / run_id).mkdir(parents=True)
    EventLog(RUNS / run_id / "events.jsonl", run_id).emit("queued", message=f"{req.preset} on {ds.parent.name}@{ds.name}")
    argv = ["--run-id", run_id, "--preset", req.preset, "--dataset", str(ds), "--epochs", str(req.epochs),
            "--lr", str(req.lr), "--rank", str(req.rank), "--max-steps", str(req.max_steps)]
    jobs.put((run_id, argv))
    return {"id": run_id, "status": "queued"}


@app.get("/runs")
def list_runs() -> list[dict]:
    if not RUNS.exists():
        return []
    ids = sorted((p.name for p in RUNS.iterdir() if (p / "events.jsonl").exists()), reverse=True)
    return [summary(i) for i in ids]


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    if not (RUNS / run_id / "events.jsonl").exists():
        raise HTTPException(404, "no such run")
    return summary(run_id) | {"events": [e for e in events(run_id) if e["type"] == "metric"]}


@app.post("/runs/{run_id}/cancel")
def cancel(run_id: str) -> dict:
    if not (RUNS / run_id).exists():
        raise HTTPException(404, "no such run")
    (RUNS / run_id / "cancel").touch()
    if proc := current.get(run_id):
        proc.terminate()
    return {"id": run_id, "status": "cancelling"}


@app.get("/runs/{run_id}/events")
async def stream_events(run_id: str) -> StreamingResponse:
    """Server-sent events: every JobEvent so far, then new ones as they are written."""
    path = RUNS / run_id / "events.jsonl"
    if not path.exists():
        raise HTTPException(404, "no such run")

    async def gen():
        pos = 0
        while True:
            with path.open(encoding="utf-8") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            for line in filter(None, chunk.splitlines()):
                yield f"data: {line}\n\n"
                if json.loads(line)["type"] in TERMINAL:
                    return
            await asyncio.sleep(1)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/models")
def list_models() -> list[dict]:
    refs = sorted(RUNS.glob("*/model_ref*.json"), reverse=True) if RUNS.exists() else []
    return [json.loads(p.read_text()) for p in refs]


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8102)


if __name__ == "__main__":
    main()
