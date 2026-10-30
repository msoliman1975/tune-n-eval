"""Train service REST API.

    bash run.sh api            # http://127.0.0.1:8102, docs at /docs

Jobs (training runs, Ollama exports) run one at a time as subprocesses under a machine-wide GPU
lock that the Eval service also takes, so training and evaluation never share the GPU. Run state
lives on disk (artifacts/train/{run}/events.jsonl), so it survives an API restart. A run names a
trainable registry model (`model`, e.g. gemma-4-e4b) or, for older scripts, a `preset`.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .presets import PRESETS
from .sft import ARTIFACTS, REGISTRY_URL, RUNS, EventLog

GPU_LOCK = Path("/tmp/agritune-gpu.lock")
SERVICE_DIR = Path(__file__).resolve().parents[2]
TERMINAL = {"succeeded", "failed", "cancelled"}
GEN_CACHE = ARTIFACTS / "eval" / "gen_cache"  # the Eval service's cached answers, one file per model id
OLLAMA_URL = "http://127.0.0.1:11434"
MLFLOW_URL = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

app = FastAPI(title="agritune-train", version="0.2.0")
jobs: "queue.Queue[tuple[str, str, list[str]]]" = queue.Queue()
current: dict[str, subprocess.Popen] = {}


def worker() -> None:
    while True:
        run_id, module, argv = jobs.get()
        log = EventLog(RUNS / run_id / "events.jsonl", run_id)
        if module == "agritune_train.sft" and (RUNS / run_id / "cancel").exists():
            log.emit("cancelled", message="cancelled before start")
            continue
        logname = "stdout.log" if module == "agritune_train.sft" else "export.log"
        with GPU_LOCK.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)  # waits while an evaluation holds the GPU
            with (RUNS / run_id / logname).open("w") as out:
                proc = subprocess.Popen([sys.executable, "-m", module, *argv],
                                        cwd=SERVICE_DIR, stdout=out, stderr=subprocess.STDOUT)
                current[run_id] = proc
                code = proc.wait()
                current.pop(run_id, None)
        if module == "agritune_train.export":
            status = "succeeded" if code == 0 else "failed"
            (RUNS / run_id / "export.json").write_text(json.dumps({"status": status, "exit_code": code,
                                                                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}))
            continue
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
    exp = RUNS / run_id / "export.json"
    export = json.loads(exp.read_text()) if exp.exists() else None
    ollama = sorted((RUNS / run_id).glob("model_ref.*.json"))
    if ollama:
        export = (export or {}) | {"status": "succeeded", "ollama_tag": json.loads(ollama[0].read_text()).get("ollama_tag")}
    return {
        "id": run_id, "status": status, "created_at": ev[0]["ts"] if ev else None,
        "message": ev[0].get("message") if ev else None,
        "step": metrics[-1].get("step") if metrics else 0, "total": metrics[-1].get("total") if metrics else None,
        "result": done.get("metrics") if done else None,
        "model_ref": json.loads(ref_path.read_text()) if ref_path.exists() else None,
        "export": export, "logs": [e["message"] for e in ev if e["type"] == "log"][-5:],
    }


class RunRequest(BaseModel):
    model: str | None = None  # registry id of a trainable base model
    preset: str | None = None  # alternative: a preset name
    dataset: str  # dataset version folder, relative to the repo or absolute
    epochs: float = 2
    lr: float = 2e-4
    rank: int = 16
    max_steps: int = -1
    train_fraction: float = Field(1.0, gt=0, le=1)  # share of the train split, e.g. 0.3
    name: str | None = Field(None, max_length=120)  # the fine-tune's registry name; default from base, data, date
    tags: list[str] = []  # registry tags, added to "fine-tuned" and the dataset id


def preset_of(req: RunRequest) -> str:
    if req.model:
        try:
            r = httpx.get(f"{REGISTRY_URL}/models/{req.model}", timeout=5)
        except httpx.HTTPError:
            raise HTTPException(503, "the model registry is not running")
        if r.status_code == 404:
            raise HTTPException(404, f"no model {req.model} in the registry")
        preset = r.json().get("train_preset")
        if not preset:
            raise HTTPException(400, f"{req.model} cannot be fine-tuned here (baseline-only)")
        return preset
    if req.preset not in PRESETS:
        raise HTTPException(404, f"unknown preset {req.preset}")
    return req.preset


@app.get("/presets")
def get_presets() -> dict:
    return PRESETS


def resolve(dataset: str) -> Path:
    path = Path(dataset)
    path = path if path.is_absolute() else ARTIFACTS.parent / path
    if not (path / "dataset_ref.json").exists():
        raise HTTPException(404, f"no dataset_ref.json in {dataset}")
    return path


@app.post("/runs/estimate")
def estimate(req: RunRequest) -> dict:
    """VRAM from the M0 benchmark, duration from the dataset's measured prompt sizes."""
    name = preset_of(req)
    p = PRESETS[name]
    ref = json.loads((resolve(req.dataset) / "dataset_ref.json").read_text())
    total = ref["counts"].get("train", 0)
    rows = max(1, round(total * req.train_fraction)) if total else 0  # sft stratifies, so this is approximate
    chars = (ref.get("stats") or {}).get("avg_chars", {}).get("train")
    tokens = chars / 3.2 if chars else 350  # ~350 tokens per mango Q&A including the template
    est = None
    if p["tokens_per_s"]:
        secs = rows * tokens * req.epochs / p["tokens_per_s"]
        if req.max_steps > 0:  # a packed step is seq tokens per micro-batch x grad accumulation (4)
            secs = min(secs, req.max_steps * p["seq"] * 4 / p["tokens_per_s"])
        est = max(1, round(secs / 60))
    fits = None if p["measured_vram_gb"] is None else p["measured_vram_gb"] < 11.0
    return {"preset": name, "vram_gb": p["measured_vram_gb"], "vram_total_gb": 11.9, "fits": fits,
            "train_rows": rows, "total_train_rows": total, "avg_tokens": round(tokens), "est_minutes": est}


@app.post("/runs", status_code=202)
def create_run(req: RunRequest) -> dict:
    name = preset_of(req)
    ds = resolve(req.dataset)
    run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{name}"
    (RUNS / run_id).mkdir(parents=True)
    if req.name or req.tags:  # read by the registry when it registers the finished run
        (RUNS / run_id / "label.json").write_text(json.dumps({"name": (req.name or "").strip() or None, "tags": req.tags}))
    share =f" ({req.train_fraction:.0%} of train)" if req.train_fraction < 1 else ""
    EventLog(RUNS / run_id / "events.jsonl", run_id).emit("queued", message=f"{req.model or name} on {ds.parent.name}@{ds.name}{share}")
    who = ["--model", req.model] if req.model else ["--preset", name]
    argv = ["--run-id", run_id, *who, "--dataset", str(ds), "--epochs", str(req.epochs),
            "--lr", str(req.lr), "--rank", str(req.rank), "--max-steps", str(req.max_steps),
            "--train-fraction", str(req.train_fraction)]
    jobs.put((run_id, "agritune_train.sft", argv))
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


@app.post("/runs/{run_id}/export", status_code=202)
def export(run_id: str, quant: str = "q4_k_m") -> dict:
    """Merges the adapter, quantizes to GGUF and registers it with Ollama (then the playground can use it)."""
    if not (RUNS / run_id / "model_ref.json").exists():
        raise HTTPException(409, "the run has no finished adapter")
    (RUNS / run_id / "export.json").write_text(json.dumps({"status": "queued", "quant": quant}))
    jobs.put((run_id, "agritune_train.export", ["--run", run_id, "--quant", quant]))
    return {"id": run_id, "export": "queued"}


def folder_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def registry(method: str, path: str) -> httpx.Response:
    try:
        return httpx.request(method, f"{REGISTRY_URL}{path}", timeout=10)
    except httpx.HTTPError:
        raise HTTPException(503, "the model registry is not running")


def deletion_plan(run_id: str) -> dict:
    """Everything a fine-tune left behind: the run folder (adapter, checkpoints, GGUF), its Ollama
    models, its registry entry, its cached evaluation answers and its MLflow run."""
    run = RUNS / run_id
    if not run_id.startswith("run-") or not run.is_dir():
        raise HTTPException(404, "no such run")
    r = registry("GET", f"/models/{run_id}")
    entry = r.json() if r.status_code == 200 else None
    others = [m for m in registry("GET", "/models").json() if m["id"] != run_id]
    refs = [json.loads(p.read_text()) for p in sorted(run.glob("model_ref*.json"))]
    tags = {ref.get("ollama_tag") for ref in refs} | {(entry or {}).get("ollama_tag")}
    tags = sorted(t for t in tags if t and t.startswith("agritune/"))  # never a base model's tag (gemma4:e4b)
    shared = [t for t in tags if any(m.get("ollama_tag") == t for m in others)]
    lineage = (entry or {}).get("lineage") or next((ref.get("lineage", {}) for ref in refs if ref.get("kind") != "ollama"), {})
    cache = GEN_CACHE / f"{run_id}.jsonl"
    return {"run_id": run_id, "name": (entry or {}).get("name"), "status": summary(run_id)["status"],
            "folder": str(run.relative_to(ARTIFACTS.parent)), "folder_bytes": folder_bytes(run),
            "registry_entry": entry is not None, "evaluations": len((entry or {}).get("evaluations", [])),
            "ollama_tags": [t for t in tags if t not in shared], "ollama_kept": shared,
            "answer_cache_bytes": cache.stat().st_size if cache.exists() else 0,
            "mlflow_run_id": lineage.get("mlflow_run_id")}


@app.delete("/runs/{run_id}")
def delete_run(run_id: str, dry_run: bool = False) -> dict:
    """Deletes a fine-tune from the system and from disk. Evaluation reports that measured it stay
    (they are history); with dry_run=true it only says what would go."""
    plan = deletion_plan(run_id)
    if run_id in current or any(j[0] == run_id for j in list(jobs.queue)):
        raise HTTPException(409, "the run (or its export) is queued or running; cancel it first")
    if dry_run:
        return plan
    warnings = []
    for tag in plan["ollama_tags"]:
        try:
            r = httpx.request("DELETE", f"{OLLAMA_URL}/api/delete", json={"model": tag}, timeout=30)
            if r.status_code not in (200, 404):
                warnings.append(f"ollama {tag}: {r.status_code} {r.text[:200]}")
        except httpx.HTTPError as e:
            warnings.append(f"ollama {tag}: {e}")
    if plan["mlflow_run_id"]:  # soft delete: hidden from the UI, purged by `mlflow gc`
        try:
            httpx.post(f"{MLFLOW_URL}/api/2.0/mlflow/runs/delete", json={"run_id": plan["mlflow_run_id"]}, timeout=10)
        except httpx.HTTPError as e:
            warnings.append(f"mlflow: {e}")
    (GEN_CACHE / f"{run_id}.jsonl").unlink(missing_ok=True)
    shutil.rmtree(RUNS / run_id)  # first, so a registry sync cannot register the run again
    if plan["registry_entry"]:
        r = registry("DELETE", f"/models/{run_id}")
        if r.status_code not in (204, 404):
            warnings.append(f"registry: {r.status_code} {r.text[:200]}")
    return plan | {"deleted": True, "warnings": warnings}


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
    """Model refs written by runs and exports (the Model Registry is the catalog; this is the raw files)."""
    refs = sorted(RUNS.glob("*/model_ref*.json"), reverse=True) if RUNS.exists() else []
    return [json.loads(p.read_text()) for p in refs]


def mark_interrupted() -> None:
    """A restart of this service kills the running job and empties the in-memory queue. Training has
    no checkpoint to resume from, so such runs and exports are marked failed rather than left "started"."""
    for d in sorted(RUNS.glob("run-*")) if RUNS.exists() else []:
        last = last_event(d.name)
        if last and last.get("type") not in TERMINAL:
            EventLog(d / "events.jsonl", d.name).emit("failed", message="interrupted by a restart of the train service; launch it again")
        exp = d / "export.json"
        if exp.exists() and json.loads(exp.read_text()).get("status") in ("queued", "running"):
            exp.write_text(json.dumps({"status": "failed", "error": "interrupted by a restart of the train service"}))


def main() -> None:
    import uvicorn

    mark_interrupted()  # here, not on import: tests import the app
    uvicorn.run(app, host="127.0.0.1", port=8102)


if __name__ == "__main__":
    main()
