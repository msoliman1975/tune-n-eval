"""Supervised fine-tuning (QLoRA / LoRA) on a frozen dataset of any task, tracked in MLflow.

    bash run.sh sft --model gemma-4-e4b --dataset ../../artifacts/data/datasets/spider/v1.0
    bash run.sh sft --preset qwen-8b-qlora --dataset ../../artifacts/data/datasets/mango-sft/v0.1
    bash run.sh sft --model gemma-4-e4b --dataset ... --max-steps 30        # smoke test
    bash run.sh sft --model gemma-4-e4b --dataset ... --train-fraction 0.3  # 30% of the train split

The dataset's task (tasks/*.json) renders every row into the same messages Eval and the playground
use (system prompt, user template, context such as the database schema), with loss on the target
tokens only. Rows longer than the preset's sequence length are dropped, not truncated, so no
example loses its answer. Eval loss comes from the validation split (mango: heldout).
Writes artifacts/train/{run_id}/: adapter/, events.jsonl (JobEvents for the Studio), model_ref.json,
then asks the Model Registry to register the run.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from agritune_core.tasks import task_of

from .presets import PRESETS

REPO = Path(__file__).resolve().parents[4]
ARTIFACTS = REPO / "artifacts"
RUNS = ARTIFACTS / "train"
REGISTRY_URL = "http://127.0.0.1:8104"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_split(ref: dict, split: str) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(ARTIFACTS.parent / ref["splits"][split]).to_pylist()


def render(rows: list[dict], task, tok, end_of_turn: str, attachments) -> list[dict]:
    """Prompt/completion strings; TRL then computes loss on the completion only."""
    out = []
    for r in rows:
        msgs = task.messages(task.with_context(r, attachments))
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        out.append({"prompt": prompt, "completion": str(r[task.target]) + end_of_turn})
    return out


def subsample(rows: list[dict], fraction: float, seed: int, key: str | None = None) -> list[dict]:
    """A seeded `fraction` of the rows; stratified by `key` (e.g. db_id) so every group keeps its share."""
    if fraction >= 1:
        return rows
    groups: dict = {}
    for r in rows:
        groups.setdefault(r.get(key) if key else None, []).append(r)
    rng = random.Random(seed)
    return [r for g in groups.values() for r in rng.sample(g, max(1, round(len(g) * fraction)))]


def fit(rows: list[dict], tok, max_len: int) -> tuple[list[dict], int]:
    keep = [r for r in rows if len(tok(r["prompt"] + r["completion"], add_special_tokens=False)["input_ids"]) <= max_len]
    return keep, len(rows) - len(keep)


def no_allocator_warmup() -> None:
    """transformers reserves one block the size of the whole model before loading. Unsloth then moves
    Gemma 4's 5 GB per-layer embedding to the CPU, leaving a hole the allocator can never return, so
    training sees 0 GB free. Loading tensor by tensor avoids that."""
    import transformers.modeling_utils as mu

    mu.caching_allocator_warmup = lambda *a, **k: None


def registry_preset(model_id: str) -> tuple[str, dict]:
    import httpx

    r = httpx.get(f"{REGISTRY_URL}/models/{model_id}", timeout=5)
    if r.status_code == 404:
        raise SystemExit(f"no model {model_id} in the registry")
    r.raise_for_status()
    m = r.json()
    if not m.get("train_preset"):
        raise SystemExit(f"{model_id} is not trainable here (no train_preset)")
    return m["train_preset"], m


def register(run_id: str) -> None:
    import httpx

    try:
        httpx.post(f"{REGISTRY_URL}/sync", timeout=10)
    except httpx.HTTPError:
        print(f"registry unreachable; {run_id} is registered on its next start")


class EventLog:
    """Appends JobEvents (contracts/job-event.schema.json) to events.jsonl."""

    def __init__(self, path: Path, job_id: str):
        self.path, self.job_id = path, job_id

    def emit(self, type_: str, **kw) -> None:
        ev = {"job_id": self.job_id, "service": "train", "type": type_, "ts": now()} | kw
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def gpu_stats() -> dict:
    import pynvml
    import torch

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    return {"vram_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
            "gpu_temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
            "gpu_power_w": round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", help="registry id of a trainable base model, e.g. gemma-4-e4b")
    who.add_argument("--preset", choices=list(PRESETS))
    ap.add_argument("--dataset", required=True, help="dataset version folder holding dataset_ref.json")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps for a smoke test")
    ap.add_argument("--train-fraction", type=float, default=1.0,
                    help="train on this share of the train split (0-1], seeded, stratified by the task's attachment key")
    ap.add_argument("--eval-sample", type=int, default=64)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--run-id", help="set by the API, which creates the run folder first")
    a = ap.parse_args()
    if not 0 < a.train_fraction <= 1:
        ap.error("--train-fraction must be in (0, 1]")

    preset_name, base_entry = registry_preset(a.model) if a.model else (a.preset, None)
    p = PRESETS[preset_name]
    ds_dir = Path(a.dataset).resolve()
    ref = json.loads((ds_dir / "dataset_ref.json").read_text(encoding="utf-8"))
    task = task_of(ref)
    attachments = ARTIFACTS.parent / ref["attachments"] if ref.get("attachments") else None
    dataset_tag = f"{ref['id']}@{ref['version']}"
    run_id = a.run_id or f"run-{time.strftime('%Y%m%d-%H%M%S')}-{preset_name}"
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=bool(a.run_id))
    log = EventLog(out / "events.jsonl", run_id)
    log.emit("started", message=f"{preset_name} on {dataset_tag} ({task.ref})")

    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "agritune-sft")
    os.environ["MLFLOW_TAGS"] = json.dumps({"preset": preset_name, "dataset": dataset_tag, "task": task.ref,
                                            "dataset_sha256": ref["sha256"], "run_id": run_id})

    from unsloth import FastModel  # must come before trl/transformers

    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    no_allocator_warmup()
    try:
        model, tok = FastModel.from_pretrained(model_name=p["model"], max_seq_length=p["seq"],
                                               load_in_4bit=p["load_in_4bit"], full_finetuning=False,
                                               # all on the GPU: auto placement keeps ~10% headroom and
                                               # counts Gemma 4's per-layer embedding, spilling onto the CPU
                                               device_map={"": 0})
        model = FastModel.get_peft_model(
            model, r=a.rank, lora_alpha=a.rank, lora_dropout=0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth", random_state=a.seed,
        )
        tok = getattr(tok, "tokenizer", tok)  # Gemma returns a processor

        val_split = "validation" if "validation" in ref["splits"] else "heldout" if "heldout" in ref["splits"] else None
        all_train = [r for r in read_split(ref, "train") if task.keep(r)]
        train_rows = subsample(all_train, a.train_fraction, a.seed, (task.attachments or {}).get("key"))
        if len(train_rows) < len(all_train):
            log.emit("log", message=f"training on {len(train_rows)} of {len(all_train)} rows ({a.train_fraction:.0%})")
        eval_rows = [r for r in read_split(ref, val_split) if task.keep(r)] if val_split else []
        eval_rows = random.Random(a.seed).sample(eval_rows, min(a.eval_sample, len(eval_rows)))
        train, dropped = fit(render(train_rows, task, tok, p["end_of_turn"], attachments), tok, p["seq"])
        evals, _ = fit(render(eval_rows, task, tok, p["end_of_turn"], attachments), tok, p["seq"])
        train_ds = Dataset.from_list(train).shuffle(seed=a.seed)
        eval_ds = Dataset.from_list(evals) if evals else None
        log.emit("log", message=f"{len(train_ds)} train rows ({dropped} longer than {p['seq']} tokens dropped), "
                                f"{len(evals)} eval rows from {val_split}")

        class Events(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                metrics = {k: float(v) for k, v in (logs or {}).items() if isinstance(v, (int, float))}
                log.emit("metric", step=state.global_step, total=state.max_steps, metrics=metrics | gpu_stats())

        cfg = SFTConfig(
            output_dir=str(out / "checkpoints"), num_train_epochs=a.epochs, max_steps=a.max_steps,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, lr_scheduler_type="cosine",
            warmup_ratio=0.05, weight_decay=0.0, optim="adamw_8bit", bf16=True, seed=a.seed,
            max_length=p["seq"], packing=True, completion_only_loss=True,
            logging_steps=5, eval_strategy="steps" if eval_ds else "no", eval_steps=50, save_strategy="no",
            report_to="mlflow", run_name=run_id,
        )
        trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=eval_ds,
                             processing_class=tok, callbacks=[Events()])
        result = trainer.train()
        import mlflow
        from transformers.integrations import MLflowCallback

        last = mlflow.last_active_run()
        trainer.remove_callback(MLflowCallback)  # otherwise evaluate() opens a second MLflow run
        final_eval = trainer.evaluate() if eval_ds else {}

        model.save_pretrained(out / "adapter")
        tok.save_pretrained(out / "adapter")
        model_ref = {
            "id": f"{run_id}", "kind": "hf_adapter", "base": p["model"],
            "uri": str((out / "adapter").relative_to(ARTIFACTS.parent)), "chat_template": p["chat_template"],
            "lineage": {"dataset": dataset_tag, "task": task.ref, "preset": preset_name,
                        **({"train_fraction": a.train_fraction} if a.train_fraction < 1 else {}),
                        **({"base_model": base_entry["id"]} if base_entry else {}),
                        **({"mlflow_run_id": last.info.run_id} if last else {})},
            "created_at": now(), "workspace_id": "default",
        }
        (out / "model_ref.json").write_text(json.dumps(model_ref, indent=2), encoding="utf-8")
        summary = {"train_loss": result.training_loss, "eval_loss": final_eval.get("eval_loss"),
                   "steps": result.global_step, "runtime_s": round(result.metrics["train_runtime"]),
                   "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
                   "train_rows": len(train_ds), "dropped_too_long": dropped, "train_fraction": a.train_fraction}
        register(run_id)
        log.emit("succeeded", metrics=summary, result={"model_ref": run_id})
        print(json.dumps(summary | {"run_id": run_id, "out": str(out)}, indent=2))
    except Exception as e:
        log.emit("failed", message=f"{type(e).__name__}: {e}"[:500])
        raise


if __name__ == "__main__":
    main()
