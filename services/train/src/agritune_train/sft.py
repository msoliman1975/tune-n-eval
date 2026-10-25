"""Supervised fine-tuning (QLoRA / LoRA) on a frozen dataset, tracked in MLflow.

    bash run.sh sft --preset qwen-8b-qlora --dataset ../../artifacts/data/datasets/mango-sft/v0.1
    bash run.sh sft --preset qwen-1.7b-lora --dataset ... --max-steps 30   # smoke test

Trains on the dataset's train split (open Q&A only), with loss on the answer tokens only.
A small sample of the heldout split gives eval loss on questions worded differently from training.
Writes artifacts/train/{run_id}/: adapter/, events.jsonl (JobEvents for the Studio), model_ref.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from .presets import PRESETS, SYSTEM_PROMPT

REPO = Path(__file__).resolve().parents[4]
ARTIFACTS = REPO / "artifacts"
RUNS = ARTIFACTS / "train"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_split(ref: dict, split: str) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(ARTIFACTS.parent / ref["splits"][split]).to_pylist()


def render(rows: list[dict], tok, end_of_turn: str) -> list[dict]:
    """Prompt/completion strings; TRL then computes loss on the completion only."""
    out = []
    for r in rows:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["question"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        out.append({"prompt": prompt, "completion": r["answer"] + end_of_turn})
    return out


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
    ap.add_argument("--preset", choices=list(PRESETS), required=True)
    ap.add_argument("--dataset", required=True, help="dataset version folder holding dataset_ref.json")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps for a smoke test")
    ap.add_argument("--eval-sample", type=int, default=64)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--run-id", help="set by the API, which creates the run folder first")
    a = ap.parse_args()

    p = PRESETS[a.preset]
    ds_dir = Path(a.dataset).resolve()
    ref = json.loads((ds_dir / "dataset_ref.json").read_text(encoding="utf-8"))
    dataset_tag = f"{ref['id']}@{ref['version']}"
    run_id = a.run_id or f"run-{time.strftime('%Y%m%d-%H%M%S')}-{a.preset}"
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=bool(a.run_id))
    log = EventLog(out / "events.jsonl", run_id)
    log.emit("started", message=f"{a.preset} on {dataset_tag}")

    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "agritune-sft")
    os.environ["MLFLOW_TAGS"] = json.dumps({"preset": a.preset, "dataset": dataset_tag,
                                            "dataset_sha256": ref["sha256"], "run_id": run_id})

    from unsloth import FastModel  # must come before trl/transformers

    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    try:
        model, tok = FastModel.from_pretrained(model_name=p["model"], max_seq_length=p["seq"],
                                               load_in_4bit=p["load_in_4bit"], full_finetuning=False)
        model = FastModel.get_peft_model(
            model, r=a.rank, lora_alpha=a.rank, lora_dropout=0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth", random_state=a.seed,
        )
        tok = getattr(tok, "tokenizer", tok)  # Gemma 3 returns a processor

        train_rows = [r for r in read_split(ref, "train") if r["type"] == "open_qa"]
        eval_rows = [r for r in read_split(ref, "heldout") if r["type"] == "open_qa"]
        eval_rows = random.Random(a.seed).sample(eval_rows, min(a.eval_sample, len(eval_rows)))
        train_ds = Dataset.from_list(render(train_rows, tok, p["end_of_turn"])).shuffle(seed=a.seed)
        eval_ds = Dataset.from_list(render(eval_rows, tok, p["end_of_turn"]))
        log.emit("log", message=f"{len(train_ds)} train rows, {len(eval_ds)} eval rows")

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
            logging_steps=5, eval_strategy="steps", eval_steps=50, save_strategy="no",
            report_to="mlflow", run_name=run_id,
        )
        trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=eval_ds,
                             processing_class=tok, callbacks=[Events()])
        result = trainer.train()
        import mlflow
        from transformers.integrations import MLflowCallback

        last = mlflow.last_active_run()
        trainer.remove_callback(MLflowCallback)  # otherwise evaluate() opens a second MLflow run
        final_eval = trainer.evaluate()

        model.save_pretrained(out / "adapter")
        tok.save_pretrained(out / "adapter")
        model_ref = {
            "id": f"{run_id}", "kind": "hf_adapter", "base": p["model"],
            "uri": str((out / "adapter").relative_to(ARTIFACTS.parent)), "chat_template": p["chat_template"],
            "lineage": {"dataset": dataset_tag, "preset": a.preset,
                        **({"mlflow_run_id": last.info.run_id} if last else {})},
            "created_at": now(), "workspace_id": "default",
        }
        (out / "model_ref.json").write_text(json.dumps(model_ref, indent=2), encoding="utf-8")
        summary = {"train_loss": result.training_loss, "eval_loss": final_eval.get("eval_loss"),
                   "steps": result.global_step, "runtime_s": round(result.metrics["train_runtime"]),
                   "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2)}
        log.emit("succeeded", metrics=summary, result={"model_ref": run_id})
        print(json.dumps(summary | {"run_id": run_id, "out": str(out)}, indent=2))
    except Exception as e:
        log.emit("failed", message=f"{type(e).__name__}: {e}"[:500])
        raise


if __name__ == "__main__":
    main()
