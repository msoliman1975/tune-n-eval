"""M0 benchmark: measure peak VRAM and training throughput per preset on this GPU.

Each preset runs in its own process so VRAM is fully released between runs:
    uv run python -m agritune_train.bench --preset qwen-1.7b-lora
    uv run python -m agritune_train.bench --all
Results are appended to bench_results.jsonl.
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

PRESETS = {
    "qwen-1.7b-lora": dict(model="unsloth/Qwen3-1.7B", load_in_4bit=False, seq=4096),
    "gemma-4b-qlora": dict(model="unsloth/gemma-3-4b-it-unsloth-bnb-4bit", load_in_4bit=True, seq=2048),
    "qwen-8b-qlora": dict(model="unsloth/Qwen3-8B-unsloth-bnb-4bit", load_in_4bit=True, seq=2048),
    "gemma-12b-qlora": dict(model="unsloth/gemma-3-12b-it-unsloth-bnb-4bit", load_in_4bit=True, seq=1024),
}

STEPS, WARMUP = 25, 5
RESULTS = Path(__file__).resolve().parents[2] / "bench_results.jsonl"

PARAGRAPH = (
    "Mango trees in Egypt flower between February and April. Growers in Ismailia and Sharqia "
    "watch for powdery mildew on panicles and for mango malformation, and they monitor white "
    "mango scale on leaves through the summer before harvest from July to October. "
)


def run_one(name: str) -> dict:
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel

    p = PRESETS[name]
    torch.cuda.reset_peak_memory_stats()
    t_load = time.time()
    model, tok = FastModel.from_pretrained(
        model_name=p["model"], max_seq_length=p["seq"], load_in_4bit=p["load_in_4bit"],
        full_finetuning=False,
    )
    load_s = time.time() - t_load
    model = FastModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=0,
    )
    text_tok = getattr(tok, "tokenizer", tok)  # Gemma 3 returns a processor

    # Fixed-length rows so tokens per step are exact.
    ids = text_tok(PARAGRAPH * 400, add_special_tokens=False)["input_ids"][: p["seq"]]
    text = text_tok.decode(ids)
    ds = Dataset.from_dict({"text": [text] * (STEPS * 2)})

    step_times: list[float] = []

    class Timer(TrainerCallback):
        def on_step_begin(self, *a, **k):
            torch.cuda.synchronize()
            self.t = time.time()

        def on_step_end(self, *a, **k):
            torch.cuda.synchronize()
            step_times.append(time.time() - self.t)

    cfg = dict(
        output_dir=f"/tmp/agritune-bench/{name}", per_device_train_batch_size=1,
        gradient_accumulation_steps=1, max_steps=STEPS, learning_rate=2e-4, logging_steps=5,
        report_to="none", save_strategy="no", packing=False, dataset_text_field="text",
        bf16=True, optim="adamw_8bit",
    )
    # TRL renamed max_seq_length -> max_length; pass whichever this version accepts.
    params = inspect.signature(SFTConfig.__init__).parameters
    cfg["max_length" if "max_length" in params else "max_seq_length"] = p["seq"]

    trainer = SFTTrainer(model=model, train_dataset=ds, args=SFTConfig(**cfg),
                         processing_class=text_tok, callbacks=[Timer()])
    trainer.train()

    timed = step_times[WARMUP:]
    step_s = sum(timed) / len(timed)
    return {
        "preset": name, "model": p["model"], "seq": p["seq"], "4bit": p["load_in_4bit"],
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "step_s": round(step_s, 3), "tokens_per_s": round(p["seq"] / step_s),
        "load_s": round(load_s, 1), "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    if a.all:
        for name in PRESETS:
            print(f"=== {name}", flush=True)
            subprocess.run([sys.executable, "-m", "agritune_train.bench", "--preset", name])
        return

    try:
        res = run_one(a.preset)
    except Exception as e:  # record OOMs and load failures as results too
        res = {"preset": a.preset, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    with RESULTS.open("a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
