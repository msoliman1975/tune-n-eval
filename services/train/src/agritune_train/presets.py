"""Model presets, with the numbers M0 measured on the RTX 5070 Ti Laptop (12 GB).

A preset is how the Train service fine-tunes one base model; the Model Registry links each
trainable base to its preset (`train_preset`). gemma-12b-qlora was dropped: it does not fit in
12 GB (see docs/runbook-m0.md). Gemma 4 E4B numbers come from bench.py at seq 4096.
"""

import json
from pathlib import Path

PRESETS = {
    "qwen-1.7b-lora": dict(model="unsloth/Qwen3-1.7B", load_in_4bit=False, seq=4096,
                           chat_template="qwen3", end_of_turn="<|im_end|>",
                           measured_vram_gb=4.08, tokens_per_s=2665),
    "gemma-4b-qlora": dict(model="unsloth/gemma-3-4b-it-unsloth-bnb-4bit", load_in_4bit=True, seq=2048,
                           chat_template="gemma3", end_of_turn="<end_of_turn>",
                           measured_vram_gb=5.02, tokens_per_s=1273),
    "gemma4-e4b-qlora": dict(model="unsloth/gemma-4-E4B-it-unsloth-bnb-4bit", load_in_4bit=True, seq=4096,
                             chat_template="gemma4", end_of_turn="<turn|>",
                             # smoke run on spider@v1.0, 2026-10-26: ~44 s per step of 4 packed sequences
                             measured_vram_gb=10.25, tokens_per_s=317),
    "qwen-8b-qlora": dict(model="unsloth/Qwen3-8B-unsloth-bnb-4bit", load_in_4bit=True, seq=2048,
                          chat_template="qwen3", end_of_turn="<|im_end|>",
                          measured_vram_gb=7.88, tokens_per_s=719),
}

# Mango legacy datasets (no task) keep their original system prompt through the mango-qa task.
_PROMPTS = Path(__file__).resolve().parents[4] / "contracts" / "prompts.json"
SYSTEM_PROMPT = json.loads(_PROMPTS.read_text(encoding="utf-8"))["system"]
