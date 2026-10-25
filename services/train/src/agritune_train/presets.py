"""Model presets, with the numbers M0 measured on the RTX 5070 Ti Laptop (12 GB).

gemma-12b-qlora was dropped: it does not fit in 12 GB (see docs/runbook-m0.md).
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
    "qwen-8b-qlora": dict(model="unsloth/Qwen3-8B-unsloth-bnb-4bit", load_in_4bit=True, seq=2048,
                          chat_template="qwen3", end_of_turn="<|im_end|>",
                          measured_vram_gb=7.88, tokens_per_s=719),
}

# Same system prompt in training and in every evaluation, for base and fine-tuned models alike.
_PROMPTS = Path(__file__).resolve().parents[4] / "contracts" / "prompts.json"
SYSTEM_PROMPT = json.loads(_PROMPTS.read_text(encoding="utf-8"))["system"]
