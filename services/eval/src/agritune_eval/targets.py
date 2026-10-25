"""HF target: a 4-bit base checkpoint, optionally with a PEFT adapter, on the local GPU.

The base is loaded once; the adapter is switched on and off so base and fine-tuned answers come
from the exact same weights, quantization, prompt and decoding.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from . import PROMPTS

LETTERS = ["A", "B", "C", "D"]


class HFTarget:
    def __init__(self, base: str, adapter: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(adapter or base)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(base, device_map="cuda:0", dtype=torch.bfloat16)
        self.has_adapter = adapter is not None
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.letter_ids = [self.tok.encode(x, add_special_tokens=False)[0] for x in LETTERS]

    def use(self, adapter: bool):
        """Context manager: adapter on (fine-tuned) or off (base)."""
        if adapter or not self.has_adapter:
            return nullcontext()
        return self.model.disable_adapter()

    def prompt(self, user: str) -> str:
        msgs = [{"role": "system", "content": PROMPTS["system"]}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    @torch.inference_mode()
    def generate(self, questions: list[str], max_new_tokens: int = 400, batch: int = 8) -> list[str]:
        order = sorted(range(len(questions)), key=lambda i: len(questions[i]))
        out: list[str] = [""] * len(questions)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            enc = self.tok([self.prompt(questions[i]) for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to("cuda:0")
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            for i, row in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
                out[i] = self.tok.decode(row, skip_special_tokens=True).strip()
        return out

    @torch.inference_mode()
    def mcq(self, items: list[dict], batch: int = 8) -> list[list[str]]:
        """For each item, the original-order letter picked under each of the 4 rotations of its choices.

        A pick is the letter with the highest next-token probability. Rotating the choices cancels the
        position bias of small models (e.g. always answering A).
        """
        jobs = [(i, r) for i in range(len(items)) for r in range(4)]
        picks: list[list[str]] = [[] for _ in items]
        for s in range(0, len(jobs), batch):
            chunk = jobs[s:s + batch]
            prompts = [self.prompt(mcq_text(items[i], r)) for i, r in chunk]
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
            logits = self.model(**enc).logits[:, -1, self.letter_ids]  # left padding: last position is real
            for (i, r), j in zip(chunk, logits.argmax(-1).tolist()):
                picks[i].append(LETTERS[(j + r) % 4])  # map the shown position back to the original letter
        return picks


def mcq_text(it: dict, rotation: int = 0) -> str:
    """Choice shown at position p is original choice (p + rotation) % 4."""
    lang = "ar" if it["lang"] == "ar" else "en"
    shown = [it["choices"][(p + rotation) % 4] for p in range(4)]
    choices = "\n".join(f"{L}) {c}" for L, c in zip(LETTERS, shown))
    return f"{it['question']}\n\n{choices}\n\n{PROMPTS['mcq_instruction'][lang]}"
