"""Claude judge for open answers, graded against the reference with a versioned rubric.

Verdicts are cached by (rubric, question, reference, answer), so re-running a report, or
evaluating an answer twice, costs nothing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os

import anthropic
from pydantic import BaseModel, Field

from . import ARTIFACTS, RUBRICS, SERVICE_DIR

CACHE = ARTIFACTS / "eval" / "judge_cache.jsonl"
CRITERIA = ["correctness", "completeness", "actionability", "safety", "language"]
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0)}


class Verdict(BaseModel):
    correctness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    actionability: int = Field(ge=1, le=5)
    safety: int = Field(ge=1, le=5)
    language: int = Field(ge=1, le=5)
    rationale: str


def load_env() -> None:
    env = SERVICE_DIR / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


class Judge:
    def __init__(self, rubric: str = "mango_advice_v1", model: str = "claude-opus-5-5", concurrency: int = 6):
        load_env()
        self.rubric_id, self.model = rubric, model
        self.rubric = (RUBRICS / f"{rubric}.md").read_text(encoding="utf-8")
        self.client = anthropic.AsyncAnthropic(max_retries=4)
        self.sem = asyncio.Semaphore(concurrency)
        self.cache: dict[str, dict] = {}
        if CACHE.exists():
            for line in CACHE.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row["verdict"]
        self.spent = {"input_tokens": 0, "output_tokens": 0}

    def key(self, q: str, ref: str, ans: str) -> str:
        return hashlib.sha256(json.dumps([self.rubric_id, self.model, q, ref, ans]).encode()).hexdigest()

    def pending(self, triples: list[tuple[str, str, str]]) -> int:
        return sum(self.key(*t) not in self.cache for t in triples)

    async def grade(self, q: str, ref: str, ans: str) -> dict:
        k = self.key(q, ref, ans)
        if k in self.cache:
            return self.cache[k]
        user = (f"<question>\n{q}\n</question>\n\n<reference_answer>\n{ref}\n</reference_answer>\n\n"
                f"<candidate_answer>\n{ans or '(empty answer)'}\n</candidate_answer>")
        async with self.sem:
            resp = await self.client.beta.messages.parse(
                model=self.model, max_tokens=8000, system=self.rubric,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "low"}, output_format=Verdict,
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            )
        self.spent["input_tokens"] += resp.usage.input_tokens
        self.spent["output_tokens"] += resp.usage.output_tokens
        if resp.parsed_output is None:
            raise RuntimeError(f"judge returned no verdict (stop_reason={resp.stop_reason}, id={resp._request_id})")
        v = resp.parsed_output.model_dump()
        self.cache[k] = v
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": k, "verdict": v}, ensure_ascii=False) + "\n")
        return v

    def cost(self) -> float:
        pin, pout = PRICES.get(self.model, (0, 0))
        return self.spent["input_tokens"] / 1e6 * pin + self.spent["output_tokens"] / 1e6 * pout

    def estimate(self, n: int) -> str:
        pin, pout = PRICES.get(self.model, (0, 0))
        usd = n * (1500 / 1e6 * pin + 700 / 1e6 * pout)  # ~1.5k in (rubric + texts), ~0.7k out
        return f"{n} judge calls on {self.model}: about ${usd:.0f}"
