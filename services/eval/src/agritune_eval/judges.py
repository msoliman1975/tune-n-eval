"""Generic local judges, through Ollama with schema-constrained JSON output; verdicts are cached.

- RubricJudge: grades an answer against the reference with a task rubric (rubrics/<id>.md), 1-5.
  The mango rubric keeps its 5-criteria judge (judge.py) so mango scores stay comparable.
- FailureJudge: for a SQL answer that fails execution accuracy, names the main error category.

The default judge is Qwen3-14B: a different family from the Gemma models under test, so it does
not grade its own kin. Thinking is switched off; the schema keeps the output parseable.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel, Field

from . import ARTIFACTS, RUBRICS
from .metrics import FAILURE_CATEGORIES

CACHE = ARTIFACTS / "eval" / "judge_cache.jsonl"


class RubricVerdict(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str


class FailureVerdict(BaseModel):
    category: Literal[tuple(FAILURE_CATEGORIES)]  # type: ignore[valid-type]
    rationale: str


FAILURE_SYSTEM = ("You review text-to-SQL mistakes. A predicted SQLite query returned a different result from "
                  "the gold query. Name the single main reason, using one of these categories:\n"
                  + "\n".join(f"- {k}: {v}" for k, v in FAILURE_CATEGORIES.items())
                  + "\nAnswer in JSON with `category` and a one- or two-sentence `rationale`.")


class OllamaJudge:
    def __init__(self, model: str = "qwen3:14b", url: str = "http://127.0.0.1:11434", concurrency: int = 2):
        import httpx

        self.model, self.concurrency = model, concurrency
        self.http = httpx.Client(base_url=url, timeout=httpx.Timeout(30, read=600))
        self.cache: dict[str, dict] = {}
        if CACHE.exists():
            for line in CACHE.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row["verdict"]

    def _call(self, kind: str, system: str, user: str, schema: type[BaseModel]) -> dict:
        key = hashlib.sha256(json.dumps([kind, self.model, system, user]).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        body = {"model": self.model, "stream": False, "think": False, "format": schema.model_json_schema(),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": 0, "num_ctx": 12288, "num_predict": 512}}
        for attempt in range(3):
            r = self.http.post("/api/chat", json=body)
            r.raise_for_status()
            try:
                v = schema.model_validate_json(r.json()["message"]["content"]).model_dump()
                break
            except ValueError:
                if attempt == 2:
                    raise RuntimeError(f"{self.model} returned no valid verdict after 3 tries")
        self.cache[key] = v
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "verdict": v}, ensure_ascii=False) + "\n")
        return v

    def many(self, fn, jobs: list[tuple], progress=None) -> list[dict]:
        out: list[dict | None] = [None] * len(jobs)
        with ThreadPoolExecutor(self.concurrency) as pool:
            futs = {pool.submit(fn, *j): i for i, j in enumerate(jobs)}
            for n, f in enumerate(futs, 1):
                try:
                    out[futs[f]] = f.result()
                except Exception as e:  # one bad call must not lose the rest of the evaluation
                    out[futs[f]] = {"error": f"{type(e).__name__}: {e}"[:300]}
                if progress and n % 10 == 0:
                    progress(n)
        return out  # type: ignore[return-value]

    def rubric(self, rubric_id: str, question: str, reference: str, answer: str) -> dict:
        system = (RUBRICS / f"{rubric_id}.md").read_text(encoding="utf-8")
        user = (f"<question>\n{question}\n</question>\n\n<reference_answer>\n{reference}\n</reference_answer>\n\n"
                f"<candidate_answer>\n{answer or '(empty answer)'}\n</candidate_answer>")
        return self._call(f"rubric:{rubric_id}", system, user, RubricVerdict)

    def failure(self, question: str, schema: str, gold_sql: str, gold_result: dict, pred_sql: str,
                pred_result: dict) -> dict:
        user = (f"<schema>\n{schema[:9000]}\n</schema>\n\n<question>\n{question}\n</question>\n\n"
                f"<gold_sql>\n{gold_sql}\n</gold_sql>\n<gold_result>\n{json.dumps(gold_result, default=str)[:1500]}\n</gold_result>\n\n"
                f"<predicted_sql>\n{pred_sql or '(empty)'}\n</predicted_sql>\n"
                f"<predicted_result>\n{json.dumps(pred_result, default=str)[:1500]}\n</predicted_result>")
        return self._call("sql_failure", FAILURE_SYSTEM, user, FailureVerdict)
