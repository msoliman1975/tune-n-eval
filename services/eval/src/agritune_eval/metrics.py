"""The metric catalog, grouped by evaluation approach, and per-answer scoring.

A task lists the metrics that make sense for it (tasks/*.json "metrics.available"); the user picks
some. Each metric scores one answer at a time; aggregation, intervals and comparisons live in run.py.

Kinds: rate (0/1 per item, reported as %), score (0-1), scale (1-5), seconds (lower is better),
category (a label per failed item, reported as counts), value (a number to read per slice, e.g. a
0-100 match score; no model is "better" for a higher value, so pairs are not compared).

The structured and Strong Hire metrics are scored in structured.py.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from agritune_core.sqlexec import ExecResult, execute, extract_sql, order_matters, results_match

APPROACHES = {
    "execution": "Execution-based: run the answer and compare the result with the reference's",
    "reference": "Reference match: compare the answer's text or structure with the reference",
    "llm_judge": "LLM judge: a local model grades the answer against the reference with a rubric",
    "structured": "Structured output: parse the JSON answer and check it against the schema and the reference",
    "product": "Product formula: the product's own code turns the answer into a result, compared with a human label",
    "efficiency": "Efficiency: time and tokens per answer",
}


@dataclass(frozen=True)
class Metric:
    id: str
    name: str
    approach: str
    kind: str
    description: str
    higher_is_better: bool = True
    sql_only: bool = False
    judge: bool = False

    @property
    def compare(self) -> bool:
        """Whether a model pair is compared on this metric (not for counts or neutral values)."""
        return self.kind not in ("category", "value")


METRICS = {m.id: m for m in [
    Metric("sql_exec", "Execution accuracy", "execution", "rate",
           "The predicted query returns the same rows as the gold query on the database (as a multiset; in order "
           "when the gold query has ORDER BY; column order free).", sql_only=True),
    Metric("sql_valid", "Valid SQL", "execution", "rate",
           "The predicted query runs without an error within the time limit.", sql_only=True),
    Metric("sql_exact_set", "Exact-set match", "reference", "rate",
           "Spider's official structural match of SQL clauses, values ignored. Comparable with published "
           "numbers; a correct query written differently does not match.", sql_only=True),
    Metric("exact_match", "Exact match", "reference", "rate",
           "The answer equals the reference after case and whitespace normalization."),
    Metric("chrf", "chrF++", "reference", "score",
           "Character n-gram overlap with the reference (0-1). Free and language-agnostic; rewards wording "
           "as well as content."),
    Metric("llm_judge", "LLM judge score", "llm_judge", "scale",
           "A local judge model grades the answer against the reference with the task's rubric (1-5).", judge=True),
    Metric("sql_failure_judge", "Failure analysis", "llm_judge", "category",
           "For each answer that fails execution accuracy, a local judge names the main error: wrong table or "
           "column, join, filter, aggregation, ...", sql_only=True, judge=True),
    Metric("json_valid", "Valid JSON", "structured", "rate",
           "The answer is JSON that matches the task's JSON Schema (the product's output contract)."),
    Metric("field_match", "Field accuracy", "structured", "score",
           "Mean agreement with the reference over the task's fields: text fields match after normalization or by "
           "most words, enum fields must be equal, list fields score F1 with fuzzy item matching (0-1)."),
    Metric("label_acc", "Label accuracy", "structured", "rate",
           "The answer's label (e.g. probe or move on, the document kind) equals the reference label. Slice by the "
           "expected label to read recall per class."),
    Metric("one_question", "One question at a time", "structured", "rate",
           "The reply (or the task's message field) contains at most one question mark."),
    Metric("word_limit", "Within the word limit", "structured", "rate",
           "The spoken reply has no more words than the product's limit for this model tier."),
    Metric("quotes_verbatim", "Quotes are the candidate's words", "structured", "score",
           "Share of the answer's quotes found word for word in what the candidate said (case and punctuation "
           "ignored; '...' may skip words; at least 3 words)."),
    Metric("rubric_within_one", "Rubric scores within one point", "product", "score",
           "Per question and competency the human scored: the model's 1-4 score is within one point. Scores whose "
           "quotes are not the candidate's words are dropped first, as Strong Hire does; a missing score is a miss."),
    Metric("hire_signal_within_one", "Hire signal within one band", "product", "rate",
           "The hire signal Strong Hire computes from the model's scores is within one band of the human signal "
           "(spec bar: 85%)."),
    Metric("hire_signal_exact", "Hire signal exact", "product", "rate",
           "The computed hire signal equals the human signal."),
    Metric("gap_fit", "Gap analysis fit", "product", "rate",
           "The match score Strong Hire computes from the model's ratings falls in the band the pair expects "
           "(a matching resume high, a resume for another job low)."),
    Metric("gap_match_score", "Gap match score", "product", "value",
           "The 0-100 match score Strong Hire computes from the model's ratings. Read it per fit slice: matching "
           "pairs should score well above mismatched ones."),
    Metric("latency", "Latency", "efficiency", "seconds",
           "Seconds per answer (amortized over the batch for local GPU models).", higher_is_better=False),
    Metric("ttft", "Time to first token", "efficiency", "seconds",
           "Seconds until the first answer token arrives (thinking excluded). Streamed tasks only; Ollama and "
           "Claude models (not local transformers or Gemini).", higher_is_better=False),
]}

FAILURE_CATEGORIES = {
    "schema_linking": "wrong or missing table or column",
    "join": "wrong join path or join condition",
    "filter": "wrong WHERE / HAVING condition or literal value",
    "aggregation": "wrong aggregate, GROUP BY or COUNT",
    "ordering_limit": "wrong ORDER BY, LIMIT, or top-k logic",
    "set_or_nesting": "wrong subquery, UNION / INTERSECT / EXCEPT",
    "projection": "right rows but wrong, missing or extra selected columns",
    "invalid_sql": "the query does not run (syntax or unknown identifier)",
    "gold_questionable": "the prediction looks correct; the gold query or question is ambiguous",
    "other": "anything else",
}


def catalog(task_metrics: dict | None = None) -> list[dict]:
    allowed = set(task_metrics["available"]) if task_metrics else set(METRICS)
    default = set(task_metrics.get("default", [])) if task_metrics else set()
    return [asdict(m) | {"default": m.id in default} for m in METRICS.values() if m.id in allowed]


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().casefold()).rstrip(" .;")


def chrf(hyp: str, ref: str) -> float:
    from sacrebleu.metrics import CHRF

    return CHRF(word_order=2).sentence_score(hyp, [ref]).score / 100


def preview(r: ExecResult, n: int = 5) -> dict:
    return {"columns": r.columns, "rows": [list(map(lambda v: v if isinstance(v, (int, float, type(None))) else str(v)[:80], row))
                                           for row in r.rows[:n]], "n_rows": len(r.rows), "truncated": r.truncated,
            "error": r.error}


def score_sql(pred_text: str, gold_sql: str, db, gold: ExecResult, want: set[str], tables_json: str | None,
              timeout_s: float = 10.0) -> tuple[dict, dict]:
    """Scores one SQL answer; returns (scores, details). The gold result is computed once per item."""
    sql = extract_sql(pred_text)
    scores, details = {}, {"sql": sql}
    if want & {"sql_exec", "sql_valid", "sql_failure_judge"}:
        pr = execute(db, sql, timeout_s=timeout_s)
        details["result"] = preview(pr)
        if "sql_valid" in want:
            scores["sql_valid"] = float(pr.ok)
        if want & {"sql_exec", "sql_failure_judge"}:
            scores["sql_exec"] = float(results_match(gold, pr, order_matters(gold_sql)))
    if "sql_exact_set" in want:
        from .spider_match import exact_set_match

        v = exact_set_match(sql, gold_sql, db, tables_json)
        if v is not None:
            scores["sql_exact_set"] = v
    if "exact_match" in want:
        scores["exact_match"] = float(norm_text(sql) == norm_text(gold_sql))
    return scores, details


def score_text(pred_text: str, reference: str, want: set[str]) -> dict:
    scores = {}
    if "exact_match" in want:
        scores["exact_match"] = float(norm_text(pred_text) == norm_text(reference))
    if "chrf" in want:
        scores["chrf"] = chrf(pred_text, reference)
    return scores
