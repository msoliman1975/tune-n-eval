"""SQL execution and comparison for execution accuracy.

- connect(): read-only, so a generated DROP or UPDATE cannot change a benchmark database.
- execute(): one statement, a wall-clock timeout, a row cap.
- extract_sql(): the first SQL statement in a model's reply (code fences, "SQL:" labels, prose).
- results_match(): two result sets are equal as multisets of rows (as lists when the gold query
  has ORDER BY), allowing the columns in a different order, as in the Spider test-suite evaluation.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path


def connect(db: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{Path(db)}?mode=ro", uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")  # some benchmark DBs hold latin-1 bytes
    return conn


@dataclass
class ExecResult:
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: str | None = None
    truncated: bool = False
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


def execute(db: str | Path, sql: str, timeout_s: float = 10.0, max_rows: int = 100_000) -> ExecResult:
    if not sql.strip():
        return ExecResult(error="empty query")
    t0 = time.monotonic()
    deadline = t0 + timeout_s
    conn = connect(db)
    conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)  # non-zero aborts the query
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        cols = [d[0] for d in cur.description or []]
        return ExecResult(cols, [tuple(r) for r in rows[:max_rows]], None, len(rows) > max_rows,
                          time.monotonic() - t0)
    except sqlite3.OperationalError as e:
        msg = f"timeout after {timeout_s:g}s" if "interrupted" in str(e) else str(e)
        return ExecResult(error=msg, seconds=time.monotonic() - t0)
    except Exception as e:  # sqlite3.Warning for several statements, ProgrammingError, ...
        return ExecResult(error=f"{type(e).__name__}: {e}", seconds=time.monotonic() - t0)
    finally:
        conn.close()


THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
FENCE = re.compile(r"```[ \t]*(?:sql|sqlite)?[ \t]*\n?(.*?)```", re.S | re.I)
LABEL = re.compile(r"^\s*(?:sql(?:ite)?\s*(?:query)?\s*:|query\s*:)\s*", re.I)
START = re.compile(r"\b(WITH|SELECT)\b", re.I)


def first_statement(sql: str) -> str:
    """Text up to the first semicolon outside quotes."""
    quote = None
    for i, ch in enumerate(sql):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"`":
            quote = ch
        elif ch == ";":
            return sql[:i]
    return sql


def extract_sql(text: str) -> str:
    """The query in a model reply; the reply itself when it is already bare SQL."""
    t = THINK.sub("", text or "").strip()
    m = FENCE.search(t)
    if m:
        t = m.group(1)
    elif "```" in t:  # an unclosed fence, as when generation hits max tokens
        t = t.split("```", 1)[1].removeprefix("sql").removeprefix("sqlite")
    t = LABEL.sub("", t.strip())
    if not START.match(t.lstrip("( \n")):  # prose before the query
        m = START.search(t)
        if m:
            t = t[m.start():]
    return " ".join(first_statement(t).split())


def _norm(v):
    if isinstance(v, float):
        return round(v, 6)
    return v


def results_match(gold: ExecResult, pred: ExecResult, order_matters: bool) -> bool:
    if not (gold.ok and pred.ok):
        return False
    if len(gold.rows) != len(pred.rows):
        return False
    if not gold.rows:
        return len(gold.columns) == len(pred.columns) or not pred.columns
    ng, np_ = len(gold.rows[0]), len(pred.rows[0])
    if ng != np_:
        return False
    g = [tuple(map(_norm, r)) for r in gold.rows]
    p = [tuple(map(_norm, r)) for r in pred.rows]
    target = g if order_matters else Counter(g)
    orders = permutations(range(ng)) if ng <= 6 else [tuple(range(ng))]
    for perm in orders:
        cand = [tuple(r[i] for i in perm) for r in p]
        if (cand if order_matters else Counter(cand)) == target:
            return True
    return False


def order_matters(gold_sql: str) -> bool:
    return bool(re.search(r"\border\s+by\b", gold_sql, re.I))
