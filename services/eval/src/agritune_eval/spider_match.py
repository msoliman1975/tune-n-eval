"""Spider exact-set match and hardness for any SQLite database, through the vendored official code.

Exact-set match compares the parsed clauses of the predicted and gold queries as sets, with values
ignored (Spider's default), so `SELECT a, b` and `SELECT b, a` match but a correct query written
with a different structure does not. Foreign keys come from a Spider `tables.json` next to the
databases when there is one, otherwise from the database's own FOREIGN KEY declarations.
Hardness (easy / medium / hard / extra) is Spider's classification of the gold query.
"""

from __future__ import annotations

import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

INNER = re.compile(r"\binner\s+join\b", re.I)


def normalize(sql: str) -> str:
    """The official parser does not know `INNER JOIN`, a pure synonym of `JOIN`, and scores such a query
    as empty. That is the only rewrite: LEFT / CROSS JOIN mean something else and stay unparsed (0)."""
    return INNER.sub("JOIN", sql)


EMPTY_SQL = {"except": None, "from": {"conds": [], "table_units": []}, "groupBy": [], "having": [],
             "intersect": None, "limit": None, "orderBy": [], "select": [False, []], "union": None, "where": []}


def _nltk_ready() -> None:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def _entry_from_sqlite(db: str) -> dict:
    """The tables.json entry Spider would have for this database (names and foreign keys only)."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        cols = [[-1, "*"]]
        index = {}
        for ti, t in enumerate(tables):
            for c in conn.execute(f'PRAGMA table_info("{t}")'):
                index[(t.lower(), c[1].lower())] = len(cols)
                cols.append([ti, c[1]])
        fks = []
        for t in tables:
            for _id, _seq, ref, frm, to, *_ in conn.execute(f'PRAGMA foreign_key_list("{t}")'):
                a, b = index.get((t.lower(), str(frm).lower())), index.get((ref.lower(), str(to or "").lower()))
                if a is not None and b is not None:
                    fks.append([a, b])
        return {"table_names_original": tables, "column_names_original": cols, "foreign_keys": fks}
    finally:
        conn.close()


@lru_cache(maxsize=256)
def _db(db: str, tables_json: str | None):
    _nltk_ready()
    from .vendor.spider.evaluation import build_foreign_key_map
    from .vendor.spider.process_sql import Schema, get_schema

    entry = None
    if tables_json and Path(tables_json).exists():
        db_id = Path(db).stem
        entry = next((e for e in _tables(tables_json) if e["db_id"] == db_id), None)
    return Schema(get_schema(db)), build_foreign_key_map(entry or _entry_from_sqlite(db))


@lru_cache(maxsize=4)
def _tables(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hardness(gold: str, db: str | Path, tables_json: str | None = None) -> str | None:
    from .vendor.spider.evaluation import Evaluator
    from .vendor.spider.process_sql import get_sql

    schema, _ = _db(str(db), tables_json)
    try:
        return Evaluator().eval_hardness(get_sql(schema, normalize(gold)))
    except Exception:  # the official parser cannot read every query
        return None


def exact_set_match(pred: str, gold: str, db: str | Path, tables_json: str | None = None) -> float | None:
    """1.0 or 0.0; None when the official parser cannot read the gold query (not scored)."""
    import copy

    from .vendor.spider.evaluation import Evaluator, build_valid_col_units, rebuild_sql_col, rebuild_sql_val
    from .vendor.spider.process_sql import get_sql

    schema, kmap = _db(str(db), tables_json)
    try:
        g_sql = get_sql(schema, normalize(gold))
    except Exception:
        return None
    try:
        p_sql = get_sql(schema, normalize(pred).replace("value", "1"))  # "value" -> "1" as the official script does
    except Exception:
        p_sql = copy.deepcopy(EMPTY_SQL)
    try:
        g = rebuild_sql_col(build_valid_col_units(g_sql["from"]["table_units"], schema), rebuild_sql_val(g_sql), kmap)
        p = rebuild_sql_col(build_valid_col_units(p_sql["from"]["table_units"], schema), rebuild_sql_val(p_sql), kmap)
        return float(bool(Evaluator().eval_exact_match(p, g)))
    except Exception:
        return 0.0
