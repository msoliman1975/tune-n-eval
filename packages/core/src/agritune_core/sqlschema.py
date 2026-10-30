"""Renders a SQLite database as prompt context: CREATE TABLE statements with keys, plus sample rows.

The DDL is rebuilt from PRAGMA metadata rather than copied from sqlite_master, so every database
reads the same way whatever style its original CREATE statements used. Sample rows show the model
real values ('USA' vs 'United States'), which matters for filters.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .sqlexec import connect

PLAIN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# https://sqlite.org/lang_keywords.html: a table named "order" (BIRD's financial) must be quoted
KEYWORDS = set("""ABORT ACTION ADD AFTER ALL ALTER ALWAYS ANALYZE AND AS ASC ATTACH AUTOINCREMENT BEFORE BEGIN BETWEEN BY
CASCADE CASE CAST CHECK COLLATE COLUMN COMMIT CONFLICT CONSTRAINT CREATE CROSS CURRENT CURRENT_DATE CURRENT_TIME
CURRENT_TIMESTAMP DATABASE DEFAULT DEFERRABLE DEFERRED DELETE DESC DETACH DISTINCT DO DROP EACH ELSE END ESCAPE EXCEPT
EXCLUDE EXCLUSIVE EXISTS EXPLAIN FAIL FILTER FIRST FOLLOWING FOR FOREIGN FROM FULL GENERATED GLOB GROUP GROUPS HAVING IF
IGNORE IMMEDIATE IN INDEX INDEXED INITIALLY INNER INSERT INSTEAD INTERSECT INTO IS ISNULL JOIN KEY LAST LEFT LIKE LIMIT
MATCH MATERIALIZED NATURAL NO NOT NOTHING NOTNULL NULL NULLS OF OFFSET ON OR ORDER OTHERS OUTER OVER PARTITION PLAN
PRAGMA PRECEDING PRIMARY QUERY RAISE RANGE RECURSIVE REFERENCES REGEXP REINDEX RELEASE RENAME REPLACE RESTRICT RETURNING
RIGHT ROLLBACK ROW ROWS SAVEPOINT SELECT SET TABLE TEMP TEMPORARY THEN TIES TO TRANSACTION TRIGGER UNBOUNDED UNION UNIQUE
UPDATE USING VACUUM VALUES VIEW VIRTUAL WHEN WHERE WINDOW WITH WITHOUT""".split())


def ident(name: str) -> str:
    return name if PLAIN.match(name) and name.upper() not in KEYWORDS else '"' + name.replace('"', '""') + '"'


def cell(v, max_chars: int) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bytes):
        return "<blob>"
    s = " ".join(str(v).split())  # one line per row
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def tables(conn) -> list[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                        "ORDER BY rowid").fetchall()
    return [r[0] for r in rows]


def table_ddl(conn, table: str) -> str:
    cols = conn.execute(f"PRAGMA table_info({ident(table)})").fetchall()  # cid, name, type, notnull, default, pk
    pks = [c[1] for c in sorted(cols, key=lambda c: c[5]) if c[5]]
    lines = []
    for _, name, typ, *_rest in cols:
        line = f"  {ident(name)}" + (f" {typ}" if typ else "")
        if len(pks) == 1 and name == pks[0]:
            line += " PRIMARY KEY"
        lines.append(line)
    if len(pks) > 1:
        lines.append(f"  PRIMARY KEY ({', '.join(map(ident, pks))})")
    fks: dict[int, list] = {}
    for fk_id, _seq, ref_table, col, ref_col, *_ in conn.execute(f"PRAGMA foreign_key_list({ident(table)})"):
        fks.setdefault(fk_id, []).append((col, ref_table, ref_col))
    for parts in fks.values():
        src = ", ".join(ident(p[0]) for p in parts)
        dst = ", ".join(ident(p[2]) for p in parts if p[2])
        lines.append(f"  FOREIGN KEY ({src}) REFERENCES {ident(parts[0][1])}" + (f"({dst})" if dst else ""))
    return f"CREATE TABLE {ident(table)} (\n" + ",\n".join(lines) + "\n);"


def sample_block(conn, table: str, n: int, max_chars: int) -> str:
    try:
        cur = conn.execute(f"SELECT * FROM {ident(table)} LIMIT {int(n)}")
        rows = cur.fetchall()
    except Exception:  # a broken table should not break the whole schema
        return ""
    if not rows:
        return f"/*\n0 rows in {table}\n*/"
    head = "\t".join(d[0] for d in cur.description)
    body = "\n".join("\t".join(cell(v, max_chars) for v in r) for r in rows)
    return f"/*\n{len(rows)} example rows:\nSELECT * FROM {ident(table)} LIMIT {int(n)};\n{head}\n{body}\n*/"


@lru_cache(maxsize=512)
def _render(db: str, mtime: float, sample_rows: int, max_value_chars: int) -> str:
    conn = connect(db)
    try:
        blocks = []
        for t in tables(conn):
            block = table_ddl(conn, t)
            if sample_rows > 0:
                block += "\n" + sample_block(conn, t, sample_rows, max_value_chars)
            blocks.append(block)
        return "\n\n".join(blocks)
    finally:
        conn.close()


def render_schema(db: str | Path, sample_rows: int = 3, max_value_chars: int = 40) -> str:
    """DDL (types, primary and foreign keys) for every table, each followed by `sample_rows` rows."""
    p = Path(db)
    return _render(str(p), p.stat().st_mtime, sample_rows, max_value_chars)
