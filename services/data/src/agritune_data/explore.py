"""Read-only exploration of frozen datasets: a profile of every column, one item with its rendered
prompt, the attached databases (tables, keys, rows) and exports. Used by the Studio's dataset page.

A profile reads every split once and is cached until dataset_ref.json changes (versions are immutable).
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
from collections import Counter
from functools import lru_cache
from pathlib import Path

from agritune_core.sqlexec import connect
from agritune_core.tasks import task_of

from .drafts import REPO

TOP_N = 12
CATEGORICAL_MAX = 40  # a column with at most this many distinct values gets its full value counts
CHARS_PER_TOKEN = 3.2  # the same rough estimate the freeze step and the train estimate use


def load_ref(d: Path) -> dict:
    return json.loads((d / "dataset_ref.json").read_text(encoding="utf-8"))


def split_rows(d: Path, split: str) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(d / f"{split}.parquet").to_pylist()


def fields_of(ref: dict) -> dict:
    """inputs / target / context of the dataset: from the ref, else from its task (legacy mango)."""
    if ref.get("fields"):
        return ref["fields"]
    t = task_of(ref)
    return {"inputs": t.inputs, "target": t.target, "context": list(t.context)}


def role_of(col: str, fields: dict, key: str | None) -> str:
    if col in ("id", "split", "edited"):
        return "system"
    if col == fields.get("target"):
        return "target"
    if col in fields.get("context", []):
        return "context"
    if col in fields.get("inputs", []):
        return "input (attachment key)" if col == key else "input"
    if col == "meta" or col.startswith("meta."):
        return "metadata"
    return "other"


def flatten(row: dict) -> dict:
    """Struct columns one level deep (source.doc_id) and the JSON `meta` column as meta.<key>."""
    out = {}
    for k, v in row.items():
        if k == "meta" and isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                pass
        if isinstance(v, dict):
            out |= {f"{k}.{kk}": vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def column_profile(values: list, n: int) -> dict:
    present = [v for v in values if v is not None and v != ""]
    p: dict = {"nulls": n - len(present)}
    hashable = [json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (list, dict)) else v for v in present]
    counts = Counter(hashable)
    p["distinct"] = len(counts)
    if present and all(isinstance(v, str) for v in present):
        lens = [len(v) for v in present]
        p["chars"] = {"min": min(lens), "p50": pct(lens, 0.5), "p90": pct(lens, 0.9), "max": max(lens),
                      "mean": round(statistics.fmean(lens))}
    elif present and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
        p["numbers"] = {"min": min(present), "p50": pct(present, 0.5), "max": max(present),
                        "mean": round(statistics.fmean(present), 3)}
    if counts and (len(counts) <= CATEGORICAL_MAX or len(counts) < n * 0.5):
        p["top"] = [[str(k), c] for k, c in counts.most_common(TOP_N)]
    return p


@lru_cache(maxsize=16)
def _profile(d: str, mtime: float) -> dict:
    d = Path(d)
    ref = load_ref(d)
    task = task_of(ref)
    fields = fields_of(ref)
    key = task.attachments["key"] if task.attachments else None
    splits = list(ref["splits"])
    rows = {s: split_rows(d, s) for s in splits}

    # columns, in file order, with the struct and meta keys expanded
    import pyarrow.parquet as pq

    schema = pq.read_schema(d / f"{splits[0]}.parquet")
    types = {f.name: str(f.type) for f in schema}
    for f in schema:
        if hasattr(f.type, "num_fields"):  # struct: one type per child (source.doc_id: string)
            types |= {f"{f.name}.{f.type.field(i).name}": str(f.type.field(i).type) for i in range(f.type.num_fields)}
    flat = {s: [flatten(r) for r in rs] for s, rs in rows.items()}
    cols: list[str] = []
    for rs in flat.values():
        for r in rs[:200]:
            cols += [c for c in r if c not in cols]
    every = [r for rs in flat.values() for r in rs]
    columns = []
    for c in cols:
        base = c.split(".")[0]
        columns.append({"name": c, "type": types.get(c) or ("json field" if base == "meta" else ""),
                        "role": role_of(c, fields, key)} | column_profile([r.get(c) for r in every], len(every)))

    # size of what the model reads and writes per item, in estimated tokens
    text_fields = [*fields.get("inputs", []), *fields.get("context", []), fields.get("target")]
    tokens = {s: [sum(len(str(r.get(f) or "")) for f in text_fields if f) / CHARS_PER_TOKEN for r in rs]
              for s, rs in rows.items()}
    allt = [t for ts in tokens.values() for t in ts]
    hist = None
    if allt:
        top = pct(allt, 0.99)
        width = max(50, math.ceil(top / 12 / 50) * 50)
        nb = max(1, math.ceil(top / width))
        cats = [f"{i * width}–{(i + 1) * width}" for i in range(nb)] + [f"{nb * width}+"]
        hist = {"categories": cats, "series": {s: [0] * (nb + 1) for s in splits}}
        for s, ts in tokens.items():
            for t in ts:
                hist["series"][s][min(nb, int(t // width))] += 1
    lengths = {s: {"p50": round(pct(ts, 0.5)), "p90": round(pct(ts, 0.9)), "max": round(max(ts))} for s, ts in tokens.items() if ts}

    coverage = None
    if key:
        keys = {s: Counter(r.get(key) for r in rs) for s, rs in rows.items()}
        per_key: dict[str, dict[str, int]] = {}
        for s, k in keys.items():
            for name, n in k.items():
                per_key.setdefault(str(name), {})[s] = n
        coverage = {"key": key,
                    "per_split": {s: {"keys": len(k), "top": k.most_common(TOP_N)} for s, k in keys.items()},
                    "per_key": per_key,
                    "shared": {f"{a}|{b}": len(set(keys[a]) & set(keys[b]))
                               for i, a in enumerate(splits) for b in splits[i + 1:]}}

    files = [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(d.iterdir()) if p.is_file()]
    att = None
    if ref.get("attachments"):
        root = REPO / ref["attachments"]
        all_files = [p for p in root.rglob("*") if p.is_file()]
        att = {"path": ref["attachments"], "files": len(all_files), "bytes": sum(p.stat().st_size for p in all_files)}
    return {"columns": columns, "fields": fields, "attachment_key": key, "tokens_hist": hist, "tokens": lengths,
            "coverage": coverage, "files": files, "attachments": att,
            "task": task.public() | {"ref": task.ref}}


def profile(d: Path) -> dict:
    return _profile(str(d), (d / "dataset_ref.json").stat().st_mtime)


def item(d: Path, split: str, item_id: str) -> dict:
    ref = load_ref(d)
    row = next((r for r in split_rows(d, split) if r["id"] == item_id), None)
    if row is None:
        raise KeyError(item_id)
    task = task_of(ref)
    root = REPO / ref["attachments"] if ref.get("attachments") else None
    try:
        messages, error = task.messages(task.with_context(row, root)), None
    except Exception as e:  # e.g. a legacy MCQ row the task's template does not cover
        messages, error = None, f"{type(e).__name__}: {e}"
    chars = sum(len(m["content"]) for m in messages or [])
    return {"row": row, "messages": messages, "applies": task.keep(row), "error": error,
            "est_prompt_tokens": round(chars / CHARS_PER_TOKEN)}


def _db(d: Path, key: str) -> Path:
    ref = load_ref(d)
    task = task_of(ref)
    if not (task.attachments and ref.get("attachments")):
        raise ValueError("this dataset has no attachments")
    db = task.attachment(REPO / ref["attachments"], {task.attachments["key"]: key})
    if not db.exists():
        raise KeyError(key)
    return db


@lru_cache(maxsize=4096)
def _row_count(db: str, mtime: float, table: str) -> int | None:
    """count(*) scans the table, seconds for BIRD's largest ones; counted once per database file."""
    con = connect(db)
    try:
        return con.execute(f'SELECT count(*) FROM "{table.replace(chr(34), chr(34) * 2)}"').fetchone()[0]
    except Exception:
        return None
    finally:
        con.close()


def database(d: Path, key: str) -> dict:
    """Tables of one attached SQLite database: columns, keys, row counts."""
    db = _db(d, key)
    con = connect(db)
    try:
        tables = []
        for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
            q = name.replace('"', '""')
            cols = [{"name": c[1], "type": c[2], "pk": bool(c[5]), "notnull": bool(c[3])}
                    for c in con.execute(f'PRAGMA table_info("{q}")')]
            fks = [{"column": f[3], "ref_table": f[2], "ref_column": f[4]} for f in con.execute(f'PRAGMA foreign_key_list("{q}")')]
            tables.append({"name": name, "rows": _row_count(str(db), db.stat().st_mtime, name), "columns": cols, "foreign_keys": fks})
    finally:
        con.close()
    ref = load_ref(d)
    k = task_of(ref).attachments["key"]
    used = {s: sum(1 for r in split_rows(d, s) if r.get(k) == key) for s in ref["splits"]}
    return {"key": key, "file": str(db.relative_to(REPO)), "bytes": db.stat().st_size, "tables": tables, "items": used}


def table_rows(d: Path, key: str, table: str, limit: int = 50, offset: int = 0) -> dict:
    db = _db(d, key)
    con = connect(db)
    try:
        names = {n for (n,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if table not in names:
            raise KeyError(table)
        q = table.replace('"', '""')
        total = _row_count(str(db), db.stat().st_mtime, table)
        cur = con.execute(f'SELECT * FROM "{q}" LIMIT ? OFFSET ?', (min(limit, 500), offset))
        cols = [c[0] for c in cur.description]
        rows = [[v if isinstance(v, (int, float, str)) or v is None else repr(v)[:200] for v in r] for r in cur.fetchall()]
    finally:
        con.close()
    return {"table": table, "total": total, "columns": cols, "rows": rows}


def export(d: Path, split: str, fmt: str) -> tuple[bytes, str]:
    """A split as JSONL or CSV (nested values as JSON), with its media type; Parquet is served as the file."""
    rows = split_rows(d, split)
    if fmt == "jsonl":
        return ("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows)).encode(), "application/x-ndjson"
    if fmt == "csv":
        buf = io.StringIO()
        cols = list(rows[0]) if rows else []
        w = csv.writer(buf)
        w.writerow(cols)
        for r in rows:
            w.writerow([json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else v for v in (r.get(c) for c in cols)])
        return buf.getvalue().encode("utf-8-sig"), "text/csv"
    raise ValueError(f"unknown format {fmt}")
