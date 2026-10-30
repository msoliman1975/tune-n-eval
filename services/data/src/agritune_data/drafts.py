"""Draft datasets: upload a file, map its columns to a task, attach files, browse and edit, freeze.

A draft lives in artifacts/data/drafts/{draft_id}/:
- source.<ext>: the uploaded file (JSONL, JSON, CSV or Parquet), kept as uploaded
- rows.jsonl: its rows, each with `_row` (position in the file)
- items.jsonl: the rows mapped to the task's fields plus `split`; edits and deletes change this file
- attachments/: files the task needs, e.g. one SQLite database per db_id
- draft.json: name, columns, mapping, counts, status

Freezing validates every item (required fields, attachments present, and for SQL tasks that the
gold query runs), renders context fields such as the schema into the rows, and writes an immutable
dataset version like the mango pipeline's: one Parquet file per split plus dataset_ref.json.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import threading
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from agritune_core.sqlexec import execute
from agritune_core.tasks import Task, load_task
from pydantic import BaseModel, Field

from . import ARTIFACTS, REPO

DRAFTS = ARTIFACTS / "drafts"
DATASETS = ARTIFACTS / "datasets"
SPLITS = ("train", "validation", "test")
_locks: dict[str, threading.Lock] = {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def lock(draft_id: str) -> threading.Lock:
    return _locks.setdefault(draft_id, threading.Lock())


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "dataset"


# ---------- reading uploads ----------

def read_rows(filename: str, data: bytes) -> list[dict]:
    ext = Path(filename).suffix.lower()
    if ext == ".jsonl":
        return [json.loads(l) for l in data.decode("utf-8-sig").splitlines() if l.strip()]
    if ext == ".json":
        obj = json.loads(data.decode("utf-8-sig"))
        rows = obj if isinstance(obj, list) else next((v for v in obj.values() if isinstance(v, list)), None)
        if rows is None:
            raise ValueError("a JSON upload must be a list of objects, or an object holding one")
        return rows
    if ext == ".csv":
        return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    if ext == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(io.BytesIO(data)).to_pylist()
    raise ValueError(f"unsupported file type {ext}; upload .jsonl, .json, .csv or .parquet")


def jsonl_read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def jsonl_write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def meta(draft_id: str) -> dict:
    p = DRAFTS / draft_id / "draft.json"
    if not p.exists():
        raise KeyError(draft_id)
    return json.loads(p.read_text(encoding="utf-8"))


def save_meta(draft_id: str, **kw) -> dict:
    d = meta(draft_id) | kw | {"updated_at": now()}
    (DRAFTS / draft_id / "draft.json").write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    return d


def create(filename: str, data: bytes, name: str | None = None) -> dict:
    rows = read_rows(filename, data)
    if not rows or not all(isinstance(r, dict) for r in rows):
        raise ValueError("the file holds no rows of named fields")
    name = name or Path(filename).stem
    draft_id = f"draft-{time.strftime('%Y%m%d-%H%M%S')}-{slug(name)}"
    d = DRAFTS / draft_id
    d.mkdir(parents=True)
    (d / f"source{Path(filename).suffix.lower()}").write_bytes(data)
    jsonl_write(d / "rows.jsonl", [r | {"_row": i} for i, r in enumerate(rows)])
    columns = list(dict.fromkeys(k for r in rows[:2000] for k in r))
    info = {"id": draft_id, "name": name, "filename": filename, "n_rows": len(rows), "columns": columns,
            "sample": rows[:3], "status": "uploaded", "mapping": None, "counts": {}, "attachments": None,
            "created_at": now()}
    (d / "draft.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return info


def from_dataset(ds_dir: Path, name: str | None = None) -> dict:
    """A new editable draft holding a frozen dataset's items, already mapped, with its attachments."""
    ref = json.loads((ds_dir / "dataset_ref.json").read_text(encoding="utf-8"))
    import pyarrow.parquet as pq

    rows = []
    for split, rel in ref["splits"].items():
        for r in pq.read_table(REPO / rel).to_pylist():
            m = json.loads(r.pop("meta") or "{}") if isinstance(r.get("meta"), str) else (r.pop("meta", None) or {})
            rows.append(r | {"split": split} | {f"meta.{k}": v for k, v in m.items()})
    buf = "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows).encode()
    info = create(f"{ref['id']}-{ref['version']}.jsonl", buf, name or f"{ref['id']} {ref['version']} (copy)")
    if ref.get("task"):
        task = load_task(ref["task"])
        fields = {v: v for v in [*task.inputs, task.target]}
        meta_cols = [c for c in info["columns"] if c.startswith("meta.")]
        set_mapping(info["id"], Mapping(task=task.ref, fields=fields, id_column="id", meta_columns=meta_cols,
                                        split=SplitSpec(mode="column", column="split",
                                                        values={s: s for s in ref["splits"]})))
    if ref.get("attachments"):
        attach_path(info["id"], str(REPO / ref["attachments"]))
    return meta(info["id"])


# ---------- mapping ----------

class SplitSpec(BaseModel):
    mode: Literal["column", "random", "single"] = "random"
    column: str | None = None  # mode column: the column holding a split label per row
    values: dict[str, str] = {}  # mode column: label in the file -> train | validation | test ("" drops)
    single: Literal["train", "validation", "test"] = "test"  # mode single: everything goes to one split
    validation_frac: float = Field(0.0, ge=0, lt=1)  # mode random
    test_frac: float = Field(0.1, ge=0, lt=1)  # mode random
    validation_n: int = Field(0, ge=0)  # any mode: move this many train items to validation
    group_by: str | None = None  # keep rows sharing this column's value in one split (e.g. db_id)
    seed: int = 13


class Mapping(BaseModel):
    task: str
    fields: dict[str, str]  # task field (input, target, or a context field you provide) -> column
    id_column: str | None = None
    meta_columns: list[str] = []
    split: SplitSpec = SplitSpec()


def assign_groups(keys: list[str], fracs: dict[str, float], seed: int) -> dict[str, str]:
    groups = sorted(set(keys))
    random.Random(seed).shuffle(groups)
    out, start = {}, 0
    for split, frac in fracs.items():
        n = round(len(groups) * frac)
        for g in groups[start:start + n]:
            out[g] = split
        start += n
    for g in groups[start:]:
        out[g] = "train"
    return out


def set_mapping(draft_id: str, m: Mapping) -> dict:
    task = load_task(m.task)
    d = DRAFTS / draft_id
    info = meta(draft_id)
    cols = set(info["columns"])
    need = [*task.inputs, task.target]
    missing = [f for f in [*task.required, task.target] if f not in m.fields]
    if missing:
        raise ValueError(f"map a column to every task field; missing {', '.join(missing)}")
    allowed = set(need) | set(task.context)
    unknown_fields = [f for f in m.fields if f not in allowed]
    if unknown_fields:
        raise ValueError(f"{', '.join(unknown_fields)} are not fields of {task.ref}")
    used = set(m.fields.values()) | set(m.meta_columns) | ({m.id_column} if m.id_column else set())
    used |= {c for c in (m.split.column, m.split.group_by) if c}
    absent = sorted(c for c in used if c not in cols)
    if absent:
        raise ValueError(f"no column {', '.join(absent)} in the file")

    with lock(draft_id):
        rows = jsonl_read(d / "rows.jsonl")
        items, seen, dropped = [], Counter(), 0
        for r in rows:
            base = str(r.get(m.id_column)) if m.id_column and r.get(m.id_column) not in (None, "") else f"r{r['_row']}"
            seen[base] += 1
            item_id = base if seen[base] == 1 else f"{base}-{seen[base]}"
            split = None
            if m.split.mode == "column":
                split = m.split.values.get(str(r.get(m.split.column)), "")
                if split not in SPLITS:
                    dropped += 1
                    continue
            elif m.split.mode == "single":
                split = m.split.single
            item = {"id": item_id, "split": split, "_row": r["_row"]}
            item |= {f: r.get(c) for f, c in m.fields.items()}
            item["meta"] = {c: r.get(c) for c in m.meta_columns}
            items.append(item)

        group = (lambda it: str(rows[it["_row"]].get(m.split.group_by))) if m.split.group_by else (lambda it: it["id"])
        if m.split.mode == "random":
            fracs = {"test": m.split.test_frac, "validation": m.split.validation_frac}
            by = assign_groups([group(it) for it in items], fracs, m.split.seed)
            for it in items:
                it["split"] = by[group(it)]
        if m.split.validation_n:
            train = [it for it in items if it["split"] == "train"]
            groups = sorted({group(it) for it in train})
            random.Random(m.split.seed + 1).shuffle(groups)
            size = Counter(group(it) for it in train)
            chosen, n = set(), 0
            for g in groups:
                if n >= m.split.validation_n:
                    break
                chosen.add(g)
                n += size[g]
            for it in train:
                if group(it) in chosen:
                    it["split"] = "validation"
        jsonl_write(d / "items.jsonl", items)
    counts = dict(Counter(it["split"] for it in items))
    return save_meta(draft_id, status="mapped", mapping=m.model_dump(), task=task.ref,
                     counts={s: counts.get(s, 0) for s in SPLITS}, dropped_unlabelled=dropped, edits=0,
                     attachments=attachment_status(draft_id, task, items))


# ---------- attachments ----------

def _link_tree(src: Path, dst: Path) -> int:
    n = 0
    for p in src.rglob("*"):
        if p.is_file():
            t = dst / p.relative_to(src)
            t.parent.mkdir(parents=True, exist_ok=True)
            if t.exists():
                t.unlink()
            try:
                os.link(p, t)  # same filesystem: no copy
            except OSError:
                shutil.copy2(p, t)
            n += 1
    return n


def attach_zip(draft_id: str, data: bytes) -> dict:
    meta(draft_id)
    dst = DRAFTS / draft_id / "attachments"
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if not n.endswith("/") and not n.startswith("__MACOSX")]
        tops = {n.split("/", 1)[0] for n in names}
        strip = next(iter(tops)) + "/" if len(tops) == 1 and all("/" in n for n in names) else ""
        for n in names:
            rel = Path(n[len(strip):])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe path in zip: {n}")
            t = dst / rel
            t.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as fsrc, t.open("wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
    return refresh_attachments(draft_id)


def attach_path(draft_id: str, src: str) -> dict:
    """Links (or copies) a folder on this machine into the draft; for attachments too big to upload."""
    meta(draft_id)
    p = Path(src).expanduser()
    if not p.is_dir():
        raise ValueError(f"{src} is not a folder on the server")
    _link_tree(p, DRAFTS / draft_id / "attachments")
    return refresh_attachments(draft_id)


def attachment_status(draft_id: str, task: Task, items: list[dict]) -> dict | None:
    if not task.attachments:
        return None
    root = DRAFTS / draft_id / "attachments"
    key = task.attachments["key"]
    needed = sorted({str(it.get(key)) for it in items if it.get(key) not in (None, "")})
    missing = [k for k in needed if not task.attachment(root, {key: k}).exists()]
    files = sum(1 for p in root.rglob("*") if p.is_file()) if root.exists() else 0
    return {"type": task.attachments["type"], "key": key, "files": files, "needed": len(needed),
            "missing": missing[:50], "n_missing": len(missing)}


def refresh_attachments(draft_id: str) -> dict:
    info = meta(draft_id)
    if not info.get("task"):
        files = sum(1 for p in (DRAFTS / draft_id / "attachments").rglob("*") if p.is_file())
        return save_meta(draft_id, attachments={"files": files, "note": "map the columns to a task to check them"})
    items = jsonl_read(DRAFTS / draft_id / "items.jsonl")
    return save_meta(draft_id, attachments=attachment_status(draft_id, load_task(info["task"]), items))


# ---------- browsing and editing ----------

def list_items(draft_id: str, split: str | None = None, q: str | None = None, offset: int = 0,
               limit: int = 50) -> dict:
    items = jsonl_read(DRAFTS / draft_id / "items.jsonl")
    if split:
        items = [it for it in items if it["split"] == split]
    if q:
        ql = q.lower()
        items = [it for it in items if ql in json.dumps(it, ensure_ascii=False).lower()]
    return {"total": len(items), "items": items[offset:offset + limit]}


def edit_item(draft_id: str, item_id: str, fields: dict) -> dict:
    info = meta(draft_id)
    task = load_task(info["task"])
    allowed = {*task.inputs, task.target, *task.context, "split", "meta"}
    bad = [k for k in fields if k not in allowed]
    if bad:
        raise ValueError(f"cannot edit {', '.join(bad)}")
    if "split" in fields and fields["split"] not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}")
    with lock(draft_id):
        path = DRAFTS / draft_id / "items.jsonl"
        items = jsonl_read(path)
        for it in items:
            if it["id"] == item_id:
                it.update(fields)
                it["edited"] = True
                jsonl_write(path, items)
                break
        else:
            raise KeyError(item_id)
    counts = Counter(i["split"] for i in items)
    save_meta(draft_id, edits=info.get("edits", 0) + 1, counts={s: counts.get(s, 0) for s in SPLITS})
    return it


def delete_item(draft_id: str, item_id: str) -> None:
    info = meta(draft_id)
    with lock(draft_id):
        path = DRAFTS / draft_id / "items.jsonl"
        items = jsonl_read(path)
        keep = [it for it in items if it["id"] != item_id]
        if len(keep) == len(items):
            raise KeyError(item_id)
        jsonl_write(path, keep)
    counts = Counter(i["split"] for i in keep)
    save_meta(draft_id, edits=info.get("edits", 0) + 1, counts={s: counts.get(s, 0) for s in SPLITS})


def preview(draft_id: str, n: int = 3, item_id: str | None = None) -> list[dict]:
    """Items rendered exactly as Train and Eval will see them, with a rough token count."""
    info = meta(draft_id)
    task = load_task(info["task"])
    items = jsonl_read(DRAFTS / draft_id / "items.jsonl")
    if item_id:
        items = [it for it in items if it["id"] == item_id]
    else:  # one per split first, so every split shows up
        firsts = {}
        for it in items:
            firsts.setdefault(it["split"], it)
        items = list(firsts.values()) + [it for it in items if it not in firsts.values()]
    out = []
    for it in items[:n]:
        try:
            row = task.with_context(it, DRAFTS / draft_id / "attachments")
            msgs = task.messages(row)
            chars = sum(len(m["content"]) for m in msgs) + len(str(row.get(task.target) or ""))
            out.append({"id": it["id"], "split": it["split"], "messages": msgs, "target": row.get(task.target),
                        "chars": chars, "est_tokens": round(chars / 3.2)})
        except Exception as e:
            out.append({"id": it["id"], "split": it["split"], "error": f"{type(e).__name__}: {e}"})
    return out


# ---------- freezing ----------

def check_item(task: Task, it: dict, root: Path, check_gold: bool) -> tuple[dict | None, str | None]:
    for f in [*task.required, task.target]:
        if it.get(f) in (None, "") or (isinstance(it.get(f), str) and not it[f].strip()):
            return None, f"empty {f}"
    try:
        row = task.with_context(it, root)
    except FileNotFoundError:
        return None, "attachment missing"
    except Exception as e:
        return None, f"context: {type(e).__name__}"
    if check_gold and task.attachments and task.attachments["type"] == "sqlite_dbs":
        r = execute(task.attachment(root, it), str(it[task.target]), timeout_s=30)
        if not r.ok:
            return None, "gold query fails" + (" (timeout)" if r.error.startswith("timeout") else "")
    return row, None


def freeze(draft_id: str, ds_id: str, version: str, drop_invalid: bool = False, check_gold: bool = True) -> None:
    """Runs in a thread; progress and the outcome go to draft.json (status freezing / frozen / freeze_failed)."""
    info = meta(draft_id)
    out = DATASETS / ds_id / version
    try:
        task = load_task(info["task"])
        root = DRAFTS / draft_id / "attachments"
        items = jsonl_read(DRAFTS / draft_id / "items.jsonl")
        good, reasons, examples = {s: [] for s in SPLITS}, Counter(), []
        for i, it in enumerate(items):
            row, why = check_item(task, it, root, check_gold)
            if why:
                reasons[why] += 1
                if len(examples) < 20:
                    examples.append({"id": it["id"], "reason": why})
            else:
                good[it["split"]].append(row)
            if i % 500 == 0:
                save_meta(draft_id, progress={"checked": i, "total": len(items)})
        report = {"checked": len(items), "invalid": sum(reasons.values()), "reasons": dict(reasons), "examples": examples}
        if reasons and not drop_invalid:
            save_meta(draft_id, status="freeze_failed", validation=report, progress=None,
                      error=f"{report['invalid']} invalid items; fix or delete them, or freeze with drop_invalid")
            return

        import pyarrow as pa
        import pyarrow.parquet as pq

        out.mkdir(parents=True)
        cols = ["id", "split", *dict.fromkeys([*task.inputs, *task.context, task.target])]
        paths, digest = {}, hashlib.sha256()
        for split in SPLITS:
            if not good[split]:
                continue
            table = [{c: (str(r.get(c)) if r.get(c) is not None else None) for c in cols}
                     | {"meta": json.dumps(r.get("meta") or {}, ensure_ascii=False, default=str),
                        "edited": bool(r.get("edited"))} for r in good[split]]
            p = out / f"{split}.parquet"
            pq.write_table(pa.Table.from_pylist(table), p)
            digest.update(p.read_bytes())
            paths[split] = str(p.relative_to(REPO))
        def chars(r: dict) -> int:
            return sum(len(m["content"]) for m in task.messages(r)) + len(str(r.get(task.target) or ""))

        stats = {"avg_chars": {s: round(sum(map(chars, good[s])) / len(good[s])) for s in SPLITS if good[s]}}
        att = None
        if task.attachments and root.exists():
            _link_tree(root, out / "attachments")
            att = str((out / "attachments").relative_to(REPO))
        m = info["mapping"]
        split_by = {"column": f"column:{m['split']['column']}", "random": "random", "single": f"single:{m['split']['single']}"}
        split_by = split_by[m["split"]["mode"]] + (f"+group:{m['split']['group_by']}" if m["split"].get("group_by") else "")
        ref = {"id": ds_id, "version": version, "task": task.ref, "sha256": digest.hexdigest(), "splits": paths,
               "counts": {s: len(good[s]) for s in SPLITS if good[s]}, "split_by": split_by, "stats": stats,
               "fields": {"inputs": task.inputs, "target": task.target, "context": list(task.context)},
               "attachments": att, "source": {"draft": draft_id, "filename": info["filename"], "rows": info["n_rows"],
                                              "edits": info.get("edits", 0)},
               "validation": report, "created_at": now(), "workspace_id": "default"}
        (out / "dataset_ref.json").write_text(json.dumps(ref, indent=2, ensure_ascii=False), encoding="utf-8")
        save_meta(draft_id, status="frozen", frozen_as=f"{ds_id}@{version}", validation=report, progress=None, error=None)
    except Exception as e:
        if out.exists() and not (out / "dataset_ref.json").exists():
            shutil.rmtree(out)  # never leave a half-written version behind
        save_meta(draft_id, status="freeze_failed", error=f"{type(e).__name__}: {e}", progress=None)
        raise
