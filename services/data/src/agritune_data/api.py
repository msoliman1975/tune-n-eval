"""Data service REST API.

    bash run.sh api            # http://127.0.0.1:8101, docs at /docs

- Tasks: the task specs datasets are mapped to (tasks/*.json).
- Drafts: upload a file -> map columns to a task -> attach files -> browse/edit -> freeze (drafts.py).
- Datasets: frozen versions, their items, attachments and rendered context (e.g. a DB schema).
- Mango pipeline (read side): sources and generation progress; the stages run from the CLI.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from agritune_core.tasks import list_tasks, load_task, task_of
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from . import ARTIFACTS, drafts, explore
from .collect import RAW, load_sources
from .drafts import DATASETS, DRAFTS, REPO, Mapping
from .parse import PARSED

app = FastAPI(title="agritune-data", version="0.2.0")


def jsonl(path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def bad_request(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except KeyError as e:
        raise HTTPException(404, f"not found: {e}")
    except (ValueError, FileExistsError) as e:
        raise HTTPException(400, str(e))


# ---------- tasks ----------

@app.get("/tasks")
def tasks() -> list[dict]:
    return [t.public() for t in list_tasks()]


@app.get("/tasks/{ref}")
def task(ref: str) -> dict:
    return bad_request(load_task, ref).public()


# ---------- drafts ----------

@app.post("/drafts", status_code=201)
async def upload(file: UploadFile = File(...), name: str | None = Form(None)) -> dict:
    return bad_request(drafts.create, file.filename or "upload.jsonl", await file.read(), name)


class FromDataset(BaseModel):
    dataset: str  # dataset version folder, relative to the repo
    name: str | None = None


@app.post("/drafts/from-dataset", status_code=201)
def from_dataset(req: FromDataset) -> dict:
    """An editable copy of a frozen dataset, to fix items and freeze as a new version."""
    p = resolve_dataset(req.dataset)
    return bad_request(drafts.from_dataset, p, req.name)


@app.get("/drafts")
def list_drafts() -> list[dict]:
    if not DRAFTS.exists():
        return []
    out = []
    for p in sorted(DRAFTS.glob("*/draft.json"), reverse=True):
        d = json.loads(p.read_text(encoding="utf-8"))
        out.append({k: d.get(k) for k in ("id", "name", "filename", "n_rows", "status", "task", "counts",
                                           "frozen_as", "created_at", "updated_at")})
    return out


@app.get("/drafts/{draft_id}")
def get_draft(draft_id: str) -> dict:
    return bad_request(drafts.meta, draft_id)


@app.delete("/drafts/{draft_id}", status_code=204)
def delete_draft(draft_id: str) -> None:
    bad_request(drafts.meta, draft_id)
    import shutil

    shutil.rmtree(DRAFTS / draft_id)


@app.get("/drafts/{draft_id}/columns/{column}")
def column_values(draft_id: str, column: str, top: int = 50) -> dict:
    """Distinct values of one uploaded column with counts, e.g. to map split labels."""
    info = bad_request(drafts.meta, draft_id)
    if column not in info["columns"]:
        raise HTTPException(404, f"no column {column}")
    from collections import Counter

    counts = Counter(str(r.get(column)) for r in drafts.jsonl_read(DRAFTS / draft_id / "rows.jsonl"))
    return {"column": column, "distinct": len(counts), "values": dict(counts.most_common(top))}


@app.put("/drafts/{draft_id}/mapping")
def set_mapping(draft_id: str, m: Mapping) -> dict:
    """Maps columns to the task's fields and assigns splits. Re-mapping discards item edits."""
    return bad_request(drafts.set_mapping, draft_id, m)


@app.post("/drafts/{draft_id}/attachments")
async def attach_zip(draft_id: str, file: UploadFile = File(...)) -> dict:
    """A zip laid out as the task says, e.g. <db_id>/<db_id>.sqlite (one top folder is stripped)."""
    return bad_request(drafts.attach_zip, draft_id, await file.read())


class AttachPath(BaseModel):
    path: str


@app.post("/drafts/{draft_id}/attachments/path")
def attach_path(draft_id: str, req: AttachPath) -> dict:
    """Links a folder on this machine, for attachments too big to upload through the browser."""
    return bad_request(drafts.attach_path, draft_id, req.path)


@app.get("/drafts/{draft_id}/items")
def draft_items(draft_id: str, split: str | None = None, q: str | None = None, offset: int = 0, limit: int = 50) -> dict:
    bad_request(drafts.meta, draft_id)
    return drafts.list_items(draft_id, split, q, offset, min(limit, 500))


@app.patch("/drafts/{draft_id}/items/{item_id}")
def edit_item(draft_id: str, item_id: str, fields: dict = Body(...)) -> dict:
    return bad_request(drafts.edit_item, draft_id, item_id, fields)


@app.delete("/drafts/{draft_id}/items/{item_id}", status_code=204)
def delete_item(draft_id: str, item_id: str) -> None:
    bad_request(drafts.delete_item, draft_id, item_id)


@app.get("/drafts/{draft_id}/preview")
def preview(draft_id: str, n: int = 3, item_id: str | None = None) -> list[dict]:
    info = bad_request(drafts.meta, draft_id)
    if not info.get("task"):
        raise HTTPException(400, "map the columns to a task first")
    return drafts.preview(draft_id, min(n, 20), item_id)


class FreezeRequest(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    version: str = Field(pattern=r"^v[0-9]+\.[0-9]+$")
    drop_invalid: bool = False  # drop invalid items instead of failing
    check_gold: bool = True  # SQL tasks: run every gold query against its database


@app.post("/drafts/{draft_id}/freeze", status_code=202)
def freeze(draft_id: str, req: FreezeRequest) -> dict:
    info = bad_request(drafts.meta, draft_id)
    if info["status"] not in ("mapped", "freeze_failed"):
        raise HTTPException(409, f"draft is {info['status']}; map it first" if info["status"] == "uploaded" else
                            f"draft is {info['status']}")
    if (DATASETS / req.id / req.version).exists():
        raise HTTPException(409, f"{req.id}@{req.version} exists; versions are immutable, pick a new one")
    drafts.save_meta(draft_id, status="freezing", error=None)
    threading.Thread(target=drafts.freeze, args=(draft_id, req.id, req.version, req.drop_invalid, req.check_gold),
                     daemon=True).start()
    return {"id": draft_id, "status": "freezing"}


# ---------- frozen datasets ----------

def resolve_dataset(p: str) -> Path:
    path = Path(p)
    path = path if path.is_absolute() else REPO / path
    if not (path / "dataset_ref.json").exists():
        raise HTTPException(404, f"no dataset_ref.json in {p}")
    return path


def dataset_dir(ds_id: str, version: str) -> Path:
    d = DATASETS / ds_id / version
    if not (d / "dataset_ref.json").exists():
        raise HTTPException(404, "no such dataset")
    return d


@app.get("/datasets")
def datasets() -> list[dict]:
    refs = sorted(DATASETS.glob("*/*/dataset_ref.json"), reverse=True) if DATASETS.exists() else []
    out = []
    for p in refs:
        ref = json.loads(p.read_text(encoding="utf-8"))
        out.append(ref | {"task": task_of(ref).ref, "path": str(p.parent.relative_to(REPO))})
    return out


@app.get("/datasets/{ds_id}/{version}")
def dataset(ds_id: str, version: str) -> dict:
    d = dataset_dir(ds_id, version)
    ref = json.loads((d / "dataset_ref.json").read_text(encoding="utf-8"))
    return ref | {"task": task_of(ref).ref, "path": str(d.relative_to(REPO))}


@app.get("/datasets/{ds_id}/{version}/items")
def items(ds_id: str, version: str, split: str = "train", q: str | None = None, limit: int = 50, offset: int = 0,
          field: str | None = None, value: str | None = None) -> dict:
    """Items of one split; `q` searches every field, `field`=`value` filters one (e.g. db_id)."""
    import pyarrow.parquet as pq

    path = dataset_dir(ds_id, version) / f"{split}.parquet"
    if not path.exists():
        raise HTTPException(404, "no such dataset split")
    rows = pq.read_table(path).to_pylist()
    if field and value is not None:
        rows = [r for r in rows if str(r.get(field)) == value]
    if q:
        ql = q.lower()
        rows = [r for r in rows if ql in json.dumps(r, ensure_ascii=False, default=str).lower()]
    return {"total": len(rows), "items": rows[offset:offset + min(limit, 500)]}


@app.get("/datasets/{ds_id}/{version}/profile")
def dataset_profile(ds_id: str, version: str) -> dict:
    """Every column profiled (type, role, nulls, distinct, lengths, top values), prompt sizes per split,
    attachment coverage and overlap between splits, and the files on disk."""
    return explore.profile(dataset_dir(ds_id, version))


@app.get("/datasets/{ds_id}/{version}/items/{item_id}")
def dataset_item(ds_id: str, version: str, item_id: str, split: str) -> dict:
    """One item in full, with the messages a model gets for it."""
    return bad_request(explore.item, dataset_dir(ds_id, version), split, item_id)


@app.get("/datasets/{ds_id}/{version}/download")
def download(ds_id: str, version: str, split: str, fmt: str = "jsonl"):
    d = dataset_dir(ds_id, version)
    if not (d / f"{split}.parquet").exists():
        raise HTTPException(404, "no such dataset split")
    name = f"{ds_id}-{version}-{split}"
    if fmt == "parquet":
        return FileResponse(d / f"{split}.parquet", filename=f"{name}.parquet")
    body, media = bad_request(explore.export, d, split, fmt)
    return Response(body, media_type=media, headers={"content-disposition": f'attachment; filename="{name}.{fmt}"'})


@app.get("/datasets/{ds_id}/{version}/attachments/{key}")
def attachment_detail(ds_id: str, version: str, key: str) -> dict:
    """One attached database: its tables, columns, keys, row counts and how many items use it."""
    return bad_request(explore.database, dataset_dir(ds_id, version), key)


@app.get("/datasets/{ds_id}/{version}/attachments/{key}/tables/{table}")
def attachment_table(ds_id: str, version: str, key: str, table: str, limit: int = 50, offset: int = 0) -> dict:
    """Rows of one table, read-only."""
    return bad_request(explore.table_rows, dataset_dir(ds_id, version), key, table, limit, offset)


@app.get("/datasets/{ds_id}/{version}/attachments")
def attachments(ds_id: str, version: str) -> dict:
    """The attachment keys (e.g. db_ids) a dataset ships, for the playground's database picker."""
    d = dataset_dir(ds_id, version)
    ref = json.loads((d / "dataset_ref.json").read_text(encoding="utf-8"))
    t = task_of(ref)
    if not (t.attachments and ref.get("attachments")):
        return {"type": None, "keys": []}
    root = REPO / ref["attachments"]
    key = t.attachments["key"]
    keys = sorted(p.name for p in root.iterdir() if p.is_dir() and t.attachment(root, {key: p.name}).exists())
    return {"type": t.attachments["type"], "key": key, "keys": keys}


@app.get("/datasets/{ds_id}/{version}/context")
def context(ds_id: str, version: str, key: str) -> dict:
    """Context fields rendered for one attachment, e.g. the schema of one database."""
    d = dataset_dir(ds_id, version)
    ref = json.loads((d / "dataset_ref.json").read_text(encoding="utf-8"))
    t = task_of(ref)
    if not (t.attachments and ref.get("attachments")):
        raise HTTPException(400, "this dataset has no attachments")
    row = {t.attachments["key"]: key}
    try:
        full = t.with_context(row, REPO / ref["attachments"])
    except FileNotFoundError:
        raise HTTPException(404, f"no attachment {key}")
    return {k: full[k] for k in t.context}


# ---------- mango document pipeline (read side) ----------

@app.get("/sources")
def sources() -> list[dict]:
    """The source manifest joined with download and parse status."""
    raw = {r["doc_id"]: r for r in jsonl(RAW / "index.jsonl")}
    parsed = {r["doc_id"]: r for r in jsonl(PARSED / "index.jsonl")}
    return [s | {"download": raw.get(s["doc_id"], {}).get("status"), "parse": parsed.get(s["doc_id"], {}).get("status"),
                 "chars": parsed.get(s["doc_id"], {}).get("chars")} for s in load_sources()]


@app.get("/pipeline")
def pipeline() -> dict:
    done = jsonl(ARTIFACTS / "synth_done.jsonl")
    chunks = ARTIFACTS / "chunks.jsonl"
    return {"chunks": sum(1 for _ in chunks.open()) if chunks.exists() else 0, "chunks_synthesized": len(done),
            "items": sum(d["items"] for d in done), "dropped": sum(d["dropped"] for d in done),
            "input_tokens": sum(d["input_tokens"] for d in done), "output_tokens": sum(d["output_tokens"] for d in done)}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8101)


if __name__ == "__main__":
    main()
