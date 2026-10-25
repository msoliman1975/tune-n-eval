"""Data service REST API (read side for M1; the pipeline itself runs from the CLI).

    bash run.sh api            # http://127.0.0.1:8101, docs at /docs
"""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException

from . import ARTIFACTS
from .collect import RAW, load_sources
from .freeze import DATASETS
from .parse import PARSED

app = FastAPI(title="agritune-data", version="0.1.0")


def jsonl(path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


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


@app.get("/datasets")
def datasets() -> list[dict]:
    refs = sorted(DATASETS.glob("*/*/dataset_ref.json"), reverse=True) if DATASETS.exists() else []
    return [json.loads(p.read_text(encoding="utf-8")) | {"path": str(p.parent.relative_to(ARTIFACTS.parents[1]))}
            for p in refs]


@app.get("/datasets/{ds_id}/{version}/items")
def items(ds_id: str, version: str, split: str = "train", limit: int = 50, offset: int = 0) -> list[dict]:
    import pyarrow.parquet as pq

    path = DATASETS / ds_id / version / f"{split}.parquet"
    if not path.exists():
        raise HTTPException(404, "no such dataset split")
    return pq.read_table(path).slice(offset, limit).to_pylist()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8101)


if __name__ == "__main__":
    main()
