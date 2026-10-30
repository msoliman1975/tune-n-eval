"""Model Registry REST API.

    bash run.sh api            # http://127.0.0.1:8104, docs at /docs

On start it adds missing seed models and registers finished training runs; Train also calls
POST /sync when a run succeeds, and Eval posts each evaluation's summary to the models it measured.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import store
from .store import ModelEntry


@asynccontextmanager
async def lifespan(_app):
    store.seed()
    store.sync_runs()
    yield


app = FastAPI(title="agritune-registry", version="0.1.0", lifespan=lifespan)


def view(m: ModelEntry) -> dict:
    return m.model_dump() | {"servable": m.servable, "trainable": bool(m.train_preset)}


def entry(model_id: str) -> ModelEntry:
    try:
        return store.get(model_id)
    except KeyError:
        raise HTTPException(404, f"no model {model_id}")


@app.get("/models")
def list_models(kind: str | None = None, trainable: bool | None = None, servable: bool | None = None) -> list[dict]:
    out = []
    for m in store.all_models():
        if kind and m.kind != kind:
            continue
        if trainable is not None and bool(m.train_preset) != trainable:
            continue
        if servable is not None and m.servable != servable:
            continue
        out.append(view(m))
    return out


@app.get("/models/{model_id}")
def get_model(model_id: str) -> dict:
    return view(entry(model_id))


def check_name(m: ModelEntry) -> None:
    if store.name_taken(m.name, m.id):
        raise HTTPException(409, f"another model is already named {m.name!r}; names tell models apart")


@app.post("/models", status_code=201)
def create(m: ModelEntry) -> dict:
    if store.path(m.id).exists():
        raise HTTPException(409, f"model {m.id} exists; use PUT to replace it")
    check_name(m)
    return view(store.put(m))


@app.put("/models/{model_id}")
def replace(model_id: str, m: ModelEntry) -> dict:
    if m.id != model_id:
        raise HTTPException(400, "id in the path and the body differ")
    old = store.get(model_id) if store.path(model_id).exists() else None
    if old:  # keep history the client may not have sent back
        m.created_at = old.created_at
        m.evaluations = m.evaluations or old.evaluations
    if not old or old.name != m.name:
        check_name(m)
    return view(store.put(m))


@app.patch("/models/{model_id}")
def update(model_id: str, fields: dict) -> dict:
    fields.pop("id", None)
    try:
        m = ModelEntry.model_validate(entry(model_id).model_dump() | fields)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if "name" in fields:
        check_name(m)
    return view(store.put(m))


@app.delete("/models/{model_id}", status_code=204)
def remove(model_id: str) -> None:
    """Removes the registry entry only. A fine-tune is deleted through the Train service
    (DELETE /runs/{id}), which also removes its files; its run folder would register it again."""
    m = entry(model_id)
    if m.kind == "hf_adapter" and (store.RUNS / model_id / "model_ref.json").exists():
        raise HTTPException(409, f"{model_id} is a fine-tune with files on disk; delete it with DELETE /api/train/runs/{model_id}")
    store.delete(model_id)


class EvalSummary(BaseModel):
    eval_id: str
    dataset: str
    task: str | None = None
    split: str | None = None
    n: int | None = None
    metrics: dict[str, float] = {}
    created_at: str | None = None


@app.post("/models/{model_id}/evaluations")
def add_evaluation(model_id: str, s: EvalSummary) -> dict:
    m = entry(model_id)
    m.evaluations = [e for e in m.evaluations if e.get("eval_id") != s.eval_id] + [s.model_dump()]
    return view(store.put(m))


@app.post("/sync")
def sync() -> dict:
    return {"seeded": store.seed(), "registered": store.sync_runs()}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8104)


if __name__ == "__main__":
    main()
