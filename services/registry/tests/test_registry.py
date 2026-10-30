import json
import os
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
os.environ["AGRITUNE_REGISTRY_DIR"] = str(TMP / "models")
os.environ["AGRITUNE_RUNS_DIR"] = str(TMP / "train")

from fastapi.testclient import TestClient  # noqa: E402

from agritune_registry.api import app  # noqa: E402


def make_run(run_id: str, base: str, tag: str | None = None) -> None:
    d = TMP / "train" / run_id
    d.mkdir(parents=True)
    (d / "model_ref.json").write_text(json.dumps({
        "id": run_id, "kind": "hf_adapter", "base": base, "uri": f"artifacts/train/{run_id}/adapter",
        "chat_template": "gemma4", "lineage": {"dataset": "spider@v1.0", "preset": "gemma4-e4b-qlora"}}))
    if tag:
        (d / "model_ref.q4_k_m.json").write_text(json.dumps({"id": f"{run_id}-q4", "ollama_tag": tag}))


def test_seed_sync_crud_and_evaluations():
    make_run("run-1-gemma4-e4b-qlora", "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit")
    with TestClient(app) as c:  # the lifespan seeds and syncs
        models = {m["id"]: m for m in c.get("/models").json()}
        assert {"gemma-4-e4b", "gemma-4-12b", "claude-sonnet-5-5", "gemini-flash"} <= set(models)
        ft = models["run-1-gemma4-e4b-qlora"]
        assert ft["kind"] == "hf_adapter" and ft["parent"] == "gemma-4-e4b" and ft["lineage"]["run_id"] == ft["id"]
        assert not ft["servable"] and models["gemma-4-12b"]["servable"] and models["gemma-4-e4b"]["trainable"]
        assert {m["id"] for m in c.get("/models", params={"trainable": True}).json()} >= {"gemma-4-e4b", "qwen3-8b"}
        assert all(m["kind"] == "api" for m in c.get("/models", params={"kind": "api"}).json())

        # an export found by a later sync makes the fine-tune servable
        (TMP / "train" / "run-1-gemma4-e4b-qlora" / "model_ref.q4_k_m.json").write_text(json.dumps({"ollama_tag": "agritune/x:1"}))
        assert c.post("/sync").json()["registered"] == []
        assert c.get("/models/run-1-gemma4-e4b-qlora").json()["servable"]

        new = {"id": "llama-local", "name": "Llama", "kind": "ollama", "ollama_tag": "llama3.2:3b"}
        assert c.post("/models", json=new).status_code == 201
        assert c.post("/models", json=new).status_code == 409
        assert c.post("/models", json={"id": "bad", "name": "x", "kind": "api", "provider": "anthropic"}).status_code == 422
        assert c.post("/models", json={"id": "Bad Id", "name": "x", "kind": "ollama", "ollama_tag": "a"}).status_code == 422
        assert c.patch("/models/llama-local", json={"notes": "hi"}).json()["notes"] == "hi"
        assert c.patch("/models/llama-local", json={"ollama_tag": None}).status_code == 422

        s = {"eval_id": "eval-1", "dataset": "spider@v1.0", "metrics": {"sql_exec": 0.61}}
        c.post("/models/llama-local/evaluations", json=s)
        c.post("/models/llama-local/evaluations", json=s | {"metrics": {"sql_exec": 0.62}})  # same id replaces
        evs = c.get("/models/llama-local").json()["evaluations"]
        assert len(evs) == 1 and evs[0]["metrics"]["sql_exec"] == 0.62
        assert c.put("/models/llama-local", json=new).json()["evaluations"] == evs  # PUT keeps history

        assert c.delete("/models/llama-local").status_code == 204
        assert c.get("/models/llama-local").status_code == 404


def test_names_tags_and_delete_guard():
    make_run("run-20261026-140500-gemma4-e4b-qlora", "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit")
    labelled = "run-20261026-150000-gemma4-e4b-qlora"
    make_run(labelled, "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit")
    (TMP / "train" / labelled / "label.json").write_text(json.dumps({"name": "E4B best", "tags": ["Best", " v2 "]}))
    with TestClient(app) as c:
        a = c.get("/models/run-20261026-140500-gemma4-e4b-qlora").json()
        assert a["name"] == "Gemma 4 E4B · spider@v1.0 · Oct 26 14:05"
        assert a["tags"] == ["fine-tuned", "spider"]
        b = c.get(f"/models/{labelled}").json()
        assert b["name"] == "E4B best" and b["tags"] == ["fine-tuned", "spider", "best", "v2"]
        assert c.patch(f"/models/{labelled}", json={"name": "gemma 4 e4b · spider@v1.0 · oct 26 14:05"}).status_code == 409
        assert c.patch(f"/models/{labelled}", json={"name": "  "}).status_code == 422
        assert c.patch(f"/models/{labelled}", json={"notes": "kept"}).status_code == 200
        assert c.delete(f"/models/{labelled}").status_code == 409  # its run folder would register it again
