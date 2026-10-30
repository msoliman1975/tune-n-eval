"""Model entries on disk, the seed catalog, and the sync that registers finished training runs."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from agritune_core import ARTIFACTS
from pydantic import BaseModel, Field, model_validator

ROOT = Path(os.environ.get("AGRITUNE_REGISTRY_DIR", ARTIFACTS / "registry" / "models"))
RUNS = Path(os.environ.get("AGRITUNE_RUNS_DIR", ARTIFACTS / "train"))
ID = r"^[a-z0-9][a-z0-9._-]*$"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ModelEntry(BaseModel):
    """contracts/model-entry.schema.json"""

    id: str = Field(pattern=ID)
    name: str
    kind: Literal["hf_base", "hf_adapter", "ollama", "api"]
    provider: Literal["local", "ollama", "anthropic", "google"] = "local"
    hf_id: str | None = None  # hf_base: the checkpoint; hf_adapter: the base checkpoint it was trained on
    adapter_uri: str | None = None  # hf_adapter: artifact path of the adapter folder
    ollama_tag: str | None = None  # kind ollama, or an hf model exported to Ollama
    api_model: str | None = None  # kind api: the provider's model id
    chat_template: str | None = None
    train_preset: str | None = None  # set when the Train service can fine-tune this model
    parent: str | None = None  # registry id of the model a fine-tune started from
    lineage: dict = Field(default_factory=dict)  # dataset, task, run_id, preset, mlflow_run_id
    paid: bool = False
    options: dict = Field(default_factory=dict)  # provider request options, e.g. {"effort": "low"} for Claude
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    evaluations: list[dict] = Field(default_factory=list)  # appended by the Eval service
    created_at: str = Field(default_factory=now)
    updated_at: str = Field(default_factory=now)

    @model_validator(mode="after")
    def check(self):
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("a model needs a name")
        self.tags = list(dict.fromkeys(t.strip().lower()[:40] for t in self.tags if t.strip()))
        need = {"hf_base": ["hf_id"], "hf_adapter": ["hf_id", "adapter_uri"], "ollama": ["ollama_tag"],
                "api": ["api_model"]}[self.kind]
        missing = [f for f in need if not getattr(self, f)]
        if missing:
            raise ValueError(f"kind {self.kind} needs {', '.join(missing)}")
        if self.kind == "api" and self.provider not in ("anthropic", "google"):
            raise ValueError("an api model needs provider anthropic or google")
        if self.kind == "ollama":
            self.provider = "ollama"
        return self

    @property
    def servable(self) -> bool:
        """Usable in the playground: answers come from Ollama or an API, without loading weights here."""
        return self.kind in ("ollama", "api") or bool(self.ollama_tag)


def path(model_id: str) -> Path:
    if not re.match(ID, model_id):
        raise KeyError(model_id)
    return ROOT / f"{model_id}.json"


def get(model_id: str) -> ModelEntry:
    p = path(model_id)
    if not p.exists():
        raise KeyError(model_id)
    return ModelEntry.model_validate_json(p.read_text(encoding="utf-8"))


def put(entry: ModelEntry) -> ModelEntry:
    ROOT.mkdir(parents=True, exist_ok=True)
    entry.updated_at = now()
    tmp = path(entry.id).with_suffix(".tmp")
    tmp.write_text(entry.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path(entry.id))
    return entry


def all_models() -> list[ModelEntry]:
    if not ROOT.exists():
        return []
    return [ModelEntry.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(ROOT.glob("*.json"))]


def delete(model_id: str) -> None:
    p = path(model_id)
    if not p.exists():
        raise KeyError(model_id)
    p.unlink()


SEEDS = [
    dict(id="gemma-4-e4b", name="Gemma 4 E4B (instruct)", kind="hf_base", hf_id="unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
         chat_template="gemma4", train_preset="gemma4-e4b-qlora", ollama_tag="gemma4:e4b", tags=["gemma", "base"]),
    dict(id="gemma-4-12b", name="Gemma 4 12B (instruct, Ollama Q4)", kind="ollama", ollama_tag="gemma4:12b",
         tags=["gemma", "baseline-only"], notes="Does not train on 12 GB; evaluated through Ollama."),
    dict(id="qwen3-8b", name="Qwen3 8B", kind="hf_base", hf_id="unsloth/Qwen3-8B-unsloth-bnb-4bit",
         chat_template="qwen3", train_preset="qwen-8b-qlora", tags=["qwen", "base"]),
    dict(id="qwen3-1.7b", name="Qwen3 1.7B", kind="hf_base", hf_id="unsloth/Qwen3-1.7B",
         chat_template="qwen3", train_preset="qwen-1.7b-lora", tags=["qwen", "base", "smoke-test"]),
    dict(id="gemma-3-4b", name="Gemma 3 4B (instruct)", kind="hf_base", hf_id="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
         chat_template="gemma3", train_preset="gemma-4b-qlora", tags=["gemma", "base"]),
    dict(id="qwen3-14b", name="Qwen3 14B (Ollama Q4)", kind="ollama", ollama_tag="qwen3:14b",
         tags=["qwen", "judge"], notes="Local generator and judge."),
    dict(id="claude-sonnet-5-5", name="Claude Sonnet 5.5", kind="api", provider="anthropic",
         api_model="claude-sonnet-5-5", paid=True, tags=["api"], notes="Needs ANTHROPIC_API_KEY in services/eval/.env."),
    dict(id="gemini-flash", name="Gemini Flash (latest)", kind="api", provider="google", api_model="gemini-flash-latest",
         paid=True, tags=["api"],
         notes="Needs GEMINI_API_KEY in services/eval/.env. Check the model id against your key's model list."),
]


def seed() -> list[str]:
    """Adds seed models that are missing; never overwrites an entry (users may have edited it)."""
    added = []
    for s in SEEDS:
        if not path(s["id"]).exists():
            put(ModelEntry(**s))
            added.append(s["id"])
    return added


def default_name(base: str, lineage: dict, run_id: str) -> str:
    """e.g. "Gemma 4 E4B · spider@v1.0 · 30% · Oct 26 14:05": base, data, share, and when it was launched."""
    base = base.split(" (")[0]
    share = f" · {lineage['train_fraction']:.0%}" if lineage.get("train_fraction") else ""
    try:
        launched = " · " + datetime.strptime("-".join(run_id.split("-")[1:3]), "%Y%m%d-%H%M%S").strftime("%b %d %H:%M")
    except ValueError:
        launched = ""
    return f"{base} · {lineage.get('dataset', '?')}{share}{launched}"


def name_taken(name: str, except_id: str | None = None) -> bool:
    return any(m.name.casefold() == name.strip().casefold() and m.id != except_id for m in all_models())


def unique_name(name: str) -> str:
    n, out = 2, name
    while name_taken(out):
        out, n = f"{name} ({n})", n + 1
    return out


def sync_runs() -> list[str]:
    """Registers every finished training run (model_ref.json) that the registry does not know yet,
    and records Ollama exports (model_ref.<quant>.json) on the run's entry."""
    if not RUNS.exists():
        return []
    by_hf = {m.hf_id: m for m in all_models() if m.kind == "hf_base"}
    added = []
    for ref_path in sorted(RUNS.glob("*/model_ref.json")):
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        run_id = ref["id"]
        exports = [json.loads(p.read_text()) for p in sorted(ref_path.parent.glob("model_ref.*.json"))]
        tag = next((e.get("ollama_tag") for e in exports if e.get("ollama_tag")), None)
        if path(run_id).exists():
            entry = get(run_id)
            if tag and entry.ollama_tag != tag:
                entry.ollama_tag = tag
                put(entry)
            continue
        parent = by_hf.get(ref["base"])
        lineage = dict(ref.get("lineage", {})) | {"run_id": run_id}
        label_path = ref_path.parent / "label.json"  # name and tags given when the run was launched
        label = json.loads(label_path.read_text(encoding="utf-8")) if label_path.exists() else {}
        dataset = lineage.get("dataset", "?")
        name = unique_name(label.get("name") or default_name(parent.name if parent else ref["base"], lineage, run_id))
        tags = ["fine-tuned", dataset.split("@")[0], *label.get("tags", [])]
        put(ModelEntry(id=run_id, name=name, kind="hf_adapter", hf_id=ref["base"], adapter_uri=ref["uri"],
                       chat_template=ref.get("chat_template"), parent=parent.id if parent else None,
                       lineage=lineage, ollama_tag=tag, tags=tags, created_at=ref.get("created_at", now())))
        added.append(run_id)
    return added
