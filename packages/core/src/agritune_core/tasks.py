"""Task specs (tasks/<id>@<version>.json): how a dataset row becomes chat messages and a target.

The same Task renders training examples, evaluation prompts and playground prompts, so a model is
always asked in the words it was trained on. A spec has:

- system, user_template: the prompt; `{name}` placeholders are filled from the row.
- inputs: the row fields a user supplies (e.g. question, db_id); target: the field to learn/score.
- optional: inputs that may be empty (e.g. BIRD's evidence); `[[...]]` in the template is a segment
  rendered only when every placeholder inside it has a value.
- context: fields derived from attachments, e.g. `schema` rendered from the row's SQLite database.
- attachments: how attachment files are laid out, e.g. "{db_id}/{db_id}.sqlite".
- metrics: the metric ids that make sense for the task, and the default selection.

Tasks that come from another product (e.g. the Strong Hire tasks, tasks/sh-*.json) carry their
prompts in the row instead of a template, because the product builds them in code:

- messages_field: the row field holding the chat messages as JSON: a list of {role, content}, or
  {"default": [...], "<tier>": [...]} when the product sends a different prompt to some models
  (e.g. "small"). A registry model's `prompt_tier` picks the variant; any other tier gets "default".
- output: {"format": "json", "schema": "<path under tasks/>"}: the answer is JSON that must match
  this JSON Schema. Targets that support it constrain decoding to the schema.
- scoring: settings the task's metrics read (field lists, label paths, thresholds).
- slices: row meta keys to report scores by (e.g. the expected label).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import TASKS_DIR

VAR = re.compile(r"(?<!\{)\{(\w+)\}(?!\})")
OPTIONAL = re.compile(r"\[\[(.*?)\]\]", re.S)
LEGACY_TASK = "mango-qa@v1"  # datasets frozen before tasks existed (mango-sft) carry no task


@dataclass
class Task:
    id: str
    version: str
    name: str
    user_template: str
    target: str
    inputs: list[str]
    optional: list[str] = field(default_factory=list)
    system: str = ""
    description: str = ""
    context: dict = field(default_factory=dict)
    attachments: dict | None = None
    decoding: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    rubric: str | None = None
    row_filter: dict = field(default_factory=dict)
    messages_field: str | None = None
    output: dict = field(default_factory=dict)
    scoring: dict = field(default_factory=dict)
    slices: list[str] = field(default_factory=list)

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    @property
    def variables(self) -> list[str]:
        return VAR.findall(self.user_template)

    @property
    def json_schema(self) -> dict | None:
        """The answer's JSON Schema, for tasks whose output format is json."""
        if self.output.get("format") != "json" or not self.output.get("schema"):
            return None
        return json.loads((TASKS_DIR / self.output["schema"]).read_text(encoding="utf-8"))

    @property
    def required(self) -> list[str]:
        """Inputs every row must have."""
        return [f for f in self.inputs if f not in self.optional]

    def keep(self, row: dict) -> bool:
        """Rows the task applies to (mango datasets mix open questions and MCQ)."""
        return all(row.get(k) == v for k, v in self.row_filter.items())

    def user(self, row: dict) -> str:
        if self.messages_field:  # the last user message, e.g. for a judge prompt
            return next((m["content"] for m in reversed(self.messages(row)) if m["role"] == "user"), "")

        def filled(seg: re.Match) -> str:
            return seg.group(1) if all(str(row.get(v) or "").strip() for v in VAR.findall(seg.group(1))) else ""

        text = OPTIONAL.sub(filled, self.user_template)
        return VAR.sub(lambda m: str(row.get(m.group(1)) or ""), text).replace("{{", "{").replace("}}", "}")

    def messages(self, row: dict, tier: str | None = None) -> list[dict]:
        if self.messages_field:
            return row_messages(row.get(self.messages_field), tier)
        msgs = [{"role": "system", "content": self.system}] if self.system else []
        return msgs + [{"role": "user", "content": self.user(row)}]

    def tiers(self, row: dict) -> list[str]:
        """The prompt variants a row carries besides the default."""
        if not self.messages_field:
            return []
        value = parse_json(row.get(self.messages_field))
        return sorted(k for k in value if k != "default") if isinstance(value, dict) else []

    def attachment(self, root: str | Path, row: dict) -> Path:
        if not self.attachments:
            raise ValueError(f"task {self.ref} has no attachments")
        return Path(root) / self.attachments["layout"].format(**row)

    def with_context(self, row: dict, attachments_root: str | Path | None) -> dict:
        """The row plus every context field it does not already carry (frozen datasets carry them)."""
        out = dict(row)
        for name, spec in self.context.items():
            if out.get(name):
                continue
            if spec["type"] == "sqlite_schema":
                from .sqlschema import render_schema

                out[name] = render_schema(self.attachment(attachments_root, row), spec.get("sample_rows", 3),
                                          spec.get("max_value_chars", 40))
            else:
                raise ValueError(f"unknown context type {spec['type']}")
        return out

    def public(self) -> dict:
        return {k: v for k, v in self.__dict__.items()} | {"ref": self.ref, "variables": self.variables}


def parse_json(value):
    """Frozen datasets store every field as a string; JSON fields come back parsed."""
    return json.loads(value) if isinstance(value, str) and value.strip()[:1] in ("[", "{") else value


def row_messages(value, tier: str | None = None) -> list[dict]:
    value = parse_json(value)
    if isinstance(value, dict):
        value = value.get(tier) if tier and value.get(tier) else value["default"]
    if not isinstance(value, list) or not all(isinstance(m, dict) and {"role", "content"} <= set(m) for m in value):
        raise ValueError("messages must be a JSON list of {role, content}")
    return [{"role": m["role"], "content": m["content"]} for m in value]


def load_task(ref: str) -> Task:
    path = TASKS_DIR / f"{ref}.json"
    if not path.exists():
        raise KeyError(f"no task {ref} (looked for {path})")
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec.pop("$schema", None)
    return Task(**spec)


def list_tasks() -> list[Task]:
    return [load_task(p.stem) for p in sorted(TASKS_DIR.glob("*@*.json"))]


def task_of(dataset_ref: dict) -> Task:
    return load_task(dataset_ref.get("task") or LEGACY_TASK)
