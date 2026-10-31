"""Freezes the built-in Strong Hire datasets (datasets/strong-hire/*.jsonl) through the data service, as
test-only datasets `<id>@v1.0`, ready for Evaluations -> New. Run once after `bash studio/up.sh`:

    python3 scripts/import_strong_hire.py                 # all nine
    python3 scripts/import_strong_hire.py sh-score sh-probe

Each file pairs with the task of the same id (tasks/<id>@v1.json). An existing id@version is left alone;
to load a rebuilt file, bump VERSION. See docs/runbook-strong-hire.md.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from import_nl2sql import call, upload

REPO = Path(__file__).resolve().parents[1]
FILES = REPO / "datasets" / "strong-hire"
VERSION = "v1.0"

# dataset id -> the row columns kept as metadata (slices and filters in the Studio)
META = {
    "sh-extract-posting": ["role_family", "planted_instruction", "prompt_refs"],
    "sh-extract-resume": ["file_kind", "planted_instruction", "prompt_refs"],
    "sh-input-check": ["expected", "doc_kind", "expected_input", "prompt_refs"],
    "sh-gap": ["fit", "mode", "posting_family", "resume_family", "prompt_refs"],
    "sh-score": ["interview_type", "quality", "gold_signal", "mode", "prompt_refs"],
    "sh-probe": ["expected", "interview_type", "quality", "prompt_refs"],
    "sh-answer-check": ["expected", "transcript", "prompt_refs"],
    "sh-interviewer-turn": ["move", "interview_type", "prompt_refs"],
    "sh-coach-turn": ["candidate", "style", "prompt_refs"],
}


def build(ds_id: str) -> None:
    tag = f"{ds_id}@{VERSION}"
    if any(d["id"] == ds_id and d["version"] == VERSION for d in call("GET", "/datasets")):
        print(f"{tag} exists, skipped")
        return
    draft = upload(FILES / f"{ds_id}.jsonl", f"{ds_id} {VERSION}")
    did = draft["id"]
    print(f"{tag}: draft {did}, {draft['n_rows']} rows")
    m = call("PUT", f"/drafts/{did}/mapping", {
        "task": f"{ds_id}@v1", "fields": {"title": "title", "messages": "messages", "reference": "reference"},
        "id_column": "id", "meta_columns": META[ds_id], "split": {"mode": "single", "single": "test"}})
    print(f"  mapped: {m['counts']}")
    call("POST", f"/drafts/{did}/freeze", {"id": ds_id, "version": VERSION, "drop_invalid": False})
    while (d := call("GET", f"/drafts/{did}"))["status"] == "freezing":
        time.sleep(2)
    if d["status"] != "frozen":
        raise SystemExit(f"  {d['status']}: {d.get('error')} {json.dumps(d.get('validation') or {})[:400]}")
    print(f"  frozen as {d['frozen_as']}")


def main() -> None:
    for ds_id in sys.argv[1:] or list(META):
        if ds_id not in META:
            raise SystemExit(f"unknown dataset {ds_id}; one of {', '.join(META)}")
        build(ds_id)


if __name__ == "__main__":
    main()
