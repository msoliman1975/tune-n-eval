"""Builds and freezes the prepared NL→SQL datasets through the data service (what the Studio's
Datasets -> New dataset flow does by hand): upload -> map -> link the database folder -> freeze.

    python3 scripts/import_nl2sql.py bird               # after scripts/prepare_bird.py
    python3 scripts/import_nl2sql.py spider-variants    # after scripts/prepare_spider_variants.py

Items whose gold SQL does not run (or takes over 30 s) are dropped at freeze; the count and reasons
are kept in the dataset's validation report. An existing id@version is left alone.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DATA = "http://127.0.0.1:8101"
DL = Path.home() / "agritune-downloads"
VERSION = "v1.0"

SPIDER_LIKE = {"question": "question", "db_id": "db_id", "sql": "query"}
DATASETS = {
    "bird": [dict(
        id="bird", file=DL / "bird/agritune/bird.jsonl", db=DL / "bird/agritune/database", task="nl2sql-sqlite@v2",
        fields=SPIDER_LIKE | {"evidence": "evidence"}, meta=["difficulty", "source"],
        split={"mode": "column", "column": "split", "values": {"train": "train", "dev": "test"},
               "validation_n": 300, "group_by": "db_id", "seed": 13})],
    "spider-variants": [
        dict(id=name, file=DL / f"spider-variants/agritune/{name}.jsonl", db=DL / f"spider-variants/agritune/{name}-db",
             task="nl2sql-sqlite@v1", fields=SPIDER_LIKE, meta=meta, split={"mode": "single", "single": "test"})
        for name, meta in (("spider-syn", ["spider_question", "spider_id"]), ("spider-realistic", []),
                           ("spider-dk", ["category"]))],
}


def call(method: str, path: str, body: dict | None = None, data: bytes | None = None, ctype: str | None = None):
    if body is not None:
        data, ctype = json.dumps(body).encode(), "application/json"
    req = urllib.request.Request(DATA + path, data=data, method=method, headers={"Content-Type": ctype} if ctype else {})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path}: {e.code} {e.read().decode()[:400]}")


def upload(file: Path, name: str) -> dict:
    b = uuid.uuid4().hex
    parts = [f'--{b}\r\nContent-Disposition: form-data; name="name"\r\n\r\n{name}\r\n'.encode(),
             f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{file.name}"\r\n'
             f"Content-Type: application/jsonl\r\n\r\n".encode() + file.read_bytes() + b"\r\n",
             f"--{b}--\r\n".encode()]
    return call("POST", "/drafts", data=b"".join(parts), ctype=f"multipart/form-data; boundary={b}")


def build(spec: dict) -> None:
    tag = f"{spec['id']}@{VERSION}"
    if any(d["id"] == spec["id"] and d["version"] == VERSION for d in call("GET", "/datasets")):
        print(f"{tag} exists, skipped")
        return
    draft = upload(spec["file"], f"{spec['id']} {VERSION}")
    did = draft["id"]
    print(f"{tag}: draft {did}, {draft['n_rows']} rows")
    m = call("PUT", f"/drafts/{did}/mapping", {"task": spec["task"], "fields": spec["fields"], "id_column": "id",
                                               "meta_columns": spec["meta"], "split": spec["split"]})
    print(f"  mapped: {m['counts']}")
    a = call("POST", f"/drafts/{did}/attachments/path", {"path": str(spec["db"])})
    print(f"  attachments: {json.dumps(a.get('attachments') or a)[:200]}")
    call("POST", f"/drafts/{did}/freeze", {"id": spec["id"], "version": VERSION, "drop_invalid": True})
    while (d := call("GET", f"/drafts/{did}"))["status"] == "freezing":
        p = d.get("progress") or {}
        print(f"  freezing: checked {p.get('checked', 0)} of {p.get('total', '?')}", flush=True)
        time.sleep(30)
    if d["status"] != "frozen":
        raise SystemExit(f"  {d['status']}: {d.get('error')}")
    v = d["validation"]
    print(f"  frozen as {d['frozen_as']}; dropped {v['invalid']} of {v['checked']}: {v['reasons']}")


def main() -> None:
    which = sys.argv[1:] or list(DATASETS)
    for w in which:
        for spec in DATASETS[w]:
            build(spec)


if __name__ == "__main__":
    main()
