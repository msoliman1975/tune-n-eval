"""Upload -> map -> attach -> preview -> edit -> freeze, through the REST API, on a tiny SQL dataset."""

import io
import json
import os
import shutil
import sqlite3
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRATCH = REPO / "artifacts" / "_test-data"
shutil.rmtree(SCRATCH, ignore_errors=True)
os.environ["AGRITUNE_DATA_DIR"] = str(SCRATCH)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from agritune_data.api import app  # noqa: E402

c = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def cleanup():
    yield
    shutil.rmtree(SCRATCH, ignore_errors=True)


def db_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for db_id in ("shop", "zoo"):
            tmp = SCRATCH / "tmp" / f"{db_id}.sqlite"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(tmp)
            con.executescript("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT); INSERT INTO t VALUES (1, 'a'), (2, 'b');")
            con.commit(); con.close()
            z.write(tmp, f"database/{db_id}/{db_id}.sqlite")  # one top folder, stripped on upload
    return buf.getvalue()


ROWS = ([{"qid": f"s{i}", "question": f"q{i}?", "db": "shop", "query": "SELECT name FROM t", "part": "train",
          "hardness": "easy"} for i in range(8)]
        + [{"qid": f"z{i}", "question": f"zq{i}?", "db": "zoo", "query": "SELECT count(*) FROM t", "part": "train",
            "hardness": "medium"} for i in range(4)]
        + [{"qid": "d0", "question": "dev?", "db": "zoo", "query": "SELECT id FROM t", "part": "dev", "hardness": "hard"},
           {"qid": "x0", "question": "other", "db": "zoo", "query": "SELECT 1", "part": "ignored"}])


def wait_frozen(draft_id: str) -> dict:
    for _ in range(100):
        d = c.get(f"/drafts/{draft_id}").json()
        if d["status"] != "freezing":
            return d
        time.sleep(0.1)
    raise AssertionError("freeze did not finish")


def test_draft_to_dataset():
    assert "nl2sql-sqlite@v1" in {t["ref"] for t in c.get("/tasks").json()}

    body = "\n".join(json.dumps(r) for r in ROWS).encode()
    r = c.post("/drafts", files={"file": ("mini.jsonl", body)}, data={"name": "Mini SQL"})
    assert r.status_code == 201, r.text
    d = r.json()
    assert d["n_rows"] == 14 and {"qid", "question", "db", "query", "part"} <= set(d["columns"])
    did = d["id"]

    mapping = {"task": "nl2sql-sqlite@v1", "fields": {"question": "question", "db_id": "db", "sql": "query"},
               "id_column": "qid", "meta_columns": ["hardness"],
               "split": {"mode": "column", "column": "part", "values": {"train": "train", "dev": "test"},
                         "validation_n": 3, "group_by": "db"}}
    assert c.put(f"/drafts/{did}/mapping", json=mapping | {"fields": {"question": "question"}}).status_code == 400
    assert c.put(f"/drafts/{did}/mapping", json=mapping | {"id_column": "nope"}).status_code == 400
    d = c.put(f"/drafts/{did}/mapping", json=mapping).json()
    # the whole zoo group (4 rows) moves to validation: groups never straddle splits
    assert d["counts"] == {"train": 8, "validation": 4, "test": 1} and d["dropped_unlabelled"] == 1
    assert d["attachments"]["n_missing"] == 2

    d = c.post(f"/drafts/{did}/attachments", files={"file": ("dbs.zip", db_zip())}).json()
    assert d["attachments"]["n_missing"] == 0 and d["attachments"]["files"] == 2

    p = c.get(f"/drafts/{did}/preview", params={"n": 3}).json()
    assert {x["split"] for x in p} == {"train", "validation", "test"}
    assert p[0]["messages"][1]["content"].startswith("Database schema:\nCREATE TABLE t (")
    assert "2 example rows" in p[0]["messages"][1]["content"] and p[0]["target"]

    page = c.get(f"/drafts/{did}/items", params={"split": "train", "limit": 5}).json()
    assert page["total"] == 8 and len(page["items"]) == 5
    assert c.get(f"/drafts/{did}/items", params={"q": "dev?"}).json()["total"] == 1

    # a broken gold query fails the freeze, unless invalid items are dropped
    assert c.patch(f"/drafts/{did}/items/s1", json={"sql": "SELECT nope FROM t"}).json()["edited"]
    assert c.patch(f"/drafts/{did}/items/s1", json={"bogus": 1}).status_code == 400
    assert c.delete(f"/drafts/{did}/items/s2").status_code == 204
    assert c.delete(f"/drafts/{did}/items/s2").status_code == 404

    assert c.post(f"/drafts/{did}/freeze", json={"id": "mini-sql", "version": "v1"}).status_code == 422  # bad version
    assert c.post(f"/drafts/{did}/freeze", json={"id": "mini-sql", "version": "v1.0"}).status_code == 202
    d = wait_frozen(did)
    assert d["status"] == "freeze_failed" and d["validation"]["reasons"] == {"gold query fails": 1}

    c.post(f"/drafts/{did}/freeze", json={"id": "mini-sql", "version": "v1.0", "drop_invalid": True})
    d = wait_frozen(did)
    assert d["status"] == "frozen" and d["frozen_as"] == "mini-sql@v1.0", d

    ref = c.get("/datasets/mini-sql/v1.0").json()
    assert ref["task"] == "nl2sql-sqlite@v1" and ref["counts"] == {"train": 6, "validation": 4, "test": 1}
    assert ref["split_by"] == "column:part+group:db" and ref["source"]["edits"] == 2
    rows = c.get("/datasets/mini-sql/v1.0/items", params={"split": "test"}).json()["items"]
    assert rows[0]["id"] == "d0" and rows[0]["schema"].startswith("CREATE TABLE t")
    assert json.loads(rows[0]["meta"]) == {"hardness": "hard"}
    assert c.get("/datasets/mini-sql/v1.0/attachments").json()["keys"] == ["shop", "zoo"]
    assert c.get("/datasets/mini-sql/v1.0/context", params={"key": "zoo"}).json()["schema"].startswith("CREATE TABLE t")
    assert c.post(f"/drafts/{did}/freeze", json={"id": "mini-sql", "version": "v1.0"}).status_code == 409

    # a copy of the frozen dataset is a new draft, already mapped and attached
    copy = c.post("/drafts/from-dataset", json={"dataset": ref["path"]}).json()
    assert copy["status"] == "mapped" and copy["counts"] == {"train": 6, "validation": 4, "test": 1}
    assert copy["attachments"]["n_missing"] == 0
    assert any(x["id"] == did for x in c.get("/drafts").json())


def test_strong_hire_dataset_freezes_as_the_import_script_does():
    """scripts/import_strong_hire.py's mapping, on a built-in Strong Hire file: JSON fields stay JSON."""
    import sys

    sys.path.insert(0, str(REPO / "scripts"))  # the script imports its sibling import_nl2sql
    try:
        import import_strong_hire as mod
    finally:
        sys.path.pop(0)
    body = (REPO / "datasets" / "strong-hire" / "sh-probe.jsonl").read_bytes()
    did = c.post("/drafts", files={"file": ("sh-probe.jsonl", body)}, data={"name": "sh-probe"}).json()["id"]
    mapping = {"task": "sh-probe@v1", "fields": {"title": "title", "messages": "messages", "reference": "reference"},
               "id_column": "id", "meta_columns": mod.META["sh-probe"], "split": {"mode": "single", "single": "test"}}
    d = c.put(f"/drafts/{did}/mapping", json=mapping).json()
    assert d["counts"]["test"] == len(body.splitlines())
    p = c.get(f"/drafts/{did}/preview", params={"n": 1}).json()[0]
    assert p["messages"][0]["role"] == "system" and p["messages"][-1]["content"].startswith("Interview type:")
    c.post(f"/drafts/{did}/freeze", json={"id": "sh-probe", "version": "v1.0"})
    d = wait_frozen(did)
    assert d["status"] == "frozen", d
    row = c.get("/datasets/sh-probe/v1.0/items", params={"split": "test"}).json()["items"][0]
    assert json.loads(row["reference"])["expected"]["action"] in ("probe", "move_on")
    assert "default" in json.loads(row["messages"]) and json.loads(row["meta"])["expected"]
