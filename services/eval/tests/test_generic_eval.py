"""The generic evaluator end to end on a tiny SQL dataset, with canned answers instead of a GPU model,
then the answers explorer API on its output."""

import json
import shutil
import sqlite3
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from agritune_eval import REPO, run
from agritune_eval.api import EVALS, app
from agritune_eval.metrics import score_sql
from agritune_eval.spider_match import exact_set_match, hardness
from agritune_eval.targets import Answer
from agritune_core.sqlexec import execute

SCRATCH = REPO / "artifacts" / "_test-eval"
EVAL_ID = "eval-00000000-test"

ITEMS = [  # (id, question, gold)
    ("q1", "Names of all singers?", "SELECT name FROM singer"),
    ("q2", "How many singers?", "SELECT count(*) FROM singer"),
    ("q3", "Oldest singer?", "SELECT name FROM singer ORDER BY age DESC LIMIT 1"),
    ("q4", "Singers older than 30 by country?",
     "SELECT country, count(*) FROM singer WHERE age > 30 GROUP BY country"),
]
GOOD = {"q1": "```sql\nSELECT name FROM singer;\n```", "q2": "SELECT COUNT(*) FROM singer",
        "q3": "SELECT name FROM singer ORDER BY age DESC LIMIT 1",
        "q4": "SELECT country, COUNT(*) FROM singer WHERE age > 30 GROUP BY country"}
BAD = {"q1": "SELECT name, age FROM singer", "q2": "SELECT count(*) FROM singers",  # wrong columns; bad table
       "q3": "SELECT name FROM singer ORDER BY age ASC LIMIT 1", "q4": GOOD["q4"]}


@pytest.fixture(scope="module")
def dataset():
    shutil.rmtree(SCRATCH, ignore_errors=True)
    db = SCRATCH / "ds" / "v1.0" / "attachments" / "music" / "music.sqlite"
    db.parent.mkdir(parents=True)
    c = sqlite3.connect(db)
    c.executescript("CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INT, country TEXT);"
                    "INSERT INTO singer VALUES (1,'Ann',25,'EG'),(2,'Bo',41,'US'),(3,'Cy',35,'EG');")
    c.commit(); c.close()
    rows = [{"id": i, "split": "test", "question": q, "db_id": "music", "schema": None, "sql": s, "meta": "{}",
             "edited": False} for i, q, s in ITEMS]
    d = SCRATCH / "ds" / "v1.0"
    pq.write_table(pa.Table.from_pylist(rows), d / "test.parquet")
    (d / "dataset_ref.json").write_text(json.dumps({
        "id": "mini", "version": "v1.0", "task": "nl2sql-sqlite@v1", "splits": {"test": str((d / "test.parquet").relative_to(REPO))},
        "counts": {"test": 4}, "attachments": str((d / "attachments").relative_to(REPO))}))
    yield d
    shutil.rmtree(SCRATCH, ignore_errors=True)
    shutil.rmtree(EVALS / EVAL_ID, ignore_errors=True)


def test_spider_match(dataset):
    db = dataset / "attachments" / "music" / "music.sqlite"
    assert exact_set_match("SELECT name FROM singer WHERE age = 3", "select name from singer where age = 99", db) == 1.0
    assert exact_set_match("SELECT age FROM singer", "SELECT name FROM singer", db) == 0.0
    assert exact_set_match("not sql at all", "SELECT name FROM singer", db) == 0.0
    join = "SELECT T1.name FROM singer AS T1 {} singer AS T2 ON T1.id = T2.id"
    assert exact_set_match(join.format("INNER JOIN"), join.format("JOIN"), db) == 1.0  # a synonym, not an error
    assert exact_set_match(join.format("LEFT JOIN"), join.format("JOIN"), db) == 0.0
    assert hardness("SELECT name FROM singer", db) == "easy"
    assert hardness("SELECT country, count(*) FROM singer WHERE age > 30 GROUP BY country", db) == "medium"


def test_score_sql(dataset):
    db = dataset / "attachments" / "music" / "music.sqlite"
    gold_sql = "SELECT name FROM singer ORDER BY age DESC"
    gold = execute(db, gold_sql)
    want = {"sql_exec", "sql_valid", "sql_exact_set", "exact_match"}
    s, d = score_sql("```sql\nSELECT name FROM singer ORDER BY age DESC\n```", gold_sql, db, gold, want, None)
    assert s == {"sql_valid": 1.0, "sql_exec": 1.0, "sql_exact_set": 1.0, "exact_match": 1.0} and d["result"]["n_rows"] == 3
    s, _ = score_sql("SELECT name FROM singer ORDER BY age", gold_sql, db, gold, want, None)
    assert s["sql_exec"] == 0.0 and s["sql_valid"] == 1.0  # order matters under ORDER BY
    s, d = score_sql("SELECT nope FROM singer", gold_sql, db, gold, want, None)
    assert s["sql_valid"] == 0.0 and "no such column" in d["result"]["error"]


def test_run_and_explorer(dataset, monkeypatch):
    models = [{"id": "m-good", "kind": "ollama", "ollama_tag": "x", "name": "good", "paid": False},
              {"id": "m-bad", "kind": "ollama", "ollama_tag": "y", "name": "bad", "paid": False}]
    order = [i for i, _, _ in sorted(ITEMS)]

    def fake_generate(models_, convs, max_new, batch, ev):
        assert all(c[1]["content"].startswith("Database schema:\nCREATE TABLE singer") for c in convs)
        return {"m-good": [Answer(GOOD[i], 0.5, 900, 12) for i in order],
                "m-bad": [Answer(BAD[i], 1.5, 900, 15) for i in order]}

    monkeypatch.setattr(run, "generate_all", fake_generate)
    monkeypatch.setattr(run, "post_registry", lambda report: None)
    out = EVALS / EVAL_ID
    out.mkdir(parents=True, exist_ok=True)
    spec = {"id": EVAL_ID, "dataset": str(dataset.relative_to(REPO)), "splits": ["test"], "limit": 0, "seed": 7,
            "models": models, "metrics": ["sql_exec", "sql_valid", "sql_exact_set", "latency"]}
    (out / "spec.json").write_text(json.dumps(spec))
    monkeypatch.setattr(sys, "argv", ["run", "--spec", str(out / "spec.json")])
    run.main()

    rep = json.loads((out / "report.json").read_text())
    head = {(s["model"], s["metric"]): s["value"] for s in rep["scores"] if s["slice"]["key"] == "all"}
    assert head[("m-good", "sql_exec")] == 1.0 and head[("m-bad", "sql_exec")] == 0.25
    assert head[("m-good", "sql_valid")] == 1.0 and head[("m-bad", "sql_valid")] == 0.75
    assert head[("m-good", "latency")] == 0.5
    assert {s["slice"]["value"] for s in rep["scores"] if s["slice"]["key"] == "hardness"} >= {"easy", "medium"}
    lat = next(c for c in rep["comparisons"] if c["metric"] == "latency" and c["slice"]["key"] == "all")
    assert lat["a"] == "m-good" and lat["delta"] == 1.0 and lat["verdict"] == "a_better"  # faster is better
    assert rep["usage"]["m-good"]["input_tokens"] == 3600

    with TestClient(app) as c:
        listed = next(e for e in c.get("/evaluations").json() if e["id"] == EVAL_ID)
        assert listed["kind"] == "generic" and listed["headline"] == {"m-good": 1.0, "m-bad": 0.25}
        page = c.get(f"/evaluations/{EVAL_ID}/items", params={"model": "m-bad", "outcome": "fail"}).json()
        assert {it["id"] for it in page["items"]} == {"q1", "q2", "q3"}
        assert page["items"][0]["results"]["m-good"]["scores"]["sql_exec"] == 1.0
        assert c.get(f"/evaluations/{EVAL_ID}/items", params={"outcome": "disagree"}).json()["total"] == 3
        assert c.get(f"/evaluations/{EVAL_ID}/items", params={"slice": "hardness:medium"}).json()["total"] >= 1
        full = c.get(f"/evaluations/{EVAL_ID}/items/q2").json()
        assert full["messages"][1]["content"].endswith("Question: How many singers?")
        assert "no such table" in full["results"]["m-bad"]["details"]["result"]["error"]
        cat = c.get("/metrics", params={"task": "nl2sql-sqlite@v1"}).json()
        assert {m["id"] for m in cat["metrics"]} >= {"sql_exec", "sql_failure_judge"}
        assert "chrf" not in {m["id"] for m in cat["metrics"]}
