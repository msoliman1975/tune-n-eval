import sqlite3

import pytest

from agritune_core.sqlexec import ExecResult, execute, extract_sql, order_matters, results_match
from agritune_core.sqlschema import render_schema
from agritune_core.tasks import list_tasks, load_task


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop" / "shop.sqlite"
    path.parent.mkdir()
    c = sqlite3.connect(path)
    c.executescript("""
        CREATE TABLE country (code TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE "order item" (order_id INT, line INT, qty REAL, country TEXT,
                                   PRIMARY KEY (order_id, line), FOREIGN KEY (country) REFERENCES country(code));
        CREATE TABLE empty_t (x);
        INSERT INTO country VALUES ('EG', 'Egypt'), ('US', 'United States of America and a very long name indeed'),
                                   ('FR', 'France'), ('DE', 'Germany');
        INSERT INTO "order item" VALUES (1, 1, 2.5, 'EG'), (1, 2, 1, 'US');
    """)
    c.commit()
    c.close()
    return path


def test_schema_has_keys_and_samples(db):
    s = render_schema(db, sample_rows=3, max_value_chars=20)
    assert "CREATE TABLE country (\n  code TEXT PRIMARY KEY,\n  name TEXT\n);" in s
    assert 'CREATE TABLE "order item"' in s
    assert "PRIMARY KEY (order_id, line)" in s
    assert "FOREIGN KEY (country) REFERENCES country(code)" in s
    assert "3 example rows:" in s and "Germany" not in s  # LIMIT 3
    assert "United States of Am…" in s  # truncated to 20 chars
    assert "0 rows in empty_t" in s
    assert render_schema(db, sample_rows=0).count("example rows") == 0


def test_schema_quotes_keyword_names(tmp_path):
    path = tmp_path / "bank.sqlite"  # BIRD's financial and beer_factory name tables "order" and "transaction"
    c = sqlite3.connect(path)
    c.executescript('CREATE TABLE "order" (order_id INT PRIMARY KEY, "desc" TEXT); INSERT INTO "order" VALUES (1, \'a\');')
    c.close()
    s = render_schema(path)
    assert 'CREATE TABLE "order" (\n  order_id INT PRIMARY KEY,\n  "desc" TEXT\n);' in s
    assert '1 example rows:\nSELECT * FROM "order" LIMIT 3;' in s


def test_execute_read_only_and_errors(db):
    r = execute(db, "SELECT name FROM country WHERE code = 'EG'")
    assert r.ok and r.rows == [("Egypt",)] and r.columns == ["name"]
    assert not execute(db, "DELETE FROM country").ok  # read-only connection
    assert "no such table" in execute(db, "SELECT * FROM nope").error
    assert not execute(db, "SELECT 1; SELECT 2").ok
    slow = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT count(*) FROM n"
    assert execute(db, slow, timeout_s=0.3).error.startswith("timeout")


@pytest.mark.parametrize("text, sql", [
    ("SELECT * FROM t;", "SELECT * FROM t"),
    ("```sql\nSELECT a\nFROM t\n```\nThis query selects a.", "SELECT a FROM t"),
    ("SQL: select 1; select 2", "select 1"),
    ("Here is the query:\nSELECT x FROM t WHERE y = 'a;b';", "SELECT x FROM t WHERE y = 'a;b'"),
    ("<think>hmm</think>```sqlite\nWITH c AS (SELECT 1) SELECT * FROM c", "WITH c AS (SELECT 1) SELECT * FROM c"),
    ("", ""),
])
def test_extract_sql(text, sql):
    assert extract_sql(text) == sql


def test_results_match():
    g = ExecResult(["a", "b"], [(1, "x"), (2, "y")])
    assert results_match(g, ExecResult(["b", "a"], [("y", 2), ("x", 1)]), order_matters=False)
    assert not results_match(g, ExecResult(["a", "b"], [(2, "y"), (1, "x")]), order_matters=True)
    assert results_match(g, ExecResult(["a", "b"], [(1, "x"), (2, "y")]), order_matters=True)
    assert not results_match(g, ExecResult(["a"], [(1,), (2,)]), order_matters=False)
    assert results_match(ExecResult(["n"], [(2,)]), ExecResult(["c"], [(2.0,)]), order_matters=False)
    assert not results_match(g, ExecResult(error="boom"), order_matters=False)
    assert order_matters("select a from t ORDER  BY a") and not order_matters("select a from t")


def test_tasks_render(db):
    tasks = {t.ref: t for t in list_tasks()}
    assert {"nl2sql-sqlite@v1", "qa@v1", "mango-qa@v1"} <= set(tasks)
    t = load_task("nl2sql-sqlite@v1")
    row = t.with_context({"question": "How many countries?", "db_id": "shop"}, db.parent.parent)
    msgs = t.messages(row)
    assert msgs[0]["role"] == "system" and "SQLite" in msgs[0]["content"]
    assert msgs[1]["content"].startswith("Database schema:\nCREATE TABLE country")
    assert msgs[1]["content"].endswith("Question: How many countries?")
    m = load_task("mango-qa@v1")
    assert m.keep({"type": "open_qa"}) and not m.keep({"type": "mcq"})


def test_optional_inputs():
    t = load_task("nl2sql-sqlite@v2")
    assert t.required == ["question", "db_id"]
    row = {"question": "How many?", "db_id": "x", "schema": "CREATE TABLE a (b);"}
    assert t.user(row) == "Database schema:\nCREATE TABLE a (b);\n\nQuestion: How many?"
    assert t.user(row | {"evidence": "  "}) == t.user(row)  # a blank hint leaves the line out
    assert t.user(row | {"evidence": "rate = a / b"}).endswith("Question: How many?\nExternal knowledge: rate = a / b")
    v1 = load_task("nl2sql-sqlite@v1")
    assert v1.user(row) == t.user(row) and v1.required == v1.inputs  # v2 renders a v1 row the same
