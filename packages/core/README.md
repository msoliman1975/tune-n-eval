# agritune-core

Shared by the Data, Train, Eval and Registry services (installed editable from `packages/core`):

- `tasks`: task specs from `tasks/*.json`. A task turns a dataset row into chat messages and a
  target, the same way for training, evaluation and the Studio playground.
- `sqlschema`: renders a SQLite database as `CREATE TABLE` statements with keys plus sample rows,
  the schema context of the `nl2sql-sqlite` task.
- `sqlexec`: read-only SQL execution with a timeout, SQL extraction from model output, and
  result-set comparison for execution accuracy.

```bash
cd packages/core && uv run --group dev pytest -q
```
