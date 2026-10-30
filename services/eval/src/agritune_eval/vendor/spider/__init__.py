"""Spider's official evaluation (exact-set match and hardness), vendored unchanged except imports.

Source: https://github.com/taoyds/test-suite-sql-eval (evaluation.py, process_sql.py), Apache-2.0,
see LICENSE in this folder. Only the import lines differ: relative imports, and exec_eval is not
vendored because AgriTune scores execution itself (agritune_core.sqlexec).
"""
