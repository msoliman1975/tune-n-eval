# Runbook: the generic platform (any task), first use case NL→SQL

The platform is no longer mango-specific: prepare a dataset for any **task**, pick a model,
fine-tune, and every base and fine-tuned model lands in the **Model Registry**; evaluate any set of
models with chosen **metrics**, read every answer and score, compare models pairwise.

## Start everything

```bash
cd ~/agritune
bash infra/up.sh            # postgres, redis, mlflow (:5000), qdrant
bash studio/up.sh           # registry :8104, data :8101, train :8102, eval :8103, Studio :8100
```

Studio: http://localhost:8100. Ollama (system service, :11434) serves Gemma 4 12B, Gemma 4 E4B,
the Qwen3-14B judge, and exported fine-tunes.

## Pieces

| Piece | Where | What it does |
|---|---|---|
| Task specs | `tasks/<id>@<v>.json` | How a row becomes a prompt and a target: system prompt, user template, inputs, target, context (e.g. the DB schema), attachments, decoding, applicable metrics. One spec drives training, evaluation and the playground, so a model is always asked in the words it was trained on. |
| Core package | `packages/core` | Task rendering, SQLite schema rendering (DDL + keys + 3 sample rows), read-only SQL execution with timeout, SQL extraction, result-set comparison. Installed editable in every service. |
| Data service | `services/data` | Drafts: upload JSONL/JSON/CSV/Parquet → map columns to task fields → splits (by column, random, single; groups kept together; carve N validation items) → attach files (zip or a folder on this machine) → preview prompts → browse, edit, delete items → freeze (validates every item, runs every gold SQL) → immutable `id@version`. The mango document pipeline is unchanged. |
| Model Registry | `services/registry` | One JSON per model: `hf_base` (trainable if it has a `train_preset`), `hf_adapter` (fine-tune; `parent` = its base), `ollama`, `api` (Anthropic, Google; `paid`). Train registers each finished run; Eval appends each evaluation's summary. |
| Train service | `services/train` | Fine-tunes a registry model (`model: gemma-4-e4b`) on a dataset of any task, loss on the target only, rows longer than the preset's sequence dropped (never truncated). Export to Ollama from the run page. |
| Eval service | `services/eval` | Generic evaluations (`run.py`): N models × one dataset × chosen metrics; mango evaluations (`evaluate.py`) unchanged. Playground endpoint `POST /generate`. |
| Strong Hire tasks | `tasks/sh-*`, `datasets/strong-hire/` | Nine built-in tasks and datasets that measure models on Strong Hire's LLM jobs, with JSON-schema output, per-tier prompts and the product's own scoring formulas. See `docs/runbook-strong-hire.md`. |

## Exploring a dataset

Datasets → a dataset opens the explorer (`#/datasets/<id>/<v>?tab=…`):
- **Overview**: metadata (task, fields, source, validation, sha256), files with downloads (JSONL, CSV,
  Parquet), prompt size per split, and which databases each split uses and shares.
- **Data**: every item, table or cards, searchable, filterable by any value. Each item opens with
  all fields, the exact prompt the model gets, and the raw JSON.
- **Columns**: type, role (input, target, context, metadata), empty and distinct counts, lengths and
  most common values; a value opens its items.
- **Databases** (NL→SQL): tables, columns, keys and row counts, browse table rows, and the
  schema exactly as rendered into prompts.
- **Task** (prompt template, metrics) and **Usage** (models fine-tuned and evaluations run on it).

Data service endpoints: `/datasets/{id}/{v}/profile`, `/items/{item}`, `/download`,
`/attachments/{key}`, `/attachments/{key}/tables/{table}` (read-only SQLite).

## NL→SQL datasets

| Dataset | Splits | Task | What it measures |
|---|---|---|---|
| `spider@v1.0` | 8,343 train / 313 val / 1,034 test | `nl2sql-sqlite@v1` | the classic benchmark; no database shared between splits |
| `bird@v1.0` | train / 300 val / dev as test | `nl2sql-sqlite@v2` | large, messy real databases; each question has an "evidence" hint; slices by `difficulty` |
| `spider-syn@v1.0` | 1,034 test | v1 | Spider dev with schema words swapped for synonyms (`meta.spider_id` = the spider item) |
| `spider-realistic@v1.0` | 508 test | v1 | Spider dev with explicit column names removed |
| `spider-dk@v1.0` | 534 test | v1 | questions needing domain knowledge; slices by `category` |

The three Spider variants re-ask Spider dev questions on Spider dev databases, so they are test-only
and a Spider fine-tune has never seen their databases. `nl2sql-sqlite@v2` adds an optional
`evidence` input: the "External knowledge:" line is left out when an item has none (task specs can
mark inputs `optional` and wrap template text in `[[...]]`).

```bash
# downloads: BIRD https://bird-bench.github.io (CC BY-SA 4.0; train.zip 8.9 GB, dev.zip 0.35 GB) into
# ~/agritune-downloads/bird; Spider-Syn and Spider-DK from github.com/ygan (MIT), Spider-Realistic from
# zenodo.org/records/5205322 (CC BY-SA 4.0) into ~/agritune-downloads/spider-variants/{syn,dk,realistic}
python3 scripts/prepare_bird.py ~/agritune-downloads/bird
python3 scripts/prepare_spider_variants.py ~/agritune-downloads/spider-variants ~/agritune-downloads/spider/spider_data
python3 scripts/import_nl2sql.py            # upload, map, link databases, freeze (drops items whose gold SQL fails)
```

## Naming, tagging and deleting fine-tunes

- A fine-tune is named `<base> · <dataset> · <share> · <launch time>` unless a name is given at
  launch (Training → Name, Tags); tags `fine-tuned` and the dataset id are added. Rename and retag on
  the model's page; names are unique. Models filters by kind, tag and text.
- **Delete fine-tune** (model page or run page) removes the run folder (adapter, checkpoints, GGUF),
  its Ollama model, registry entry, cached evaluation answers and MLflow run (soft delete), after
  showing the size. Evaluation reports that measured it stay. API: `DELETE /api/train/runs/{id}`
  (`?dry_run=true` lists what would go). The registry refuses to remove a fine-tune whose run folder
  exists, since the next sync would register it again.

## Metrics, by approach

| Approach | Metric | Notes |
|---|---|---|
| Execution | `sql_exec` execution accuracy | headline: predicted and gold SQL return the same rows (multiset; ordered if gold has ORDER BY; columns in any order) |
| Execution | `sql_valid` | the query runs without error within 10 s |
| Reference | `sql_exact_set` | Spider's official exact-set match (vendored, Apache-2.0); values ignored |
| Reference | `exact_match`, `chrf` | text tasks |
| LLM judge | `llm_judge` | rubric 1-5 against the reference (`rubrics/`), local judge |
| LLM judge | `sql_failure_judge` | one main error category per failed SQL answer: schema linking, join, filter, aggregation, ordering/limit, set/nesting, projection, invalid SQL, gold questionable |
| Efficiency | `latency` | seconds per answer (amortized per batch for local GPU models), p50/p90 |

Every score has a 95% bootstrap interval; every model pair gets a paired difference on the same
items. Slices: split, and Spider hardness (easy/medium/hard/extra) for SQL tasks.

## NL→SQL walkthrough (what was done for Spider)

```bash
# 1. Spider 1.0 from https://yale-lily.github.io/spider (CC BY-SA 4.0), unpacked to ~/agritune-downloads/spider
python3 scripts/prepare_spider.py ~/agritune-downloads/spider/spider_data   # -> spider.jsonl, tables.json next to the DBs
```

2. Studio → Datasets → upload `spider.jsonl` → task `nl2sql-sqlite@v1` → question / db_id / query →
   splits from column `split` (train → train, dev → test), 300 validation items grouped by `db_id`
   → link folder `spider_data/database` → freeze `spider@v1.0`. Freeze found 3 train items whose
   gold SQL does not run (a missing table, two ORDER BY before INTERSECT); they were deleted.
   Result: 8,343 train / 313 validation / 1,034 test, ~880 tokens per training example.
3. Baseline: Evaluations → New → Gemma 4 E4B + Gemma 4 12B on test, all SQL metrics + failure analysis.
4. Fine-tune: Training → Gemma 4 E4B on `spider@v1.0`, 1 epoch (~6.4 h for all 8,343 rows at
   ~317 tokens/s; "Training data (%)" = 30 trains on a seeded, per-database stratified 30% in ~2 h).
   Or unattended (`FRACTION=0.3` optional):
   `systemd-run --user --unit=agritune-nl2sql bash ~/agritune/scripts/nl2sql-finetune.sh`.
5. Compare: evaluation of base vs fine-tuned vs 12B (cached answers are reused), then Export to
   Ollama and try it in the Playground.

## Design decisions

- **The schema is in every prompt** (CREATE TABLE with types and keys, then 3 example rows per
  table), rebuilt from PRAGMA metadata so every database reads the same. Training uses the same
  rendering, so the fine-tune learns to read the schema it will get at runtime.
- **Execution accuracy is the headline**; exact-set match is reported for comparability with
  published Spider numbers but penalizes correct queries written differently.
- **One normalization before exact-set match**: `INNER JOIN` → `JOIN`. Spider's parser does not
  know `INNER` and scores such a query as empty; Gemma 4 writes `INNER JOIN` almost always, so
  without it exact-set match measured the parser, not the model. LEFT/CROSS JOIN stay unparsed (0).
- **The judge is Qwen3-14B**, a different family from the Gemma models under test, and only
  classifies failures; it does not decide correctness.
- **Base and fine-tuned answers come from the same 4-bit weights** with the adapter off and on.
  Gemma 4 12B runs through Ollama (Q4), as it does not train on 12 GB.
- **Paid models need `confirm_paid`** (API 402 otherwise) and show a cost estimate first; all
  answers are cached per model and prompt, so nothing is paid for twice.
- **The playground answers through Ollama or the APIs** (no weights loaded in the API), refuses
  local models while the GPU runs a job, and runs the SQL read-only.
- Gemma 4 E4B holds ~10 GB in transformers 4-bit (its per-layer embeddings stay bf16), so Eval
  uses batch 4.
- **Training Gemma 4 E4B on 12 GB** needs two things (sft.py): `device_map={"": 0}` (auto placement
  spills to the CPU, which bitsandbytes rejects) and no transformers allocator warm-up (it reserves
  one model-sized block; Unsloth then moves the 5 GB per-layer embedding to the CPU and the hole is
  never returned, so the loss sees 0 GB free). Peak 10.25 GB at seq 4096.
- **Training data (%)** subsamples the train split with the run's seed, stratified by the task's
  attachment key (`db_id` for NL→SQL) so every database keeps its share; recorded in the
  adapter's registry lineage as `train_fraction`.

## API models (Claude Sonnet, Gemini)

Built and wired; keys live in `services/eval/.env` (re-read on every call, the file wins):

```
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

then `systemctl --user restart agritune-eval` and run an evaluation with a small sample (e.g. 5)
and "I accept the API cost". Sonnet 5.5 runs with its defaults (adaptive thinking); set
`{"effort": "low"}` in the model's options to cut cost. The Gemini model id (`gemini-flash-latest`)
is a placeholder to check against your key's model list.

## Known issues

- Restarting the eval service no longer loses an evaluation: on start it re-queues evaluations that
  were queued or running, and they resume from cached answers (saved every 64). Restarting the train
  service marks a running fine-tune failed (there is no checkpoint to resume), so check
  `GET /api/train/runs` before restarting it.
- On battery the GPU is power-capped (about 29 W against about 80 W on the charger) and generation runs several times slower.

- The mango M1 run was stopped on 2026-10-25 at 243 of 244 chunks; resume with
  `systemd-run --user --unit=agritune-m1 bash ~/agritune/scripts/m1-slice.sh` (every stage resumes).
- Mango evaluations still use the mango-specific `evaluate.py` (MCQ by letter logits, chrF++,
  5-criteria judge); the generic evaluator covers mango-qa open questions with chrF++ and the judge.
- HF targets load each fine-tune's base again per adapter; fine for a few adapters per evaluation.
