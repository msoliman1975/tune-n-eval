# Runbook: evaluating Strong Hire's models

Strong Hire (a voice interview simulator, separate repo) calls an LLM for nine text jobs. Each job is a
built-in task here (`tasks/sh-*@v1.json`) with a built-in dataset (`datasets/strong-hire/sh-*.jsonl`),
so any registry model can be measured on what Strong Hire actually asks of it. Speech to text and text
to speech are out of scope.

## The tasks

| Task | Strong Hire job (role) | Items | Labels come from | Main metric | Also |
|---|---|---|---|---|---|
| `sh-extract-posting` | Job posting to JobPosting JSON (extractor) | 10 | hand-checked JSON | `field_match` | `json_valid` |
| `sh-extract-resume` | CV (.txt, .pdf, .docx text) to Resume JSON (extractor) | 15 | hand-checked JSON | `field_match` | `json_valid` |
| `sh-input-check` | Is this really a job description / a CV? (extractor) | 94 | by construction | `label_acc` (accept or refuse) | `field_match` (document kind) |
| `sh-gap` | Rate a CV against a posting (planner) | 151 | job family and seniority of each pair | `gap_fit` | `gap_match_score` per fit slice |
| `sh-score` | Score a finished interview (scorer) | 30 | human gold set | `hire_signal_within_one` (spec bar 85%) | `hire_signal_exact`, `rubric_within_one`, `quotes_verbatim` |
| `sh-probe` | Follow up or move on after an answer (interviewer) | 182 | vague-answer marks in the scripted transcripts | `label_acc` | per `expected` slice |
| `sh-answer-check` | Does a turn belong in the interview? (interviewer) | 38 | hand-written, labeled | `label_acc` | per `expected` slice |
| `sh-interviewer-turn` | The interviewer's next spoken line (interviewer) | 152 | product rules | `ttft` (time to first word) | `one_question`, `word_limit` |
| `sh-coach-turn` | One story-coach reply (coach) | 40 | product rules | `quotes_verbatim` | `one_question`, `json_valid` |

Every task also reports `latency`, tokens and cost. Slices (Evaluation page, "by slice") split each
score by the row's labels: the expected label, the interview type, the gap fit, the coach candidate.

How the metrics work (definitions in the Studio's metric picker, code in
`services/eval/src/agritune_eval/structured.py`):

- **Product formulas.** Strong Hire never lets the model pick the hire signal or the match score: its
  code computes them from the model's ratings. `hire_signal_*`, `rubric_within_one` and `gap_*` run
  that same code (ported; `test_hire_signal_matches_strong_hire_on_the_gold_set` reproduces Strong
  Hire's own result of 26/30 exact and 30/30 within one band from the human scores). Scores backed by
  quotes the candidate never said are dropped first, and ratings of 2-3 without CV evidence are capped,
  exactly as the product does.
- **JSON.** `json_valid` checks the answer against the product's contract (`tasks/schemas/strong-hire/`).
  Ollama and Claude models are constrained to the schema, as Strong Hire constrains them; local
  transformers models (the `hf_base` entries) are not, so they fail more often. Strong Hire retries
  once on a schema error; the evaluation measures the first attempt.
- **Gap bands.** `gap_fit` expects 60-100 for a matching pair (same job family, seniority within one
  level) and 0-40 for an unrelated family. Adjacent families (SWE and data/ML, PM and TPM, ...) and big
  seniority gaps have no band; read `gap_match_score` by the `fit` slice: matches should sit well above
  mismatches. The bands are in each row's reference (`scripts/build_strong_hire.py`, `BANDS`).

## The models

Strong Hire's model per role, by its profile (`config/models.*.yaml` there):

| Role | `claude` profile | `local` profile | `tiny` profile |
|---|---|---|---|
| extractor, interviewer, coach | Claude Haiku 5.5 | Qwen2.5 3B (small tier) | Qwen2.5 3B |
| planner, scorer | Claude Sonnet 5.5 | Qwen2.5 7B (medium tier) | Qwen2.5 3B |

Registry entries (seeded on start): `qwen2.5-3b`, `qwen2.5-7b` (Ollama, the same Q4_K_M weights and
8192 context Strong Hire uses), `claude-haiku-5-5`, `claude-sonnet-5-5`. The Claude entries use the
models' default thinking and effort, which is what Strong Hire sends; set `options` on a copy of an entry
to try others (e.g. `{"effort": "low"}`).

**Prompt tier.** Strong Hire sends small models shorter prompts for some jobs (`*.small` prompts, and a
shorter transcript window for the interviewer). Rows carry both variants and a model's `prompt_tier`
(Models -> a model -> Prompt tier) picks one: `qwen2.5-3b` is `small`, `qwen2.5-7b` is `medium`. Any
other local model gets the default prompt until you set its tier, e.g. set Gemma 3 4B to `small` to
compare it fairly with Qwen2.5 3B. The tier also sets the interviewer's word limit (30 words small, 35
otherwise).

## Run it

```bash
ollama pull qwen2.5:3b-instruct
ollama pull qwen2.5:7b-instruct
bash studio/up.sh                               # seeds the new registry models
python3 scripts/import_strong_hire.py           # freezes the nine datasets as sh-*@v1.0 (test split)
```

Claude needs `ANTHROPIC_API_KEY` in `services/eval/.env` (see runbook-generic, "API models").

Then Evaluations -> New: pick a `sh-*` dataset, the models, keep the default metrics. Start with
`Limit` 10 to check the setup, then run the full set (answers are cached, so nothing runs twice).
Sensible model sets, per Strong Hire role:

- extractor tasks, `sh-probe`, `sh-answer-check`, `sh-interviewer-turn`, `sh-coach-turn`: Qwen2.5 3B,
  Claude Haiku 5.5, plus any small local model you want to try (Gemma 3 4B, Qwen3 1.7B, ...).
- `sh-gap`, `sh-score`: Qwen2.5 7B, Claude Sonnet 5.5 (and Qwen2.5 3B for the tiny profile).
- `sh-interviewer-turn` measures `ttft`, so run it on Ollama and Claude models only (local
  transformers models are batched, with no time to first token), with nothing else using the GPU.

## Limits

- One call per item. The product's retries (schema errors, lost quotes in the scorer, guarded
  interviewer replies) are not simulated, so these numbers are a floor for the product.
- `sh-interviewer-turn` probes give the model no "missing" hints (the transcripts do not record them).
- `sh-coach-turn` keeps the story empty: the coach's own part updates are not replayed into later turns.
- `sh-answer-check` turns are placed after the first core question of a transcript, in rotation.
- Not for training: the target of these tasks is a scoring reference, not an answer to learn.

## Rebuilding the datasets

The JSONL files are built from Strong Hire's code and fixtures and committed here. After Strong Hire
changes a prompt, a contract or a fixture, rebuild with a Strong Hire checkout next to this repo, then
import under a new version (`VERSION` in `scripts/import_strong_hire.py`):

```bash
uv run --project ../strong-hire python scripts/build_strong_hire.py --strong-hire ../strong-hire
```

Each row records the prompt files it used (`prompt_refs`), so a result traces back to a prompt version.
