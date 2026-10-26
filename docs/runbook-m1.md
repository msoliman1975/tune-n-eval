# M1 runbook: thin slice (data → train → eval → report)

Exit criterion: the first measured knowledge gain of a fine-tuned Qwen3-8B over its base.

## Start everything

```bash
cd ~/agritune
bash infra/up.sh            # postgres, redis, mlflow (:5000), qdrant
bash studio/up.sh           # data :8101, train :8102, eval :8103, Studio :8100 (systemd user units)
```

Studio: http://localhost:8100 (opens from a Windows browser). Ollama runs as a system service on :11434.
Logs: `journalctl --user -u agritune-train -f`. Stop: `bash studio/up.sh stop`.

## No API budget: everything runs on local models

- Generation: `qwen3:14b` through Ollama (bake-off on 6 chunks: it kept numbers and Egyptian
  specifics and returned no facts for a title page, where `gemma3:12b` invented "what does the
  text say" questions). About 60 s per chunk, so 244 chunks take about 4 hours.
- Judge: `gemma3:12b` through Ollama, a different family from the Qwen model under test.
- chrF++ against the reference scores every open answer for free, with or without a judge.
- Claude stays available: `--model claude-opus-5-5` on synth, `--judge-model claude-opus-5-5` on evaluate.

The whole slice runs unattended:

```bash
systemd-run --user --unit=agritune-m1 bash ~/agritune/scripts/m1-slice.sh
journalctl --user -u agritune-m1 -f
```

## The slice, step by step

```bash
cd ~/agritune/services/data
bash run.sh collect && bash run.sh parse && bash run.sh chunk     # done: 34 of 40 sources usable, 506 chunks
bash run.sh synth --per-doc 12                                   # plan: 244 chunks
bash run.sh synth --per-doc 12 --yes                             # qwen3:14b by default, ~4 h
bash run.sh freeze --id mango-sft --version v0.1
```

Then in the Studio: Training → launch `qwen-8b-qlora` on `mango-sft@v0.1` → watch the loss →
"Run evaluation" (tick the judge box for open answers) → Evaluations shows the scorecard.
The same from the CLI:

```bash
cd ~/agritune/services/train
bash run.sh sft --preset qwen-8b-qlora --dataset ~/agritune/artifacts/data/datasets/mango-sft/v0.1
bash run.sh export --run <run-id>                       # Q4_K_M GGUF + `ollama create`
cd ../eval && bash run.sh evaluate --model-ref ~/agritune/artifacts/train/<run-id>/model_ref.json \
  --dataset ~/agritune/artifacts/data/datasets/mango-sft/v0.1 --judge
```

Only if you switch to Claude: put `ANTHROPIC_API_KEY=...` in `services/data/.env` and
`services/eval/.env` (gitignored).

## Design decisions made in M1

- **Split by fact, not only by document.** The blueprint split train and eval by source document.
  With SFT only (no continued pretraining until M2), the model would then be tested on facts it
  was never taught, and the gain would be near zero by construction. Now each fact gets training
  Q&A (EN + AR) and separately worded eval items: `heldout` measures what was taught, and
  `heldout_unseen` (3 whole documents) measures whether the gain carries over to new documents.
- **MCQ is scored over the 4 rotations of the choices**, so small models' position bias (e.g.
  always "A") cannot fake or hide a gain.
- **Base and fine-tuned answers come from the same 4-bit weights** with the adapter switched off
  and on, the same system prompt (`contracts/prompts.json`) and greedy decoding with thinking off.
- **Judge**: Claude Opus 5.5, effort low, pointwise against the reference with
  `rubrics/mango_advice_v1.md`; verdicts cached. Pairwise A/B with position swap comes in M3.
- **Claude calls** use `claude-opus-5-5` with the server-side refusal fallback (`fallbacks: "default"`).
- **One GPU job at a time** across Train and Eval, through a file lock at `/tmp/agritune-gpu.lock`.
- **Job queue**: an in-process worker thread per service for M1; Dramatiq + Redis is deferred.

## Status (2026-10-25)

- Built and smoke-tested on a throwaway `fixture@v0.0` dataset with Qwen3-1.7B: SFT (loss
  falls, MLflow run logged), GGUF export to Ollama, MCQ evaluation, all four REST APIs and the
  Studio (launch → live SSE loss → evaluate → scorecard).
- 2026-10-25 01:18: `scripts/m1-slice.sh` started as unit `agritune-m1` (generate → freeze →
  train 8B → evaluate → export). Licensing decision: all 34 usable sources, for local research use.

## Known issues

- 5 sources did not download: four Australian NT government documents return 403 to any
  scripted client, and one SpringerOpen link serves HTML. Paclobutrazol (flowering control)
  has no remaining dedicated source.
- Source licences are mixed: 11 state no licence, the Arabic news and agri-site pages and the
  Florida postharvest manual are "copyright, free to read", two Pakistan papers are CC BY-NC-ND.
  See `services/data/sources/m1-sources.yaml`.
- `transformers` warns that both `max_new_tokens` and `max_length` are set; harmless.
- The Studio bundle is 1.3 MB (ECharts); fine for a local tool.
