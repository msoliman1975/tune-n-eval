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

## The slice, step by step

```bash
cd ~/agritune/services/data
bash run.sh collect && bash run.sh parse && bash run.sh chunk     # done: 34 of 40 sources usable, 506 chunks
bash run.sh synth --per-doc 12                                   # estimate: 244 chunks, about $19
bash run.sh synth --per-doc 12 --yes --limit 5                   # look at items.jsonl before spending more
bash run.sh synth --per-doc 12 --yes
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
  --dataset ~/agritune/artifacts/data/datasets/mango-sft/v0.1 --yes
```

Credentials: synthesis and the judge call Claude. Put `ANTHROPIC_API_KEY=...` in
`services/data/.env` and `services/eval/.env` (gitignored), or export it before `studio/up.sh`.

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
- Not yet run: synthesis (needs credentials), the real 8B training run, the judged evaluation.

## Known issues

- 5 sources did not download: four Australian NT government documents return 403 to any
  scripted client, and one SpringerOpen link serves HTML. Paclobutrazol (flowering control)
  has no remaining dedicated source.
- Source licences are mixed: 11 state no licence, the Arabic news and agri-site pages and the
  Florida postharvest manual are "copyright, free to read", two Pakistan papers are CC BY-NC-ND.
  See `services/data/sources/m1-sources.yaml`.
- `transformers` warns that both `max_new_tokens` and `max_length` are set; harmless.
- The Studio bundle is 1.3 MB (ECharts); fine for a local tool.
