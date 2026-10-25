# agritune-eval

Measures a fine-tuned model against its base on a frozen dataset's heldout splits.

```bash
bash run.sh evaluate --model-ref ../../artifacts/train/<run>/model_ref.json \
                     --dataset ../../artifacts/data/datasets/mango-sft/v0.1        # MCQ + judge cost estimate
bash run.sh evaluate ... --yes                                                   # also judge open answers
```

Base and fine-tuned answers come from the same 4-bit weights with the adapter switched off and on,
the same system prompt (`contracts/prompts.json`) and greedy decoding with thinking off.
Open answers are graded by Claude with `rubrics/mango_advice_v1.md`; verdicts are cached in
`artifacts/eval/judge_cache.jsonl`. The judge needs `ANTHROPIC_API_KEY` (or `services/eval/.env`).
