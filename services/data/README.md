# agritune-data

Turns mango sources into a frozen, split dataset. Each stage reads the previous stage's output
under `artifacts/data/` and skips work it has already done, so any stage can be re-run.

```bash
bash run.sh collect                      # sources/m1-sources.yaml -> raw/
bash run.sh parse                        # raw/ -> parsed/*.md
bash run.sh chunk                        # parsed/ -> chunks.jsonl
bash run.sh synth                        # cost estimate only
bash run.sh synth --yes                  # chunks.jsonl -> items.jsonl (calls Claude)
bash run.sh freeze --id mango-sft --version v0.1
```

`synth` needs Claude credentials: `ANTHROPIC_API_KEY` in the environment (or `services/data/.env`).

## Splits

Claude extracts the key facts of each chunk and writes, per fact, two training Q&A (EN and AR)
and two eval items (an open question and an MCQ) worded differently from the training ones.

- `train`: training Q&A from all documents except the unseen ones.
- `heldout`: eval items about facts that `train` covers. This measures knowledge gain.
- `heldout_unseen`: eval items from a few whole documents with no training items. This measures
  whether the gain carries over to mango facts the model was not taught.
