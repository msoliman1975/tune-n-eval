# agritune-registry

The Model Registry (:8104): every model the platform can train or evaluate, one JSON file per
model in `artifacts/registry/models/`.

| kind | what it is | how Eval reaches it |
|---|---|---|
| `hf_base` | a Hugging Face checkpoint; `train_preset` set when Train can fine-tune it | transformers on the local GPU |
| `hf_adapter` | a LoRA adapter on top of `parent` (registered by Train when a run succeeds) | the parent's weights with the adapter on |
| `ollama` | a model served by the local Ollama | Ollama `/api/chat` |
| `api` | a hosted model (`provider` anthropic or google); `paid: true` | the provider's API, key in `services/eval/.env` |

An `hf_*` entry with an `ollama_tag` (after a GGUF export) can also be used in the playground.
Eval appends a summary of every evaluation to each model it measured (`evaluations`).

```bash
bash run.sh api                          # http://127.0.0.1:8104/docs
uv run --group dev pytest -q
```
