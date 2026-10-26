#!/usr/bin/env bash
# Runs the whole M1 slice unattended, with local models only (no API cost):
#   generate (Qwen3-14B) -> freeze -> SFT Qwen3-8B -> evaluate (MCQ, chrF++, Gemma 3 12B judge) -> export to Ollama
#
#   systemd-run --user --unit=agritune-m1 bash ~/agritune/scripts/m1-slice.sh   # survives closed shells
#   journalctl --user -u agritune-m1 -f
# Every stage resumes or skips work already done, so re-running after a failure is safe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
VERSION=${VERSION:-v0.1}
DS="$ROOT/artifacts/data/datasets/mango-sft/$VERSION"
gpu() { flock /tmp/agritune-gpu.lock "$@"; }  # same lock the Train and Eval APIs take
say() { echo "[$(date +%H:%M:%S)] $*"; }

cd "$ROOT/services/data"
say "generate: qwen3:14b, 12 chunks per document"
for attempt in 1 2 3; do  # failed chunks are not marked done, so a second pass retries only those
  gpu bash run.sh synth --yes --model qwen3:14b --per-doc 12
done
ollama stop qwen3:14b || true

if [ ! -d "$DS" ]; then
  say "freeze mango-sft@$VERSION"
  bash run.sh freeze --id mango-sft --version "$VERSION"
fi

cd "$ROOT/services/train"
RUN=$(ls -d "$ROOT"/artifacts/train/run-*-qwen-8b-qlora 2>/dev/null | while read -r d; do
        grep -q "mango-sft@$VERSION" "$d/model_ref.json" 2>/dev/null && basename "$d"; done | tail -1 || true)
if [ -z "$RUN" ]; then
  say "train qwen-8b-qlora on mango-sft@$VERSION"
  RUN="run-$(date +%Y%m%d-%H%M%S)-qwen-8b-qlora"
  gpu bash run.sh sft --preset qwen-8b-qlora --dataset "$DS" --run-id "$RUN"
fi
say "trained $RUN"

cd "$ROOT/services/eval"
say "evaluate $RUN vs base"
gpu bash run.sh evaluate --model-ref "$ROOT/artifacts/train/$RUN/model_ref.json" --dataset "$DS" --judge
ollama stop gemma3:12b || true

cd "$ROOT/services/train"
if ! ls "$ROOT/artifacts/train/$RUN"/model_ref.q4_k_m.json >/dev/null 2>&1; then
  say "export $RUN to Ollama"
  gpu bash run.sh export --run "$RUN"
fi
say "M1 slice done: open http://localhost:8100/#/evals"
