#!/usr/bin/env bash
# NL->SQL slice, unattended: benchmark the Gemma 4 E4B preset, smoke-train, fine-tune on spider@v1.0,
# evaluate fine-tuned vs base (and Gemma 4 12B), export to Ollama for the playground.
#
#   systemd-run --user --unit=agritune-nl2sql bash ~/agritune/scripts/nl2sql-finetune.sh
#   journalctl --user -u agritune-nl2sql -f
# Training and evaluation go through the service APIs, so they show up live in the Studio and take
# the same GPU lock; if a baseline evaluation is still running, everything waits for it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
DS=artifacts/data/datasets/spider/v1.0
EPOCHS=${EPOCHS:-1}
FRACTION=${FRACTION:-1}  # share of the train split, e.g. FRACTION=0.3
API=http://127.0.0.1:8100/api
say() { echo "[$(date +%H:%M:%S)] $*"; }
json() { python3 -c "import sys,json; d=json.load(sys.stdin); print($1)"; }

wait_eval() {  # until an evaluation is terminal; prints its status
  while true; do
    s=$(curl -s "$API/eval/evaluations/$1" | json "d['status']")
    case "$s" in succeeded|failed|cancelled) echo "$s"; return;; esac
    sleep 60
  done
}
wait_run() {
  while true; do
    s=$(curl -s "$API/train/runs/$1" | json "d['status']")
    case "$s" in succeeded|failed|cancelled) echo "$s"; return;; esac
    sleep 60
  done
}

for id in $(curl -s "$API/eval/evaluations" | json "' '.join(e['id'] for e in d if e['status'] in ('queued','running'))"); do
  say "waiting for evaluation $id"; say "$id: $(wait_eval "$id")"
done

cd "$ROOT/services/train"
if ! grep -q '"preset": "gemma4-e4b-qlora", "model"' bench_results.jsonl 2>/dev/null; then
  say "benchmark gemma4-e4b-qlora (seq 4096)"
  flock /tmp/agritune-gpu.lock bash run.sh bench --preset gemma4-e4b-qlora | tail -15
fi

SMOKE="run-smoke-$(date +%H%M%S)-gemma4-e4b-qlora"
say "smoke test: 10 steps -> $SMOKE"
flock /tmp/agritune-gpu.lock bash run.sh sft --model gemma-4-e4b --dataset "$ROOT/$DS" --max-steps 10 --run-id "$SMOKE" | tail -12
curl -s "$API/registry/models/$SMOKE" | json "'registered:', d['id'], 'parent', d['parent'], d['lineage']"
curl -s -X DELETE "$API/registry/models/$SMOKE" -o /dev/null && rm -rf "$ROOT/artifacts/train/$SMOKE"

say "fine-tune gemma-4-e4b on spider@v1.0, $EPOCHS epoch(s), train fraction $FRACTION"
RUN=$(curl -s -H 'content-type: application/json' "$API/train/runs" \
  -d "{\"model\": \"gemma-4-e4b\", \"dataset\": \"$DS\", \"epochs\": $EPOCHS, \"train_fraction\": $FRACTION}" | json "d['id']")
say "run $RUN: $(wait_run "$RUN")"
curl -s "$API/train/runs/$RUN" | json "d['result']"

say "evaluate $RUN vs gemma-4-e4b vs gemma-4-12b on spider test"
EV=$(curl -s -H 'content-type: application/json' "$API/eval/evaluations" -d "{
  \"models\": [\"gemma-4-e4b\", \"$RUN\", \"gemma-4-12b\"], \"dataset\": \"$DS\", \"splits\": [\"test\"],
  \"metrics\": [\"sql_exec\", \"sql_exact_set\", \"sql_valid\", \"latency\", \"sql_failure_judge\"]}" | json "d['id']")
say "evaluation $EV: $(wait_eval "$EV")"

say "export $RUN to Ollama"
curl -s -X POST "$API/train/runs/$RUN/export" >/dev/null
while [ "$(curl -s "$API/train/runs/$RUN" | json "(d.get('export') or {}).get('status')")" = "queued" ]; do sleep 60; done
curl -s "$API/train/runs/$RUN" | json "d['export']"
say "done: http://localhost:8100/#/evals/$EV"
