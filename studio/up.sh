#!/usr/bin/env bash
# Starts the Data, Train and Eval APIs and the Studio gateway as systemd user units, so they keep
# running after the shell (or the `wsl` call) that started them exits.
#   bash studio/up.sh            # then open http://localhost:8100
#   bash studio/up.sh stop
#   journalctl --user -u agritune-train -f     # logs
# The infra stack (infra/up.sh) and Ollama start separately.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UV=~/.local/bin/uv
NAMES=(data train eval studio)
declare -A DIRS=([data]=services/data [train]=services/train [eval]=services/eval [studio]=studio/api)
declare -A MODS=([data]=agritune_data.api [train]=agritune_train.api [eval]=agritune_eval.api [studio]=agritune_studio_api)
declare -A PORTS=([data]=8101 [train]=8102 [eval]=8103 [studio]=8100)

# user units must outlive the login session
loginctl show-user "$USER" -p Linger 2>/dev/null | grep -q yes || sudo loginctl enable-linger "$USER"

if [ "${1:-}" = "stop" ]; then
  for name in "${NAMES[@]}"; do systemctl --user stop "agritune-$name" 2>/dev/null && echo "stopped $name" || true; done
  exit 0
fi

for name in "${NAMES[@]}"; do
  if systemctl --user is-active --quiet "agritune-$name"; then echo "$name already running"; continue; fi
  systemctl --user reset-failed "agritune-$name" 2>/dev/null || true
  systemd-run --user --quiet --unit="agritune-$name" --working-directory="$ROOT/${DIRS[$name]}" \
    --setenv=PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" "$UV" run python -m "${MODS[$name]}"
done
for name in "${NAMES[@]}"; do
  for _ in $(seq 60); do curl -fsS "http://127.0.0.1:${PORTS[$name]}/openapi.json" >/dev/null 2>&1 && break; sleep 1; done
  curl -fsS "http://127.0.0.1:${PORTS[$name]}/openapi.json" >/dev/null 2>&1 \
    && echo "$name up on :${PORTS[$name]}" || echo "$name FAILED: journalctl --user -u agritune-$name"
done
echo "Studio: http://localhost:8100"
