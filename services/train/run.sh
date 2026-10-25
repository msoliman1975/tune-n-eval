#!/usr/bin/env bash
# Runs a module of agritune_train from this directory, e.g. `bash run.sh triton_check`.
cd "$(dirname "$0")"
exec ~/.local/bin/uv run python -m "agritune_train.$1" "${@:2}"
