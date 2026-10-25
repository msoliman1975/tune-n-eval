#!/usr/bin/env bash
# Runs a module of agritune_eval from this directory, e.g. `bash run.sh evaluate --help`.
cd "$(dirname "$0")"
exec ~/.local/bin/uv run python -m "agritune_eval.$1" "${@:2}"
