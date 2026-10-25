#!/usr/bin/env bash
# Runs a module of agritune_data from this directory, e.g. `bash run.sh collect`.
cd "$(dirname "$0")"
exec ~/.local/bin/uv run python -m "agritune_data.$1" "${@:2}"
