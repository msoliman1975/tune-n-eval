#!/usr/bin/env bash
# Runs a module of agritune_registry from this directory, e.g. `bash run.sh api`.
cd "$(dirname "$0")"
exec ~/.local/bin/uv run python -m "agritune_registry.$1" "${@:2}"
