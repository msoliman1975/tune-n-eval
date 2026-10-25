#!/usr/bin/env bash
# Diagnoses the Triton crash: versions, bundled ptxas, and a faulthandler traceback.
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY -c 'import triton, torch; print("triton", triton.__version__, "torch", torch.__version__)' 2>/dev/null
PTXAS=$(find .venv -path '*triton*' -name ptxas -type f | head -1)
echo "ptxas: $PTXAS"; [ -n "$PTXAS" ] && "$PTXAS" --version | tail -2
echo "--- run"
TRITON_CACHE_DIR=/tmp/triton-diag $PY -X faulthandler -m agritune_train.triton_check 2>&1 | grep -vi warn | tail -25
echo "exit ${PIPESTATUS[0]}"
