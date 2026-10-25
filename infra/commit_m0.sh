#!/usr/bin/env bash
# Commits the M0 work in progress to the local repo.
set -e
cd ~/agritune
git config user.name >/dev/null || git config user.name "msoli"
git config user.email >/dev/null || git config user.email "mohamed75soliman@gmail.com"
git add -A
git status --short
git commit -q -F - <<'EOF'
M0 in progress: WSL training stack, infra compose, GPU benchmark

- Monorepo layout from the AgriTune blueprint (contracts, services, studio, infra)
- Train service: Unsloth + TRL + MLflow, torch pinned to cu129 for driver 577.13
- Benchmark, GPU and Triton check scripts
- Compose stack: Postgres, Redis, MLflow, Qdrant on localhost
- Runbook with status and resume steps after the NVIDIA driver update

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git log --oneline -1
