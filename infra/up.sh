#!/usr/bin/env bash
# Starts the shared infrastructure, creating infra/.env with a random Postgres password on first run.
set -euo pipefail
cd "$(dirname "$0")"
# Drop inherited Windows PATH entries: Rancher Desktop's Windows credential helper breaks image pulls.
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/' | paste -sd: -); export PATH
[ -f .env ] || { echo "POSTGRES_PASSWORD=$(openssl rand -hex 16)" > .env; chmod 600 .env; }
sg docker -c "docker compose --env-file .env up -d --wait" 2>&1 | tail -8
sg docker -c "docker compose ps --format 'table {{.Service}}\t{{.Status}}\t{{.Ports}}'"
curl -fsS http://127.0.0.1:5000/health && echo " <- mlflow"
curl -fsS http://127.0.0.1:6333/healthz && echo " <- qdrant"
