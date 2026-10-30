"""Shared AgriTune pieces used by every service. Installed editable, so these paths point at the repo."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ARTIFACTS = REPO / "artifacts"
TASKS_DIR = REPO / "tasks"
