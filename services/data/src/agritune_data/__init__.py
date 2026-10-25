"""AgriTune Data service."""

from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[2]
REPO = SERVICE_DIR.parents[1]
ARTIFACTS = REPO / "artifacts" / "data"
SOURCES = SERVICE_DIR / "sources" / "m1-sources.yaml"
