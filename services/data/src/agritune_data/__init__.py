"""AgriTune Data service."""

import os
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[2]
REPO = SERVICE_DIR.parents[1]
ARTIFACTS = Path(os.environ.get("AGRITUNE_DATA_DIR", REPO / "artifacts" / "data"))  # tests use a scratch folder
SOURCES = SERVICE_DIR / "sources" / "m1-sources.yaml"
