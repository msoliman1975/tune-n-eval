"""AgriTune Eval service. Knows nothing about how a model was trained: it reads ModelRefs."""

import json
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[2]
REPO = SERVICE_DIR.parents[1]
ARTIFACTS = REPO / "artifacts"
RUBRICS = REPO / "rubrics"
PROMPTS = json.loads((REPO / "contracts" / "prompts.json").read_text(encoding="utf-8"))
