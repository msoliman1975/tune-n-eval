"""Exports a run's adapter to a quantized GGUF and registers it with Ollama.

    bash run.sh export --run run-20261025-...-qwen-8b-qlora [--quant q4_k_m] [--tag agritune/qwen8b-mango:v1]

Unsloth merges the adapter into the base and converts with llama.cpp (built on first use).
Writes artifacts/train/{run}/gguf_gguf/ (Unsloth's naming) and a second ModelRef (kind "ollama") next to the run's
model_ref.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from agritune_core.tasks import load_task

from .presets import SYSTEM_PROMPT
from .sft import RUNS, ARTIFACTS, now, register


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--quant", default="q4_k_m")
    ap.add_argument("--tag", help="Ollama tag; default agritune/<preset>:<run timestamp>")
    a = ap.parse_args()

    run = RUNS / a.run
    ref = json.loads((run / "model_ref.json").read_text(encoding="utf-8"))
    preset = ref["lineage"]["preset"]
    tag = a.tag or f"agritune/{preset}:{a.run.split('-')[1]}-{a.run.split('-')[2]}"
    out = run / "gguf"

    from unsloth import FastModel

    model, tok = FastModel.from_pretrained(model_name=str(run / "adapter"), load_in_4bit=True)
    model.save_pretrained_gguf(str(out), tok, quantization_method=a.quant)
    out = next((d for d in (Path(f"{out}_gguf"), out) if d.exists()), out)  # Unsloth appends "_gguf"
    ggufs = sorted(out.rglob("*.gguf"), key=lambda p: p.stat().st_size)
    if not ggufs:
        raise SystemExit(f"no GGUF written under {out}")
    gguf = next((g for g in ggufs if a.quant.upper() in g.name.upper()), ggufs[-1])

    # Unsloth writes a Modelfile with the right chat template; add the task's system prompt as the
    # default (Eval and the playground send it explicitly anyway).
    system = load_task(ref["lineage"]["task"]).system if ref["lineage"].get("task") else SYSTEM_PROMPT
    generated = next(out.rglob("Modelfile"), None)
    body = generated.read_text(encoding="utf-8") if generated else ""
    lines = [l for l in body.splitlines() if not l.startswith(("FROM ", "SYSTEM "))]
    modelfile = out / "Modelfile.agritune"
    modelfile.write_text("\n".join([f"FROM {gguf}", f'SYSTEM """{system}"""', *lines]) + "\n", encoding="utf-8")
    subprocess.run(["ollama", "create", tag, "-f", str(modelfile)], check=True)

    ollama_ref = {
        "id": f"{a.run}-{a.quant}", "kind": "ollama", "base": ref["base"],
        "uri": str(gguf.relative_to(ARTIFACTS.parent)), "ollama_tag": tag, "chat_template": ref["chat_template"],
        "lineage": ref["lineage"] | {"parent": ref["id"]}, "created_at": now(), "workspace_id": "default",
    }
    (run / f"model_ref.{a.quant}.json").write_text(json.dumps(ollama_ref, indent=2), encoding="utf-8")
    register(a.run)  # the registry records the Ollama tag on the run's entry
    print(json.dumps({"gguf": str(gguf), "size_gb": round(gguf.stat().st_size / 1024**3, 2), "ollama_tag": tag}, indent=2))


if __name__ == "__main__":
    main()
