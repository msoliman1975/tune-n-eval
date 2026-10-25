"""Downloads every source in the manifest to artifacts/data/raw/, skipping files already fetched.

    bash run.sh collect [--sources path.yaml]
Writes raw/index.jsonl with one row per source: status, sha256, bytes, content type.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import httpx
import yaml

from . import ARTIFACTS, SOURCES

RAW = ARTIFACTS / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AgriTune-research/0.1"}


def load_sources(path: Path = SOURCES) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["sources"]


def fetch(client: httpx.Client, src: dict) -> dict:
    dest = RAW / f"{src['doc_id']}.{src['format']}"
    row = {"doc_id": src["doc_id"], "url": src["url"], "path": str(dest.relative_to(ARTIFACTS))}
    if dest.exists():
        data = dest.read_bytes()
        return row | {"status": "cached", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    try:
        r = client.get(src["url"])
        r.raise_for_status()
    except httpx.HTTPError as e:
        return row | {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}
    ctype = r.headers.get("content-type", "")
    if src["format"] == "pdf" and not r.content.startswith(b"%PDF"):
        return row | {"status": "error", "error": f"expected a PDF, got {ctype or 'unknown type'}"}
    dest.write_bytes(r.content)
    return row | {"status": "ok", "sha256": hashlib.sha256(r.content).hexdigest(),
                  "bytes": len(r.content), "content_type": ctype}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(SOURCES))
    a = ap.parse_args()

    sources = load_sources(Path(a.sources))
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    # long read timeout: one source is a ~80 MB PDF
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=httpx.Timeout(30, read=600)) as client:
        for src in sources:
            row = fetch(client, src)
            rows.append(row)
            print(f"{row['status']:7} {src['doc_id']}" + (f"  {row['error']}" if "error" in row else ""), flush=True)
    with (RAW / "index.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    ok = sum(r["status"] != "error" for r in rows)
    print(f"{ok}/{len(rows)} sources on disk")


if __name__ == "__main__":
    main()
