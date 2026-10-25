"""Splits parsed Markdown into section-aware chunks for Q&A generation.

    bash run.sh chunk [--target-chars 3500]
Splits on headings, then packs paragraphs up to the target size; tiny trailing pieces merge into
the previous chunk. Writes chunks.jsonl: doc_id, chunk index, heading, text.
"""

from __future__ import annotations

import argparse
import json
import re

from . import ARTIFACTS
from .parse import PARSED

CHUNKS = ARTIFACTS / "chunks.jsonl"
HEADING = re.compile(r"^#{1,4}\s+(.*)$", re.M)


def sections(md: str) -> list[tuple[str, str]]:
    """(heading, body) pairs in document order; text before the first heading gets an empty heading."""
    out, last, heading = [], 0, ""
    for m in HEADING.finditer(md):
        out.append((heading, md[last:m.start()]))
        heading, last = m.group(1).strip(), m.end()
    out.append((heading, md[last:]))
    return [(h, b.strip()) for h, b in out if b.strip()]


def pack(paragraphs: list[str], target: int) -> list[str]:
    chunks, cur = [], ""
    for p in paragraphs:
        if cur and len(cur) + len(p) > target:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        if chunks and len(cur) < target // 4:
            chunks[-1] += "\n\n" + cur
        else:
            chunks.append(cur)
    return chunks


def chunk_doc(md: str, target: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pending_h, pending = "", []
    for heading, body in sections(md):
        # keep short sections together under the first heading so chunks stay substantive
        pending_h = pending_h or heading
        pending += [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        if sum(map(len, pending)) >= target // 2:
            out += [(pending_h, c) for c in pack(pending, target)]
            pending_h, pending = "", []
    if pending:
        out += [(pending_h, c) for c in pack(pending, target)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-chars", type=int, default=3500)
    a = ap.parse_args()

    index = [json.loads(l) for l in (PARSED / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    n = 0
    with CHUNKS.open("w", encoding="utf-8") as f:
        for doc in (d for d in index if d["status"] == "ok"):
            md = (PARSED / f"{doc['doc_id']}.md").read_text(encoding="utf-8")
            pieces = chunk_doc(md, a.target_chars)
            for i, (heading, text) in enumerate(pieces):
                f.write(json.dumps({"doc_id": doc["doc_id"], "chunk": i, "lang": doc["lang"],
                                    "license": doc["license"], "heading": heading, "text": text},
                                   ensure_ascii=False) + "\n")
            n += len(pieces)
            print(f"{len(pieces):4} chunks  {doc['doc_id']}")
    print(f"{n} chunks -> {CHUNKS}")


if __name__ == "__main__":
    main()
