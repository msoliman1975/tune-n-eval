"""Turns the official Spider 1.0 release into one upload file for the Studio's dataset flow.

    python3 scripts/prepare_spider.py ~/agritune-downloads/spider/spider_data

Writes <spider_data>/agritune/spider.jsonl with one row per question:
    id, split (train | dev), db_id, question, query, source
and copies tables.json next to the databases, where the evaluator looks for Spider's foreign-key
annotations (used by exact-set match). Then, in the Studio (Datasets -> New dataset):
    upload spider.jsonl -> task nl2sql-sqlite@v1 -> map question/db_id/query -> split by column
    `split` (train -> train, dev -> test), 300 validation items grouped by db_id -> attach the
    folder <spider_data>/database -> freeze as spider@v1.0.

Download: https://yale-lily.github.io/spider (CC BY-SA 4.0).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main() -> None:
    src = Path(sys.argv[1]).expanduser()
    out = src / "agritune"
    out.mkdir(exist_ok=True)
    rows = []
    for name, split, prefix in (("train_spider.json", "train", "train"), ("train_others.json", "train", "others"),
                                ("dev.json", "dev", "dev")):
        for i, ex in enumerate(json.loads((src / name).read_text(encoding="utf-8"))):
            rows.append({"id": f"{prefix}-{i:05d}", "split": split, "db_id": ex["db_id"],
                         "question": ex["question"].strip(), "query": " ".join(ex["query"].split()), "source": name})
    with (out / "spider.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy2(src / "tables.json", src / "database" / "tables.json")
    n = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev")}
    print(f"{len(rows)} rows ({n['train']} train, {n['dev']} dev) -> {out / 'spider.jsonl'}")
    print(f"databases: {src / 'database'} ({sum(1 for p in (src / 'database').iterdir() if p.is_dir())} folders)")


if __name__ == "__main__":
    main()
