"""Turns the official BIRD release into one upload file and one database folder for the Studio.

    python3 scripts/prepare_bird.py ~/agritune-downloads/bird      # holds train.zip and dev.zip

Unpacks both zips (and the database zips inside them), then writes into <bird>/agritune/:
    bird.jsonl   one row per question: id, split (train | dev), db_id, question, evidence, query,
                 difficulty (dev only: simple | moderate | challenging), source
    database/    every train and dev database as <db_id>/<db_id>.sqlite (hard links, no copy), with
                 tables.json (train_tables.json + dev_tables.json) where the evaluator looks for keys
Then `python3 scripts/import_nl2sql.py bird` builds and freezes bird@v1.0 through the data service
(task nl2sql-sqlite@v2: the evidence is part of the prompt; train -> train, dev -> test, 300
validation items grouped by db_id).

Download: https://bird-bench.github.io (CC BY-SA 4.0). dev is the 2024-06-27 corrected release.
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path


def unzip(z: Path, dest: Path) -> None:
    with zipfile.ZipFile(z) as f:
        for m in f.infolist():
            if m.filename.startswith("__MACOSX") or "/." in m.filename:
                continue
            f.extract(m, dest)


def unpack(src: Path, name: str) -> Path:
    """<name>.zip -> its folder, and the <name>_databases.zip inside it -> <folder>/<name>_databases/."""
    out = src / "unpacked"
    if not any(out.glob(f"{name}*/{name}.json")):
        print(f"unpacking {name}.zip")
        unzip(src / f"{name}.zip", out)
    folder = next(out.glob(f"{name}*/{name}.json")).parent
    if not (folder / f"{name}_databases").is_dir():
        print(f"unpacking {name}_databases.zip")
        unzip(folder / f"{name}_databases.zip", folder)
    return folder


def main() -> None:
    src = Path(sys.argv[1]).expanduser()
    out = src / "agritune"
    db_out = out / "database"
    db_out.mkdir(parents=True, exist_ok=True)
    rows, tables = [], []
    for name in ("train", "dev"):
        folder = unpack(src, name)
        for db in sorted(p for p in (folder / f"{name}_databases").iterdir() if p.is_dir()):
            sqlite = db / f"{db.name}.sqlite"
            if not sqlite.exists():
                continue
            (db_out / db.name).mkdir(exist_ok=True)
            target = db_out / db.name / sqlite.name
            if not target.exists():
                os.link(sqlite, target)
        tables += json.loads((folder / f"{name}_tables.json").read_text(encoding="utf-8"))
        for i, ex in enumerate(json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))):
            rows.append({"id": f"{name}-{ex.get('question_id', i):05d}", "split": name, "db_id": ex["db_id"],
                         "question": ex["question"].strip(), "evidence": (ex.get("evidence") or "").strip(),
                         "query": " ".join(ex["SQL"].split()), "difficulty": ex.get("difficulty", ""),
                         "source": f"{folder.name}/{name}.json"})
    (db_out / "tables.json").write_text(json.dumps(tables), encoding="utf-8")
    with (out / "bird.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev")}
    print(f"{len(rows)} rows ({n['train']} train, {n['dev']} dev) -> {out / 'bird.jsonl'}")
    print(f"databases: {db_out} ({sum(1 for p in db_out.iterdir() if p.is_dir())} folders)")


if __name__ == "__main__":
    main()
