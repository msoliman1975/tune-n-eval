"""Turns the Spider robustness variants into upload files and database folders for the Studio.

    python3 scripts/prepare_spider_variants.py ~/agritune-downloads/spider-variants ~/agritune-downloads/spider/spider_data

All three re-ask Spider's dev questions (same gold SQL, same databases), so they are test sets only:
a model fine-tuned on spider@v1.0 has never seen their databases either.
- Spider-Syn (MIT): every dev question with schema words replaced by synonyms ("singers" -> "vocalists").
- Spider-Realistic (CC BY-SA 4.0): 508 dev questions with explicit column names removed.
- Spider-DK (MIT): 535 questions that need domain knowledge, 3 databases of its own; `category` is the
  kind of knowledge (paper: Gan et al. 2021).
Writes <variants>/agritune/<name>.jsonl and <variants>/agritune/<name>-db/ (hard links to only the
databases it uses, plus tables.json). Then `python3 scripts/import_nl2sql.py spider-variants`.

Inputs (fetched by hand or by the commands in docs/runbook-generic.md):
    syn/dev.json, realistic/spider-realistic.json, dk/Spider-DK.json, dk/tables.json
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

DK_REPO = "https://raw.githubusercontent.com/ygan/Spider-DK/main/database"
DK_DBS = ("new_concert_singer", "new_orchestra", "new_pets_1")
DK_KIND = {"0": "none (original question)", "1": "columns implied, not named", "2": "simple inference",
           "3": "synonym for a cell value", "4": "a word implies a condition", "5": "conflicts with another domain"}


def dk_category(t) -> str:
    return " + ".join(DK_KIND.get(c, c) for c in str(t))  # "34" = types 3 and 4


def link_dbs(rows: list[dict], dbs: list[Path], tables: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    used = {r["db_id"] for r in rows}
    for db_id in sorted(used):
        src = next((d / db_id / f"{db_id}.sqlite" for d in dbs if (d / db_id / f"{db_id}.sqlite").exists()), None)
        if src is None:
            raise SystemExit(f"database {db_id} not found in {', '.join(map(str, dbs))}")
        (out / db_id).mkdir(exist_ok=True)
        if not (out / db_id / src.name).exists():
            os.link(src, out / db_id / src.name)
    (out / "tables.json").write_text(json.dumps([t for t in json.loads(tables.read_text(encoding="utf-8"))
                                                 if t["db_id"] in used]), encoding="utf-8")
    return len(used)


def main() -> None:
    src, spider = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
    out = src / "agritune"
    out.mkdir(exist_ok=True)
    spider_db, spider_tables = spider / "database", spider / "tables.json"

    syn = [{"id": f"syn-{i:05d}", "db_id": ex["db_id"], "question": ex["SpiderSynQuestion"].strip(),
            "query": " ".join(ex["query"].split()), "spider_question": ex["SpiderQuestion"].strip(),
            "spider_id": f"dev-{i:05d}"}  # the same index as spider@v1.0's test items
           for i, ex in enumerate(json.loads((src / "syn" / "dev.json").read_text(encoding="utf-8")))]
    realistic = [{"id": f"realistic-{i:05d}", "db_id": ex["db_id"], "question": ex["question"].strip(),
                  "query": " ".join(ex["query"].split())}
                 for i, ex in enumerate(json.loads((src / "realistic" / "spider-realistic.json").read_text(encoding="utf-8")))]

    dk_db = src / "dk" / "database"
    for db in DK_DBS:  # Spider-DK's own databases (versions of concert_singer, orchestra, pets_1)
        f = dk_db / db / f"{db}.sqlite"
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{DK_REPO}/{db}/{db}.sqlite", f)
    dk = [{"id": f"dk-{i:05d}", "db_id": ex["db_id"], "question": ex["question"].strip(),
           "query": " ".join(ex["query"].split()), "category": dk_category(ex["type"])}
          for i, ex in enumerate(json.loads((src / "dk" / "Spider-DK.json").read_text(encoding="utf-8")))]

    for name, rows, dbs, tables in (("spider-syn", syn, [spider_db], spider_tables),
                                    ("spider-realistic", realistic, [spider_db], spider_tables),
                                    ("spider-dk", dk, [dk_db, spider_db], src / "dk" / "tables.json")):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n = link_dbs(rows, dbs, tables, out / f"{name}-db")
        print(f"{name}: {len(rows)} rows, {n} databases -> {out / name}.jsonl")


if __name__ == "__main__":
    main()
