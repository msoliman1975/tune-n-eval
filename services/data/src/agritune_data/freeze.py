"""Splits items.jsonl into train / heldout / heldout_unseen and freezes them as a DatasetRef.

    bash run.sh freeze --id mango-sft --version v0.1 [--unseen-docs 3] [--eval-cap 400]

- heldout_unseen: every eval item from `--unseen-docs` whole documents (chosen with a fixed seed,
  balanced across languages). Their training items are dropped.
- train: training items from the remaining documents.
- heldout: eval items from the remaining documents, sampled down to `--eval-cap` per type.
Writes artifacts/data/datasets/{id}/{version}/ with one Parquet file per split, a source
manifest, and dataset_ref.json. A version that already exists is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from . import ARTIFACTS
from .collect import load_sources
from .synth import ITEMS

DATASETS = ARTIFACTS / "datasets"
SEED = 13
# DatasetItem fields; `role` is internal to the pipeline and is not frozen.
FIELDS = ["id", "fact_id", "lang", "type", "topic", "question", "answer", "choices",
          "source", "generator", "review"]


def write(rows: list[dict], path) -> None:
    pq.write_table(pa.Table.from_pylist([{k: r.get(k) for k in FIELDS} for r in rows]), path)


def sample(rows: list[dict], cap: int, rng: random.Random) -> list[dict]:
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    out = []
    for typ in sorted(by_type):
        group = sorted(by_type[typ], key=lambda r: r["id"])
        out += rng.sample(group, cap) if len(group) > cap else group
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="mango-sft")
    ap.add_argument("--version", required=True)
    ap.add_argument("--unseen-docs", type=int, default=3)
    ap.add_argument("--eval-cap", type=int, default=400, help="max heldout items per type")
    a = ap.parse_args()

    out = DATASETS / a.id / a.version
    if out.exists():
        raise SystemExit(f"{out} exists; versions are immutable, pick a new one")
    items = [json.loads(l) for l in ITEMS.read_text(encoding="utf-8").splitlines()]
    seen_ids, deduped = set(), []
    for it in items:  # a re-run chunk can append the same ids twice
        if it["id"] not in seen_ids:
            seen_ids.add(it["id"]); deduped.append(it)
    items = deduped

    rng = random.Random(SEED)
    sources = {s["doc_id"]: s for s in load_sources()}
    docs_by_lang = defaultdict(list)
    for d in sorted({it["source"]["doc_id"] for it in items}):
        docs_by_lang[sources[d]["lang"]].append(d)
    unseen: list[str] = []
    langs = sorted(docs_by_lang)
    for i in range(a.unseen_docs):
        pool = [d for d in docs_by_lang[langs[i % len(langs)]] if d not in unseen]
        if pool:
            unseen.append(rng.choice(pool))

    in_unseen = lambda it: it["source"]["doc_id"] in unseen  # noqa: E731
    splits = {
        "train": [it for it in items if it["role"] == "train" and not in_unseen(it)],
        "heldout": sample([it for it in items if it["role"] == "eval" and not in_unseen(it)], a.eval_cap, rng),
        "heldout_unseen": [it for it in items if it["role"] == "eval" and in_unseen(it)],
    }

    out.mkdir(parents=True)
    paths, digest = {}, hashlib.sha256()
    for name in sorted(splits):
        path = out / f"{name}.parquet"
        write(splits[name], path)
        digest.update(path.read_bytes())
        paths[name] = str(path.relative_to(ARTIFACTS.parent))
    used = sorted({it["source"]["doc_id"] for it in items})
    manifest = [sources[d] | {"unseen": d in unseen} for d in used]
    pq.write_table(pa.Table.from_pylist(manifest), out / "manifest.parquet")

    ref = {"id": a.id, "version": a.version, "sha256": digest.hexdigest(), "splits": paths,
           "counts": {k: len(v) for k, v in splits.items()}, "split_by": "fact+source_doc",
           "source_docs": str((out / "manifest.parquet").relative_to(ARTIFACTS.parent)),
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "workspace_id": "default"}
    (out / "dataset_ref.json").write_text(json.dumps(ref, indent=2, ensure_ascii=False), encoding="utf-8")

    for name, rows in splits.items():
        langs_ = defaultdict(int)
        for r in rows:
            langs_[f"{r['lang']}/{r['type']}"] += 1
        print(f"{name:15} {len(rows):5}  " + "  ".join(f"{k}={v}" for k, v in sorted(langs_.items())))
    print(f"unseen docs: {', '.join(unseen)}\nfrozen {a.id}@{a.version} -> {out}")


if __name__ == "__main__":
    main()
