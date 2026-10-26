"""Generates training and eval items from chunks with a local model (Ollama) or Claude.

    bash run.sh synth                 # print the plan and an estimate, generate nothing
    bash run.sh synth --yes           # run with the default local model; resumes where it stopped
    bash run.sh synth --yes --model claude-opus-5-5   # same prompt and schema on Claude (paid)
    bash run.sh synth --yes --limit 5 # try a few chunks first
    bash run.sh synth --per-doc 12    # cap chunks per document so long manuals don't dominate

For each chunk Claude extracts its key facts and writes, per fact, two training Q&A (EN, AR) and
two eval items (open question + MCQ) worded differently, in one language that alternates by chunk.
Appends to items.jsonl and synth_done.jsonl (one row per finished chunk, with token usage).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from typing import Literal

import anthropic
from pydantic import BaseModel

from . import ARTIFACTS, SERVICE_DIR
from .chunk import CHUNKS
from .parse import arabic_ratio

ITEMS = ARTIFACTS / "items.jsonl"
DONE = ARTIFACTS / "synth_done.jsonl"
PROMPT_VERSION = "facts-v1"
# $ per 1M tokens (input, output); used only for the estimate and the running total.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0)}


class QA(BaseModel):
    question: str
    answer: str


class MCQ(BaseModel):
    question: str
    choices: list[str]
    correct: Literal["A", "B", "C", "D"]


class Fact(BaseModel):
    fact: str
    topic: str
    train_en: QA
    train_ar: QA
    eval_open: QA
    eval_mcq: MCQ


class ChunkFacts(BaseModel):
    facts: list[Fact]


SYSTEM = """You write training and evaluation data for a model that advises mango growers in Egypt.

You get one passage from an agricultural source. Pick the key facts in it that a grower or
extension officer would need: varieties, timings, rates and doses, symptoms, thresholds,
practices, and their reasons. Skip bibliographies, tables of contents, author lists, funding
notes and anything not about mango production. Return an empty list if the passage has nothing
substantive.

For each fact:
- fact: the fact in one plain English sentence.
- topic: a short slash path, e.g. "pests/white_scale", "irrigation/salinity", "postharvest/hot_water".
- train_en and train_ar: one question and answer each, in English and in Modern Standard Arabic.
  The two may be translations of each other.
- eval_open and eval_mcq: an open question with its reference answer, and a multiple-choice
  question with four choices (A-D) and the correct letter. Write both in the eval language given
  in the request. They must test the same fact from a different angle with different wording,
  so that a model which only memorized the training question cannot match them by surface form.
  MCQ distractors must be plausible to a grower and clearly wrong according to the passage.

Rules for every item:
- Use only what the passage states. Copy numbers, doses, units and timings exactly; never invent one.
- Questions must stand alone: no "according to the text", "in this study" or "the authors".
  Name the crop, pest or variety so the question makes sense on its own.
- Answers are complete, direct and practical, two to five sentences, and say when advice depends
  on conditions (region, season, soil). Keep the passage's caveats.
- Where the passage concerns Egypt, keep the Egyptian specifics (governorates, local varieties,
  local months or rules).
- Choices are the bare option text, without "A)" prefixes."""


META = re.compile(r"\b(the|this) (text|passage|document|study|article|authors?)\b|according to the|"
                  r"\b(ل?لنص|النص|المادة العلمية|هذه الدراسة|المقال|المقالة|الفقرة|المؤلفون?)\b",  # \b: not النصف
                  re.I)


def chunk_key(c: dict) -> str:
    return f"{c['doc_id']}#{c['chunk']}"


def eval_lang(c: dict) -> str:
    return "ar" if c["chunk"] % 2 else "en"


def to_items(c: dict, facts: list[Fact], model: str) -> tuple[list[dict], int]:
    """Flattens facts into item rows; returns (items, dropped). `role` is train or eval."""
    base = {"source": {"doc_id": c["doc_id"], "chunk": c["chunk"], "license": c["license"]},
            "generator": {"model": model, "prompt_version": PROMPT_VERSION},
            "review": {"status": "pending"}}
    lang_e = eval_lang(c)
    items, dropped = [], 0
    for k, f in enumerate(facts):
        fid = f"{c['doc_id']}-c{c['chunk']}-f{k}"
        cand = [
            ("train", "en", "open_qa", f.train_en.question, f.train_en.answer, None),
            ("train", "ar", "open_qa", f.train_ar.question, f.train_ar.answer, None),
            ("eval", lang_e, "open_qa", f.eval_open.question, f.eval_open.answer, None),
            ("eval", lang_e, "mcq", f.eval_mcq.question, f.eval_mcq.correct, f.eval_mcq.choices),
        ]
        for role, lang, typ, q, ans, choices in cand:
            q, ans = q.strip(), ans.strip()
            ok = bool(q and ans)
            if typ == "mcq":
                ok &= choices is not None and len(choices) == 4 and len({x.strip() for x in choices}) == 4
            # questions must stand alone: drop ones that point at "the text" (local models slip here)
            ok &= not META.search(q)
            # the language check catches items written in the wrong language
            r = arabic_ratio(q + " " + (ans if typ == "open_qa" else " ".join(choices or [])))
            ok &= r > 0.5 if lang == "ar" else r < 0.2
            if not ok:
                dropped += 1
                continue
            item = base | {"id": f"{fid}-{role}-{typ}-{lang}", "fact_id": fid, "role": role,
                           "lang": lang, "type": typ, "topic": f.topic, "question": q, "answer": ans}
            if choices:
                item["choices"] = [x.strip() for x in choices]
            items.append(item)
    return items, dropped


def user_prompt(c: dict) -> str:
    return (f"Eval language: {'Modern Standard Arabic' if eval_lang(c) == 'ar' else 'English'}\n"
            f"Source: {c['doc_id']} (section: {c['heading'] or 'none'})\n\n<passage>\n{c['text']}\n</passage>")


def is_claude(model: str) -> bool:
    return model.startswith("claude-")


async def run_chunk(clients: dict, c: dict, model: str) -> tuple[list[Fact], dict]:
    return await (run_claude if is_claude(model) else run_ollama)(clients, c, model)


async def run_ollama(clients: dict, c: dict, model: str) -> tuple[list[Fact], dict]:
    """Local model through Ollama, constrained to the same JSON schema. No API cost."""
    body = {"model": model, "stream": False, "format": ChunkFacts.model_json_schema(),
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(c)}],
            "options": {"temperature": 0.3, "num_ctx": 8192, "num_predict": 6000}}
    if model.startswith("qwen3"):
        body["think"] = False  # the schema already structures the work; thinking would triple the time
    r = await clients["ollama"].post("/api/chat", json=body)
    r.raise_for_status()
    data = r.json()
    usage = {"input_tokens": data.get("prompt_eval_count", 0), "output_tokens": data.get("eval_count", 0),
             "stop_reason": data.get("done_reason", "stop"), "model": model,
             "seconds": round(data.get("total_duration", 0) / 1e9, 1)}
    try:
        return ChunkFacts.model_validate_json(data["message"]["content"]).facts, usage
    except ValueError:
        return [], usage | {"stop_reason": "invalid_json"}


async def run_claude(clients: dict, c: dict, model: str) -> tuple[list[Fact], dict]:
    client: anthropic.AsyncAnthropic = clients["claude"]
    user = user_prompt(c)
    resp = await client.beta.messages.parse(
        model=model,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "medium"},
        output_format=ChunkFacts,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
             "stop_reason": resp.stop_reason, "model": resp.model, "request_id": resp._request_id}
    if resp.stop_reason != "end_turn" or resp.parsed_output is None:
        return [], usage
    return resp.parsed_output.facts, usage


def estimate(chunks: list[dict], model: str) -> str:
    # ~3.5 chars/token for English, ~2.5 for Arabic; output ~900 tokens per fact incl. thinking, ~4 facts.
    tin = sum(len(c["text"]) / (2.5 if c["lang"] == "ar" else 3.5) + 600 for c in chunks)
    tout = len(chunks) * 4 * (900 if is_claude(model) else 500)
    head = (f"{len(chunks)} chunks -> ~{len(chunks) * 4} facts, ~{len(chunks) * 8} train Q&A, "
            f"~{len(chunks) * 8} eval items\n~{tin / 1e6:.2f}M input + ~{tout / 1e6:.2f}M output tokens on {model}: ")
    if not is_claude(model):
        return head + "no API cost (local model); time depends on GPU throughput, see the per-chunk seconds"
    pin, pout = PRICES.get(model, (0, 0))
    return head + f"about ${tin / 1e6 * pin + tout / 1e6 * pout:.0f}"


async def main_async(a: argparse.Namespace) -> None:
    items_path, done_path = ITEMS, DONE
    if a.tag:  # a side run (e.g. a model bake-off) that must not touch the main items file
        items_path, done_path = ARTIFACTS / f"items.{a.tag}.jsonl", ARTIFACTS / f"synth_done.{a.tag}.jsonl"
    chunks = [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines()]
    if a.docs:
        chunks = [c for c in chunks if c["doc_id"] in a.docs.split(",")]
    done = set()
    if done_path.exists():
        done = {json.loads(l)["key"] for l in done_path.read_text(encoding="utf-8").splitlines()}
    if a.per_doc:  # spread the budget across sources: keep at most N chunks per document, evenly spaced
        by_doc: dict[str, list[dict]] = {}
        for c in chunks:
            by_doc.setdefault(c["doc_id"], []).append(c)
        chunks = [cs[round(i * len(cs) / a.per_doc)] if len(cs) > a.per_doc else cs[i]
                  for cs in by_doc.values() for i in range(min(len(cs), a.per_doc))]
    todo = [c for c in chunks if chunk_key(c) not in done][: a.limit or None]
    print(f"{len(done)} chunks already done; to do now:")
    print(estimate(todo, a.model))
    if not a.yes or not todo:
        print("Dry run. Re-run with --yes to generate." if todo else "Nothing to do.")
        return

    import httpx

    clients = {"ollama": httpx.AsyncClient(base_url=a.ollama_url, timeout=httpx.Timeout(30, read=1800))}
    if is_claude(a.model):
        clients["claude"] = anthropic.AsyncAnthropic(max_retries=4)
    sem = asyncio.Semaphore(a.concurrency)
    totals = {"items": 0, "dropped": 0, "in": 0, "out": 0, "failed": 0}
    lock = asyncio.Lock()
    t0 = time.time()

    async def one(c: dict) -> None:
        async with sem:
            try:
                facts, usage = await run_chunk(clients, c, a.model)
            except anthropic.BadRequestError as e:
                raise SystemExit(f"Request rejected; fix before re-running: {e.message}")
            except (anthropic.APIStatusError, anthropic.APIConnectionError, httpx.HTTPError) as e:
                async with lock:
                    totals["failed"] += 1
                print(f"failed  {chunk_key(c)}  {type(e).__name__}: {e}"[:200], flush=True)
                return  # not marked done, so the next run retries it
        if usage["stop_reason"] == "invalid_json":
            async with lock:
                totals["failed"] += 1
            print(f"failed  {chunk_key(c)}  output did not match the schema", flush=True)
            return
        items, dropped = to_items(c, facts, usage["model"])
        async with lock:
            with items_path.open("a", encoding="utf-8") as f:
                f.writelines(json.dumps(i, ensure_ascii=False) + "\n" for i in items)
            with done_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": chunk_key(c), "facts": len(facts), "items": len(items),
                                    "dropped": dropped} | usage) + "\n")
            totals["items"] += len(items); totals["dropped"] += dropped
            totals["in"] += usage["input_tokens"]; totals["out"] += usage["output_tokens"]
        note = "" if usage["stop_reason"] in ("end_turn", "stop") else f"  stop={usage['stop_reason']}"
        secs = f"  {usage['seconds']}s" if "seconds" in usage else ""
        print(f"{len(facts):2} facts {len(items):3} items  {chunk_key(c)}{secs}{note}", flush=True)

    await asyncio.gather(*(one(c) for c in todo))
    pin, pout = PRICES.get(a.model, (0, 0))
    cost = totals["in"] / 1e6 * pin + totals["out"] / 1e6 * pout
    print(f"\n{totals['items']} items written, {totals['dropped']} dropped by checks, "
          f"{totals['failed']} chunks failed (re-run to retry). {time.time() - t0:.0f}s, "
          f"{totals['out']} output tokens" + (f", spent about ${cost:.2f}." if is_claude(a.model) else "."))


def load_env() -> None:
    env = SERVICE_DIR / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually generate")
    ap.add_argument("--limit", type=int, default=0, help="only this many chunks")
    ap.add_argument("--per-doc", type=int, default=0, help="at most this many chunks per document (0 = all)")
    ap.add_argument("--concurrency", type=int, default=2, help="parallel requests (Claude handles ~6)")
    ap.add_argument("--model", default="qwen3:14b", help="an Ollama model, or a claude-* model id")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--tag", help="write items.<tag>.jsonl instead of items.jsonl (for side runs)")
    ap.add_argument("--docs", help="comma-separated doc_ids to restrict to")
    a = ap.parse_args()
    load_env()
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
