"""Model targets: how Eval gets answers from each kind of registry model.

- HFTarget: a 4-bit checkpoint on the local GPU, optionally with a PEFT adapter. The base is loaded
  once and the adapter switched on and off, so base and fine-tuned answers come from the exact
  same weights, quantization, prompt and decoding.
- OllamaTarget: a model served by the local Ollama (e.g. Gemma 4 12B, or an exported fine-tune).
- AnthropicTarget / GeminiTarget: hosted models; they cost money, so Eval only calls them on request.

Every target answers chat messages with greedy (or provider-default) decoding, thinking off where
the model allows it, and reports per-answer latency and token counts.

Two per-task options (Gen): `json_schema` asks the provider for JSON that matches the schema
(Ollama `format`, Anthropic `output_config.format`, Gemini JSON mime type; local transformers models
get no constraint and are scored on what they write), and `stream` streams the answer so the time
to the first answer token (`ttft_s`) is measured. Thinking tokens never count as the first token.
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import PROMPTS, SERVICE_DIR

LETTERS = ["A", "B", "C", "D"]


@dataclass
class Answer:
    text: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    ttft_s: float | None = None  # seconds to the first answer token, when the answer was streamed


@dataclass(frozen=True)
class Gen:
    """Per-task generation options, the same for every model in an evaluation."""

    json_schema: dict | None = None
    stream: bool = False

    def cache_extra(self) -> list:
        """What these options add to a cached answer's key; nothing for the defaults, so older caches stay valid."""
        out = []
        if self.json_schema:
            out.append({"json_schema": self.json_schema})
        if self.stream:
            out.append("stream")
        return out


# JSON Schema keywords the Anthropic structured-output format does not accept. They are removed from the
# schema sent; json_valid still checks the answer against the full schema.
API_UNSUPPORTED = {"minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                   "multipleOf", "minItems", "maxItems", "uniqueItems", "pattern"}


def api_schema(schema):
    if isinstance(schema, dict):
        return {k: api_schema(v) for k, v in schema.items() if k not in API_UNSUPPORTED}
    if isinstance(schema, list):
        return [api_schema(v) for v in schema]
    return schema


def load_env() -> None:
    """Reads services/eval/.env on every call; the file wins, so a corrected key takes effect without a restart."""
    env = SERVICE_DIR / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"')


def chat_text(tok, messages: list[dict]) -> str:
    """The exact string a local model is fed: its chat template applied, ready for the answer."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@lru_cache(maxsize=4)
def _tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def as_sent(model: dict, messages: list[dict], max_new_tokens: int, repo: Path, gen: Gen = Gen()) -> dict:
    """What `model` (a registry entry) receives for these messages, built by the same code that sends it."""
    kind = model["kind"]
    if kind in ("hf_base", "hf_adapter"):
        path = str(repo / model["adapter_uri"]) if kind == "hf_adapter" else model["hf_id"]
        return {"format": "text", "label": "chat template applied (the exact token string)",
                "text": chat_text(_tokenizer(path), messages)}
    if kind == "ollama":
        body = OllamaTarget.request(model["ollama_tag"], messages, max_new_tokens, gen=gen)
        return {"format": "json", "label": "POST /api/chat (Ollama applies the model's chat template)", "body": body}
    if model.get("provider") == "anthropic":
        return {"format": "json", "label": "Anthropic messages." + ("stream" if gen.stream else "create") + "(...)",
                "body": AnthropicTarget.request(model["api_model"], messages, max_new_tokens, model.get("options"), gen)}
    return {"format": "json", "label": f"POST models/{model['api_model']}:generateContent",
            "body": GeminiTarget.request(messages, max_new_tokens, model.get("options"), gen)}


class HFTarget:
    def __init__(self, base: str, adapter: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(adapter or base)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(base, device_map="cuda:0", dtype=torch.bfloat16)
        self.has_adapter = adapter is not None
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.letter_ids = [self.tok.encode(x, add_special_tokens=False)[0] for x in LETTERS]

    def use(self, adapter: bool):
        """Context manager: adapter on (fine-tuned) or off (base)."""
        if adapter or not self.has_adapter:
            return nullcontext()
        return self.model.disable_adapter()

    def render(self, messages: list[dict]) -> str:
        return chat_text(self.tok, messages)

    def prompt(self, user: str) -> str:
        return self.render([{"role": "system", "content": PROMPTS["system"]}, {"role": "user", "content": user}])

    def _generate(self, prompts: list[str], max_new_tokens: int, batch: int) -> list[Answer]:
        torch = self.torch
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out: list[Answer | None] = [None] * len(prompts)
        with torch.inference_mode():
            for s in range(0, len(order), batch):
                idx = order[s:s + batch]
                enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                               add_special_tokens=False).to("cuda:0")
                t0 = time.time()
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
                per = (time.time() - t0) / len(idx)  # batched: latency is amortized over the batch
                n_in = enc["attention_mask"].sum(1).tolist()
                for k, (i, row) in enumerate(zip(idx, gen[:, enc["input_ids"].shape[1]:])):
                    n_out = int((row != self.tok.pad_token_id).sum())
                    out[i] = Answer(self.tok.decode(row, skip_special_tokens=True).strip(), per, int(n_in[k]), n_out)
        return out  # type: ignore[return-value]

    def chat(self, conversations: list[list[dict]], max_new_tokens: int = 400, batch: int = 8,
             progress=None, gen: Gen = Gen()) -> list[Answer]:
        """Batched greedy decoding. No schema constraint and no streaming: `gen` is accepted and ignored."""
        out: list[Answer] = []
        step = max(batch * 8, 1)
        for s in range(0, len(conversations), step):  # chunks, so progress can be reported
            out += self._generate([self.render(m) for m in conversations[s:s + step]], max_new_tokens, batch)
            if progress:
                progress(len(out))
        return out

    def generate(self, questions: list[str], max_new_tokens: int = 400, batch: int = 8) -> list[str]:
        """Mango legacy path: questions with the mango system prompt."""
        return [a.text for a in self._generate([self.prompt(q) for q in questions], max_new_tokens, batch)]

    def mcq(self, items: list[dict], batch: int = 8) -> list[list[str]]:
        """For each item, the original-order letter picked under each of the 4 rotations of its choices.

        A pick is the letter with the highest next-token probability. Rotating the choices cancels the
        position bias of small models (e.g. always answering A).
        """
        torch = self.torch
        jobs = [(i, r) for i in range(len(items)) for r in range(4)]
        picks: list[list[str]] = [[] for _ in items]
        with torch.inference_mode():
            for s in range(0, len(jobs), batch):
                chunk = jobs[s:s + batch]
                prompts = [self.prompt(mcq_text(items[i], r)) for i, r in chunk]
                enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
                logits = self.model(**enc).logits[:, -1, self.letter_ids]  # left padding: last position is real
                for (i, r), j in zip(chunk, logits.argmax(-1).tolist()):
                    picks[i].append(LETTERS[(j + r) % 4])  # map the shown position back to the original letter
        return picks


def mcq_text(it: dict, rotation: int = 0) -> str:
    """Choice shown at position p is original choice (p + rotation) % 4."""
    lang = "ar" if it["lang"] == "ar" else "en"
    shown = [it["choices"][(p + rotation) % 4] for p in range(4)]
    choices = "\n".join(f"{L}) {c}" for L, c in zip(LETTERS, shown))
    return f"{it['question']}\n\n{choices}\n\n{PROMPTS['mcq_instruction'][lang]}"


class OllamaTarget:
    """One request at a time: the GPU serves one model, and latency stays per answer, not amortized."""

    def __init__(self, tag: str, url: str = "http://127.0.0.1:11434", num_ctx: int = 8192):
        import httpx

        self.tag, self.num_ctx = tag, num_ctx
        self.http = httpx.Client(base_url=url, timeout=httpx.Timeout(30, read=900))
        self.think: bool | None = False
        self.warm = False

    @staticmethod
    def request(tag: str, messages: list[dict], max_new_tokens: int, num_ctx: int = 8192, think: bool | None = False,
                gen: Gen = Gen()) -> dict:
        body = {"model": tag, "messages": messages, "stream": gen.stream, "keep_alive": "10m",
                "options": {"temperature": 0, "seed": 0, "num_ctx": num_ctx, "num_predict": max_new_tokens}}
        if think is not None:
            body["think"] = think
        if gen.json_schema:
            body["format"] = gen.json_schema  # Ollama constrains decoding to the schema
        return body

    def one(self, messages: list[dict], max_new_tokens: int, gen: Gen = Gen()) -> Answer:
        body = self.request(self.tag, messages, max_new_tokens, self.num_ctx, self.think, gen)
        t0 = time.time()
        if gen.stream:
            return self._stream(body, messages, max_new_tokens, gen, t0)
        r = self.http.post("/api/chat", json=body)
        if r.status_code == 400 and "think" in r.text and self.think is not None:  # model without a thinking switch
            self.think = None
            return self.one(messages, max_new_tokens, gen)
        if r.status_code != 200:
            return Answer("", time.time() - t0, error=f"ollama {r.status_code}: {r.text[:300]}")
        j = r.json()
        return Answer(j["message"]["content"].strip(), time.time() - t0, j.get("prompt_eval_count"), j.get("eval_count"))

    def _stream(self, body: dict, messages: list[dict], max_new_tokens: int, gen: Gen, t0: float) -> Answer:
        import json

        parts, ttft, last = [], None, {}
        with self.http.stream("POST", "/api/chat", json=body) as r:
            if r.status_code != 200:
                text = r.read().decode(errors="replace")
                if r.status_code == 400 and "think" in text and self.think is not None:
                    self.think = None
                    return self.one(messages, max_new_tokens, gen)
                return Answer("", time.time() - t0, error=f"ollama {r.status_code}: {text[:300]}")
            for line in r.iter_lines():
                if not line.strip():
                    continue
                last = json.loads(line)
                piece = (last.get("message") or {}).get("content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    parts.append(piece)
        return Answer("".join(parts).strip(), time.time() - t0, last.get("prompt_eval_count"), last.get("eval_count"),
                      None, ttft)

    def chat(self, conversations: list[list[dict]], max_new_tokens: int = 400, progress=None, gen: Gen = Gen(),
             **_) -> list[Answer]:
        if conversations and not self.warm:  # load the model first, so its load time is not counted as the first answer's latency
            self.one(conversations[0], 1)
            self.warm = True
        out = []
        for i, m in enumerate(conversations):
            out.append(self.one(m, max_new_tokens, gen))
            if progress and (i + 1) % 10 == 0:
                progress(i + 1)
        return out

    def unload(self) -> None:
        self.http.post("/api/generate", json={"model": self.tag, "keep_alive": 0})


def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


class AnthropicTarget:
    """Claude through the Messages API. No sampling parameters (current models reject them); thinking
    and effort stay at the model's defaults unless the registry entry sets `options`: `effort`
    ("low" .. "max") and `thinking` (sent as is, e.g. {"type": "between_tools"} on Sonnet 5.5)."""

    def __init__(self, model: str, options: dict | None = None, concurrency: int = 4):
        load_env()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set; add it to services/eval/.env")
        import anthropic

        self.client = anthropic.Anthropic(max_retries=4)
        self.model, self.options, self.concurrency = model, options or {}, concurrency

    @staticmethod
    def request(model: str, messages: list[dict], max_new_tokens: int, options: dict | None = None,
                gen: Gen = Gen()) -> dict:
        """The messages.create (or messages.stream) keyword arguments."""
        options = options or {}
        system, msgs = split_system(messages)
        config = {"effort": options["effort"]} if options.get("effort") else {}
        if gen.json_schema:
            config["format"] = {"type": "json_schema", "schema": api_schema(gen.json_schema)}
        kw = {"output_config": config} if config else {}
        if options.get("thinking"):
            kw["thinking"] = options["thinking"]
        return {"model": model, "max_tokens": max(16000, max_new_tokens), "system": system, "messages": msgs} | kw

    def one(self, messages: list[dict], max_new_tokens: int, gen: Gen = Gen()) -> Answer:
        req = self.request(self.model, messages, max_new_tokens, self.options, gen)
        t0, ttft = time.time(), None
        try:
            if gen.stream:
                with self.client.messages.stream(**req) as stream:
                    for piece in stream.text_stream:  # text only: thinking blocks are not the answer
                        if piece and ttft is None:
                            ttft = time.time() - t0
                    resp = stream.get_final_message()
            else:
                resp = self.client.messages.create(**req)
        except Exception as e:
            return Answer("", time.time() - t0, error=f"{type(e).__name__}: {e}"[:300])
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        err = "refused" if resp.stop_reason == "refusal" else None
        return Answer(text, time.time() - t0, resp.usage.input_tokens, resp.usage.output_tokens, err, ttft)

    def chat(self, conversations: list[list[dict]], max_new_tokens: int = 400, progress=None, gen: Gen = Gen(),
             **_) -> list[Answer]:
        return parallel(lambda m, n: self.one(m, n, gen), conversations, max_new_tokens, self.concurrency, progress)


class GeminiTarget:
    """Gemini through the REST generateContent endpoint (no SDK needed)."""

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model: str, options: dict | None = None, concurrency: int = 4):
        load_env()
        self.key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.key:
            raise RuntimeError("GEMINI_API_KEY is not set; add it to services/eval/.env")
        import httpx

        self.http = httpx.Client(timeout=httpx.Timeout(30, read=300))
        self.model, self.options, self.concurrency = model, options or {}, concurrency

    @staticmethod
    def request(messages: list[dict], max_new_tokens: int, options: dict | None = None, gen: Gen = Gen()) -> dict:
        """No streaming (ttft is not measured for Gemini); a JSON schema asks for JSON output only."""
        system, msgs = split_system(messages)
        config = {"temperature": 0, "maxOutputTokens": max(8192, max_new_tokens)}
        if gen.json_schema:
            config["responseMimeType"] = "application/json"
        body = {"contents": [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                             for m in msgs],
                "generationConfig": config | (options or {})}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def one(self, messages: list[dict], max_new_tokens: int, gen: Gen = Gen()) -> Answer:
        body = self.request(messages, max_new_tokens, self.options, gen)
        t0 = time.time()
        for attempt in range(4):
            r = self.http.post(self.URL.format(model=self.model), json=body, headers={"x-goog-api-key": self.key})
            if r.status_code not in (429, 500, 503):
                break
            time.sleep(2 ** attempt * 2)
        if r.status_code != 200:
            return Answer("", time.time() - t0, error=f"gemini {r.status_code}: {r.text[:300]}")
        j = r.json()
        cand = (j.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
        u = j.get("usageMetadata", {})
        return Answer(text.strip(), time.time() - t0, u.get("promptTokenCount"),
                      (u.get("candidatesTokenCount") or 0) + (u.get("thoughtsTokenCount") or 0),
                      None if text else f"no text (finishReason {cand.get('finishReason')})")

    def chat(self, conversations: list[list[dict]], max_new_tokens: int = 400, progress=None, gen: Gen = Gen(),
             **_) -> list[Answer]:
        return parallel(lambda m, n: self.one(m, n, gen), conversations, max_new_tokens, self.concurrency, progress)


def parallel(fn, conversations, max_new_tokens, concurrency, progress) -> list[Answer]:
    from concurrent.futures import ThreadPoolExecutor

    out: list[Answer | None] = [None] * len(conversations)
    with ThreadPoolExecutor(concurrency) as pool:
        futs = {pool.submit(fn, m, max_new_tokens): i for i, m in enumerate(conversations)}
        for n, f in enumerate(futs, 1):
            out[futs[f]] = f.result()
            if progress and n % 10 == 0:
                progress(n)
    return out  # type: ignore[return-value]
