"""Scoring for tasks whose answer is JSON (task spec `output.format: json`) or short spoken text.

The reference of such a task is a JSON object. It always has `expected` (the reference answer, when
the task has one) and may carry what a metric needs besides it: the source texts quotes must come
from, rubric weights, a word limit. The task spec's `scoring` block says where to look.

Generic metrics (any task):
- json_valid: the answer parses as JSON and matches the task's JSON Schema.
- field_match: per-field agreement with `expected` (`scoring.fields`: path, type text | enum | list).
- label_acc: every path in `scoring.labels` equals `expected` (a classification is right).
- one_question: the text at `scoring.text_path` (or the whole answer) asks at most one question.
- word_limit: the text has at most `reference.max_words` words ({tier: n} for a limit per model tier).
- quotes_verbatim: share of quotes (`scoring.quotes_path`) found word for word in one of
  `reference.source_texts` (case, punctuation and spacing ignored; "..." may skip words).

Strong Hire formulas (ported from strong-hire, where code, not the model, turns ratings into results):
- rubric_within_one: per (question, competency) the human scored, the model's 1-4 score is within
  one point. Scores whose quotes are not the candidate's words are dropped first, as the product does.
- hire_signal_within_one / hire_signal_exact: the hire signal computed from the model's scores
  (weighted competency and value averages, band floors, hard rules) against the human signal.
- gap_match_score: the GA-1 match score (0-100) computed from the model's requirement and competency
  ratings, with the evidence rule. gap_fit: the score falls in the band the pair's fit expects.

Paths: dotted keys; `[]` after a key walks a list, e.g. `per_question[].scores[].quotes[]`.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict

from agritune_core.tasks import parse_json

# ---------- parsing ----------

FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_answer(text: str):
    """The JSON value in an answer: the whole text, a ```json fence, or the outermost {...}; None if none parses."""
    text = (text or "").strip()
    candidates = [text] + FENCE.findall(text)
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{"): text.rfind("}") + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def pluck(obj, path: str) -> list:
    """Every value at `path`; an empty list when the path does not exist."""
    values = [obj]
    for part in path.split("."):
        many = part.endswith("[]")
        key = part[:-2] if many else part
        nxt = []
        for v in values:
            if key:
                if not isinstance(v, dict) or key not in v:
                    continue
                v = v[key]
            if many:
                nxt += v if isinstance(v, list) else []
            else:
                nxt.append(v)
        values = nxt
    return values


def first(obj, path: str):
    found = pluck(obj, path)
    return found[0] if found else None


# ---------- text matching (strong-hire's normalize / grounding and quote checker) ----------

_NON_WORD = re.compile(r"[^a-z0-9+#]+")


def normalize(text: str) -> str:
    return " " + _NON_WORD.sub(" ", str(text).lower()).strip() + " "


def grounding(value: str, source: str) -> float:
    """1.0 when the normalized value is in the normalized source, else the share of its words there."""
    needle = normalize(value)
    if needle.strip() and needle in source:
        return 1.0
    words = [w for w in needle.split() if len(w) > 2]
    return sum(f" {w} " in source for w in words) / len(words) if words else 0.0


def text_score(expected, actual, threshold: float = 0.6) -> float:
    if not expected or not actual:
        return 1.0 if not expected and not actual else 0.0
    exp, act = normalize(expected), normalize(actual)
    if exp in act or act in exp:
        return 1.0
    return 1.0 if max(grounding(expected, act), grounding(actual, exp)) >= threshold else 0.0


def _same_item(a: str, b: str, threshold: float) -> bool:
    return max(grounding(a, normalize(b)), grounding(b, normalize(a))) >= threshold


def list_f1(expected: list, actual: list, threshold: float = 0.6) -> float:
    expected = [str(x) for x in expected or []]
    actual = [str(x) for x in actual or []]
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    recall = sum(any(_same_item(e, a, threshold) for a in actual) for e in expected) / len(expected)
    precision = sum(any(_same_item(a, e, threshold) for e in expected) for a in actual) / len(actual)
    return 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)


QUOTE_MIN_WORDS = 3
_ELLIPSIS = re.compile(r"\.\.\.|…")
_NOT_WORD = re.compile(r"[^\w']+")
_APOSTROPHES = str.maketrans({0x2018: "'", 0x2019: "'"})


def quote_words(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).translate(_APOSTROPHES)
    tokens = [t.strip("'") for t in _NOT_WORD.split(text.lower())]
    return " " + " ".join(t for t in tokens if t) + " "


def quote_ok(quote: str, sources: list[str]) -> bool:
    """The quote's words appear in order in one source text (each part of an "a ... b" quote)."""
    parts = [p for p in (quote_words(p) for p in _ELLIPSIS.split(str(quote))) if p.strip()]
    if not parts or sum(len(p.split()) for p in parts) < QUOTE_MIN_WORDS:
        return False
    for text in sources:
        pos, ok = 0, True
        for part in parts:
            found = text.find(part, pos)
            if found < 0:
                ok = False
                break
            pos = found + len(part) - 1
        if ok:
            return True
    return False


# ---------- generic metrics ----------

def schema_errors(value, schema: dict) -> list[str]:
    from jsonschema import Draft202012Validator

    v = Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.absolute_path)) or '(root)'}: {e.message}"[:200] for e in v.iter_errors(value)][:5]


def field_match(expected: dict, actual: dict, fields: list[dict]) -> tuple[float, dict]:
    per = {}
    for f in fields:
        path, kind, thr = f["path"], f.get("type", "text"), f.get("threshold", 0.6)
        exp, act = first(expected, path), first(actual, path) if isinstance(actual, dict) else None
        if kind == "enum":
            per[path] = float((exp or None) == (act or None))
        elif kind == "list":  # "skills" is a list; "roles[].company" collects one value per list item
            lp = path if "[]" in path else path + "[]"
            per[path] = list_f1(pluck(expected, lp), pluck(actual, lp) if isinstance(actual, dict) else [], thr)
        else:
            per[path] = text_score(exp, act, thr)
    return (sum(per.values()) / len(per) if per else 0.0), per


def same_label(a, b) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().casefold() == b.strip().casefold()
    return a == b


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


# ---------- Strong Hire: hire signal (strong_worker.scoring.signal) ----------

SIGNALS = ["No Hire", "Lean No Hire", "Lean Hire", "Hire", "Strong Hire"]  # rank order, worst first
BAND_FLOORS = [("Strong Hire", 3.5), ("Hire", 3.0), ("Lean Hire", 2.6), ("Lean No Hire", 2.0)]


def rubric_point(average: float) -> int:
    return max(1, min(4, math.floor(average + 0.5)))


def _weighted(averages: dict[str, float], weights: dict[str, float]) -> float:
    total = sum(weights.get(k, 1.0) for k in averages)
    return sum(a * weights.get(k, 1.0) for k, a in averages.items()) / total


def hire_signal(comp_avgs: dict[str, float], weights: dict[str, float], value_avgs: dict[str, float],
                value_weights: dict[str, float], values_share: float) -> str:
    comp = _weighted(comp_avgs, weights)
    values = {v: a for v, a in value_avgs.items() if v in value_weights}
    average = comp if not values else (1 - values_share) * comp + values_share * _weighted(values, value_weights)
    signal = next((name for name, floor in BAND_FLOORS if average >= floor), "No Hire")
    points = [rubric_point(a) for a in comp_avgs.values()]
    if signal == "Strong Hire" and min(points) < 3:
        signal = "Hire"
    if min(points) == 1 and SIGNALS.index(signal) > SIGNALS.index("Lean No Hire"):
        signal = "Lean No Hire"
    return signal


def kept_scores(card: dict, ref: dict) -> tuple[dict, dict, dict, dict, int, int]:
    """The model's scores the product keeps: asked questions, allowed competencies and values, valid quotes.

    Returns (per-question competency scores, per-question value scores, overall competency scores,
    overall value scores, quotes checked, quotes valid)."""
    asked, comps, vals = set(ref["asked"]), set(ref["competencies"]), set(ref.get("values") or [])
    sources = [quote_words(t) for t in ref["source_texts"]]
    checked = valid = 0

    def good(quotes) -> bool:
        nonlocal checked, valid
        ok = [q for q in quotes or [] if quote_ok(q, sources)]
        checked += len(quotes or [])
        valid += len(ok)
        return bool(ok)

    per_q, per_v, overall, overall_v = {}, {}, {}, {}
    for qs in card.get("per_question") or []:
        ref_id = qs.get("question_ref")
        if ref_id not in asked:
            continue
        for s in qs.get("scores") or []:
            key = (ref_id, s.get("competency"))
            if key[1] in comps and key not in per_q and isinstance(s.get("score"), int) and good(s.get("quotes")):
                per_q[key] = s["score"]
        for s in qs.get("value_scores") or []:
            key = (ref_id, s.get("value"))
            if key[1] in vals and key not in per_v and isinstance(s.get("score"), int) and good(s.get("quotes")):
                per_v[key] = s["score"]
    for s in card.get("competency_scores") or []:
        c = s.get("competency")
        if c in comps and c not in overall and isinstance(s.get("score"), int) and good(s.get("quotes")):
            overall[c] = s["score"]
    for s in card.get("value_scores") or []:
        v = s.get("value")
        if v in vals and v not in overall_v and isinstance(s.get("score"), int) and good(s.get("quotes")):
            overall_v[v] = s["score"]
    return per_q, per_v, overall, overall_v, checked, valid


def averages(per: dict, overall: dict) -> dict[str, float]:
    """Mean of per-question scores by name; a name scored only overall keeps that score."""
    by = defaultdict(list)
    for (_, name), score in per.items():
        by[name].append(score)
    out = {name: sum(v) / len(v) for name, v in by.items()}
    for name, score in overall.items():
        out.setdefault(name, float(score))
    return out


def score_card(card: dict, ref: dict) -> tuple[dict, dict]:
    per_q, per_v, overall, overall_v, checked, valid = kept_scores(card, ref)
    gold = [(g["question"], g["competency"], g["score"]) for g in ref["gold_scores"]]
    hits = [abs(per_q[(q, c)] - s) <= 1 if (q, c) in per_q else False for q, c, s in gold]
    scores = {"rubric_within_one": sum(hits) / len(hits) if hits else 0.0}
    comp_avgs = averages(per_q, overall)
    details = {"kept_scores": len(per_q), "quotes_checked": checked, "quotes_valid": valid,
               "competency_averages": {k: round(v, 2) for k, v in comp_avgs.items()}}
    if comp_avgs:
        signal = hire_signal(comp_avgs, ref.get("weights") or {}, averages(per_v, overall_v),
                             ref.get("value_weights") or {}, ref.get("values_share", 0.0))
    else:
        signal = None  # the product refuses to score a transcript with no kept score
    details |= {"hire_signal": signal, "gold_signal": ref["gold_signal"]}
    gap = abs(SIGNALS.index(signal) - SIGNALS.index(ref["gold_signal"])) if signal else 99
    scores["hire_signal_within_one"] = float(gap <= 1)
    scores["hire_signal_exact"] = float(gap == 0)
    return scores, details


# ---------- Strong Hire: gap analysis match score (strong_worker.gap.scoring) ----------

POINTS = (0, 35, 70, 100)
REQUIREMENT_WEIGHTS = {"must_have": 2.0, "nice_to_have": 1.0}
REQUIREMENT_SHARE = 0.7
EVIDENCE_MATCH = 0.6
_GAP_WORD = re.compile(r"[a-z0-9][a-z0-9+#.%]*")


def gap_words(text: str) -> set[str]:
    return {w.rstrip(".") for w in _GAP_WORD.findall(str(text).lower())} - {""}


def gap_score(assessment: dict, ref: dict) -> tuple[int, dict]:
    """The match score the product computes from this assessment."""
    resume = gap_words(ref["resume_text"])

    def checked(rating, evidence) -> int:
        rating = max(0, min(3, rating if isinstance(rating, int) else 0))
        found = gap_words(evidence or "")
        ok = bool(found) and len(found & resume) / len(found) >= EVIDENCE_MATCH
        return 1 if rating > 1 and not ok else rating

    by_id = {}
    for r in assessment.get("requirements") or []:
        by_id.setdefault(str(r.get("id", "")).strip().lower(), r)
    reqs, unrated = [], 0
    for req in ref["requirements"]:
        r = by_id.get(req["id"])
        if r is None:
            unrated += 1
        reqs.append((POINTS[checked(r.get("rating"), r.get("evidence")) if r else 0], REQUIREMENT_WEIGHTS[req["kind"]]))
    weights = ref.get("competency_weights") or {}
    comps = {}
    for c in assessment.get("competencies") or []:
        if c.get("competency") in ref["competencies"] and c["competency"] not in comps:
            rating = c.get("rating") if isinstance(c.get("rating"), int) else 0
            comps[c["competency"]] = POINTS[max(0, min(3, rating))]

    def mean(pairs):
        w = sum(p[1] for p in pairs)
        return sum(v * wt for v, wt in pairs) / w if w > 0 else None

    req = mean(reqs)
    comp = mean([(v, weights.get(c, 1.0)) for c, v in comps.items()])
    if req is None and comp is None:
        total = 0.0
    elif req is None or comp is None:
        total = req if comp is None else comp
    else:
        total = REQUIREMENT_SHARE * req + (1 - REQUIREMENT_SHARE) * comp
    return round(total), {"requirement_score": req, "competency_score": comp, "unrated_requirements": unrated}


# ---------- entry point ----------

STRUCTURED = {"json_valid", "field_match", "label_acc", "one_question", "word_limit", "quotes_verbatim",
              "rubric_within_one", "hire_signal_within_one", "hire_signal_exact", "gap_fit", "gap_match_score"}


def score_structured(task, text: str, reference, want: set[str], tier: str | None = None) -> tuple[dict, dict]:
    """(scores, details) for one answer by a model with this prompt tier. A metric that does not apply to
    an item is left out."""
    want = want & STRUCTURED
    if not want:
        return {}, {}
    cfg = task.scoring or {}
    ref = parse_json(reference) if isinstance(reference, str) else reference
    ref = ref if isinstance(ref, dict) else {"expected": ref}
    expected = ref.get("expected")
    is_json = task.output.get("format") == "json"
    value = parse_answer(text) if is_json else None
    scores, details = {}, {}
    if is_json:
        details["parsed"] = value is not None
    if "json_valid" in want and is_json:
        schema = task.json_schema
        errors = ["not JSON"] if value is None else (schema_errors(value, schema) if schema else [])
        scores["json_valid"] = float(not errors)
        if errors:
            details["schema_errors"] = errors
    obj = value if isinstance(value, dict) else {}
    if "field_match" in want and cfg.get("fields") and isinstance(expected, dict):
        if isinstance(value, dict):
            scores["field_match"], details["fields"] = field_match(expected, obj, cfg["fields"])
        else:  # no answer object: a failed extraction, not a partial one (empty fields would match empty ones)
            scores["field_match"] = 0.0
    if "label_acc" in want and cfg.get("labels") and isinstance(expected, dict):
        got = {p: first(obj, p) for p in cfg["labels"]}
        scores["label_acc"] = float(all(same_label(got[p], first(expected, p)) for p in cfg["labels"]))
        details["labels"] = got
    spoken = (first(obj, cfg["text_path"]) if cfg.get("text_path") else text) or ""
    if "one_question" in want:
        scores["one_question"] = float(str(spoken).count("?") <= 1)
    limit = ref.get("max_words")
    if isinstance(limit, dict):  # the limit the product gives this model's tier
        limit = limit.get(tier or "default") or limit.get("default")
    if "word_limit" in want and limit:
        n = count_words(str(spoken))
        scores["word_limit"] = float(n <= int(limit))
        details["words"] = n
    if "quotes_verbatim" in want and cfg.get("quotes_path") and ref.get("source_texts"):
        quotes = [q for q in pluck(obj, cfg["quotes_path"]) if isinstance(q, str)]
        if quotes:
            sources = [quote_words(t) for t in ref["source_texts"]]
            bad = [q for q in quotes if not quote_ok(q, sources)]
            scores["quotes_verbatim"] = 1 - len(bad) / len(quotes)
            if bad:
                details["unbacked_quotes"] = bad[:5]
    if want & {"rubric_within_one", "hire_signal_within_one", "hire_signal_exact"} and "gold_scores" in ref:
        s, d = score_card(obj, ref)
        scores |= {k: v for k, v in s.items() if k in want}
        details["scorecard"] = d
    if want & {"gap_fit", "gap_match_score"} and "requirements" in ref:
        score, d = gap_score(obj, ref)
        if value is None:
            d["note"] = "no JSON: scored as no ratings"
        if "gap_match_score" in want:
            scores["gap_match_score"] = float(score)
        band = ref.get("band")
        if "gap_fit" in want and band:
            scores["gap_fit"] = float(band[0] <= score <= band[1])
        details["gap"] = d | {"match_score": score, "band": band}
    return scores, details
