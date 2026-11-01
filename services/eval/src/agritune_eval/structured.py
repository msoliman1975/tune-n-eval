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

Strong Hire formulas (ported from strong-hire, where code, not the model, turns ratings into results).
Scores whose quotes are not the candidate's words are dropped first, as the product does.
- scorecard_usable: the product could compute a hire signal (valid scorecard, at least one kept score).
- hire_signal_kappa: quadratic weighted kappa of the computed signal against the human signal over the
  usable scorecards: agreement on the 5-band scale beyond what guessing gets (1 perfect, 0 chance).
  hire_signal_error: mean distance in bands; hire_signal_bias: signed (+ = more generous than the human).
- hire_signal_within_one / hire_signal_exact: per transcript, an unusable scorecard is a miss. Easy to
  reach by guessing on 5 bands: read them next to the baselines in the report.
- competency_kappa / competency_error: each competency's score (mean of its per-question scores, or the
  overall score when the model gave only that) against the human's mean for it, on the rubric points.
- rubric_within_one: per (question, competency) the human scored, the model's per-question score is
  within one point; a missing per-question score is a miss.

Schema rule: as in the product, an answer that does not match the task's schema is rejected, so it
counts as no answer on every metric (it can still score json_valid 0). The product's retry after a
rejection is not simulated.
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


def dicts(value) -> list[dict]:
    """The objects in a list; nothing for anything else. Unconstrained models write any shape."""
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def strs(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [x for x in value if isinstance(x, str)] if isinstance(value, list) else []


def name(value) -> str | None:
    return value if isinstance(value, str) else None


def rating(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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


def weighted_kappa(pairs: list[tuple[int, int]], k: int) -> float | None:
    """Quadratic weighted kappa of (model, human) labels on an ordered scale of k categories."""
    if len(pairs) < 2:
        return None
    n = len(pairs)
    rows, cols = [0] * k, [0] * k
    for a, b in pairs:
        rows[a] += 1
        cols[b] += 1
    w = lambda i, j: (i - j) ** 2 / (k - 1) ** 2  # noqa: E731
    observed = sum(w(a, b) for a, b in pairs) / n
    expected = sum(w(i, j) * rows[i] * cols[j] for i in range(k) for j in range(k)) / (n * n)
    return 1 - observed / expected if expected else None


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
        quotes = strs(quotes)
        ok = [q for q in quotes if quote_ok(q, sources)]
        checked += len(quotes)
        valid += len(ok)
        return bool(ok)

    per_q, per_v, overall, overall_v = {}, {}, {}, {}
    for qs in dicts(card.get("per_question")):
        ref_id = name(qs.get("question_ref"))
        if ref_id not in asked:
            continue
        for s in dicts(qs.get("scores")):
            key = (ref_id, name(s.get("competency")))
            if key[1] in comps and key not in per_q and rating(s.get("score")) is not None and good(s.get("quotes")):
                per_q[key] = s["score"]
        for s in dicts(qs.get("value_scores")):
            key = (ref_id, name(s.get("value")))
            if key[1] in vals and key not in per_v and rating(s.get("score")) is not None and good(s.get("quotes")):
                per_v[key] = s["score"]
    for s in dicts(card.get("competency_scores")):
        c = name(s.get("competency"))
        if c in comps and c not in overall and rating(s.get("score")) is not None and good(s.get("quotes")):
            overall[c] = s["score"]
    for s in dicts(card.get("value_scores")):
        v = name(s.get("value"))
        if v in vals and v not in overall_v and rating(s.get("score")) is not None and good(s.get("quotes")):
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


def gold_competencies(ref: dict) -> dict[str, float]:
    by = defaultdict(list)
    for g in ref["gold_scores"]:
        by[g["competency"]].append(g["score"])
    return {c: sum(v) / len(v) for c, v in by.items()}


def score_card(card: dict, ref: dict) -> tuple[dict, dict, dict]:
    """(scores, details, kappa pairs) for one scorecard."""
    per_q, per_v, overall, overall_v, checked, valid = kept_scores(card, ref)
    gold = [(g["question"], g["competency"], g["score"]) for g in ref["gold_scores"]]
    hits = [abs(per_q[(q, c)] - s) <= 1 if (q, c) in per_q else False for q, c, s in gold]
    scores = {"rubric_within_one": sum(hits) / len(hits) if hits else 0.0}
    comp_avgs = averages(per_q, overall)
    gold_comp = gold_competencies(ref)
    both = [c for c in gold_comp if c in comp_avgs]
    pairs = {"competency_kappa": [[rubric_point(comp_avgs[c]) - 1, rubric_point(gold_comp[c]) - 1, 4] for c in both]}
    if both:
        scores["competency_error"] = sum(abs(comp_avgs[c] - gold_comp[c]) for c in both) / len(both)
    details = {"kept_scores": len(per_q), "kept_overall": len(overall), "quotes_checked": checked,
               "quotes_valid": valid, "competency_averages": {k: round(v, 2) for k, v in comp_avgs.items()},
               "human_competency_averages": {k: round(v, 2) for k, v in gold_comp.items()}}
    if comp_avgs:
        signal = hire_signal(comp_avgs, ref.get("weights") or {}, averages(per_v, overall_v),
                             ref.get("value_weights") or {}, ref.get("values_share", 0.0))
    else:
        signal = None  # the product refuses to score a transcript with no kept score
    details |= {"hire_signal": signal, "gold_signal": ref["gold_signal"]}
    scores["scorecard_usable"] = float(signal is not None)
    g = SIGNALS.index(ref["gold_signal"])
    if signal is None:
        scores["hire_signal_within_one"] = scores["hire_signal_exact"] = 0.0
    else:
        p = SIGNALS.index(signal)
        scores |= {"hire_signal_within_one": float(abs(p - g) <= 1), "hire_signal_exact": float(p == g),
                   "hire_signal_error": float(abs(p - g)), "hire_signal_bias": float(p - g)}
        pairs["hire_signal_kappa"] = [[p, g, 5]]
    return scores, details, pairs


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

    def checked(value, evidence) -> int:
        r = max(0, min(3, rating(value) or 0))
        found = gap_words(evidence if isinstance(evidence, str) else "")
        ok = bool(found) and len(found & resume) / len(found) >= EVIDENCE_MATCH
        return 1 if r > 1 and not ok else r

    by_id = {}
    for r in dicts(assessment.get("requirements")):
        by_id.setdefault(str(r.get("id", "")).strip().lower(), r)
    reqs, unrated = [], 0
    for req in ref["requirements"]:
        r = by_id.get(req["id"])
        if r is None:
            unrated += 1
        reqs.append((POINTS[checked(r.get("rating"), r.get("evidence")) if r else 0], REQUIREMENT_WEIGHTS[req["kind"]]))
    weights = ref.get("competency_weights") or {}
    comps = {}
    for c in dicts(assessment.get("competencies")):
        comp_name = name(c.get("competency"))
        if comp_name in ref["competencies"] and comp_name not in comps:
            comps[comp_name] = POINTS[max(0, min(3, rating(c.get("rating")) or 0))]

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

SCORER = {"scorecard_usable", "hire_signal_kappa", "hire_signal_error", "hire_signal_bias", "hire_signal_within_one",
          "hire_signal_exact", "competency_kappa", "competency_error", "rubric_within_one"}
STRUCTURED = {"json_valid", "field_match", "label_acc", "one_question", "word_limit", "quotes_verbatim",
              "gap_fit", "gap_match_score", *SCORER}
# Metrics computed over a whole set of items from per-item (model, human, categories) pairs, not averaged.
SET_LEVEL = {"hire_signal_kappa", "competency_kappa"}


def score_structured(task, text: str, reference, want: set[str], tier: str | None = None) -> tuple[dict, dict]:
    """(scores, details) for one answer by a model with this prompt tier. A metric that does not apply to
    an item is left out. Set-level metrics (SET_LEVEL) leave their pairs in details["pairs"]."""
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
    usable = True
    if is_json:
        schema = task.json_schema
        errors = ["not JSON"] if value is None else (schema_errors(value, schema) if schema else [])
        if "json_valid" in want:
            scores["json_valid"] = float(not errors)
        if errors:
            details["schema_errors"] = errors
            usable = False  # the product rejects it: it counts as no answer below, for every model alike
            details["unusable"] = "does not match the schema" if value is not None else "not JSON"
    obj = value if usable and isinstance(value, dict) else {}
    if "field_match" in want and cfg.get("fields") and isinstance(expected, dict):
        if obj:
            scores["field_match"], details["fields"] = field_match(expected, obj, cfg["fields"])
        else:  # no answer object: a failed extraction, not a partial one (empty fields would match empty ones)
            scores["field_match"] = 0.0
    if "label_acc" in want and cfg.get("labels") and isinstance(expected, dict):
        got = {p: first(obj, p) for p in cfg["labels"]}
        scores["label_acc"] = float(all(same_label(got[p], first(expected, p)) for p in cfg["labels"]))
        details["labels"] = got
    spoken = (first(obj, cfg["text_path"]) if cfg.get("text_path") else text) or ""
    if "one_question" in want:  # an answer the product rejects says nothing, so it fails rather than passes
        scores["one_question"] = float(usable and str(spoken).count("?") <= 1)
    limit = ref.get("max_words")
    if isinstance(limit, dict):  # the limit the product gives this model's tier
        limit = limit.get(tier or "default") or limit.get("default")
    if "word_limit" in want and limit:
        n = count_words(str(spoken))
        scores["word_limit"] = float(usable and n <= int(limit))
        details["words"] = n
    if "quotes_verbatim" in want and cfg.get("quotes_path") and ref.get("source_texts"):
        quotes = [q for q in pluck(obj, cfg["quotes_path"]) if isinstance(q, str)]
        if not usable:
            scores["quotes_verbatim"] = 0.0
        elif quotes:
            sources = [quote_words(t) for t in ref["source_texts"]]
            bad = [q for q in quotes if not quote_ok(q, sources)]
            scores["quotes_verbatim"] = 1 - len(bad) / len(quotes)
            if bad:
                details["unbacked_quotes"] = bad[:5]
    if want & SCORER and "gold_scores" in ref:
        s, d, pairs = score_card(obj, ref)
        scores |= {k: v for k, v in s.items() if k in want}
        details["scorecard"] = d
        details["pairs"] = {k: v for k, v in pairs.items() if k in want and v}
    if want & {"gap_fit", "gap_match_score"} and "requirements" in ref:
        band = ref.get("band")
        if not obj:  # rejected or not JSON: no match score (a 0 would "pass" a mismatch band)
            if "gap_fit" in want and band:
                scores["gap_fit"] = 0.0
        else:
            score, d = gap_score(obj, ref)
            if "gap_match_score" in want:
                scores["gap_match_score"] = float(score)
            if "gap_fit" in want and band:
                scores["gap_fit"] = float(band[0] <= score <= band[1])
            details["gap"] = d | {"match_score": score, "band": band}
    return scores, details


# ---------- guessing baselines ----------

def baselines(task, references: list, want: set[str]) -> dict:
    """What always giving the same answer would score on this item set, for the metrics where guessing
    does well: the best constant answer per metric. Shown next to the models in the report."""
    refs = [r if isinstance(r, dict) else parse_json(r) for r in references]
    refs = [r for r in refs if isinstance(r, dict)]
    out = {}
    gold = [SIGNALS.index(r["gold_signal"]) for r in refs if "gold_signal" in r]
    if gold:
        def best(metric, f, higher=True):
            vals = {SIGNALS[c]: sum(f(c, g) for g in gold) / len(gold) for c in range(5)}
            answer = (max if higher else min)(vals, key=vals.get)
            out[metric] = {"answer": f"always {answer}", "value": round(vals[answer], 4)}

        best("hire_signal_within_one", lambda c, g: abs(c - g) <= 1)
        best("hire_signal_exact", lambda c, g: c == g)
        best("hire_signal_error", lambda c, g: abs(c - g), higher=False)
        out["hire_signal_kappa"] = {"answer": "any constant answer", "value": 0.0}
        comp = [v for r in refs if "gold_scores" in r for v in gold_competencies(r).values()]
        pts = [g["score"] for r in refs if "gold_scores" in r for g in r["gold_scores"]]
        if comp:
            errs = {p: sum(abs(p - v) for v in comp) / len(comp) for p in (1, 2, 3, 4)}
            p = min(errs, key=errs.get)
            out["competency_error"] = {"answer": f"always {p}", "value": round(errs[p], 4)}
            hits = {p: sum(abs(p - v) <= 1 for v in pts) / len(pts) for p in (1, 2, 3, 4)}
            p = max(hits, key=hits.get)
            out["rubric_within_one"] = {"answer": f"always {p}", "value": round(hits[p], 4)}
            out["competency_kappa"] = {"answer": "any constant answer", "value": 0.0}
    labels = task.scoring.get("labels") if task.scoring else None
    if labels:
        seen = [tuple(str(first(r.get("expected") or {}, p)) for p in labels) for r in refs]
        if seen:
            top = max(set(seen), key=seen.count)
            out["label_acc"] = {"answer": f"always {', '.join(top)}", "value": round(seen.count(top) / len(seen), 4)}
    return {k: v for k, v in out.items() if k in want}
