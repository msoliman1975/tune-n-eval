"""The Strong Hire tasks: structured metrics, the product formulas ported from Strong Hire, the model
request bodies (JSON schema, streaming, thinking options) and an evaluation run with per-tier prompts."""

import json
import shutil
import sys
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agritune_core.tasks import load_task
from agritune_eval import REPO, run
from agritune_eval.api import EVALS
from agritune_eval.structured import gap_score, parse_answer, pluck, quote_ok, quote_words, score_structured
from agritune_eval.targets import AnthropicTarget, Answer, Gen, GeminiTarget, OllamaTarget, api_schema

DATA = REPO / "datasets" / "strong-hire"


def rows(ds: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / f"{ds}.jsonl").read_text(encoding="utf-8").splitlines()]


def score(task: str, answer, reference, tier=None) -> tuple[dict, dict]:
    t = load_task(task)
    text = answer if isinstance(answer, str) else json.dumps(answer)
    return score_structured(t, text, reference, set(t.metrics["available"]), tier)


def test_parsing_and_paths():
    assert parse_answer('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_answer('Sure! {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    assert parse_answer("no json here") is None
    card = {"per_question": [{"scores": [{"quotes": ["a", "b"]}, {"quotes": ["c"]}]}, {"scores": []}]}
    assert pluck(card, "per_question[].scores[].quotes[]") == ["a", "b", "c"]
    assert pluck({"x": None}, "x.y") == [] and pluck({"roles": [{"company": "A"}]}, "roles[].company") == ["A"]


def test_quotes_follow_the_product_rule():
    said = [quote_words("I wrote the retry logic and ran the cut-over on a Saturday.")]
    assert quote_ok("I wrote the retry logic", said)
    assert quote_ok("i WROTE ... the cut-over", said)  # an ellipsis may skip words, in order
    assert not quote_ok("the cut-over ... I wrote", said)
    assert not quote_ok("retry logic", said)  # fewer than 3 words cannot back a score
    assert not quote_ok("I designed the retry logic", said)


def test_extraction_scores_its_own_reference_fully():
    for ds in ("sh-extract-posting", "sh-extract-resume"):
        for r in rows(ds):
            expected = json.loads(r["reference"])["expected"]
            s, d = score(f"{ds}@v1", expected, r["reference"])
            assert s == {"json_valid": 1.0, "field_match": 1.0}, (r["id"], d)
    r = rows("sh-extract-posting")[0]
    wrong = json.loads(r["reference"])["expected"] | {"title": "Chef", "must_have_skills": []}
    s, d = score("sh-extract-posting@v1", wrong, r["reference"])
    assert s["field_match"] == pytest.approx(0.8) and d["fields"]["title"] == 0.0
    s, d = score("sh-extract-posting@v1", "I cannot help with that.", r["reference"])
    assert s == {"json_valid": 0.0, "field_match": 0.0} and d["schema_errors"] == ["not JSON"]


def test_labels_and_one_question():
    r = next(x for x in rows("sh-input-check") if x["expected"] == "refuse")
    exp = json.loads(r["reference"])["expected"]
    s, _ = score("sh-input-check@v1", exp | {"reason": "A cover letter."}, r["reference"])
    assert s == {"json_valid": 1.0, "label_acc": 1.0, "field_match": 1.0}
    s, _ = score("sh-input-check@v1", {"matches": True, "looks_like": "resume", "reason": ""}, r["reference"])
    assert s["label_acc"] == 0.0
    p = next(x for x in rows("sh-probe") if x["expected"] == "probe")
    s, _ = score("sh-probe@v1", {"action": "probe", "missing": []}, p["reference"])
    assert s == {"json_valid": 1.0, "label_acc": 1.0}
    s, d = score("sh-probe@v1", {"action": "PROBE", "missing": []}, p["reference"])  # the enum is lower case
    assert s == {"json_valid": 0.0, "label_acc": 0.0} and d["unusable"] == "does not match the schema"
    c = rows("sh-coach-turn")[1]
    said = json.loads(c["reference"])["source_texts"][-1]
    reply = {"message": "Good. What did you do yourself? And what changed?", "missing": [], "skills": [],
             "parts": [{"part": "context", "text": "x", "quotes": [" ".join(said.split()[:5]), "words nobody said here"]}]}
    s, d = score("sh-coach-turn@v1", reply, c["reference"])
    assert s["one_question"] == 0.0 and s["quotes_verbatim"] == 0.5 and d["unbacked_quotes"] == ["words nobody said here"]


def test_word_limit_follows_the_tier():
    r = rows("sh-interviewer-turn")[0]
    line = " ".join(["word"] * 33) + "?"
    assert score("sh-interviewer-turn@v1", line, r["reference"])[0]["word_limit"] == 1.0  # 35 words allowed
    assert score("sh-interviewer-turn@v1", line, r["reference"], "small")[0]["word_limit"] == 0.0  # 30 for small


def gold_card(ref: dict, quote: str) -> dict:
    per, vals = defaultdict(list), defaultdict(list)
    for g in ref["gold_scores"]:
        per[g["question"]].append({"competency": g["competency"], "score": g["score"], "justification": "j", "quotes": [quote]})
    for g in ref["gold_value_scores"]:
        vals[g["question"]].append({"value": g["value"], "score": g["score"], "justification": "j", "quotes": [quote]})
    g0 = ref["gold_scores"][0]
    return {"hire_signal": "Hire", "rationale": "r", "scorer_model": "m", "rubric_version": "v",
            "competency_scores": [{"competency": g0["competency"], "score": g0["score"], "justification": "j", "quotes": [quote]}],
            "per_question": [{"question_ref": q, "question_text": "t", "scores": s, "value_scores": vals.get(q, [])}
                             for q, s in per.items()]}


def test_hire_signal_matches_strong_hire_on_the_gold_set():
    """Strong Hire's own test: the signal computed from the human rubric scores matches the human signal
    exactly on 26 of 30 transcripts and within one band on all 30 (test_signal_matches_goldset)."""
    exact = within = 0
    for r in rows("sh-score"):
        ref = json.loads(r["reference"])
        quote = " ".join(max(ref["source_texts"], key=len).split()[:6])
        s, d = score("sh-score@v1", gold_card(ref, quote), r["reference"])
        assert s["json_valid"] == 1.0 and s["rubric_within_one"] == 1.0 and s["quotes_verbatim"] == 1.0, d
        exact += s["hire_signal_exact"]
        within += s["hire_signal_within_one"]
    assert (exact, within) == (26, 30)


def test_scores_with_invented_quotes_are_dropped():
    r = rows("sh-score")[0]
    ref = json.loads(r["reference"])
    s, d = score("sh-score@v1", gold_card(ref, "a sentence the candidate never said"), r["reference"])
    assert s["quotes_verbatim"] == 0.0 and s["rubric_within_one"] == 0.0 and s["hire_signal_within_one"] == 0.0
    assert d["scorecard"]["hire_signal"] is None  # the product refuses to score with no kept score


def test_gap_score_follows_the_evidence_rule():
    r = next(x for x in rows("sh-gap") if x["fit"] == "match")
    ref = json.loads(r["reference"])
    evidence = " ".join(ref["resume_text"].split()[:8])
    full = {"requirements": [{"id": q["id"], "rating": 3, "evidence": evidence} for q in ref["requirements"]],
            "competencies": [{"competency": c, "rating": 3} for c in ref["competencies"]]}
    assert gap_score(full, ref)[0] == 100
    unbacked = {**full, "requirements": [{"id": q["id"], "rating": 3, "evidence": "launched rockets to mars"}
                                         for q in ref["requirements"]]}
    assert gap_score(unbacked, ref)[0] == round(0.7 * 35 + 0.3 * 100)  # ratings capped at 1 without evidence
    s, d = score("sh-gap@v1", full | {"strengths": [], "gaps": [], "probe_areas": [],
                                      "session_plan": [{"interview_type": "behavioral", "reason": "r", "priority": 1,
                                                        "focus_topics": ["ownership"]}]}, r["reference"])
    assert s["gap_match_score"] == 100.0 and s["gap_fit"] == 1.0
    assert score("sh-gap@v1", {"requirements": []}, r["reference"])[0]["gap_fit"] == 0.0  # no ratings: score 0
    adjacent = next(x for x in rows("sh-gap") if x["fit"] == "adjacent")
    assert "gap_fit" not in score("sh-gap@v1", full, adjacent["reference"])[0]  # no band, read the value only


def test_request_bodies():
    schema = load_task("sh-probe@v1").json_schema
    gen = Gen(json_schema=schema, stream=True)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    body = OllamaTarget.request("qwen2.5:3b-instruct", msgs, 256, gen=gen)
    assert body["format"] == schema and body["stream"] is True
    assert "format" not in OllamaTarget.request("x", msgs, 256) and OllamaTarget.request("x", msgs, 256)["stream"] is False
    req = AnthropicTarget.request("claude-haiku-5-5", msgs, 256, {"effort": "low", "thinking": {"type": "between_tools"}}, gen)
    fmt = req["output_config"]["format"]
    assert fmt["type"] == "json_schema" and req["output_config"]["effort"] == "low"
    assert req["thinking"] == {"type": "between_tools"} and req["system"] == "s"
    assert "maxItems" not in json.dumps(fmt["schema"]) and "maxItems" in json.dumps(schema)  # stripped for the API only
    assert "thinking" not in AnthropicTarget.request("claude-sonnet-5-5", msgs, 256)  # model defaults, as Strong Hire sends
    assert api_schema({"a": {"minLength": 1, "type": "string"}}) == {"a": {"type": "string"}}
    assert GeminiTarget.request(msgs, 256, None, gen)["generationConfig"]["responseMimeType"] == "application/json"


def test_cache_keys_unchanged_for_plain_tasks():
    """Answers cached before generation options existed stay valid for tasks that use none."""
    import hashlib

    cache = run.GenCache({"id": "_test-cache", "kind": "ollama", "ollama_tag": "t"})
    msgs = [{"role": "user", "content": "q"}]
    old = hashlib.sha256(json.dumps([cache.fp, msgs, 400], ensure_ascii=False).encode()).hexdigest()
    assert cache.key(msgs, 400) == old and cache.key(msgs, 400, Gen(stream=True)) != old


SCRATCH = REPO / "artifacts" / "_test-sh"
EVAL_ID = "eval-00000000-sh-test"


def test_run_with_tiers_schema_and_ttft(monkeypatch):
    src = [r for r in rows("sh-probe") if "small" in json.loads(r["messages"])][:6]
    d = SCRATCH / "sh-probe" / "v1.0"
    d.mkdir(parents=True, exist_ok=True)
    table = [{"id": r["id"], "split": "test", "title": r["title"], "messages": r["messages"], "reference": r["reference"],
              "meta": json.dumps({"expected": r["expected"]}), "edited": False} for r in src]
    pq.write_table(pa.Table.from_pylist(table), d / "test.parquet")
    (d / "dataset_ref.json").write_text(json.dumps({"id": "sh-probe", "version": "v1.0", "task": "sh-probe@v1",
                                                    "splits": {"test": str((d / "test.parquet").relative_to(REPO))}}))
    models = [{"id": "small-m", "kind": "ollama", "ollama_tag": "s", "name": "small", "paid": False, "prompt_tier": "small"},
              {"id": "big-m", "kind": "api", "provider": "anthropic", "api_model": "claude-haiku-5-5", "name": "big",
               "paid": True}]
    seen = {}

    def fake_generate(models_, convs, max_new, batch, ev, gen=None, tier_convs=None):
        seen.update(gen=gen, tiers=tier_convs, convs=convs)
        right = [json.dumps({"action": json.loads(r["reference"])["expected"]["action"], "missing": []}) for r in src]
        return {"small-m": [Answer("not json", 2.0, 100, 5, None, 0.4) for _ in src],
                "big-m": [Answer(a, 1.0, 100, 5, None, 0.2) for a in right]}

    monkeypatch.setattr(run, "generate_all", fake_generate)
    monkeypatch.setattr(run, "post_registry", lambda report: None)
    out = EVALS / EVAL_ID
    out.mkdir(parents=True, exist_ok=True)
    spec = {"id": EVAL_ID, "dataset": str(d.relative_to(REPO)), "splits": ["test"], "models": models,
            "metrics": ["json_valid", "label_acc", "latency", "ttft"]}
    (out / "spec.json").write_text(json.dumps(spec))
    monkeypatch.setattr(sys, "argv", ["run", "--spec", str(out / "spec.json")])
    try:
        run.main()
        assert seen["gen"].json_schema["title"] == "ProbeDecision" and not seen["gen"].stream
        assert set(seen["tiers"]) == {"small"} and seen["tiers"]["small"][0] != seen["convs"][0]
        rep = json.loads((out / "report.json").read_text())
        head = {(s["model"], s["metric"]): s["value"] for s in rep["scores"] if s["slice"]["key"] == "all"}
        assert head[("big-m", "label_acc")] == 1.0 and head[("small-m", "json_valid")] == 0.0
        assert head[("big-m", "ttft")] == 0.2 and rep["decoding"]["prompt_tiers"] == ["small"]
        assert {s["slice"]["key"] for s in rep["scores"]} >= {"expected"}
        item = json.loads((out / "items.jsonl").read_text().splitlines()[0])
        assert item["tier_messages"]["small"][0]["content"] != item["messages"][0]["content"]
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


@pytest.mark.parametrize("ds", sorted(p.stem for p in DATA.glob("sh-*.jsonl")))
def test_malformed_answers_score_without_errors(ds):
    """Unconstrained models write any shape (lists of strings, numbers for objects, nulls): it scores, never raises."""
    ref = rows(ds)[0]["reference"]
    odd = ["", "null", "[]", '"text"', "42", "{}", '{"a": [1, "x", null]}',
           json.dumps({k: v for k in ("per_question", "competency_scores", "value_scores", "requirements", "competencies",
                                      "parts", "roles", "skills", "message", "action", "conduct", "matches", "looks_like")
                       for v in [["ownership", 3, None, {"competency": ["x"], "score": "4", "quotes": "a b c d"}]]}),
           json.dumps({"per_question": [{"question_ref": ["q1"], "scores": {"competency": "impact"}}],
                       "requirements": [{"id": None, "rating": True, "evidence": 7}], "parts": [{"quotes": [1, None]}]})]
    for text in odd:
        scores, _ = score(f"{ds}@v1", text, ref)
        assert all(isinstance(v, float) for v in scores.values()), (text, scores)



def test_schema_breaking_answers_count_as_no_answer_for_every_model():
    """Strong Hire rejects an answer that breaks its contract. The Claude request cannot carry every schema
    limit (e.g. non-empty strings), so a Claude answer can break it where an Ollama answer cannot; both are
    scored by the full schema, so neither gains from it."""
    r = rows("sh-score")[0]
    ref = json.loads(r["reference"])
    good = gold_card(ref, " ".join(max(ref["source_texts"], key=len).split()[:6]))
    s, _ = score("sh-score@v1", good, r["reference"])
    assert s["scorecard_usable"] == 1.0 and s["json_valid"] == 1.0
    blank_value = good | {"value_scores": [{"value": "", "score": 3, "justification": "", "quotes": []}]}
    s, d = score("sh-score@v1", blank_value, r["reference"])
    assert s["json_valid"] == 0.0 and s["scorecard_usable"] == 0.0 and s["hire_signal_within_one"] == 0.0
    assert "hire_signal_error" not in s and d["unusable"] == "does not match the schema"
    g = next(x for x in rows("sh-gap") if x["fit"] == "mismatch")
    assert score("sh-gap@v1", "not json", g["reference"])[0] == {"json_valid": 0.0, "gap_fit": 0.0}  # not a pass
    c = rows("sh-coach-turn")[0]
    assert score("sh-coach-turn@v1", {"message": "Hi"}, c["reference"])[0]["one_question"] == 1.0  # valid, no question
    assert score("sh-coach-turn@v1", {"message": ""}, c["reference"])[0]["one_question"] == 0.0  # rejected: says nothing


def test_overall_only_scorecards_break_the_scorer_output_schema():
    """Strong Hire's ScorerOutput requires per-question scores (FB-1, FB-2): a card with only overall
    competency scores is rejected, as in the product. A card covering only some questions is valid, and
    the competency metrics still compare the overall scores of the questions it skipped."""
    r = rows("sh-score")[0]
    ref = json.loads(r["reference"])
    quote = " ".join(max(ref["source_texts"], key=len).split()[:6])
    human = defaultdict(list)
    for g in ref["gold_scores"]:
        human[g["competency"]].append(g["score"])
    overall = [{"competency": c, "score": round(sum(v) / len(v)), "justification": "j", "quotes": [quote]}
               for c, v in human.items()]
    card = {"hire_signal": "Hire", "rationale": "r", "scorer_model": "m", "rubric_version": "v",
            "competency_scores": overall, "per_question": []}
    s, d = score("sh-score@v1", card, r["reference"])
    assert s["json_valid"] == 0.0 and s["scorecard_usable"] == 0.0 and d["unusable"] == "does not match the schema"
    without = {k: v for k, v in card.items() if k != "per_question"}
    assert score("sh-score@v1", without, r["reference"])[0]["json_valid"] == 0.0
    one = gold_card(ref, quote)["per_question"][:1]
    s, d = score("sh-score@v1", card | {"per_question": one}, r["reference"])
    assert s["json_valid"] == 1.0 and s["scorecard_usable"] == 1.0 and s["competency_error"] < 0.5
    assert len(d["pairs"]["competency_kappa"]) == len(human)


def test_baselines_show_what_guessing_scores():
    from agritune_eval.structured import baselines

    b = baselines(load_task("sh-score@v1"), [r["reference"] for r in rows("sh-score")],
                  {"hire_signal_within_one", "hire_signal_kappa", "rubric_within_one", "competency_error"})
    assert b["hire_signal_within_one"] == {"answer": "always Lean No Hire", "value": pytest.approx(20 / 30, abs=1e-4)}
    assert b["rubric_within_one"]["answer"] == "always 3" and b["rubric_within_one"]["value"] > 0.9
    assert b["hire_signal_kappa"]["value"] == 0.0
    probe = baselines(load_task("sh-probe@v1"), [r["reference"] for r in rows("sh-probe")], {"label_acc"})
    assert probe["label_acc"]["answer"] == "always move_on" and probe["label_acc"]["value"] == pytest.approx(110 / 182, abs=1e-4)


def test_kappa_separates_a_judge_from_a_guesser(monkeypatch):
    """Over the full scorer set: a scorer that writes the human scores against one that scores everything 3.
    Within one point barely separates them; kappa does, and the report carries the guessing baselines."""
    src = rows("sh-score")
    d = SCRATCH / "sh-score" / "v1.0"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([{"id": r["id"], "split": "test", "title": r["title"], "messages": r["messages"],
                                          "reference": r["reference"], "meta": "{}", "edited": False} for r in src]),
                   d / "test.parquet")
    (d / "dataset_ref.json").write_text(json.dumps({"id": "sh-score", "version": "v1.0", "task": "sh-score@v1",
                                                    "splits": {"test": str((d / "test.parquet").relative_to(REPO))}}))

    def card(r, lazy):
        ref = json.loads(r["reference"])
        c = gold_card(ref, " ".join(max(ref["source_texts"], key=len).split()[:6]))
        for q in c["per_question"] if lazy else []:
            for x in q["scores"]:
                x["score"] = 3
        return json.dumps(c)

    monkeypatch.setattr(run, "generate_all", lambda models, convs, *a, **k: {
        "judge": [Answer(card(r, False), 1.0, 10, 10) for r in src], "lazy": [Answer(card(r, True), 1.0, 10, 10) for r in src]})
    monkeypatch.setattr(run, "post_registry", lambda report: None)
    out = EVALS / EVAL_ID
    out.mkdir(parents=True, exist_ok=True)
    metrics = load_task("sh-score@v1").metrics["available"]
    models = [{"id": m, "kind": "ollama", "ollama_tag": m, "name": m, "paid": False} for m in ("judge", "lazy")]
    (out / "spec.json").write_text(json.dumps({"id": EVAL_ID, "dataset": str(d.relative_to(REPO)), "splits": ["test"],
                                               "models": models, "metrics": metrics}))
    monkeypatch.setattr(sys, "argv", ["run", "--spec", str(out / "spec.json")])
    try:
        run.main()
        rep = json.loads((out / "report.json").read_text())
        head = {(s["model"], s["metric"]): s for s in rep["scores"] if s["slice"]["key"] == "all"}
        assert head[("lazy", "rubric_within_one")]["value"] > 0.9  # guessing looks fine here...
        assert head[("judge", "hire_signal_kappa")]["value"] > 0.9  # ...but not on agreement beyond chance
        assert head[("lazy", "hire_signal_kappa")]["value"] < 0.5
        assert head[("judge", "hire_signal_kappa")]["ci95"][0] is not None and head[("judge", "hire_signal_kappa")]["n"] == 30
        assert head[("lazy", "hire_signal_bias")]["value"] > 0.5  # "3" everywhere is too generous on this set
        assert not any(c["metric"] in ("hire_signal_kappa", "hire_signal_bias") for c in rep["comparisons"])
        err = next(c for c in rep["comparisons"] if c["metric"] == "hire_signal_error" and c["slice"]["key"] == "all")
        assert err["verdict"] == "a_better"  # the judge, lower error
        assert rep["baselines"]["hire_signal_within_one"]["answer"] == "always Lean No Hire"
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
