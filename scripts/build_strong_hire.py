"""Builds the built-in Strong Hire datasets (datasets/strong-hire/*.jsonl) and copies their output
schemas (tasks/schemas/strong-hire/). The output is committed, so evaluating needs no Strong Hire
checkout; run this only to rebuild after Strong Hire changes its prompts, contracts or fixtures.

    uv run --project ../strong-hire python scripts/build_strong_hire.py --strong-hire ../strong-hire

It runs inside Strong Hire's environment and calls Strong Hire's own code to build each prompt, so a
row holds exactly the messages the product sends. No model is called: a recording stand-in for the
gateway answers the few functions that call one. Each row carries up to three prompt variants:

- default: what Strong Hire sends a large-tier model (the claude profile, 200K context).
- medium: a medium-tier model (local-mid, Qwen2.5 7B; 8192 context), kept only where it differs.
- small: a small-tier model (local-small, Qwen2.5 3B; 8192 context; the *.small prompts), kept only
  where it differs.

Labels come from Strong Hire's fixtures (hand-checked extraction JSON, the human gold set, the
scripted transcripts' vague-answer marks) and from the hand-written sources in
datasets/strong-hire/sources/. Every person and company in them is made up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

TNE = Path(__file__).resolve().parents[1]
OUT = TNE / "datasets" / "strong-hire"
SCHEMAS = TNE / "tasks" / "schemas" / "strong-hire"
SOURCES = OUT / "sources"

# tier name -> (Strong Hire capability tier, context window); None is the default prompt
TIERS = {"default": (None, 200_000), "medium": ("medium", 8192), "small": ("small", 8192)}

# Families and seniority of the resume fixtures, read from each CV (for the gap analysis pairs).
RESUMES = {
    "backend-senior": ("swe", "senior"), "bold-subsections": ("swe", "staff_principal"),
    "date-column": ("swe", "staff_principal"), "docx-bold-subsections": ("pm", "senior"),
    "docx-table": ("design", "senior"), "docx-textbox-sidebar": ("swe", "senior"),
    "ml-engineer-staff": ("data_ml", "staff_principal"), "multipage-split-role": ("swe", "senior"),
    "new-grad-swe": ("swe", "new_grad"), "product-designer-senior": ("design", "senior"),
    "product-manager-mid": ("pm", "mid"), "same-company-two-roles": ("data_ml", "staff_principal"),
    "sidebar-two-column": ("data_ml", "senior"), "table-layout": ("tpm", "mid"),
    "tpm-career-changer": ("tpm", "mid"),
}
ADJACENT = {frozenset(p) for p in (("swe", "data_ml"), ("pm", "tpm"), ("pm", "design"), ("swe", "tpm"))}
LEVELS = ["new_grad", "mid", "senior", "staff_principal"]
# Expected match score (0-100) by fit. A matching CV rated mostly 2-3 lands well above 60; a CV for
# another job family rated mostly 0-1 stays under 40. Adjacent and partial pairs get no band.
BANDS = {"match": [60, 100], "mismatch": [0, 40]}
PLANTED = {"swe-northwind-newgrad", "product-manager-mid"}  # fixtures with an instruction aimed at the model


class Stop(Exception):
    pass


class Recorder:
    """Stands in for strong_core.gateway: reports a tier and context window, records every call."""

    def __init__(self, tier, window: int):
        from strong_core.gateway import CapabilityTier

        self.tier = CapabilityTier(tier) if tier else CapabilityTier.LARGE
        self.window = window
        self.calls: list[list] = []
        self.profile = "recorder"
        self.config = SimpleNamespace(alias_for=lambda role: "recorder")

    def capabilities(self, role):
        return SimpleNamespace(tier=self.tier, context_window=self.window, supports_tools=False, json_mode=True,
                               options={}, sample_rate=24000, voice_gender=None, streaming=True)

    async def complete(self, role, messages, **_):
        self.calls.append(list(messages))
        raise Stop()


@contextmanager
def tier_prompts(tier):
    """load_prompt picks the variant for this tier (strong_core.prompts reads the active tier per role)."""
    import strong_core.prompts as prompts
    from strong_core.gateway import CapabilityTier

    saved = prompts.active_tier
    prompts.active_tier = lambda role: CapabilityTier(tier) if tier else None
    try:
        yield
    finally:
        prompts.active_tier = saved


def plain(messages) -> list[dict]:
    return [{"role": str(m.role), "content": m.content} for m in messages]


def refs(messages) -> list[str]:
    return sorted({m.prompt_ref for m in messages if getattr(m, "prompt_ref", None)})


def variants(build) -> tuple[dict, list[str]]:
    """build(recorder) -> Strong Hire messages, for each tier; variants equal to the default are dropped."""
    out, prompt_refs = {}, set()
    for name, (tier, window) in TIERS.items():
        with tier_prompts(tier):
            msgs = build(Recorder(tier, window))
        prompt_refs |= set(refs(msgs))
        out[name] = plain(msgs)
    keep = {"default": out["default"]} | {k: v for k, v in out.items() if k != "default" and v != out["default"]}
    return keep, sorted(prompt_refs)


def recorded(coro_fn):
    """Messages of the first model call an async Strong Hire function makes."""

    def build(rec):
        try:
            asyncio.run(coro_fn(rec))
        except Stop:
            pass
        if not rec.calls:
            raise RuntimeError("the function made no model call")
        return rec.calls[0]

    return build


def row(item_id: str, title: str, msgs: dict, prompt_refs: list[str], reference: dict, **meta) -> dict:
    return {"id": item_id, "title": title, "messages": json.dumps(msgs, ensure_ascii=False),
            "reference": json.dumps(reference, ensure_ascii=False), "prompt_refs": ",".join(prompt_refs), **meta}


# ---------- datasets ----------

def extract_postings() -> list[dict]:
    from strong_worker.inputs.accuracy import fixtures_dir, load_files
    from strong_worker.inputs.extract import build_job_posting_messages
    from strong_worker.inputs.sanitize import sanitize_untrusted

    out = []
    for case in load_files(fixtures_dir() / "postings"):
        expected = json.loads(case.expected)
        msgs, pr = variants(lambda rec, t=case.text: build_job_posting_messages(sanitize_untrusted(t).text))
        out.append(row(f"posting-{case.name}", f"Job posting: {expected.get('title')} ({case.name})", msgs, pr,
                       {"expected": expected}, role_family=expected.get("role_family"),
                       planted_instruction=str(case.name in PLANTED).lower()))
    return out


def extract_resumes() -> list[dict]:
    from strong_worker.inputs.accuracy import fixtures_dir, load_files
    from strong_worker.inputs.extract import build_resume_messages
    from strong_worker.inputs.sanitize import sanitize_untrusted

    out = []
    for case in load_files(fixtures_dir() / "resumes"):
        msgs, pr = variants(lambda rec, t=case.text: build_resume_messages(sanitize_untrusted(t).text))
        out.append(row(f"resume-{case.name}", f"CV: {case.name} ({case.kind})", msgs, pr,
                       {"expected": json.loads(case.expected)}, file_kind=case.kind,
                       planted_instruction=str(case.name in PLANTED).lower()))
    return out


def input_check() -> list[dict]:
    from strong_worker.inputs.accuracy import fixtures_dir, load_files
    from strong_worker.inputs.check import CHECK_CHARS, build_check_messages
    from strong_worker.inputs.sanitize import sanitize_untrusted

    docs = [(f"posting-{c.name}", "job_posting", c.text) for c in load_files(fixtures_dir() / "postings")]
    docs += [(f"resume-{c.name}", "resume", c.text) for c in load_files(fixtures_dir() / "resumes")]
    hand = json.loads((SOURCES / "input-check-docs.json").read_text(encoding="utf-8"))["docs"]
    docs += [(d["id"], d["kind"], d["text"]) for d in hand]
    out = []
    for doc_id, kind, text in docs:
        for expected_input in ("job_posting", "resume"):
            clean = sanitize_untrusted(text).text[:CHECK_CHARS]
            msgs, pr = variants(lambda rec, e=expected_input, t=clean: build_check_messages(e, t))
            accept = kind == expected_input
            gave = "job description" if expected_input == "job_posting" else "CV"
            out.append(row(f"{doc_id}-as-{expected_input}", f"{doc_id} given as a {gave}", msgs, pr,
                           {"expected": {"matches": accept, "looks_like": kind}},
                           expected_input=expected_input, doc_kind=kind,
                           expected=("accept" if accept else "refuse")))
    return out


def gap_pairs() -> list[dict]:
    from strong_evals.gap import INPUTS_DIR, load_posting, load_profile, load_resume
    from strong_worker.gap.analysis import build_messages, rated_competencies, requirements_of
    from strong_worker.gap.text import resume_text

    postings = sorted(p.stem for p in (INPUTS_DIR / "postings").glob("*.json"))
    pairs = [(p, r, None) for p in postings for r in sorted(RESUMES)]
    pairs.append(("swe-stripe-backend", "backend-senior", "examples/example-corp.json"))  # company mode
    out = []
    for p_name, r_name, profile_path in pairs:
        posting, resume, profile = load_posting(p_name), load_resume(r_name), load_profile(profile_path)
        fam, level = RESUMES[r_name]
        p_fam, p_level = posting.role_family.value, posting.level.value if posting.level else None
        if fam == p_fam:
            gap = abs(LEVELS.index(level) - LEVELS.index(p_level)) if p_level else 0
            fit = "match" if gap <= 1 else "partial"
        else:
            fit = "adjacent" if frozenset((fam, p_fam)) in ADJACENT else "mismatch"
        msgs, pr = variants(lambda rec, a=posting, b=resume, c=profile: build_messages(a, b, c))
        comps = rated_competencies(posting)
        ref = {"requirements": [{"id": q.id, "kind": q.kind.value} for q in requirements_of(posting)],
               "competencies": [c.value for c in comps],
               "competency_weights": {c.value: profile.weight(c) for c in comps},
               "resume_text": resume_text(resume), "fit": fit, "band": BANDS.get(fit)}
        suffix = "-company" if profile_path else ""
        out.append(row(f"gap-{p_name}-x-{r_name}{suffix}", f"{posting.title} x {r_name}" + (" (company mode)" if suffix else ""),
                       msgs, pr, ref, fit=fit, posting_family=p_fam, resume_family=fam,
                       mode="company" if profile_path else "generic"))
    return out


def scorer_items() -> list[dict]:
    from strong_core.prompts import load_prompt
    from strong_core.schemas import Competency, Speaker, session_competencies
    from strong_evals.goldset import load_goldset
    from strong_evals.transcripts import load_transcripts
    from strong_worker.scoring.harness import find_profile
    from strong_worker.scoring.scorer import ROLE, SessionScorer, _allowed_values, _asked_questions, rubric_prompt

    gold = load_goldset()
    out = []
    for t in load_transcripts():
        if t.id not in gold:
            continue
        g, profile = gold[t.id], find_profile(t.brief)
        asked = _asked_questions(t.brief, t.turns)
        allowed = _allowed_values(t.brief, profile)

        def build(rec, t=t, asked=asked, allowed=allowed, profile=profile):
            system = load_prompt(ROLE, rubric_prompt(t.brief)).message("system")
            return [system, SessionScorer(rec)._input(t.brief, t.turns, profile, asked, allowed)]

        msgs, pr = variants(build)
        ref = {"asked": [q.id for q in asked],
               "competencies": [c.value for c in session_competencies(t.brief.session)],
               "values": sorted(allowed),
               "weights": {c.value: profile.weight(c) for c in Competency},
               "value_weights": dict(profile.value_weights), "values_share": profile.values_share,
               "source_texts": [x.text for x in t.turns if x.speaker == Speaker.CANDIDATE],
               "gold_scores": [{"question": q, "competency": c.value, "score": s} for (q, c), s in sorted(g.scores.items())],
               "gold_value_scores": [{"question": q, "value": v, "score": s} for (q, v), s in sorted(g.value_scores.items())],
               "gold_signal": g.hire_signal.value}
        out.append(row(f"score-{t.id}", t.title, msgs, pr, ref, interview_type=t.brief.session.interview_type.value,
                       quality=t.quality, gold_signal=g.hire_signal.value,
                       mode="generic" if t.brief.generic_mode else "company"))
    return out


def core_answers(t):
    """(index, turn, question) for every candidate answer in CORE that has a question."""
    from strong_core.schemas import Phase, Speaker

    for i, turn in enumerate(t.turns):
        if turn.speaker == Speaker.CANDIDATE and turn.phase == Phase.CORE and turn.question_ref:
            yield i, turn, t.question(turn.question_ref)


def probe_decisions() -> list[dict]:
    from strong_evals.transcripts import load_transcripts
    from strong_interview.interviewer import Interviewer

    out = []
    for t in load_transcripts():
        for i, turn, q in core_answers(t):
            expected = "probe" if i in t.vague_answers else "move_on"
            build = recorded(lambda rec, t=t, q=q, i=i, turn=turn: Interviewer(rec, t.brief).decide(q, turn.text, t.turns[: i + 1]))
            msgs, pr = variants(build)
            out.append(row(f"probe-{t.id}-t{i}", f"{t.id} turn {i}: {turn.text[:70]}", msgs, pr,
                           {"expected": {"action": expected}}, expected=expected,
                           interview_type=t.brief.session.interview_type.value, quality=t.quality))
    return out


def answer_checks() -> list[dict]:
    from strong_core.schemas import Phase
    from strong_evals.transcripts import load_transcripts
    from strong_interview.interviewer import Interviewer

    turns = json.loads((SOURCES / "answer-check-turns.json").read_text(encoding="utf-8"))["turns"]
    spots = [(t, i, q) for t in load_transcripts() for i, _, q in list(core_answers(t))[:1]]
    out = []
    for n, case in enumerate(turns):
        t, i, q = spots[n % len(spots)]
        context = [*t.turns[:i], t.turns[i].model_copy(update={"text": case["text"]})]
        build = recorded(lambda rec, t=t, q=q, ctx=context, text=case["text"]:
                         Interviewer(rec, t.brief).check_answer(text, ctx, Phase.CORE, q))
        msgs, pr = variants(build)
        out.append(row(f"check-{case['id']}", case["text"][:80], msgs, pr, {"expected": {"conduct": case["conduct"]}},
                       expected=case["conduct"], transcript=t.id))
    return out


def interviewer_turns() -> list[dict]:
    from strong_core.schemas import Phase, Speaker
    from strong_evals.transcripts import load_transcripts
    from strong_interview.controller import Move, MoveKind
    from strong_interview.interviewer import MAX_WORDS, SMALL_MAX_WORDS, Interviewer

    out = []
    for t in load_transcripts():
        for i, turn, q in core_answers(t):
            nxt = t.turns[i + 1] if i + 1 < len(t.turns) else None
            if nxt is None or nxt.speaker != Speaker.INTERVIEWER or nxt.phase != Phase.CORE or not nxt.question_ref:
                continue
            if nxt.question_ref == turn.question_ref:
                move = Move(kind=MoveKind.PROBE, phase=Phase.CORE, question=q)
            else:
                move = Move(kind=MoveKind.ASK, phase=Phase.CORE, question=t.question(nxt.question_ref))
            msgs, pr = variants(lambda rec, t=t, m=move, i=i: Interviewer(rec, t.brief).turn_messages(m, t.turns[: i + 1]))
            out.append(row(f"turn-{t.id}-t{i}", f"{t.id} after turn {i}: {move.kind.value}", msgs, pr,
                           {"max_words": {"default": MAX_WORDS, "medium": MAX_WORDS, "small": SMALL_MAX_WORDS}},
                           move=move.kind.value, interview_type=t.brief.session.interview_type.value))
    return out


def coach_turns() -> list[dict]:
    from strong_api.coach.engine import ASK_LINES, asks, opener, turn_messages
    from strong_core.db.models import CoachSession
    from strong_core.db.models import Story as StoryRow
    from strong_core.schemas import CoachSpeaker, CoachStyle
    from strong_evals.coach import load_candidates

    out = []
    for cid, cand in load_candidates().items():
        for style in CoachStyle:
            session = CoachSession(style=style)
            story = StoryRow(title="New story", skills=[])
            turns = [SimpleNamespace(speaker=CoachSpeaker.COACH, text=opener(story, is_new=True, target_skill=None))]
            said = []
            for k, answer in enumerate(cand.script):
                order = asks(story)
                msgs, pr = variants(lambda rec, s=session, st=story, tt=list(turns), a=answer, o=order:
                                    turn_messages(s, st, tt, a, o))
                said.append(answer)
                out.append(row(f"coach-{cid}-{style.value}-t{k}", f"{cid} ({style.value}) turn {k}: {answer[:60]}", msgs, pr,
                               {"source_texts": list(said)}, candidate=str(cand.kind.value if hasattr(cand.kind, "value") else cand.kind),
                               style=style.value))
                turns += [SimpleNamespace(speaker=CoachSpeaker.CANDIDATE, text=answer),
                          SimpleNamespace(speaker=CoachSpeaker.COACH, text=ASK_LINES[order[0]])]
    return out


DATASETS = {
    "sh-extract-posting": (extract_postings, "job_posting"),
    "sh-extract-resume": (extract_resumes, "resume"),
    "sh-input-check": (input_check, "input_check"),
    "sh-gap": (gap_pairs, "gap_assessment"),
    "sh-score": (scorer_items, "scorer_output"),
    "sh-probe": (probe_decisions, "probe_decision"),
    "sh-answer-check": (answer_checks, "answer_check"),
    "sh-interviewer-turn": (interviewer_turns, None),
    "sh-coach-turn": (coach_turns, "coach_reply"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strong-hire", required=True, type=Path, help="a Strong Hire checkout (its environment runs this)")
    ap.add_argument("only", nargs="*", help="dataset ids to build (default: all)")
    a = ap.parse_args()
    root = a.strong_hire.resolve()
    os.environ["MODEL_PROFILE"] = "fake"  # nothing is called; the profile only has to load
    logging.disable(logging.WARNING)  # the stand-in gateway "fails" every call; the product logs each
    os.chdir(root)
    OUT.mkdir(parents=True, exist_ok=True)
    SCHEMAS.mkdir(parents=True, exist_ok=True)
    for ds_id in a.only or list(DATASETS):
        build, schema = DATASETS[ds_id]
        rows = build()
        path = OUT / f"{ds_id}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if schema:
            shutil.copyfile(root / "schemas" / f"{schema}.schema.json", SCHEMAS / f"{schema}.schema.json")
        tiers = sorted({k for r in rows for k in json.loads(r["messages"])})
        print(f"{ds_id}: {len(rows)} rows, prompt variants {', '.join(tiers)} -> {path.relative_to(TNE)}", file=sys.stderr)


if __name__ == "__main__":
    main()
