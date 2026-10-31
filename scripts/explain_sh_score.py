"""Explains a Strong Hire scorer evaluation (task sh-score@v1) beyond its headline numbers:

    python3 scripts/explain_sh_score.py eval-20261031-105839

Per model: schema validity and its errors, scorecards with no usable score, quote validity, scores outside
the 1-4 rubric, how spread out the scores and signals are (a scorer that always says "3" or "Lean No Hire"
looks good on the within-one metrics), stricter agreement measures (mean error in bands and points,
quadratic weighted kappa), each model against always-the-same-answer baselines, and a paired comparison.
Reads artifacts/eval/<id>/{items,results}.jsonl; calls no model.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SIGNALS = ["No Hire", "Lean No Hire", "Lean Hire", "Hire", "Strong Hire"]


def qwk(a: list[int], b: list[int], k: int) -> float | None:
    """Quadratic weighted kappa: agreement on an ordered scale, corrected for chance (1 perfect, 0 chance)."""
    if len(a) < 2:
        return None
    n = len(a)
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1
    ha, hb = Counter(a), Counter(b)
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * obs[i][j]
            den += w * ha[i] * hb[j] / n
    return 1 - num / den if den else None


def main() -> None:
    eval_id = sys.argv[1]
    d = Path(__file__).resolve().parents[1] / "artifacts" / "eval" / eval_id
    items = {it["id"]: it for it in map(json.loads, (d / "items.jsonl").read_text().splitlines())}
    results = [json.loads(line) for line in (d / "results.jsonl").read_text().splitlines()]
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)
    gold_sig = {i: SIGNALS.index(json.loads(it["reference"])["gold_signal"]) for i, it in items.items()}
    gold_scores = {i: {(g["question"], g["competency"]): g["score"] for g in json.loads(it["reference"])["gold_scores"]}
                   for i, it in items.items()}

    print(f"{eval_id}: {len(items)} transcripts; gold signals {dict(Counter(SIGNALS[g] for g in gold_sig.values()))}")
    for c in range(5):
        hits = sum(abs(c - g) <= 1 for g in gold_sig.values())
        print(f"  baseline: always '{SIGNALS[c]}' -> within one band {hits}/{len(items)}")
    print()

    signals = {}
    for model, rs in by_model.items():
        print(f"== {model}")
        errors = [r for r in rs if r.get("error")]
        invalid = [r for r in rs if r["scores"].get("json_valid") == 0.0]
        none = [r for r in rs if (r["details"].get("scorecard") or {}).get("hire_signal") is None]
        print(f"  API/generation errors {len(errors)}; schema-invalid {len(invalid)}; no usable score (signal not computed) {len(none)}")
        errs = Counter(e.split(":")[0] + ": " + e.split(":", 1)[1].strip()[:60]
                       for r in invalid for e in r["details"].get("schema_errors", []))
        for e, n in errs.most_common(6):
            print(f"    {n:3d} x {e}")
        chk = sum((r["details"].get("scorecard") or {}).get("quotes_checked", 0) for r in rs)
        ok = sum((r["details"].get("scorecard") or {}).get("quotes_valid", 0) for r in rs)
        print(f"  quotes valid {ok}/{chk}")
        raw, out_of_range, n_scored = [], 0, []
        pred_pts, gold_pts = [], []
        for r in rs:
            try:
                card = json.loads(r["output"])
            except (json.JSONDecodeError, TypeError):
                continue
            pq = card.get("per_question") if isinstance(card, dict) else None
            mine = {}
            for q in pq if isinstance(pq, list) else []:
                for s in (q.get("scores") if isinstance(q, dict) and isinstance(q.get("scores"), list) else []):
                    if isinstance(s, dict) and isinstance(s.get("score"), int):
                        raw.append(s["score"])
                        out_of_range += not 1 <= s["score"] <= 4
                        mine[(q.get("question_ref"), s.get("competency"))] = s["score"]
            n_scored.append(len(mine))
            for key, g in gold_scores[r["item"]].items():
                if key in mine and 1 <= mine[key] <= 4:
                    pred_pts.append(mine[key] - 1)
                    gold_pts.append(g - 1)
        print(f"  rubric scores written {len(raw)} (mean per transcript {sum(n_scored) / max(len(n_scored), 1):.1f}); "
              f"outside 1-4: {out_of_range}; distribution {dict(sorted(Counter(raw).items()))}")
        if pred_pts:
            mae = sum(abs(a - b) for a, b in zip(pred_pts, gold_pts)) / len(pred_pts)
            k = qwk(pred_pts, gold_pts, 4)
            print(f"  on the {len(pred_pts)} scores matching a human score: mean error {mae:.2f} points, "
                  f"weighted kappa {k:.2f}" if k is not None else "")
        sig = {r["item"]: (r["details"].get("scorecard") or {}).get("hire_signal") for r in rs}
        signals[model] = sig
        pairs = [(SIGNALS.index(s), gold_sig[i]) for i, s in sig.items() if s]
        print(f"  computed signals {dict(Counter(SIGNALS[p] for p, _ in pairs))}")
        if pairs:
            w = sum(abs(p - g) <= 1 for p, g in pairs)
            e = sum(p == g for p, g in pairs)
            mae = sum(abs(p - g) for p, g in pairs) / len(pairs)
            bias = sum(p - g for p, g in pairs) / len(pairs)
            k = qwk([p for p, _ in pairs], [g for _, g in pairs], 5)
            print(f"  signal: within one {w}/{len(items)}, exact {e}/{len(items)}, mean error {mae:.2f} bands, "
                  f"bias {bias:+.2f} bands (+ = too generous), weighted kappa {k if k is None else round(k, 2)}")
        print()

    models = list(signals)
    for a_i, a in enumerate(models):
        for b in models[a_i + 1:]:
            better = worse = 0
            for i in items:
                sa, sb = signals[a].get(i), signals[b].get(i)
                ea = abs(SIGNALS.index(sa) - gold_sig[i]) if sa else 9
                eb = abs(SIGNALS.index(sb) - gold_sig[i]) if sb else 9
                better += ea < eb
                worse += ea > eb
            print(f"{a} vs {b}: {a} closer to the human signal on {better} transcripts, farther on {worse}, "
                  f"tied on {len(items) - better - worse}")


if __name__ == "__main__":
    main()
