"""Bootstrap confidence intervals and paired comparisons."""

from __future__ import annotations

import numpy as np

B = 2000
rng = np.random.default_rng(0)


def mean_ci(x: list[float]) -> tuple[float, list[float]]:
    a = np.asarray(x, dtype=float)
    if len(a) == 0:
        return float("nan"), [float("nan"), float("nan")]
    boots = a[rng.integers(0, len(a), (B, len(a)))].mean(1)
    return float(a.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def paired_delta(a: list[float], b: list[float]) -> dict:
    """b minus a on the same items; the verdict says whether the 95% interval excludes zero."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    delta, ci = mean_ci(d.tolist())
    verdict = "b_better" if ci[0] > 0 else "a_better" if ci[1] < 0 else "no_clear_difference"
    return {"delta": delta, "ci95": ci, "verdict": verdict}
