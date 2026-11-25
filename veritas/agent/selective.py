"""Verification-gated selective classification with a Learn-then-Test guarantee
(proposal, Method part 5).

A prediction is accepted when its score (e.g. the verification margin of its
claim) is at least lambda. Learn-then-Test picks lambda on a calibration set so
that, with probability at least 1 - delta, the error rate among accepted
predictions is at most alpha. Each threshold is tested with an exact binomial
p-value, and the family-wise error is controlled in one of two ways:

- few distinct score levels (the verification margin takes at most four values):
  Bonferroni over all levels (each tested at delta / #levels), and the loosest
  certified level is kept. Fixed-sequence testing is a poor fit here because the
  strictest level often holds too few answers to be certified, which would stop the
  sequence before the looser, well-populated levels are reached.
- continuous scores (e.g. oracle confidence, ~one level per answer): fixed-sequence
  testing from strict to loose, stopping at the first level that cannot be certified.

Both are valid Learn-then-Test procedures (Angelopoulos et al., 2021).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom


MAX_BONFERRONI_LEVELS = 10


def ltt_threshold(score: np.ndarray, wrong: np.ndarray, alpha: float, delta: float,
                  grid: np.ndarray | None = None) -> float | None:
    """Loosest certified threshold, or None if no threshold can be certified."""
    grid = np.unique(score)[::-1] if grid is None else np.sort(grid)[::-1]
    best = None
    if len(grid) <= MAX_BONFERRONI_LEVELS:
        for lam in grid:
            acc = score >= lam
            n = int(acc.sum())
            if n and binom.cdf(int(wrong[acc].sum()), n, alpha) <= delta / len(grid):
                best = lam  # grid runs strict -> loose, so the last certified level is the loosest
        return best
    for lam in grid:
        acc = score >= lam
        n = int(acc.sum())
        if n == 0:
            continue
        k = int(wrong[acc].sum())
        # H0: selective risk > alpha. p = P(Bin(n, alpha) <= k)
        if binom.cdf(k, n, alpha) <= delta:
            best = lam
        else:
            break
    return best


def evaluate(score: np.ndarray, wrong: np.ndarray, alpha: float, delta: float,
             n_splits: int = 200, seed: int = 0) -> dict:
    """Repeated 50/50 calibration/test splits: mean coverage and selective risk on
    the test half, and the share of splits where the risk stayed <= alpha."""
    rng = np.random.default_rng(seed)
    cov, risk, ok = [], [], []
    for _ in range(n_splits):
        p = rng.permutation(len(score))
        cal, te = p[: len(p) // 2], p[len(p) // 2:]
        lam = ltt_threshold(score[cal], wrong[cal], alpha, delta)
        acc = np.zeros(len(te), bool) if lam is None else score[te] >= lam
        c = acc.mean()
        r = wrong[te][acc].mean() if acc.any() else 0.0
        cov.append(c)
        risk.append(r)
        ok.append(r <= alpha)
    return {"coverage": float(np.mean(cov)), "selective_risk": float(np.mean(risk)),
            "risk_le_alpha": float(np.mean(ok))}


def risk_coverage(score: np.ndarray, wrong: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Curve of (coverage, selective risk) when accepting the top-scored predictions."""
    order = np.argsort(-score, kind="stable")
    w = wrong[order].astype(float)
    n = np.arange(1, len(w) + 1)
    return n / len(w), np.cumsum(w) / n
