"""Region edits used to test a claim: diffusion inpainting and masking baselines.

Every method takes standardised series ``X (B, C, T)``, boolean masks of the
same shape (True = region to replace) and returns edited copies. Necessity
edits replace the claimed region (pushing towards ``target`` classes where the
method supports it); sufficiency edits replace the complement of the claim.

Baselines (the TimeSAE critique says masking leaves the data manifold):
- zero:   region set to 0 (the training mean after standardisation)
- noise:  region replaced by N(0, 1) noise
- interp: linear interpolation across the region, per channel
- splice: the same region copied from a random training series of the target
          class (for sufficiency: of any class); realistic-looking but ignores
          the surrounding context
- hybrid: splice, then diffusion re-fits the spliced region to its context
          (SDEdit from `strength` x the full noise level)
"""
from __future__ import annotations

import numpy as np

METHODS = ("zero", "noise", "interp", "splice", "diffusion", "hybrid")


def _interp(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    out = X.copy()
    t = np.arange(X.shape[-1])
    for b in range(len(X)):
        for c in np.flatnonzero(M[b].any(axis=-1)):
            keep = ~M[b, c]
            if keep.sum() >= 2:
                out[b, c, ~keep] = np.interp(t[~keep], t[keep], X[b, c, keep])
            else:
                out[b, c] = 0.0
    return out


def edit(method: str, X: np.ndarray, M: np.ndarray, target: np.ndarray | None,
         rng: np.random.Generator, pool: tuple[np.ndarray, np.ndarray] | None = None,
         diffusion=None, guidance: float = 1.0, n_steps: int = 25, seed: int = 0,
         batch_size: int = 64, strength: float = 0.5, donor: np.ndarray | None = None) -> np.ndarray:
    """Return edited copies of X. ``target`` = class per row (None or the
    diffusion model's null index for unconditional). ``donor`` = class of the
    spliced content per row when it differs from ``target`` (e.g. an other-class
    donor with an unconditional repair)."""
    if method == "zero":
        return np.where(M, 0.0, X).astype(np.float32)
    if method == "noise":
        return np.where(M, rng.standard_normal(X.shape), X).astype(np.float32)
    if method == "interp":
        return _interp(X, M).astype(np.float32)
    if method == "splice":
        Xp, yp = pool
        cls = donor if donor is not None else target
        donors = np.empty(len(X), dtype=int)
        for b in range(len(X)):
            # a target outside the pool's classes (e.g. the null index) means "any class"
            cand = np.flatnonzero(yp == cls[b]) if cls is not None else np.array([], int)
            donors[b] = rng.choice(cand if len(cand) else len(yp))
        return np.where(M, Xp[donors], X).astype(np.float32)
    if method == "diffusion":
        c = np.full(len(X), diffusion.null) if target is None else target
        return diffusion.inpaint(X, M, c, n_steps=n_steps, guidance=guidance, seed=seed,
                                 batch_size=batch_size)
    if method == "hybrid":
        # splice a donor region, then let diffusion re-fit it to the context (SDEdit)
        spliced = edit("splice", X, M, target, rng, pool=pool, donor=donor)
        c = np.full(len(X), diffusion.null) if target is None else target
        return diffusion.inpaint(X, M, c, n_steps=n_steps, guidance=guidance, seed=seed,
                                 batch_size=batch_size, init=spliced, strength=strength)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
