"""Synthetic benchmark with planted, known evidence (proposal RQ2).

Each series has 3 channel groups of 2 channels. Within a group both channels are
noisy, differently scaled copies of one latent signal, so they are strongly
coupled; a pair of edges between neighbouring groups adds weak coupling. The
class is decided by ONE pattern planted into one group's latent signal inside
ONE of the K=8 time windows. Window and group vary per series and are stored as
ground-truth evidence (``ev_window``, ``ev_group``), so verification methods can
be scored against the truth.

Because the pattern enters through the shared latent, it appears in both
channels of the group with channel-specific gain: a counterfactual edit that
changes one channel and not its partner leaves the data manifold, which is what
the graph-aware diffusion model is meant to avoid.

Domain 1 (shift split test set) changes background frequencies, noise level and
channel gains while keeping the class patterns.
"""
from __future__ import annotations

import numpy as np

from .base import DatasetMeta, TSDataset, register, stratified_split

T, K = 512, 8
W = T // K
GROUPS = {"g0": [0, 1], "g1": [2, 3], "g2": [4, 5]}
CLASS_NAMES = ["burst", "plateau", "chirp", "dip"]
CONTEXT_CLASS_NAMES = ["weak burst", "strong burst", "weak dip", "strong dip"]
CONTEXT_AMP = (1.0, 3.0)      # pattern amplitude relative to the series scale
SCALE_RANGE = (0.4, 2.5)      # per-series scale, log-uniform


def _pattern(cls: int, rng: np.random.Generator, amp: float) -> np.ndarray:
    """One window-length pattern for a class, with mild random variation."""
    t = np.arange(W)
    centre = W / 2 + rng.uniform(-6, 6)
    env = np.exp(-0.5 * ((t - centre) / (W / 6)) ** 2)
    if cls == 0:  # high-frequency burst
        f = rng.uniform(0.22, 0.28)
        return amp * env * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    if cls == 1:  # flat-topped plateau (amplitude up)
        ramp = np.clip(np.minimum(t - 8, W - 8 - t) / 6, 0, 1)
        return amp * 0.9 * ramp
    if cls == 2:  # chirp: frequency sweeps upward
        f0, f1 = 0.03, rng.uniform(0.15, 0.2)
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * W))
        return amp * env * np.sin(phase)
    # cls == 3: dip (amplitude down)
    return -amp * 1.2 * env


def _background(rng: np.random.Generator, domain: int) -> np.ndarray:
    """Latent signal for one group: a few sinusoids plus AR(1) noise."""
    t = np.arange(T)
    lo, hi = ((0.004, 0.02), (0.012, 0.04))[domain]
    z = sum(
        rng.uniform(0.3, 0.8) * np.sin(2 * np.pi * rng.uniform(lo, hi) * t + rng.uniform(0, 2 * np.pi))
        for _ in range(3)
    )
    e = rng.normal(0, 0.15 if domain == 0 else 0.25, T)
    ar = np.zeros(T)
    for i in range(1, T):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    return z + ar


def _context_pattern(cls: int, rng: np.random.Generator) -> np.ndarray:
    """synth_context: burst (classes 0, 1) or dip (2, 3), weak or strong
    RELATIVE to the series' own scale (the whole series is scaled afterwards)."""
    rel = CONTEXT_AMP[cls % 2] * rng.uniform(0.9, 1.1)
    if cls < 2:
        return _pattern(0, rng, amp=rel)
    return _pattern(3, rng, amp=rel / 1.2)  # _pattern's dip multiplies amp by 1.2


def _series(cls: int, ev_w: int, ev_g: int, rng: np.random.Generator, domain: int,
            variant: str = "local", scale: float = 1.0) -> np.ndarray:
    names = sorted(GROUPS)
    latents = {g: _background(rng, domain) for g in names}
    # weak coupling between neighbouring groups (matches the g0-g1, g1-g2 edges)
    latents["g1"] = latents["g1"] + 0.3 * latents["g0"]
    latents["g2"] = latents["g2"] + 0.3 * latents["g1"]
    g = names[ev_g]
    if variant == "context":
        latents[g][ev_w * W:(ev_w + 1) * W] += _context_pattern(cls, rng)
    else:
        latents[g][ev_w * W:(ev_w + 1) * W] += _pattern(cls, rng, amp=rng.uniform(1.2, 1.8))
    # class-independent distractor: one sharp spike at a random place
    dg = names[rng.integers(3)]
    latents[dg][rng.integers(T)] += rng.choice([-1, 1]) * rng.uniform(1.5, 2.5)

    x = np.zeros((6, T))
    gain_lo, gain_hi = ((0.8, 1.2), (0.5, 1.5))[domain]
    for name, chans in GROUPS.items():
        for c in chans:
            x[c] = rng.uniform(gain_lo, gain_hi) * latents[name] + rng.normal(0, 0.1, T)
    return x * scale


def generate(n_source: int = 2400, n_shift: int = 800, seed: int = 0,
             variant: str = "local") -> TSDataset:
    """variant='local' -> synth_planted; variant='context' -> synth_context."""
    rng = np.random.default_rng(seed)
    n = n_source + n_shift
    domain = np.r_[np.zeros(n_source, int), np.ones(n_shift, int)]
    class_names = CONTEXT_CLASS_NAMES if variant == "context" else CLASS_NAMES
    y = rng.integers(len(class_names), size=n)
    ev_window = rng.integers(K, size=n)
    ev_group = rng.integers(len(GROUPS), size=n)
    scale = (np.exp(rng.uniform(np.log(SCALE_RANGE[0]), np.log(SCALE_RANGE[1]), size=n))
             if variant == "context" else np.ones(n))
    X = np.stack([
        _series(y[i], ev_window[i], ev_group[i], rng, domain[i], variant, scale[i])
        for i in range(n)
    ]).astype(np.float32)

    src = np.flatnonzero(domain == 0)
    tr_rel, te_rel = stratified_split(y[src], test_frac=0.3, seed=seed)
    tr, te = src[tr_rel], src[te_rel]
    shift_te = np.flatnonzero(domain == 1)

    context = variant == "context"
    meta = DatasetMeta(
        name="synth_context" if context else "synth_planted",
        domain="synthetic",
        fs=None,
        channel_names=[f"ch{c}" for c in range(6)],
        class_names=class_names,
        channel_groups=GROUPS,
        channel_edges=[(0, 1), (2, 3), (4, 5), (1, 2), (3, 4)],
        group_fields=["domain", "ev_window", "ev_group", "scale"],
        splits=["standard", "shift"],
        description=(
            "Context-dependent planted-evidence benchmark: class = burst or dip that is "
            f"weak ({CONTEXT_AMP[0]}x) or strong ({CONTEXT_AMP[1]}x) relative to the series' own "
            f"scale (log-uniform in {SCALE_RANGE}), planted in one (window, group) of a "
            f"{K}-window grid. Absolute amplitude alone is ambiguous; the context decides."
            if context else
            "Planted-evidence benchmark: class = pattern type planted in one "
            f"(window, channel group) of a {K}-window grid; ground truth stored."
        ),
        source="generated by veritas.data.synthetic",
        license="CC0",
        notes=f"T={T}, K={K}, window={W} samples; ev_group indexes sorted group names"
              + ("; a region spliced from another series keeps its donor's absolute "
                 "amplitude, so its class relative to the new context can change" if context else ""),
    )
    return TSDataset(
        X=X, y=y, meta=meta,
        groups={"domain": domain, "ev_window": ev_window, "ev_group": ev_group, "scale": scale},
        split_idx={"standard": {"train": tr, "test": te}, "shift": {"train": tr, "test": shift_te}},
    )


@register("synth_planted")
def build_synth_planted() -> TSDataset:
    return generate()


@register("synth_context")
def build_synth_context() -> TSDataset:
    return generate(variant="context")
