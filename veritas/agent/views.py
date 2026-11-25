"""What the LLM agent sees: a compact text view of one series (proposal, Method part 2).

A raw series is far too long for a small LLM's context, so the agent reads two
tool outputs instead, both computed on the per-channel standardised series:

- a cell table: for every (time window, channel group) cell of the evidence
  grid, four summary numbers averaged over the group's channels
    lvl   mean level
    amp   standard deviation (amplitude)
    hf    standard deviation of the first difference (fast-changing energy)
    frq   dominant frequency of the first difference, in cycles per window
  Differencing removes slow background drift, so short oscillatory patterns
  show up in hf and frq instead of being hidden by it.
- neighbour votes: labels of the k nearest training series in MiniRocket
  embedding space (a cheap stand-in for a foundation-model embedding)

The agent answers with a label and one evidence cell, in a fixed format that
``parse_answer`` reads back.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..grid import PROPERTIES, Claim, EvidenceGrid

FEATURES = ("lvl", "amp", "hf", "frq")


def cell_features(X: np.ndarray, grid: EvidenceGrid) -> np.ndarray:
    """(n, C, T) standardised series -> (n, K, G, 4) cell features."""
    names = grid.group_names
    out = np.zeros((len(X), grid.K, len(names), len(FEATURES)), dtype=np.float32)
    for k in range(grid.K):
        t0, t1 = grid.window_bounds(k)
        seg = X[:, :, t0:t1]
        d = np.diff(seg, axis=-1)
        spec = np.abs(np.fft.rfft(d - d.mean(-1, keepdims=True), axis=-1))
        spec[..., 0] = 0
        frq = spec.argmax(-1).astype(np.float32)          # cycles per window
        per_chan = np.stack([seg.mean(-1), seg.std(-1), d.std(-1), frq], axis=-1)  # (n, C, 4)
        for g, name in enumerate(names):
            out[:, k, g] = per_chan[:, grid.channel_groups[name]].mean(1)
    return out


class NeighbourIndex:
    """k-NN label votes in MiniRocket embedding space (cosine similarity)."""

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, n_classes: int, seed: int = 0):
        from aeon.transformations.collection.convolution_based import MiniRocket

        self.tf = MiniRocket(random_state=seed).fit(X_train.astype(np.float64))
        self.E = self._embed(X_train)
        self.y = np.asarray(y_train)
        self.n_classes = n_classes

    def _embed(self, X: np.ndarray) -> np.ndarray:
        E = np.asarray(self.tf.transform(X.astype(np.float64)), dtype=np.float32)
        if not hasattr(self, "mu"):
            self.mu, self.sd = E.mean(0), E.std(0) + 1e-6
        E = (E - self.mu) / self.sd
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def votes(self, X: np.ndarray, k: int = 10, exclude_self: bool = False,
              batch: int = 512) -> np.ndarray:
        """(n, n_classes) neighbour counts. ``exclude_self`` for training
        series, whose own label must not leak into their view."""
        E = self._embed(X)
        out = np.zeros((len(X), self.n_classes), dtype=np.int32)
        for s in range(0, len(E), batch):
            sim = E[s:s + batch] @ self.E.T
            if exclude_self:
                sim[np.arange(len(sim)), np.arange(s, s + len(sim))] = -np.inf
            nn = np.argpartition(-sim, k, axis=1)[:, :k]
            for i, row in enumerate(nn):
                out[s + i] = np.bincount(self.y[row], minlength=self.n_classes)
        return out


@dataclass
class Answer:
    label: int | None
    window: int | None
    group: str | None
    prop: str | None

    @property
    def claim(self) -> Claim | None:
        if self.window is None or self.group is None:
            return None
        return Claim((self.window,), (self.group,), self.prop)


def hint_line(prob: np.ndarray, class_names: list[str], top: int = 3) -> str:
    """Classifier hint: the ``top`` most probable classes of a standard classifier."""
    order = [int(c) for c in np.argsort(-prob, kind="stable")[:top] if prob[c] >= 0.01] or [int(prob.argmax())]
    return ("Classifier probabilities (InceptionTime): "
            + ", ".join(f"{class_names[c]}: {prob[c]:.2f}" for c in order) + ".")


def render_prompt(feat: np.ndarray, votes: np.ndarray, class_names: list[str],
                  group_names: list[str], dataset_note: str, claims: str = "cell",
                  hint: np.ndarray | None = None) -> str:
    """One series' text view. feat: (K, G, 4); votes: (n_classes,).
    claims='cell': the answer names one (window, group) cell.
    claims='group': the answer names one or two channel groups over the whole series.
    hint: optional (n_classes,) class probabilities of a standard classifier."""
    K = feat.shape[0]
    lines = [
        f"Classify a multichannel time series. {dataset_note}",
        f"Classes: {', '.join(f'{i}={c}' for i, c in enumerate(class_names))}.",
        "Nearest labelled neighbours (class: count): "
        + ", ".join(f"{class_names[c]}: {int(v)}" for c, v in enumerate(votes) if v > 0) + ".",
    ]
    if hint is not None:
        lines.append(hint_line(hint, class_names))
    lines += [
        f"Cell table: the series is split into {K} time windows (rows) and channel groups "
        "(columns). Each cell shows lvl/amp/hf/frq = mean level, amplitude, fast-change "
        "energy, dominant frequency of the changes (cycles per window). Values are standardised.",
        "window | " + " | ".join(group_names),
    ]
    for k in range(K):
        cells = [f"{c[0]:.1f}/{c[1]:.1f}/{c[2]:.2f}/{c[3]:.0f}" for c in feat[k]]
        lines.append(f"{k} | " + " | ".join(cells))
    if claims == "group":
        lines += [
            "Answer with the class and the ONE or TWO channel groups whose signals decide it, "
            "in exactly this format:",
            "label: <class name>",
            f"groups: <one of {', '.join(group_names)}, or two joined by +>",
            f"property: <one of {', '.join(PROPERTIES)}>",
        ]
    else:
        lines += [
            "Answer with the class and the ONE cell whose content decides it, in exactly this format:",
            "label: <class name>",
            "window: <window number>",
            f"group: <one of {', '.join(group_names)}>",
            f"property: <one of {', '.join(PROPERTIES)}>",
        ]
    return "\n".join(lines)


def format_answer(label: str, window: int, group: str, prop: str) -> str:
    return f"label: {label}\nwindow: {window}\ngroup: {group}\nproperty: {prop}"


def format_group_answer(label: str, groups: tuple[str, ...], prop: str) -> str:
    return f"label: {label}\ngroups: {'+'.join(sorted(groups))}\nproperty: {prop}"


_FIELD = re.compile(r"^\s*(label|window|groups?|property)\s*:\s*(.+?)\s*$", re.I | re.M)


def parse_answer(text: str, class_names: list[str], group_names: list[str], K: int,
                 claims: str = "cell") -> Answer:
    fields = {k.lower(): v for k, v in _FIELD.findall(text)}
    lookup = {c.lower(): i for i, c in enumerate(class_names)}
    raw = fields.get("label", "").lower().strip(" .\"'")
    label = lookup.get(raw)
    if label is None and raw.isdigit() and int(raw) < len(class_names):
        label = int(raw)  # an untuned model often answers with the class number
    prop = fields.get("property", "").strip(" .\"'")
    prop = prop if prop in PROPERTIES else None
    if claims == "group":  # one or two groups over the whole series (window 0 of a 1-window grid)
        # match known group names directly: some names contain commas, e.g. "limb_bipolar (I, II, III)"
        text = fields.get("groups", fields.get("group", ""))
        parts, rest = [], text
        for name in sorted(group_names, key=len, reverse=True):
            if name in rest:
                parts.append(name)
                rest = rest.replace(name, " ")
        leftover = re.sub(r"[+,\s.\"']|\band\b", "", rest)
        ok = 1 <= len(parts) <= 2 and not leftover
        return Answer(label, 0 if ok else None, "+".join(sorted(parts)) if ok else None, prop)
    window = None
    m = re.match(r"\d+", fields.get("window", ""))
    if m and int(m.group()) < K:
        window = int(m.group())
    group = fields.get("group", "").strip(" .\"'")
    group = group if group in group_names else None
    return Answer(label, window, group, prop)


def ranked_cells(feat: np.ndarray, top: int | None = None) -> list[tuple[int, int, str]]:
    """Cells ordered from most to least anomalous by robust z-score over the
    series' cells, each with a property word from its most deviating feature.
    Used only for the SFT warm start: plausible but unverified evidence."""
    flat = feat.reshape(-1, feat.shape[-1])
    med = np.median(flat, 0)
    mad = np.median(np.abs(flat - med), 0) + 1e-3
    z = (flat - med) / mad
    G = feat.shape[1]
    out = []
    for cell in np.argsort(-np.abs(z).max(1), kind="stable")[:top]:
        f = int(np.abs(z[cell]).argmax())
        prop = {0: "amplitude_up" if z[cell, 0] > 0 else "amplitude_down",
                1: "amplitude_up" if z[cell, 1] > 0 else "amplitude_down",
                2: "spike", 3: "frequency_shift"}[f]
        out.append((int(cell) // G, int(cell) % G, prop))
    return out


def heuristic_claim(feat: np.ndarray) -> tuple[int, int, str]:
    """The single most anomalous cell (see ``ranked_cells``)."""
    return ranked_cells(feat, top=1)[0]


def ranked_groups(feat: np.ndarray, group_names: list[str]) -> list[tuple[tuple[str, ...], str]]:
    """Sensor-level warm-start candidates: groups ranked by their mean (over
    windows) largest robust z-score, returned as [top1], [top2], [top1+top2], ...
    each with a property word. Plausible but unverified, like ``ranked_cells``."""
    flat = feat.reshape(-1, feat.shape[-1])
    med = np.median(flat, 0)
    mad = np.median(np.abs(flat - med), 0) + 1e-3
    z = ((feat - med) / mad)                           # (K, G, F)
    score = np.abs(z).max(-1).mean(0)                  # (G,)
    order = [int(g) for g in np.argsort(-score, kind="stable")]

    def prop(g: int) -> str:
        zg = z[:, g].mean(0)
        f = int(np.abs(zg).argmax())
        return {0: "amplitude_up" if zg[0] > 0 else "amplitude_down",
                1: "amplitude_up" if zg[1] > 0 else "amplitude_down",
                2: "irregular_periodicity", 3: "frequency_shift"}[f]

    a, b = order[0], order[1] if len(order) > 1 else order[0]
    out = [((group_names[a],), prop(a)), ((group_names[b],), prop(b)),
           (tuple(sorted({group_names[a], group_names[b]})), prop(a))]
    out += [((group_names[g],), prop(g)) for g in order[2:]]
    return out
