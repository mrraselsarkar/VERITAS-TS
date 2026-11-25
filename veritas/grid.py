"""Evidence grid: the space of claims an agent may make about a series.

A series of length T with C channels is cut into K equal time windows and the
channels into named groups (from DatasetMeta.channel_groups). A claim selects a
contiguous run of at most ``max_windows`` windows and at most ``max_groups``
channel groups, plus a property word from PROPERTIES. Keeping the claim space
small and discrete makes every claim checkable and lets the counterfactual
bank be precomputed for RL (proposal, Method part 4).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

PROPERTIES = (
    "amplitude_up",
    "amplitude_down",
    "frequency_shift",
    "irregular_periodicity",
    "spike",
    "trend",
    "morphology",
)


@dataclass(frozen=True)
class Claim:
    windows: tuple[int, ...]  # contiguous, sorted
    groups: tuple[str, ...]   # sorted group names
    prop: str | None = None   # property word; not used by the mask

    def key(self) -> str:
        return f"w{'-'.join(map(str, self.windows))}|g{'+'.join(self.groups)}"


@dataclass(frozen=True)
class EvidenceGrid:
    T: int
    channel_groups: dict[str, list[int]]
    K: int = 8
    max_windows: int = 2
    max_groups: int = 2

    def __post_init__(self):
        if self.T < self.K:
            raise ValueError(f"T={self.T} shorter than K={self.K} windows")

    @property
    def C(self) -> int:
        return sum(len(v) for v in self.channel_groups.values())

    @property
    def group_names(self) -> list[str]:
        return sorted(self.channel_groups)

    def window_bounds(self, k: int) -> tuple[int, int]:
        edges = np.linspace(0, self.T, self.K + 1).round().astype(int)
        return int(edges[k]), int(edges[k + 1])

    def claims(self) -> list[Claim]:
        """All allowed claims (without property), in a fixed order."""
        runs = [
            tuple(range(s, s + L))
            for L in range(1, self.max_windows + 1)
            for s in range(self.K - L + 1)
        ]
        names = self.group_names
        group_sets = [
            tuple(c) for r in range(1, min(self.max_groups, len(names)) + 1)
            for c in combinations(names, r)
        ]
        return [Claim(w, g) for w in runs for g in group_sets]

    def mask(self, claim: Claim) -> np.ndarray:
        """Boolean (C, T) mask, True inside the claimed region."""
        m = np.zeros((self.C, self.T), dtype=bool)
        t0, _ = self.window_bounds(claim.windows[0])
        _, t1 = self.window_bounds(claim.windows[-1])
        for g in claim.groups:
            m[self.channel_groups[g], t0:t1] = True
        return m

    def size(self, claim: Claim) -> float:
        """Claim size as a fraction of the grid (for the reward's size penalty)."""
        return len(claim.windows) * len(claim.groups) / (self.K * len(self.channel_groups))
