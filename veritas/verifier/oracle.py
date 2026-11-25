"""Frozen oracle: the classifier that judges counterfactual edits.

It is trained on the training split only and never sees the agent, so the
verifier's verdicts do not depend on the policy being trained (proposal,
Risks: circularity). Members are averaged at the probability level.
"""
from __future__ import annotations

import numpy as np

from ..baselines.classical import CLASSIFIERS


class Oracle:
    def __init__(self, members: tuple[str, ...] = ("multirocket_hydra",), seed: int = 0):
        self.members = members
        self.seed = seed
        self.models = []
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Oracle":
        """X must already be standardised the same way the verifier's edits are."""
        X = X.astype(np.float64)
        self.models = [CLASSIFIERS[m](self.seed).fit(X, y) for m in self.members]
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        out = []
        for s in range(0, len(X), batch_size):
            xb = X[s:s + batch_size].astype(np.float64)
            out.append(np.mean([m.predict_proba(xb) for m in self.models], axis=0))
        return np.concatenate(out)
