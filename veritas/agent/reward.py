"""Verification lookups and the verified-evidence reward (proposal, Method parts 4-5).

A claim (window, group) about series i with label l is *verified* when
  the oracle predicts l on the unedited series (otherwise there is no
  prediction for the evidence to explain, and necessity would hold trivially),
  necessity:   at least half of its necessity edits move the oracle away from l
  sufficiency: at least half of its sufficiency edits keep the oracle at l
The verdict is read from the precomputed counterfactual bank, so it is cheap
enough to use inside RL rollouts and for test-time gating.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .views import Answer


class Bank:
    def __init__(self, path: Path):
        z = np.load(path)
        self.y = z["y"]
        self.p0 = z["p0"]
        self.nec_pred, self.suf_pred = z["nec_pred"], z["suf_pred"]
        self.index = {(int(w), str(g)): j for j, (w, g) in
                      enumerate(zip(z["claim_windows"], z["claim_groups"]))}

    def __len__(self) -> int:
        return len(self.y)

    def claim_id(self, window: int | None, group: str | None) -> int | None:
        return self.index.get((window, group)) if window is not None and group is not None else None

    def pass_rates(self, i: int, j: int, label: int) -> tuple[float, float]:
        if self.p0[i].argmax() != label:
            return 0.0, 0.0
        nec = float((self.nec_pred[i, j] != label).mean())
        suf = float((self.suf_pred[i, j] == label).mean())
        return nec, suf

    def verified(self, i: int, j: int | None, label: int | None) -> bool:
        if j is None or label is None:
            return False
        nec, suf = self.pass_rates(i, j, label)
        return nec >= 0.5 and suf >= 0.5

    def margin(self, i: int, j: int | None, label: int | None) -> float:
        """Verification margin in [0, 1]: the score used for selective classification."""
        if j is None or label is None:
            return 0.0
        nec, suf = self.pass_rates(i, j, label)
        return nec * suf

    def verified_cells(self, i: int, label: int) -> np.ndarray:
        if self.p0[i].argmax() != label:
            return np.zeros(self.nec_pred.shape[1], dtype=bool)
        nec = (self.nec_pred[i] != label).mean(-1)
        suf = (self.suf_pred[i] == label).mean(-1)
        return (nec >= 0.5) & (suf >= 0.5)


def reward(ans: Answer, label: int, bank: Bank, i: int, mode: str) -> float:
    """mode 'accuracy': format + correct label (TimeMaster-style baseline).
    mode 'verified': additionally +1 when the label is correct AND its claim
    is verified, so evidence only pays when it survives the edits."""
    well_formed = ans.label is not None and ans.claim is not None and ans.prop is not None
    r = 0.1 * well_formed
    correct = ans.label == label
    r += 1.0 * correct
    if mode == "verified" and correct:
        r += 1.0 * bank.verified(i, bank.claim_id(ans.window, ans.group), label)
    return r
