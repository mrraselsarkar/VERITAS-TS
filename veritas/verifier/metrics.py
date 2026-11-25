"""Shared verifier metrics."""
from __future__ import annotations

import numpy as np


def discriminative_score(real: np.ndarray, fake: np.ndarray, groups: np.ndarray,
                         seed: int = 0, max_pairs: int = 1000) -> float:
    """Paired realism test (0.5 = edits indistinguishable from real data).

    real[i] is the unedited original of fake[i] and groups[i] identifies that
    base series. A MiniRocket classifier learns real-vs-edited on the pairs of
    half of the base series and is scored on the other half, so it can only
    succeed by detecting edit artefacts, not by recognising particular series.
    """
    from aeon.classification.convolution_based import MiniRocketClassifier

    rng = np.random.default_rng(seed)
    if len(real) > max_pairs:
        keep = rng.choice(len(real), max_pairs, replace=False)
        real, fake, groups = real[keep], fake[keep], groups[keep]
    bases = rng.permutation(np.unique(groups))
    train_bases = bases[: len(bases) // 2]
    tr = np.isin(groups, train_bases)
    X = np.concatenate([real, fake]).astype(np.float64)
    y = np.r_[np.zeros(len(real), int), np.ones(len(fake), int)]
    in_tr = np.r_[tr, tr]
    clf = MiniRocketClassifier(random_state=seed).fit(X[in_tr], y[in_tr])
    return float((clf.predict(X[~in_tr]) == y[~in_tr]).mean())
