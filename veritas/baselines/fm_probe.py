"""Frozen foundation-model embeddings + a light classifier (proposal baselines,
and the second member of the oracle ensemble).

Mantis-8M (paris-noah/Mantis-8M) is pre-trained for classification on series
of length 512 and embeds each channel separately (256 dims per channel), so
series are linearly resized to 512 and the channel embeddings concatenated.
"""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data.base import SplitData
from .classical import Result, standardize

MANTIS_LEN = 512


@lru_cache(maxsize=1)
def _mantis(device: str):
    from mantis.architecture import Mantis8M
    from mantis.trainer import MantisTrainer

    net = Mantis8M(device=device).from_pretrained("paris-noah/Mantis-8M")
    return MantisTrainer(device=device, network=net)


def mantis_embed(X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """(n, C, T) -> (n, C * 256) frozen Mantis embeddings."""
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, dtype=torch.float32)
    if Xt.shape[-1] != MANTIS_LEN:
        Xt = F.interpolate(Xt, size=MANTIS_LEN, mode="linear", align_corners=False)
    Z = _mantis(device).transform(Xt.numpy(), batch_size=batch_size)
    return np.asarray(Z, dtype=np.float32).reshape(len(X), -1)


def _head(kind: str, seed: int):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


def run(name: str, data: SplitData, seed: int = 0, norm: str = "channel") -> Result:
    """name is 'mantis_rf' or 'mantis_lr'."""
    kind = name.split("_", 1)[1]
    Xtr, Xte = standardize(data.X_train, data.X_test, norm)
    t0 = time.perf_counter()
    Ztr, Zte = mantis_embed(Xtr), mantis_embed(Xte)
    clf = _head(kind, seed).fit(Ztr, data.y_train)
    t1 = time.perf_counter()
    proba = clf.predict_proba(Zte)
    y_pred = clf.classes_[proba.argmax(axis=1)].astype(np.int64)
    t2 = time.perf_counter()
    return Result(
        name=name,
        accuracy=float(accuracy_score(data.y_test, y_pred)),
        macro_f1=float(f1_score(data.y_test, y_pred, average="macro")),
        fit_s=t1 - t0, predict_s=t2 - t1, y_pred=y_pred, proba=proba,
    )


PROBES = {"mantis_rf": run, "mantis_lr": run}
