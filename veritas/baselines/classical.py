"""Classical TSC baselines via aeon.

These are the non-LLM baselines of the proposal and also candidate members of
the frozen oracle ensemble that judges counterfactuals, so `run` keeps class
probabilities, not only labels.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ..data.base import SplitData


def _multirocket_hydra(seed: int):
    from aeon.classification.convolution_based import MultiRocketHydraClassifier
    return MultiRocketHydraClassifier(n_jobs=-1, random_state=seed)


def _minirocket(seed: int):
    from aeon.classification.convolution_based import MiniRocketClassifier
    return MiniRocketClassifier(n_jobs=-1, random_state=seed)


def _quant(seed: int):
    from aeon.classification.interval_based import QUANTClassifier
    return QUANTClassifier(random_state=seed)


CLASSIFIERS = {
    "multirocket_hydra": _multirocket_hydra,
    "minirocket": _minirocket,
    "quant": _quant,
}


@dataclass
class Result:
    name: str
    accuracy: float
    macro_f1: float
    fit_s: float
    predict_s: float
    y_pred: np.ndarray
    proba: np.ndarray | None


def standardize(X_train: np.ndarray, X_test: np.ndarray, mode: str = "channel"):
    """'channel': per-channel mean/std fitted on train (keeps relative amplitude
    across series, which matters for e.g. fault severity); 'series': z-normalise
    every series and channel on its own; 'none': leave as is."""
    if mode == "none":
        return X_train, X_test
    if mode == "series":
        def z(X):
            mu = X.mean(axis=-1, keepdims=True)
            sd = X.std(axis=-1, keepdims=True)
            return (X - mu) / np.where(sd < 1e-8, 1.0, sd)
        return z(X_train), z(X_test)
    mu = X_train.mean(axis=(0, 2), keepdims=True)
    sd = X_train.std(axis=(0, 2), keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd


def run(name: str, data: SplitData, seed: int = 0, norm: str = "channel") -> Result:
    Xtr, Xte = standardize(data.X_train, data.X_test, norm)
    Xtr, Xte = Xtr.astype(np.float64), Xte.astype(np.float64)
    clf = CLASSIFIERS[name](seed)
    t0 = time.perf_counter()
    clf.fit(Xtr, data.y_train)
    t1 = time.perf_counter()
    try:
        proba = clf.predict_proba(Xte)
        y_pred = clf.classes_[proba.argmax(axis=1)]
    except (AttributeError, NotImplementedError):
        proba, y_pred = None, clf.predict(Xte)
    t2 = time.perf_counter()
    y_pred = np.asarray(y_pred).astype(np.int64)
    return Result(
        name=name,
        accuracy=float(accuracy_score(data.y_test, y_pred)),
        macro_f1=float(f1_score(data.y_test, y_pred, average="macro")),
        fit_s=t1 - t0,
        predict_s=t2 - t1,
        y_pred=y_pred,
        proba=proba,
    )
