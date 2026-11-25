"""UCR energy datasets: ElectricDevices and Computers (domain: energy).

Both come from the UK "Powering the Nation" household study via
timeseriesclassification.com. Published reference results suggest the default
UCR train/test split separates households (HIVE-COTE 2: 76.1% on the default
ElectricDevices split vs 90.2% averaged over random resamples), so:

- ``standard``: stratified random split of train+test, same test fraction as UCR
- ``shift``:    the original UCR train/test split
"""
from __future__ import annotations

import numpy as np

from .base import RAW_DIR, DatasetMeta, TSDataset, register, stratified_split

UCR_SPECS = {
    "electricdevices": dict(
        ucr_name="ElectricDevices",
        fs=1 / 900,  # one reading per 15 min (96 per day)
        description="Household device power profiles, 96 readings/day, 7 device classes.",
        notes="Class-to-appliance mapping not given on the archive page; recover from "
              "Lines et al. (2011) before using class names in prompts.",
    ),
    "computers": dict(
        ucr_name="Computers",
        fs=1 / 120,  # one reading per 2 min (720 per day)
        description="Computer power consumption over 24 h; desktop vs laptop.",
        notes="Archive states classes are desktop and laptop; label order to be verified.",
    ),
}


def _load_ucr(spec_key: str) -> TSDataset:
    from aeon.datasets import load_classification

    spec = UCR_SPECS[spec_key]
    path = RAW_DIR / "ucr"
    path.mkdir(parents=True, exist_ok=True)
    Xtr, ytr = load_classification(spec["ucr_name"], split="train", extract_path=str(path))
    Xte, yte = load_classification(spec["ucr_name"], split="test", extract_path=str(path))
    X = np.concatenate([Xtr, Xte]).astype(np.float32)
    labels = np.concatenate([ytr, yte])
    # UCR labels are strings like "1".."7"; sort numerically when possible
    classes = sorted(np.unique(labels), key=lambda s: (len(str(s)), str(s)))
    lookup = {c: i for i, c in enumerate(classes)}
    y = np.array([lookup[v] for v in labels], dtype=np.int64)

    n_tr = len(ytr)
    default_tr, default_te = np.arange(n_tr), np.arange(n_tr, len(y))
    std_tr, std_te = stratified_split(y, test_frac=len(yte) / len(y), seed=0)

    meta = DatasetMeta(
        name=spec_key,
        domain="energy",
        fs=spec["fs"],
        channel_names=["power"],
        class_names=[f"class_{c}" for c in classes],
        channel_groups={"power": [0]},
        channel_edges=[],
        group_fields=["ucr_default_part"],
        splits=["standard", "shift"],
        description=spec["description"],
        source=f"https://timeseriesclassification.com/description.php?Dataset={spec['ucr_name']}",
        license="UCR archive terms (cite Dau et al. 2019)",
        notes=spec["notes"],
    )
    part = np.r_[np.zeros(n_tr, int), np.ones(len(yte), int)]  # 0 = UCR train, 1 = UCR test
    return TSDataset(
        X=X, y=y, meta=meta, groups={"ucr_default_part": part},
        split_idx={
            "standard": {"train": std_tr, "test": std_te},
            "shift": {"train": default_tr, "test": default_te},
        },
    )


@register("electricdevices")
def build_electricdevices() -> TSDataset:
    return _load_ucr("electricdevices")


@register("computers")
def build_computers() -> TSDataset:
    return _load_ucr("computers")
