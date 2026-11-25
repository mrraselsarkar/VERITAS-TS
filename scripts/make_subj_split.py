"""Build <dataset>_subj: a clean in-distribution test and a real subject shift.

Used for UCI-HAR and PAMAP2, whose 'standard' split mixes subjects and whose 'shift'
split holds out whole subjects. For the abstention-under-
shift experiment both an in-distribution test set and a shifted test set are
needed, with every model trained on the same data:
  standard: train = 80% of the official training windows, test = the other 20%
            (same people as training: in distribution)
  shift:    train = the same 80%, test = the official test windows (new people)

Usage:
    python scripts/make_subj_split.py --dataset pamap2
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.data.base import TSDataset, load_processed, save, stratified_split, validate  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ucihar")
    args = ap.parse_args()
    ds = load_processed(args.dataset)
    off_tr, off_te = ds.split_idx["shift"]["train"], ds.split_idx["shift"]["test"]
    tr_rel, te_rel = stratified_split(ds.y[off_tr], test_frac=0.2, seed=0)
    tr, id_te = off_tr[tr_rel], off_tr[te_rel]
    name = f"{args.dataset}_subj"
    meta = replace(ds.meta, name=name, notes=(
        ds.meta.notes + f" | {name}: standard = 80/20 stratified split of the official "
        "training windows (same subjects, in distribution); shift = same 80% for training, "
        "official test windows (unseen subjects) for testing."))
    new = TSDataset(X=ds.X, y=ds.y, meta=meta, groups=ds.groups,
                    split_idx={"standard": {"train": tr, "test": id_te},
                               "shift": {"train": tr, "test": off_te}})
    validate(new)
    save(new)
    print(f"{name}: train {len(tr)}, in-distribution test {len(id_te)}, shifted test {len(off_te)}")


if __name__ == "__main__":
    main()
