"""Run classical baselines on processed datasets and save results.

Usage:
    python scripts/run_baselines.py --datasets computers electricdevices synth_planted
    python scripts/run_baselines.py --datasets ptbxl --splits standard shift \
        --classifiers multirocket_hydra quant --seeds 0 1 2 3 4

Writes results/baselines/<dataset>/<split>/<classifier>_seed<k>.json (metrics)
and .npz (test row ids, predictions, probabilities) and appends one line per
run to results/baselines/summary.csv.
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.baselines import classical, fm_probe  # noqa: E402
from veritas.data import load_dataset, load_processed  # noqa: E402

OUT = ROOT / "results" / "baselines"
RUNNERS = {**{k: classical.run for k in classical.CLASSIFIERS}, **fm_probe.PROBES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=None, help="default: all splits of each dataset")
    ap.add_argument("--classifiers", nargs="+", default=["multirocket_hydra", "quant"],
                    choices=sorted(RUNNERS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--max-test", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--max-train", type=int, default=None,
                    help="stratified train subsample for large datasets (recorded in results)")
    ap.add_argument("--norm", default="channel", choices=["channel", "series", "none"])
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    summary = OUT / "summary.csv"
    new_file = not summary.exists()
    with summary.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["time", "dataset", "split", "classifier", "seed", "norm", "n_train",
                        "n_test", "accuracy", "macro_f1", "fit_s", "predict_s"])
        for name in args.datasets:
            splits = args.splits or load_processed(name).meta.splits
            for split in splits:
                for clf in args.classifiers:
                    for seed in args.seeds:
                        # the test subsample is fixed (seed 0) so every method sees the same series
                        data = load_dataset(name, split=split, max_test=args.max_test,
                                            max_len=args.max_len, seed=0,
                                            max_train=args.max_train)
                        res = RUNNERS[clf](clf, data, seed=seed, norm=args.norm)
                        d = OUT / name / split
                        d.mkdir(parents=True, exist_ok=True)
                        stem = d / f"{clf}_seed{seed}"
                        info = dict(dataset=name, split=split, classifier=clf, seed=seed,
                                    norm=args.norm, n_train=len(data.y_train),
                                    n_test=len(data.y_test), accuracy=res.accuracy,
                                    macro_f1=res.macro_f1, fit_s=res.fit_s,
                                    predict_s=res.predict_s)
                        stem.with_suffix(".json").write_text(json.dumps(info, indent=2))
                        np.savez_compressed(stem.with_suffix(".npz"), test_rows=data.test_rows,
                                            y_true=data.y_test, y_pred=res.y_pred,
                                            proba=res.proba if res.proba is not None else np.array([]))
                        w.writerow([datetime.now().isoformat(timespec="seconds"), name, split,
                                    clf, seed, args.norm, len(data.y_train), len(data.y_test),
                                    f"{res.accuracy:.4f}", f"{res.macro_f1:.4f}",
                                    f"{res.fit_s:.1f}", f"{res.predict_s:.1f}"])
                        fh.flush()
                        print(f"{name:16s} {split:9s} {clf:18s} seed={seed} "
                              f"acc={res.accuracy:.4f} f1={res.macro_f1:.4f} "
                              f"fit={res.fit_s:.0f}s")


if __name__ == "__main__":
    main()
