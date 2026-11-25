"""Download (if needed) and convert datasets into data/processed/.

Usage:
    python scripts/prepare_data.py                 # list datasets
    python scripts/prepare_data.py synth_planted computers
    python scripts/prepare_data.py --all [--overwrite]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veritas.data import available, build, load_processed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    names = available() if args.all else args.names
    if not names:
        print("available:", ", ".join(available()))
        return
    for name in names:
        path = build(name, overwrite=args.overwrite)
        ds = load_processed(name)
        n, C, T = ds.X.shape
        splits = {s: (len(p["train"]), len(p["test"])) for s, p in ds.split_idx.items()}
        print(f"{name:18s} n={n:6d} C={C:3d} T={T:5d} classes={len(ds.meta.class_names):2d} "
              f"splits={splits} -> {path.name}")


if __name__ == "__main__":
    main()
