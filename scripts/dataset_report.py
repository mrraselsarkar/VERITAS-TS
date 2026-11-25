"""Summarise all processed datasets into results/datasets.md.

Usage: python scripts/dataset_report.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.data import PROCESSED_DIR, load_processed  # noqa: E402

DOMAIN_ORDER = ["healthcare", "activity", "industrial", "energy", "synthetic"]


def main():
    rows = []
    for meta_path in sorted(PROCESSED_DIR.glob("*.json")):
        name = meta_path.stem
        m = json.loads(meta_path.read_text())
        ds = load_processed(name)
        splits = "; ".join(
            f"{s} {len(p['train'])}/{len(p['test'])}" for s, p in ds.split_idx.items()
        )
        counts = np.bincount(ds.y, minlength=len(m["class_names"]))
        imbalance = counts.max() / max(counts.min(), 1)
        rows.append((
            DOMAIN_ORDER.index(m["domain"]), m["domain"], name, m["C"], m["T"],
            len(m["class_names"]), m["n"], splits, len(m["channel_groups"]),
            len(m["channel_edges"]), f"{imbalance:.1f}",
            f"{m['fs']:.4g}" if m["fs"] else "-",
        ))
    rows.sort()
    lines = [
        "| Domain | Dataset | Channels | Length | Classes | Series | Splits (train/test) "
        "| Channel groups | Graph edges | Max/min class ratio | fs (Hz) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(v) for v in r[1:]) + " |")
    out = ROOT / "results" / "datasets.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
