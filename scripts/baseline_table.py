"""Accuracy table (%) from results/baselines/summary.csv -> results/baselines/table.md.

The latest run of each (dataset, split, classifier) wins.
Usage: python scripts/baseline_table.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["ptbxl", "sleepedf", "af2017", "ucihar", "pamap2", "hhar", "cwru", "paderborn", "tep",
         "electricdevices", "computers", "gpvs", "synth_planted", "synth_context"]


def main():
    rows = list(csv.DictReader((ROOT / "results" / "baselines" / "summary.csv").open()))
    latest = {(r["dataset"], r["split"], r["classifier"]): r for r in rows}
    clfs = sorted({k[2] for k in latest})
    datasets = [d for d in ORDER if any(k[0] == d for k in latest)]
    datasets += sorted({k[0] for k in latest} - set(datasets))
    lines = ["| Dataset | Split | Train size | " + " | ".join(clfs) + " |",
             "| --- | --- | --- | " + " | ".join("---" for _ in clfs) + " |"]
    for d in datasets:
        for s in ("standard", "shift"):
            runs = [latest.get((d, s, c)) for c in clfs]
            if not any(runs):
                continue
            n_train = next(r["n_train"] for r in runs if r)
            cells = [f"{float(r['accuracy']) * 100:.1f}" if r else "" for r in runs]
            lines.append(f"| {d} | {s} | {n_train} | " + " | ".join(cells) + " |")
    out = ROOT / "results" / "baselines" / "table.md"
    out.write_text("Accuracy (%) on the fixed test subsample (≤ 2,000 series).\n\n" + "\n".join(lines) + "\n",
                   encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
