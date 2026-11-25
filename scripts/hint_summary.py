"""Agents with and without the classifier hint: accuracy, checked rate (oracle), checked rate (judge 2)
and, on the synthetic benchmark, the true-evidence rate. Mean and SD over the seeds found.

Usage:
    python scripts/hint_summary.py --datasets ptbxl_group natops_group
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from seeds_summary import GROUPS  # noqa: E402
from veritas.agent.reward import Bank  # noqa: E402
from veritas.data import load_dataset  # noqa: E402


def metrics(adir: Path, run: str, bank: Bank, judge2: Bank | None, gt) -> dict | None:
    f = adir / f"pred_{run}.json"
    if not f.exists():
        return None
    P = json.loads(f.read_text())["preds"]
    ok = np.array([p["label"] == p["label_true"] for p in P])

    def checked(b: Bank) -> float:
        v = np.array([b.verified(p["bank_i"], b.claim_id(p["window"], p["group"]), p["label_true"]) for p in P])
        return float(v[ok].mean())
    m = {"accuracy": float(ok.mean()), "checked": checked(bank)}
    if judge2 is not None:
        m["checked_judge2"] = checked(judge2)
    if gt is not None:
        w, g, names = gt
        hit = np.array([p["window"] == w[p["bank_i"]] and p["group"] == names[g[p["bank_i"]]] for p in P])
        m["true_evidence"] = float(hit[ok].mean())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(GROUPS))
    args = ap.parse_args()
    out_path = ROOT / "results" / "hint_summary.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for ds in args.datasets:
        gt = None
        if ds.startswith("synth"):
            d = load_dataset(ds, max_test=2000)
            rows = np.load(ROOT / "results" / "bank" / ds / "test.npz")["rows"]
            gt = (d.groups_test["ev_window"][rows], d.groups_test["ev_group"][rows], sorted(d.meta.channel_groups))
        out[ds] = {}
        for variant in ("", "_hint"):
            name = ds + variant
            bdir = ROOT / "results" / "bank" / name
            if not (bdir / "test.npz").exists():
                continue
            bank = Bank(bdir / "test.npz")
            judge2 = Bank(bdir / "test_judge2.npz") if (bdir / "test_judge2.npz").exists() else None
            for label, run in GROUPS[ds].items():
                if "masking" in label or "occlusion" in label:
                    continue
                ms = [m for m in (metrics(ROOT / "results" / "agent" / name, run + s, bank, judge2, gt)
                                  for s in ("", "_s1", "_s2")) if m]
                if not ms:
                    continue
                agg = {k: {"mean": float(np.mean([m[k] for m in ms])),
                           "sd": float(np.std([m[k] for m in ms], ddof=1)) if len(ms) > 1 else 0.0} for k in ms[0]}
                agg["n_seeds"] = len(ms)
                key = label + (" + hint" if variant else "")
                out[ds][key] = agg
                print(f"{ds:18s} {key:34s} n={len(ms)}  " + "  ".join(
                    f"{k} {v['mean'] * 100:5.1f}±{v['sd'] * 100:4.1f}" for k, v in agg.items() if k != "n_seeds"))
        hint_json = ROOT / "results" / "hint" / f"{ds}.json"
        if hint_json.exists():
            out[ds]["InceptionTime hint alone"] = {"accuracy": {"mean": json.loads(hint_json.read_text())["acc_test"],
                                                                "sd": 0.0}}
            print(f"{ds:18s} {'InceptionTime hint alone':34s} accuracy "
                  f"{out[ds]['InceptionTime hint alone']['accuracy']['mean'] * 100:.1f}")
    out_path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
