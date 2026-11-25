"""Failure cases of the check on the synthetic data (review: failure-case analysis).

For the correct answers of the check-reward LLM agent (seed 0) the cited cell is compared with
the planted cell and with the check's verdict:
  false pass : a wrong cell passes the check
  false fail : the planted cell fails the check
False passes are split by where the wrong cell lies relative to the planted cell (same group and
a neighbouring window, same window and another group, elsewhere); false fails by which test failed.
The same breakdown is reported for all claims in the bank (every cell of every test recording).

Usage:
    python scripts/failure_cases.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.reward import Bank  # noqa: E402
from veritas.data import load_dataset  # noqa: E402

DS, RUN = "synth_planted", "grpo_verified_div"


def where(w, g, pw, pg):
    if g == pg and abs(w - pw) == 1:
        return "neighbouring window, same group"
    if w == pw and g != pg:
        return "same window, other group"
    return "elsewhere"


def main():
    d = load_dataset(DS, max_test=2000)
    names = sorted(d.meta.channel_groups)
    classes = d.meta.class_names
    bank = Bank(ROOT / "results" / "bank" / DS / "test.npz")
    rows = np.load(ROOT / "results" / "bank" / DS / "test.npz")["rows"]
    pw, pg = d.groups_test["ev_window"][rows], d.groups_test["ev_group"][rows]
    P = json.loads((ROOT / "results" / "agent" / DS / f"pred_{RUN}.json").read_text())["preds"]
    out = {"agent": {}, "bank": {}}

    # ---- the agent's answers ----
    cat, fp_where, fn_test, per_class = Counter(), Counter(), Counter(), {c: Counter() for c in classes}
    for p in P:
        if p["label"] != p["label_true"] or p["window"] is None:
            continue
        i, y = p["bank_i"], p["label"]
        true = p["window"] == pw[i] and p["group"] == names[pg[i]]
        j = bank.claim_id(p["window"], p["group"])
        ok = bank.verified(i, j, y)
        key = ("true" if true else "wrong") + (" cell, passes" if ok else " cell, fails")
        cat[key] += 1
        per_class[classes[y]][key] += 1
        if ok and not true:
            fp_where[where(p["window"], names.index(p["group"]), pw[i], pg[i])] += 1
        if true and not ok:
            nec, suf = bank.pass_rates(i, j, y)
            fn_test["Test 1 fails" if nec < 0.5 and suf >= 0.5 else
                    "Test 2 fails" if suf < 0.5 and nec >= 0.5 else "both fail"] += 1
    n = sum(cat.values())
    out["agent"] = {"n": n, "categories": dict(cat), "false_pass_where": dict(fp_where),
                    "false_fail_test": dict(fn_test), "per_class": {c: dict(v) for c, v in per_class.items()}}

    # ---- every claim in the bank (oracle correct) ----
    fp_all, fn_all, n_true, n_wrong, tp, fp = Counter(), Counter(), 0, 0, 0, 0
    fn_by_class = Counter()
    for i in range(len(bank)):
        y = int(bank.y[i])
        if bank.p0[i].argmax() != y:
            continue
        ok = bank.verified_cells(i, y)
        tj = bank.claim_id(int(pw[i]), names[int(pg[i])])
        n_true += 1
        if ok[tj]:
            tp += 1
        else:
            nec, suf = bank.pass_rates(i, tj, y)
            fn_all["Test 1 fails" if nec < 0.5 and suf >= 0.5 else
                   "Test 2 fails" if suf < 0.5 and nec >= 0.5 else "both fail"] += 1
            fn_by_class[classes[y]] += 1
        for (w, g), j in bank.index.items():
            if j == tj:
                continue
            n_wrong += 1
            if ok[j]:
                fp += 1
                fp_all[where(w, names.index(g), pw[i], pg[i])] += 1
    out["bank"] = {"planted_cell_passes": tp / n_true, "wrong_cell_passes": fp / n_wrong,
                   "false_pass_where": dict(fp_all), "false_fail_test": dict(fn_all),
                   "false_fail_by_class": dict(fn_by_class), "n_recordings": n_true}
    (ROOT / "results" / "failure_cases.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
