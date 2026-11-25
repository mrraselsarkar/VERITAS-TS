"""Does it matter that the PAMAP2 diffusion model never saw whole-sensor masks?

Compares the original model (trained on cell masks only) with a retrained model
that also saw whole-sensor masks ('prior+learned+sensor'):
  verdict agreement   share of (recording, claim) pairs with the same verified
                      / not-verified outcome for the true label
  any-verified rate   share of recordings with at least one verified claim
  agent results       verified rate of the agents' answers under each bank
  realism             detector accuracy on hybrid necessity edits of random
                      sensor claims (0.5 = indistinguishable from real data)

Usage:
    python scripts/sensormask_eval.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.reward import Bank  # noqa: E402
from veritas.data import load_dataset  # noqa: E402
from veritas.grid import EvidenceGrid  # noqa: E402
from veritas.verifier.diffusion import InpaintDiffusion  # noqa: E402
from veritas.verifier.edits import edit  # noqa: E402
from veritas.verifier.metrics import discriminative_score  # noqa: E402

RUNS = ["sft_verified", "grpo_accuracy_ver", "grpo_verified_ver", "policy_verified"]


def main():
    bdir = ROOT / "results" / "bank" / "pamap2_group"
    a, b = Bank(bdir / "test.npz"), Bank(bdir / "test_sensormask.npz")
    Va = np.stack([a.verified_cells(i, int(a.y[i])) for i in range(len(a))])
    Vb = np.stack([b.verified_cells(i, int(b.y[i])) for i in range(len(b))])
    res = {"verdict_agreement": float((Va == Vb).mean()),
           "any_verified_original": float(Va.any(1).mean()), "any_verified_sensor_masks": float(Vb.any(1).mean()),
           "agents": {}}
    print(f"verdict agreement {res['verdict_agreement']:.3f}; any verified: original "
          f"{res['any_verified_original']:.3f}, retrained {res['any_verified_sensor_masks']:.3f}")
    adir = ROOT / "results" / "agent" / "pamap2_group"
    for run in RUNS:
        P = json.loads((adir / f"pred_{run}.json").read_text())["preds"]
        ok = [p for p in P if p["label"] == p["label_true"]]
        r = [np.mean([bk.verified(p["bank_i"], bk.claim_id(p["window"], p["group"]), p["label_true"]) for p in ok])
             for bk in (a, b)]
        res["agents"][run] = {"original": float(r[0]), "sensor_masks": float(r[1])}
        print(f"{run:20s} verified of correct: original {r[0]:.3f}  retrained {r[1]:.3f}")

    # realism of necessity edits for random sensor claims, both models
    ddir = ROOT / "results" / "diffusion" / "pamap2" / "standard"
    st = np.load(ddir / "stats.npz")
    data = load_dataset("pamap2", max_test=2000)
    Xtr = ((data.X_train - st["mu"]) / st["sd"]).astype(np.float32)
    Xte = ((data.X_test - st["mu"]) / st["sd"]).astype(np.float32)
    grid = EvidenceGrid(T=Xte.shape[-1], channel_groups=data.meta.channel_groups, K=1, max_groups=2)
    claims = grid.claims()
    rng = np.random.default_rng(0)
    rows = rng.choice(len(Xte), 400, replace=False)
    M = np.stack([grid.mask(claims[rng.integers(len(claims))]) for _ in rows])
    n_cls = len(data.meta.class_names)
    target = (data.y_test[rows] + rng.integers(1, n_cls, size=len(rows))) % n_cls
    for name in ("prior+learned", "prior+learned+sensor"):
        diff = InpaintDiffusion.load(ddir / f"{name}.pt")
        Xe = edit("hybrid", Xte[rows], M, target, np.random.default_rng(1), pool=(Xtr, data.y_train),
                  diffusion=diff, guidance=2.0, strength=0.5, batch_size=128)
        acc = discriminative_score(Xte[rows], Xe, np.arange(len(rows)))
        res[f"realism_{name}"] = acc
        print(f"realism ({name}): detector accuracy {acc:.3f}")
    (ROOT / "results" / "agent" / "pamap2_group" / "sensormask_summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
