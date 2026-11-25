"""Train the inpainting diffusion verifier for one dataset (training split only).

Usage:
    python scripts/train_diffusion.py --dataset synth_planted --graph prior+learned
    python scripts/train_diffusion.py --dataset synth_planted --graph none   # RQ3 ablation

Writes results/diffusion/<dataset>/<split>/<graph>.pt plus stats.npz (the
per-channel standardisation fitted on the training split, shared with the oracle).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.data import load_dataset  # noqa: E402
from veritas.grid import EvidenceGrid  # noqa: E402
from veritas.verifier.diffusion import DiffusionConfig, InpaintDiffusion  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="standard")
    ap.add_argument("--graph", default="prior+learned", choices=["none", "prior", "prior+learned"])
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--dilation-cycle", type=int, default=6,
                    help="dilations 1..2**(cycle-1); use 8 for T~1000 so inpainting sees enough context")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sensor-masks", action="store_true", help="also train on whole-sensor masks")
    ap.add_argument("--name", default=None, help="checkpoint name (default: the graph setting)")
    args = ap.parse_args()
    name = args.name or args.graph

    data = load_dataset(args.dataset, split=args.split, max_test=None, max_len=args.max_len)
    mu = data.X_train.mean(axis=(0, 2), keepdims=True)
    sd = data.X_train.std(axis=(0, 2), keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xtr = (data.X_train - mu) / sd

    _, C, T = Xtr.shape
    grid = EvidenceGrid(T=T, channel_groups=data.meta.channel_groups, K=args.K)
    cfg = DiffusionConfig(channels=args.channels, layers=args.layers, graph=args.graph,
                          dilation_cycle=args.dilation_cycle,
                          iters=args.iters, batch_size=args.batch_size, seed=args.seed,
                          sensor_masks=args.sensor_masks)
    diff = InpaintDiffusion(cfg, C, len(data.meta.class_names), data.meta.channel_edges, grid)
    n_params = sum(p.numel() for p in diff.model.parameters())
    out_dir = ROOT / "results" / "diffusion" / args.dataset / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "stats.npz").exists():  # keep the statistics shared with existing judges
        np.savez(out_dir / "stats.npz", mu=mu, sd=sd)
    print(f"{args.dataset}/{args.split} graph={args.graph} C={C} T={T} "
          f"claims={len(grid.claims())} params={n_params / 1e6:.2f}M", flush=True)

    t0 = time.perf_counter()
    ckpt = out_dir / f"{name}.ckpt"  # resumable training state; removed when done
    losses = diff.fit(Xtr, data.y_train, log=lambda s: print(s, flush=True), ckpt_path=ckpt)
    minutes = (time.perf_counter() - t0) / 60
    diff.save(out_dir / f"{name}.pt")
    ckpt.unlink(missing_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(
        {"config": vars(args), "params_M": n_params / 1e6, "train_minutes": minutes,
         "loss_curve": losses}, indent=2))
    print(f"done in {minutes:.1f} min -> {out_dir / (name + '.pt')}", flush=True)


if __name__ == "__main__":
    main()
