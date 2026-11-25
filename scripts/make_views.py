"""Render the agent's text view for every series in a counterfactual bank.

Writes results/agent/<dataset>/<part>.jsonl with one record per bank row:
  bank_i      position in results/bank/<dataset>/<part>.npz
  prompt      the text view (cell table + neighbour votes + answer format)
  label       true class index
  sft_target  warm-start answer: true label + the heuristic (most anomalous,
              unverified) cell and property
plus features.npz (cell features and neighbour votes) for the non-LLM policy.

Usage:
    python scripts/make_views.py --dataset synth_planted
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.views import (NeighbourIndex, cell_features, format_answer,  # noqa: E402
                                 format_group_answer, heuristic_claim, ranked_groups, render_prompt)
from veritas.data import load_dataset  # noqa: E402
from veritas.grid import EvidenceGrid  # noqa: E402

NOTES = {
    "synth_planted": "Six sensor channels in three coupled groups; one short pattern in one "
                     "group and window decides the class.",
    "pamap2": "Body-worn IMUs (hand, chest, ankle) and heart rate; 2 s of a daily activity.",
    "pamap2_subj": "Body-worn IMUs (hand, chest, ankle) and heart rate; 2 s of a daily activity.",
    "ptbxl": "12-lead ECG, 10 s at 100 Hz, leads in four anatomical groups; diagnostic superclass.",
    "ucihar_subj": "Waist-worn smartphone: body acceleration, gyroscope and total acceleration; "
                   "2.56 s of a daily activity.",
    "natops": "x/y/z positions of 8 body points (hand tips, elbows, wrists, thumbs; left and right); "
              "one aircraft-handling gesture.",
    "cricket": "x/y/z acceleration of the left and right wrist; one cricket umpire signal (about 6.5 s).",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--all-rows", action="store_true")
    ap.add_argument("--claims", choices=["cell", "group"], default="cell",
                    help="group: claims are one or two channel groups over the whole series")
    ap.add_argument("--split", default="standard", help="shift: test views of the shifted test set")
    ap.add_argument("--parts", nargs="+", default=["train", "test"])
    ap.add_argument("--name", default=None, help="agent and bank directory, if not the default one")
    ap.add_argument("--hint-from", default=None,
                    help="add the classifier hint of results/hint/<experiment>.npz (make_hint.py) to the view")
    args = ap.parse_args()
    name = args.name or (args.dataset + ("_group" if args.claims == "group" else "")
                         + ("" if args.split == "standard" else f"_{args.split}"))  # agent and bank directory

    st = np.load(ROOT / "results" / "diffusion" / args.dataset / "standard" / "stats.npz")
    data = load_dataset(args.dataset, split=args.split, max_test=2000)
    Xtr = ((data.X_train - st["mu"]) / st["sd"]).astype(np.float32)
    Xte = ((data.X_test - st["mu"]) / st["sd"]).astype(np.float32)
    grid = EvidenceGrid(T=Xtr.shape[-1], channel_groups=data.meta.channel_groups, K=8)
    names, classes = grid.group_names, data.meta.class_names
    print("fitting neighbour index", flush=True)
    index = NeighbourIndex(Xtr, data.y_train, len(classes))

    out_dir = ROOT / "results" / "agent" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mode.json").write_text(json.dumps({"source": args.dataset, "claims": args.claims}))
    hints = np.load(ROOT / "results" / "hint" / f"{args.hint_from}.npz") if args.hint_from else None
    arrays = {}
    for part, X, y in (("train", Xtr, data.y_train), ("test", Xte, data.y_test)):
        if part not in args.parts:
            continue
        bank_path = ROOT / "results" / "bank" / name / f"{part}.npz"
        if bank_path.exists():
            rows = np.load(bank_path)["rows"]
        elif args.all_rows:  # bank still building and known to cover every series
            rows = np.arange(len(y))
        else:
            raise FileNotFoundError(f"{bank_path} missing (pass --all-rows if the bank covers all series)")
        feat = cell_features(X[rows], grid)
        if part == "train":  # leave-one-out votes: a series must not see its own label
            votes = index.votes(X, k=args.k, exclude_self=True)[rows]
        else:
            votes = index.votes(X[rows], k=args.k)
        arrays[f"{part}_feat"], arrays[f"{part}_votes"], arrays[f"{part}_y"] = feat, votes, y[rows]
        hint = None
        if hints is not None:
            assert np.array_equal(hints[f"{part}_rows"], rows), "hint rows differ from the bank rows"
            hint = arrays[f"{part}_hint"] = hints[f"{part}_prob"]
        with open(out_dir / f"{part}.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                if args.claims == "group":
                    groups, prop = ranked_groups(feat[i], names)[0]
                    target = format_group_answer(classes[y[r]], groups, prop)
                else:
                    w, g, prop = heuristic_claim(feat[i])
                    target = format_answer(classes[y[r]], w, names[g], prop)
                f.write(json.dumps({
                    "bank_i": i,
                    "prompt": render_prompt(feat[i], votes[i], classes, names, NOTES.get(args.dataset, ""),
                                            claims=args.claims, hint=None if hint is None else hint[i]),
                    "label": int(y[r]),
                    "sft_target": target,
                }) + "\n")
        knn_acc = (votes.argmax(1) == y[rows]).mean()
        print(f"{part}: {len(rows)} views, neighbour-vote accuracy {knn_acc:.3f}"
              + ("" if hint is None else f", hint accuracy {(hint.argmax(1) == y[rows]).mean():.3f}"), flush=True)
    np.savez_compressed(out_dir / "features.npz", **arrays)


if __name__ == "__main__":
    main()
