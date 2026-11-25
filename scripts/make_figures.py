"""Analysis figures for the paper (vector PDF, IEEE page width; --target dmkd: Springer DMKD text width).

  fig_example.pdf       qualitative example: a recording, its claim, and the two test edits
  fig_risk_real.pdf     selective error vs coverage, synthetic (in distribution, shifted) and real data
  fig_main_bars.pdf     checked rate and accuracy per dataset, accuracy vs check reward
  fig_edits.pdf         edit methods: realism vs ability to find the true evidence
  fig_sensitivity.pdf   checker AUC vs noise strength, guidance weight and number of edits
  fig_interpret.pdf     (a) per-class true-evidence rate, (b) PAMAP2 sensors cited per activity

Style: journal look (Times-like serif, full black frame, outward ticks, light horizontal
grid, black-edged bars with hatching, values printed vertically on the bars, panel labels
(a)/(b) at the top left, one shared legend above the panels). Muted red / blue / green /
purple series; hatching keeps the series apart in grey-scale print. One-hue blue ramp for heatmaps.

Usage:
    python scripts/make_figures.py [--only example risk_real ...] [--target ieee|dmkd]
"""
import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.reward import Bank  # noqa: E402
from veritas.agent.selective import risk_coverage  # noqa: E402

OUT = ROOT / "paper" / "figs"
A, BK = ROOT / "results" / "agent", ROOT / "results" / "bank"
RED, BLUE, GREEN, PURPLE = "#c44e52", "#4c72b0", "#55a868", "#8172b2"
INK, INK2, MUTED, GRID, AXIS = "#000000", "#000000", "#7f7f7f", "#e5e5e5", "#000000"
BLUE_RAMP = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#dbe4f1", "#a9bedd", "#7495c4", "#4c72b0", "#2c4a7c"])
COL, DCOL = 3.45, 7.16  # IEEE column and page widths (inches)
DMKD = False  # set by --target dmkd: single-column Springer layout, 31 pc text width
FT = 10  # font size of panel titles and figure-level labels
HATCH = ["", "/", "\\", "x"]  # series 1 (proposed) plain, the others hatched, as in the reference figure

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 10,
    "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.labelcolor": "black", "axes.titlecolor": "black",
    "xtick.color": "black", "ytick.color": "black", "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "legend.frameon": False,
    "hatch.linewidth": 0.6, "hatch.color": "black",
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


SMALL = {"a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "per", "vs", "at", "by", "with", "from"}


def tc(s: str) -> str:
    """Title Case: capitalise every word except short words (for, to, of, ...), maths and group ids (g1)."""
    out = []
    for i, w in enumerate(re.split(r"([ \n])", s)):
        core = w.strip("(),:").lower()
        if w in (" ", "\n") or (i > 0 and core in SMALL) or w.startswith("$") or re.match(r"^\(?g\d", w) \
                or re.match(r"^\([a-z]\)$", w):
            out.append(w)
        else:
            out.append(re.sub(r"(^|[-(])([a-z])", lambda m: m.group(1) + m.group(2).upper(), w))
    return "".join(out)


def panel(ax, s: str, fontsize=None):
    """Panel label '(a)' centred above the frame."""
    ax.text(0.5, 1.02, s, transform=ax.transAxes, ha="center", va="bottom", fontsize=fontsize or FT + 1)


def bar_values(ax, bars, vals, tops, fmt, pad, fontsize=8.5):
    """Values printed vertically just above each bar (or its error bar)."""
    for b, v, t in zip(bars, vals, tops):
        ax.text(b.get_x() + b.get_width() / 2, t + pad, fmt.format(v), ha="center", va="bottom",
                rotation=90, fontsize=fontsize)


def top_legend(fig, handles, labels, y=1.0):
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(labels),
               handlelength=2.2, handleheight=1.1, columnspacing=1.6)


def load_preds(name: str, run: str) -> list[dict]:
    return sorted(json.loads((A / name / f"pred_{run}.json").read_text())["preds"], key=lambda p: p["bank_i"])


def title_case_all(fig):
    """Apply tc() to every axis label, title, legend entry, categorical tick label and annotation."""
    from matplotlib.ticker import FixedFormatter
    for ax in fig.axes:
        ax.set_xlabel(tc(ax.get_xlabel()))
        ax.set_ylabel(tc(ax.get_ylabel()))
        ax.set_title(tc(ax.get_title()))
        for axis in (ax.xaxis, ax.yaxis):
            f = axis.get_major_formatter()
            if isinstance(f, FixedFormatter):
                f.seq = [tc(s) for s in f.seq]
        for t in ax.texts:
            t.set_text(tc(t.get_text()))
        if ax.get_legend():
            for t in ax.get_legend().get_texts():
                t.set_text(tc(t.get_text()))
    for leg in fig.legends:
        for t in leg.get_texts():
            t.set_text(tc(t.get_text()))


def save(fig, name: str):
    title_case_all(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"-> {OUT / name}.pdf")


# --------------------------------------------------------------------------- example
def fig_example():
    import torch  # noqa: F401  (diffusion runs on the GPU)
    from veritas.data import load_dataset
    from veritas.grid import Claim, EvidenceGrid
    from veritas.verifier.diffusion import InpaintDiffusion
    from veritas.verifier.edits import edit

    ds = "synth_planted"
    ddir = ROOT / "results" / "diffusion" / ds / "standard"
    st = np.load(ddir / "stats.npz")
    data = load_dataset(ds, max_test=2000)
    Xtr = ((data.X_train - st["mu"]) / st["sd"]).astype(np.float32)
    Xte = ((data.X_test - st["mu"]) / st["sd"]).astype(np.float32)
    names = sorted(data.meta.channel_groups)
    grid = EvidenceGrid(T=Xte.shape[-1], channel_groups=data.meta.channel_groups, K=8)
    bank = Bank(BK / ds / "test.npz")
    classes = data.meta.class_names
    oracle_members = [pickle.loads((ddir / f"oracle_{m}.pkl").read_bytes()) for m in ("multirocket_hydra", "quant")]

    def proba(X):
        return np.mean([m.predict_proba(X.astype(np.float64)) for m in oracle_members], axis=0)

    # a correct, fully verified answer that cites the planted cell, for the class 'chirp'
    pick = None
    for p in load_preds(ds, "grpo_verified_div"):
        i = p["bank_i"]
        if (p["label"] == p["label_true"] and classes[p["label"]] == "chirp"
                and p["window"] == data.groups_test["ev_window"][i] and p["group"] == names[data.groups_test["ev_group"][i]]
                and bank.margin(i, bank.claim_id(p["window"], p["group"]), p["label"]) == 1.0):
            pick = p
            break
    i, y = pick["bank_i"], pick["label"]
    M = grid.mask(Claim((pick["window"],), (pick["group"],)))
    diff = InpaintDiffusion.load(ddir / "prior+learned.pt")
    rng = np.random.default_rng(3)
    target = np.array([classes.index("dip")])
    x = Xte[i:i + 1]
    x_nec = edit("hybrid", x, M[None], target, rng, pool=(Xtr, data.y_train), diffusion=diff, guidance=2.0, strength=0.5)
    x_suf = edit("hybrid", x, ~M[None], None, rng, pool=(Xtr, data.y_train), diffusion=diff, guidance=1.0, strength=0.5, seed=7)
    P = proba(np.concatenate([x, x_nec, x_suf]))
    t0, _ = grid.window_bounds(pick["window"])
    _, t1 = grid.window_bounds(pick["window"])
    cols = [("Original recording", x[0], P[0]), ("Test 1: cell replaced", x_nec[0], P[1]),
            ("Test 2: rest replaced", x_suf[0], P[2])]
    fig, axes = plt.subplots(3, 3, figsize=(DCOL, 3.1), sharex=True, sharey=True)
    for c, (title, xs, pr) in enumerate(cols):
        verdict = classes[int(pr.argmax())]
        for g, gname in enumerate(names):
            ax = axes[g, c]
            chans = grid.channel_groups[gname]
            if gname == pick["group"]:
                ax.axvspan(t0, t1, color="#f3d9da", lw=0, zorder=0)
            for k in range(1, grid.K):
                ax.axvline(grid.window_bounds(k)[0], color=GRID, lw=0.6, zorder=0)
            ax.plot(xs[chans[0]], color=BLUE, lw=0.8)
            ax.plot(xs[chans[1]], color=RED, lw=0.8, alpha=0.9)
            ax.grid(False)
            if c == 0:
                ax.set_ylabel(f"group\n{gname}", rotation=0, ha="center", va="center", labelpad=24)
        # three short lines per title, so neighbouring titles never run into each other
        mark = {0: f"agent answer: {classes[y]}", 1: "class changed", 2: "class kept"}[c]
        axes[0, c].set_title(f"({'abc'[c]}) {title}\noracle: {verdict} ({pr.max():.2f})\n{mark}", fontsize=FT)
        axes[-1, c].set_xlabel("Time step")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    top_legend(fig, [Line2D([], [], color=BLUE, lw=1.4), Line2D([], [], color=RED, lw=1.4),
                     Patch(facecolor="#f3d9da", edgecolor="none")],
               ["Channel 1", "Channel 2", f"Cited cell (window {pick['window']}, group {pick['group']})"], y=0.985)
    # one y-scale for all panels (standardized units), numbered on the left column
    axes[0, 0].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, integer=True))
    fig.subplots_adjust(top=0.80, hspace=0.12, wspace=0.05 if not DMKD else 0.14)
    save(fig, "fig_example")


# --------------------------------------------------------------------------- shared helpers for edit figures
def _edit_setup(source: str, bank_name: str):
    from veritas.data import load_dataset
    from veritas.verifier.diffusion import InpaintDiffusion
    ddir = ROOT / "results" / "diffusion" / source / "standard"
    st = np.load(ddir / "stats.npz")
    data = load_dataset(source, max_test=2000)
    Xtr = ((data.X_train - st["mu"]) / st["sd"]).astype(np.float32)
    Xte = ((data.X_test - st["mu"]) / st["sd"]).astype(np.float32)
    members = [pickle.loads((ddir / f"oracle_{m}.pkl").read_bytes()) for m in ("multirocket_hydra", "quant")]
    rows = np.load(BK / bank_name / "test.npz")["rows"]

    def proba(X):
        return np.mean([m.predict_proba(X.astype(np.float64)) for m in members], axis=0)
    return data, Xtr, Xte[rows], rows, InpaintDiffusion.load(ddir / "prior+learned.pt"), proba


def _two_tests(x, M, y, n_cls, Xtr, ytr, diff, proba, seed):
    """One Test-1 edit (towards another class) and one Test-2 edit; returns edits and oracle probabilities."""
    from veritas.verifier.edits import edit
    rng = np.random.default_rng(seed)
    target = np.array([(y + 1 + rng.integers(n_cls - 1)) % n_cls])
    x_nec = edit("hybrid", x[None], M[None], target, rng, pool=(Xtr, ytr), diffusion=diff, guidance=2.0, strength=0.5, seed=seed)
    x_suf = edit("hybrid", x[None], ~M[None], None, rng, pool=(Xtr, ytr), diffusion=diff, guidance=1.0, strength=0.5, seed=seed + 7)
    return x_nec[0], x_suf[0], proba(np.stack([x, x_nec[0], x_suf[0]]))


# --------------------------------------------------------------------------- real-data example
def fig_example_real():
    from veritas.grid import Claim, EvidenceGrid
    ds, source = "ucihar_subj_group", "ucihar_subj"
    data, Xtr, X, rows, diff, proba = _edit_setup(source, ds)
    classes, n_cls = data.meta.class_names, len(data.meta.class_names)
    groups = list(data.meta.channel_groups)
    cgrid = EvidenceGrid(T=X.shape[-1], channel_groups=data.meta.channel_groups, K=1, max_groups=2)
    bank = Bank(BK / ds / "test.npz")
    pick = None
    for p in load_preds(ds, "grpo_verified_ver"):
        i, y = p["bank_i"], p["label"]
        if (y != p["label_true"] or classes[y] not in ("walking upstairs", "walking downstairs", "walking")
                or bank.margin(i, bank.claim_id(0, p["group"]), y) < 1.0):
            continue
        M = cgrid.mask(Claim((0,), tuple(p["group"].split("+"))))
        x_nec, x_suf, P = _two_tests(X[i], M, y, n_cls, Xtr, data.y_train, diff, proba, seed=i)
        if P[0].argmax() == y and P[1].argmax() != y and P[2].argmax() == y:  # the fresh edits agree with the bank
            pick = (p, M, x_nec, x_suf, P)
            break
    p, M, x_nec, x_suf, P = pick
    y, cited = p["label"], p["group"].split("+")
    cols = [("Original recording", X[p["bank_i"]], P[0]), ("Test 1: sensors replaced", x_nec, P[1]),
            ("Test 2: other sensor replaced" if not DMKD else "Test 2: rest replaced", x_suf, P[2])]
    # one y-scale per sensor row, so the original and the two edits can be compared directly
    fig, axes = plt.subplots(len(groups), 3, figsize=(DCOL, 3.2), sharex=True, sharey="row")
    cols_xyz = [BLUE, RED, GREEN]
    for c, (title, xs, pr) in enumerate(cols):
        for g, gname in enumerate(groups):
            ax = axes[g, c]
            if gname in cited:
                ax.set_facecolor("#f3d9da")
            for k, ch in enumerate(data.meta.channel_groups[gname]):
                ax.plot(xs[ch], color=cols_xyz[k], lw=0.8)
            ax.grid(False)
            if c == 0:
                ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, integer=True))
                ax.set_ylabel(gname.replace("_", "\n"), rotation=0, ha="center", va="center", labelpad=26)
        mark = {0: f"agent answer: {classes[y]}", 1: "class changed", 2: "class kept"}[c]
        axes[0, c].set_title(f"({'abc'[c]}) {title}\noracle: {classes[int(pr.argmax())]} ({pr.max():.2f})\n{mark}", fontsize=FT)
        axes[-1, c].set_xlabel("Time step")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    top_legend(fig, [Line2D([], [], color=c_, lw=1.4) for c_ in cols_xyz] + [Patch(facecolor="#f3d9da", edgecolor="none")],
               ["x axis", "y axis", "z axis", "Cited sensor" + ("s" if len(cited) > 1 else "")], y=0.985)
    fig.subplots_adjust(top=0.80, hspace=0.12, wspace=0.05 if not DMKD else 0.14)
    save(fig, "fig_example_real")
    print("real example:", classes[y], cited, np.round(P.max(1), 2), [classes[k] for k in P.argmax(1)])


# --------------------------------------------------------------------------- failure cases
def fig_failures():
    from veritas.grid import Claim, EvidenceGrid
    ds = "synth_planted"
    data, Xtr, X, rows, diff, proba = _edit_setup(ds, ds)
    classes, n_cls = data.meta.class_names, len(data.meta.class_names)
    names = sorted(data.meta.channel_groups)
    grid = EvidenceGrid(T=X.shape[-1], channel_groups=data.meta.channel_groups, K=8)
    bank = Bank(BK / ds / "test.npz")
    pw, pg = data.groups_test["ev_window"][rows], data.groups_test["ev_group"][rows]
    cases = {}
    for p in load_preds(ds, "grpo_verified_div"):
        i, y = p["bank_i"], p["label"]
        if y != p["label_true"] or p["window"] is None:
            continue
        true = p["window"] == pw[i] and p["group"] == names[pg[i]]
        j = bank.claim_id(p["window"], p["group"])
        nec, suf = bank.pass_rates(i, j, y)
        kind = ("fail" if true and nec < 0.5 <= suf and classes[y] == "plateau" else
                "pass" if not true and bank.verified(i, j, y) else None)
        if kind is None or kind in cases:
            continue
        M = grid.mask(Claim((p["window"],), (p["group"],)))
        x_nec, x_suf, P = _two_tests(X[i], M, y, n_cls, Xtr, data.y_train, diff, proba, seed=i)
        agrees = (P[1].argmax() == y and P[2].argmax() == y) if kind == "fail" else (P[1].argmax() != y and P[2].argmax() == y)
        if agrees:
            cases[kind] = (p, x_nec, x_suf, P)
        if len(cases) == 2:
            break
    fig, axes = plt.subplots(2, 3, figsize=(DCOL, 3.9), sharex=True, sharey=True)
    rowt = {"fail": "False fail: true cell, Test 1 fails", "pass": "False pass: wrong cell passes"}
    for r, kind in enumerate(("fail", "pass")):
        p, x_nec, x_suf, P = cases[kind]
        i = p["bank_i"]
        for c, (title, xs, pr) in enumerate((("Original", X[i], P[0]), ("Test 1", x_nec, P[1]), ("Test 2", x_suf, P[2]))):
            ax = axes[r, c]
            for g, gname in enumerate(names):
                off = -4.0 * g
                s = xs[data.meta.channel_groups[gname]].mean(0)
                ax.plot(s + off, color=INK, lw=0.7)
                for (w, gg, col, hatch) in ((p["window"], p["group"], "#f3d9da", None), (pw[i], names[pg[i]], "none", "////")):
                    if gname == gg:
                        t0, t1 = grid.window_bounds(int(w))
                        ax.add_patch(plt.Rectangle((t0, off - 2), t1 - t0, 4, facecolor=col, edgecolor=GREEN if hatch else "none",
                                                   hatch=hatch, lw=0.8, zorder=0))
            if c == 0:  # the y-axis is shared, so the group names are set once and shown on the left column
                ax.set_yticks([-4.0 * g for g in range(len(names))], names)
            ax.grid(False)
            verdict = classes[int(pr.argmax())]
            ax.set_title(f"({'abcdef'[3 * r + c]}) {title}\noracle: {verdict} ({pr.max():.2f})", fontsize=FT)
        axes[r, 1].annotate(rowt[kind] + f" (class {classes[p['label']]})", (0.5, 1.40), xycoords="axes fraction",
                            ha="center", fontsize=FT, fontweight="bold")
    for ax in axes[-1]:
        ax.set_xlabel("Time step")
    from matplotlib.patches import Patch
    top_legend(fig, [Patch(facecolor="#f3d9da", edgecolor="none"), Patch(facecolor="none", edgecolor=GREEN, hatch="////")],
               ["Cited cell", "Planted (true) cell"], y=0.985)
    fig.subplots_adjust(top=0.851, hspace=0.75, wspace=0.06)
    save(fig, "fig_failures")


# --------------------------------------------------------------------------- risk-coverage
def scores_for(name: str, run: str, votes_name: str | None = None):
    bank = Bank(BK / name / "test.npz")
    P = load_preds(name, run)
    votes = np.load(A / (votes_name or name) / "features.npz")["test_votes"]
    yh = np.array([p["label"] for p in P])
    y = np.array([p["label_true"] for p in P])
    i = np.array([p["bank_i"] for p in P])
    conf = bank.p0[i, yh]
    marg = np.array([bank.margin(ii, bank.claim_id(p["window"], p["group"]), p["label"]) for ii, p in zip(i, P)])
    vote = votes[i, yh] / votes[i].sum(1)
    return {"verification": marg + 1e-3 * conf, "oracle confidence": conf, "neighbour votes": vote}, yh != y


RISK_STYLES = {"verification": (RED, "-", "o", "Check score (proposed)"),
               "oracle confidence": (BLUE, "--", "s", "Oracle confidence"),
               "neighbour votes": (GREEN, ":", "^", "Neighbor votes")}


def risk_panel(ax, name, run, votes_name=None):
    """Draw the three risk-coverage curves; return {score: (area under risk curve, coverage at 5% error)}."""
    S, wrong = scores_for(name, run, votes_name)
    out = {}
    ax.top_risk, ax.top_all = 0.0, 0.0
    for label, s in S.items():
        cov, risk = risk_coverage(s, wrong)
        ax.top_risk = max(ax.top_risk, float(np.max(risk[cov >= 0.05])) * 100)  # ignoring the first 5 % of coverage
        ax.top_all = max(ax.top_all, float(np.max(risk)) * 100)  # the whole curve
        col, ls, mk, _ = RISK_STYLES[label]
        ax.plot(cov * 100, risk * 100, color=col, ls=ls, marker=mk, markevery=max(1, len(cov) // 8),
                markersize=5, markeredgecolor="black", markeredgewidth=0.5)
        ok = cov[risk <= 0.05]
        out[label] = (float(np.mean(risk)), float(ok.max()) if len(ok) else 0.0)
    ax.axhline(5, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.set_xlim(0, 100)
    return out, float(np.mean(wrong))


def risk_legend(fig, y):
    from matplotlib.lines import Line2D
    top_legend(fig, [Line2D([], [], color=c, ls=ls, marker=mk, markeredgecolor="black", markeredgewidth=0.5)
                     for c, ls, mk, _ in RISK_STYLES.values()], [v[3] for v in RISK_STYLES.values()], y=y)


def fig_risk_real():
    """Risk-coverage curves on the synthetic data (in distribution and shifted) and on the real datasets,
    including two real subject shifts."""
    panels = [("(a) Synthetic\n(no shift)","synth_planted", "grpo_verified_div"),
              ("(b) Synthetic\n(shifted)", "synth_planted_shift", "grpo_verified_div"),
              ("(c) PAMAP2\n(same people)", "pamap2_group", "grpo_verified_ver"),
              ("(d) PTB-XL\n(ECG)", "ptbxl_group", "grpo_verified_ver"),
              ("(e) UCI-HAR\n(new people)", "ucihar_subj_group_shift", "grpo_verified_ver"),
              ("(f) PAMAP2\n(new people)", "pamap2_subj_group_shift", "grpo_verified_ver")]
    if DMKD:  # two rows of three on the narrower single-column page
        fig, axes = plt.subplots(2, 3, figsize=(DCOL, 4.6))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 6, figsize=(DCOL, 2.45))
    summary = {}
    for ax, (title, name, run) in zip(axes, panels):
        res, base = risk_panel(ax, name, run)
        summary[name] = {"base_error": base, **{k: {"aurc": a, "cov_at_5": c} for k, (a, c) in res.items()}}
        # tight y-range: just above the highest curve (ignoring the first 5% of coverage) and the 5% line
        step = 2 if ax.top_risk < 12 else 5
        ax.set_ylim(0, step * np.ceil(max(6.0, ax.top_risk * 1.08) / step))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(step if ax.top_risk < 12 else 10))
        panel(ax, title, fontsize=FT)
        ax.set_xticks([0, 50, 100])
    for ax in axes[::3 if DMKD else 6]:
        ax.set_ylabel("Error among answered (%)")
    fig.supxlabel(tc("Coverage: share of recordings answered (%)"), y=-0.04 if not DMKD else 0.0, fontsize=FT)
    risk_legend(fig, 1.03 if not DMKD else 0.985)
    if DMKD:
        fig.subplots_adjust(wspace=0.32, hspace=0.62, top=0.86)
    else:
        fig.subplots_adjust(wspace=0.42)
    save(fig, "fig_risk_real")
    if DMKD:
        return  # the summary is written by the IEEE run
    (ROOT / "results" / "risk_real_summary.json").write_text(json.dumps(summary, indent=1))
    for k, v in summary.items():
        print(k, json.dumps(v))


# --------------------------------------------------------------------------- main bars
def fig_main():
    seeds = json.loads((ROOT / "results" / "seeds_summary.json").read_text())

    def single(name, run, key):
        r = json.loads((A / name / "summary.json").read_text())["runs"][run]
        return {"verified": r["verified_rate"], "accuracy": r["accuracy"]}[key]

    rows = [("Synthetic", "synth_planted", "GRPO accuracy reward", "GRPO check reward", None, None),
            ("PAMAP2", "pamap2_group", "GRPO accuracy reward", "GRPO check reward", None, None),
            ("PTB-XL", "ptbxl_group", "GRPO accuracy reward", "GRPO check reward", None, None),
            ("UCI-HAR", "ucihar_subj_group", "GRPO accuracy reward", "GRPO check reward", None, None),
            ("NATOPS", "natops_group", "GRPO accuracy reward", "GRPO check reward", None, None),
            ("Cricket", "cricket_group", "GRPO accuracy reward", "GRPO check reward", None, None)]
    # (a) and (b) side by side: half the height of a stacked layout
    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.45))
    series =[("Check reward (proposed)", RED, HATCH[0], 1), ("Accuracy reward", BLUE, HATCH[1], 0)]
    handles = []
    for ax, key, ylab, pl in [(axes[0], "verified", "Checked rate (%)", "(a)"), (axes[1], "accuracy", "Accuracy (%)", "(b)")]:
        x = np.arange(len(rows))
        for j, (lab, col, hatch, use_check) in enumerate(series):
            m, e = [], []
            for disp, name, sa, sv, ra, rv in rows:
                if sa:
                    d = seeds[name][sv if use_check else sa][key]
                    m.append(d["mean"] * 100)
                    e.append(d["sd"] * 100)
                else:
                    m.append(single(name, rv if use_check else ra, key) * 100)
                    e.append(0)
            xs = x + (j - 0.5) * 0.4
            bars = ax.bar(xs, m, 0.4, color=col, hatch=hatch, edgecolor="black", linewidth=0.8)
            if ax is axes[0]:
                handles.append(bars)
            has = [k for k, v in enumerate(e) if v > 0]  # error bars only where there are several seeds
            ax.errorbar(xs[has], np.array(m)[has], yerr=np.array(e)[has], fmt="none", elinewidth=0.8, capsize=2, ecolor="black")
            bar_values(ax, bars, m, np.array(m) + np.array(e), "{:.1f}", 2.5, fontsize=8)  # clear of the error-bar caps
        ax.set_xticks(x, [r[0] for r in rows], fontsize=9 if not DMKD else 8.5, rotation=30 if not DMKD else 40,
                      ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_xlabel("Dataset")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 120)  # headroom for the printed values; ticks stop at 100 %
        ax.set_yticks(range(0, 101, 20))
        panel(ax, pl)
    top_legend(fig, handles, [s[0] for s in series], y=0.97)
    fig.subplots_adjust(wspace=0.22 if not DMKD else 0.3)
    save(fig, "fig_main_bars")


# --------------------------------------------------------------------------- edit methods
def fig_edits():
    """Edit methods as three aligned horizontal bar panels: AUC (higher is better), realism
    (detector accuracy, lower is better, 0.5 = cannot tell) and channel correlation inside the edit."""
    from matplotlib.patches import Patch
    # (name, group, AUC, detector accuracy, channel correlation or None)
    # mean over three seeds of 100 recordings (results/review6_eval.json, verifier_ci)
    rows = [("Set to zero", "Masking", 0.63, 0.95, None),
            ("Fill with noise", "Masking", 0.52, 0.86, 0.00),
            ("Straight line", "Masking", 0.63, 0.85, 0.86),
            ("Copy alone", "Real content", 0.82, 0.64, 0.96),
            ("Diffusion, no graph", "Diffusion", 0.68, 0.73, 0.05),
            ("Diffusion, graph", "Diffusion", 0.78, 0.67, 0.71),
            ("Copy + diffusion, no graph", "Copy + diffusion", 0.83, 0.68, 0.73),
            ("Copy + diffusion, graph (proposed)", "Copy + diffusion", 0.86, 0.66, 0.91)]
    # y positions, top to bottom: a header row for each group, then its methods
    ys, heads, y, prev = [], [], 0.0, None
    for r in rows:
        if r[1] != prev:
            if prev is not None:
                y += 0.3
            heads.append((r[1], -y))
            y += 0.85
        ys.append(y)
        y += 1.0
        prev = r[1]
    ys = -np.array(ys)
    panels = [("(a) Finds the true evidence", "AUC $\\uparrow$", 2, (0.4, 1.0), [(0.5, "chance")]),
              ("(b) Looks real", "Detector accuracy $\\downarrow$", 3, (0.4, 1.0), [(0.5, "chance")]),
              ("(c) Keeps channels coupled", "Channel correlation $\\uparrow$", 4, (-0.1, 1.05), [(0.98, "real data")])]
    if DMKD:  # the narrow DMKD page: two-line titles and axis labels
        panels = [("(a) Finds the\ntrue evidence", "AUC $\\uparrow$", 2, (0.4, 1.0), [(0.5, "chance")]),
                  ("(b) Looks real", "Detector\naccuracy $\\downarrow$", 3, (0.4, 1.0), [(0.5, "chance")]),
                  ("(c) Keeps channels\ncoupled", "Channel\ncorrelation $\\uparrow$", 4, (-0.1, 1.05), [(0.98, "real data")])]
    # the method names stand outside the figure box, so on the narrow DMKD page the box itself is narrower
    fig, axes = plt.subplots(1, 3, figsize=(DCOL if not DMKD else DCOL - 1.5, 2.8 if not DMKD else 3.3), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, (title, xlab, k, xlim, refs) in zip(axes, panels):
        for yy, r in zip(ys, rows):
            ours = "proposed" in r[0]
            v = r[k]
            if v is None:
                ax.text(xlim[0] + 0.02, yy, "--", va="center", ha="left", fontsize=8.5, color=MUTED)
                continue
            ax.barh(yy, v - xlim[0], left=xlim[0], height=0.75, color=RED if ours else BLUE,
                    hatch="" if ours else "/", edgecolor="black", linewidth=0.8, zorder=2)
            ax.text(max(v, xlim[0]) + 0.012 * (xlim[1] - xlim[0]), yy, f"{v:.2f}", va="center", ha="left",
                    fontsize=8.5, fontweight="bold" if ours else "normal")
        for ref, lab in refs:
            ax.axvline(ref, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
            ax.text(ref, heads[0][1] + 0.55, lab, ha="center", va="bottom", fontsize=8, color=MUTED, zorder=3,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8))  # hides the dashed line behind the label
        ax.set_xlim(xlim[0], xlim[1] + (0.12 if not DMKD else 0.2) * (xlim[1] - xlim[0]))
        ax.set_xlabel(xlab)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", length=0)
        panel(ax, title)
    ticks = list(ys) + [h[1] for h in heads]
    labels = [r[0] for r in rows] + [h[0] for h in heads]
    axes[0].set_yticks(ticks, labels)
    for lab in axes[0].get_yticklabels()[len(rows):]:  # group headers
        lab.set_fontweight("bold")
        lab.set_fontstyle("italic")
    axes[0].set_ylim(ys[-1] - 0.7, heads[0][1] + 1.3)
    for ax in axes:  # thin separators above each group header
        for _, hy in heads[1:]:
            ax.axhline(hy + 0.55, color=GRID, lw=0.8, zorder=0)
    top_legend(fig, [Patch(facecolor=RED, edgecolor="black"), Patch(facecolor=BLUE, hatch="/", edgecolor="black")],
               ["Copy + diffusion with channel graph (proposed)" if not DMKD else "Copy + diffusion, graph (proposed)",
                "Other edit methods"], y=0.97)
    if DMKD:
        fig.subplots_adjust(top=0.8)
    save(fig, "fig_edits")

# --------------------------------------------------------------------------- sensitivity
def fig_sens():
    def auc(tag):
        f = ROOT / "results" / "verifier" / f"synth_planted_eval_seed0{tag}.json"
        if not f.exists():
            return None
        r = json.loads(f.read_text())["methods"]["hybrid:prior+learned"]
        return float(np.mean(list(r["auc_combined"].values()))), float(r["discriminative_acc"])
    base = auc("_hybrid_graph_ablation")
    sweeps = [("noise strength $s$", [0.25, 0.5, 0.75, 1.0], lambda v: "_sens_s%s_w2_m2" % v, 0.5),
              ("guidance weight $w$", [1, 2, 4], lambda v: "_sens_s0.5_w%s_m2" % v, 2),
              ("edits per test $m$", [1, 2, 4], lambda v: "_sens_s0.5_w2_m%s" % v, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.35), sharey=True)
    for ax, pl, (lab, vals, tagf, default) in zip(axes, "abc", sweeps):
        xs, ys = [], []
        for v in vals:
            r = base if v == default else auc(tagf(v))
            if r:
                xs.append(v)
                ys.append(r[0])
        ax.plot(xs, ys, color=BLUE, marker="o", markeredgecolor="black", markeredgewidth=0.5, zorder=3)
        if default in xs:
            ax.plot([default], [ys[xs.index(default)]], ls="none", marker="*", markersize=11, color=RED,
                    markeredgecolor="black", markeredgewidth=0.6, zorder=4)
        for xv, yv in zip(xs, ys):
            ax.text(xv, yv + 0.009, f"{yv:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xlabel(lab[0].upper() + lab[1:])
        ax.set_xticks(vals)
        span = vals[-1] - vals[0]
        ax.set_xlim(vals[0] - 0.12 * span, vals[-1] + 0.12 * span)
        ax.grid(True, axis="both")
        panel(ax, f"({pl})")
    axes[0].set_ylabel("AUC (true vs wrong evidence)")
    axes[0].set_ylim(0.75, 0.96)
    from matplotlib.lines import Line2D
    top_legend(fig, [Line2D([], [], color=BLUE, marker="o", markeredgecolor="black", markeredgewidth=0.5),
                     Line2D([], [], ls="none", marker="*", markersize=11, color=RED, markeredgecolor="black", markeredgewidth=0.6)],
               ["Checker AUC", "Default setting ($s=0.5$, $w=2$, $m=2$)"], y=0.95)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig_sensitivity")


# --------------------------------------------------------------------------- interpretability
def fig_interpret():
    from veritas.data import load_dataset
    if DMKD:  # (a) above (b): side by side, the activity names of (b) run into (a)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DCOL, 5.6), gridspec_kw={"height_ratios": [1, 1.5]})
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DCOL, 2.8), gridspec_kw={"width_ratios": [1.15, 1.3]})
    # (a) per-class true-evidence rate on the synthetic data
    d = load_dataset("synth_planted", max_test=2000)
    names = sorted(d.meta.channel_groups)
    classes = d.meta.class_names
    rows = np.load(BK / "synth_planted" / "test.npz")["rows"]
    w, g = d.groups_test["ev_window"][rows], d.groups_test["ev_group"][rows]
    runs = [("Check reward (proposed)", "grpo_verified_div", RED, HATCH[0]),
            ("Accuracy reward", "grpo_accuracy_div", BLUE, HATCH[1]),
            ("Warm start", "sft_diverse", GREEN, HATCH[2])]
    x = np.arange(len(classes))
    for j, (lab, run, col, hatch) in enumerate(runs):
        P = load_preds("synth_planted", run)
        rates = []
        for c in range(len(classes)):
            ok = [p for p in P if p["label_true"] == c and p["label"] == c]
            rates.append(100 * np.mean([p["window"] == w[p["bank_i"]] and p["group"] == names[g[p["bank_i"]]] for p in ok]))
        bars = ax1.bar(x + (j - 1) * 0.27, rates, 0.27, color=col, hatch=hatch, label=lab, edgecolor="black", linewidth=0.8)
        bar_values(ax1, bars, rates, rates, "{:.0f}", 1.5)
    ax1.set_xticks(x, [c.capitalize() for c in classes])
    ax1.set_xlabel("Class")
    ax1.set_ylabel("True-evidence rate (%)")
    ax1.set_ylim(0, 112)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    h, lb = ax1.get_legend_handles_labels()
    top_legend(fig, h, lb, y=0.95)
    panel(ax1, "(a)")

    # (b) PAMAP2: which sensors the check-reward agent cites, per activity
    meta = load_dataset("pamap2", max_test=2000).meta
    groups = sorted(meta.channel_groups)
    acts = meta.class_names
    H = np.zeros((len(acts), len(groups)))
    n = np.zeros(len(acts))
    for p in load_preds("pamap2_group", "grpo_verified_ver"):
        if p["label"] != p["label_true"] or not p["group"]:
            continue
        n[p["label"]] += 1
        for gname in p["group"].split("+"):
            H[p["label"], groups.index(gname)] += 1
    H = 100 * H / np.maximum(n[:, None], 1)
    im = ax2.imshow(H, cmap=BLUE_RAMP, vmin=0, vmax=100, aspect="auto")
    ax2.set_xticks(range(len(groups)), [tc(s.replace("_", " ")) for s in groups], rotation=35, ha="right")
    ax2.set_yticks(range(len(acts)), [tc(a) for a in acts])
    ax2.grid(False)
    for (r, c), v in np.ndenumerate(H):
        if v >= 15:
            ax2.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=8.5, color="white" if v > 60 else INK)
    ax2.set_xlabel("Sensor cited")
    panel(ax2, "(b)")
    cb = fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)
    cb.outline.set_linewidth(0.8)
    cb.ax.tick_params(labelsize=9.5)
    cb.set_label("Share of correct answers (%)", fontsize=FT)
    if DMKD:
        fig.subplots_adjust(hspace=0.42, top=0.9)
    else:
        fig.subplots_adjust(wspace=0.55)
    save(fig, "fig_interpret")


FIGS = {"example": fig_example, "main": fig_main, "edits": fig_edits,
        "sensitivity": fig_sens, "interpret": fig_interpret, "risk_real": fig_risk_real,
        "example_real": fig_example_real, "failures": fig_failures}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(FIGS))
    ap.add_argument("--target", choices=["ieee", "dmkd"], default="ieee")
    args = ap.parse_args()
    if args.target == "dmkd":  # Springer single column, 31 pc = 5.15 in, slightly smaller type
        global DMKD, DCOL, OUT, FT
        DMKD, DCOL, OUT, FT = True, 5.15, ROOT / "paper_dmkd" / "figs_redrawn", 9  # the paper uses the IEEE figures
        plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                             "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 9})
    for k in args.only:
        FIGS[k]()


if __name__ == "__main__":
    main()
