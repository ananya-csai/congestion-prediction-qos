"""Step 6 - figures for the paper, drawn ONLY from recorded output files.

    python make_figures.py        (run analyze.py first)

Writes PDF (for LaTeX) and PNG (for preview) files to outputs/figures/.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

import config as C
from features import run_tag

FIG = C.OUT_DIR / "figures"
RES = C.OUT_DIR / "results"
EXP = json.loads((C.ROOT / "experiments.json").read_text())
LOADS = list(EXP["loads"])
TH = EXP["theta"]

# fixed categorical order (one colour per entity, never re-assigned)
COL = {"predictive": "#2a78d6", "reactive": "#eb6834", "static": "#1baf7a",
       "oracle": "#eda100", "predictive_global": "#e87ba4",
       "rf": "#2a78d6", "logreg": "#eb6834", "persistence": "#1baf7a"}
LAB = {"static": "Static SP", "reactive": "Reactive", "predictive": "Predictive (RF)",
       "oracle": "Oracle", "predictive_global": "RF + global reweight",
       "rf": "Random Forest", "logreg": "Logistic reg.", "persistence": "Persistence"}
DSN = {"abilene": "Abilene", "geant": "GÉANT"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid.axis": "y", "savefig.dpi": 200,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def grouped_bars(ax, df, metric, methods, scale=1.0, ci=True):
    x = np.arange(len(LOADS))
    wdt = 0.8 / len(methods)
    for i, m in enumerate(methods):
        d = df[df.method == m].set_index("load").reindex(LOADS)
        y = d[metric].values * scale
        err = None
        if ci and f"{metric}_ci_low" in d:
            lo = d[f"{metric}_ci_low"].values * scale
            hi = d[f"{metric}_ci_high"].values * scale
            err = np.vstack([y - lo, hi - y])
        ax.bar(x + (i - (len(methods) - 1) / 2) * wdt, y, wdt * 0.92, color=COL[m],
               label=LAB[m], yerr=err, error_kw=dict(lw=0.7, capsize=1.5, ecolor=MUTED))
    ax.set_xticks(x, [f"{l}\n(×{EXP['loads'][l]})" for l in LOADS])


def fig_routing():
    S = pd.read_csv(RES / "routing_summary.csv")
    S = S[S.horizon == EXP["main_horizon"]]
    methods = ["static", "reactive", "predictive", "oracle"]
    panels = [("mlu", "mean MLU", 1, "MLU (ratio, 1 = full)"),
              ("congested", "congested intervals", 100, "Intervals (%)"),
              ("loss", "loss rate", 100, "Loss (%)"),
              ("mean_delay", "mean delay", 1, "Delay (× empty-link delay)")]
    fig, axes = plt.subplots(len(EXP["datasets"]), 4, figsize=(7.2, 3.6), sharex=True)
    for r, ds in enumerate(EXP["datasets"]):
        for c, (met, title, sc, ylab) in enumerate(panels):
            ax = axes[r, c]
            grouped_bars(ax, S[S.dataset == ds], met, methods, sc)
            ax.set_title(f"{DSN[ds]}: {title}", fontsize=7.5)
            ax.set_ylabel(ylab, fontsize=6.5)
            if r == len(EXP["datasets"]) - 1:
                ax.set_xlabel("Traffic load (demand multiplier)", fontsize=6.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    save(fig, "fig_routing_by_load")


def fig_prediction():
    P = pd.read_csv(RES / "prediction_summary.csv")
    P = P[P.horizon == EXP["main_horizon"]].rename(columns={"model": "method"})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), sharey=True)
    for ax, ds in zip(axes, EXP["datasets"]):
        d = P[P.dataset == ds].copy()
        has_sd = d["f1_sd"].notna().any()
        if has_sd:
            d["f1_ci_low"] = d["f1"] - d["f1_sd"].fillna(0)
            d["f1_ci_high"] = d["f1"] + d["f1_sd"].fillna(0)
        grouped_bars(ax, d, "f1", ["persistence", "logreg", "rf"], ci=has_sd)
        ax.set_title(f"{DSN[ds]}: test F1 by load (h=1)")
        lo = max(0, np.floor((d["f1"].min() - 0.08) * 20) / 20)
        ax.set_ylim(lo, 1.0)
    axes[0].set_ylabel("F1 score (test period)")
    for ax in axes:
        ax.set_xlabel("Traffic load (demand multiplier)")
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_prediction_f1")


def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, ds in zip(axes, EXP["datasets"]):
        out = C.OUT_DIR / run_tag(ds, EXP["main_horizon"], TH, EXP["loads"]["medium"])
        z = np.load(out / f"test_probabilities_seed{EXP['seeds'][0]}.npz")
        for m in ["rf", "logreg"]:
            pr, rc, _ = precision_recall_curve(z["y"], z[m])
            ax.plot(rc, pr, color=COL[m], lw=1.6, label=LAB[m])
        y, p = z["y"], z["persistence"]
        tp = ((p == 1) & (y == 1)).sum()
        ax.plot(tp / y.sum(), tp / max(p.sum(), 1), "o", ms=6, color=COL["persistence"],
                mec="white", mew=1, label="Persistence (single point)")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{DSN[ds]}: precision–recall (medium load, h=1)")
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.01)
        ax.grid(True, axis="both")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_pr_curves")


def fig_importance():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, ds in zip(axes, EXP["datasets"]):
        out = C.OUT_DIR / run_tag(ds, EXP["main_horizon"], TH, EXP["loads"]["medium"])
        imp = pd.read_csv(out / f"feature_importance_seed{EXP['seeds'][0]}.csv", index_col=0)
        imp = imp["importance"].sort_values()
        ax.barh(imp.index, imp.values, color=COL["rf"], height=0.7)
        ax.set_title(f"{DSN[ds]}: RF feature importance")
        ax.set_xlabel("Importance (mean decrease in Gini impurity)")
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", labelsize=6.5)
    fig.tight_layout()
    save(fig, "fig_feature_importance")


def fig_timeseries():
    ds = EXP["datasets"][0]
    out = C.OUT_DIR / run_tag(ds, EXP["main_horizon"], TH, EXP["loads"]["medium"])
    df = pd.read_csv(out / f"routing_steps_seed{EXP['seeds'][0]}_alpha{EXP['alpha']:g}_m{EXP['margin']:g}.csv")
    spd = 1440 // C.DATASETS[ds]["step_min"]
    piv = df.pivot(index="t", columns="method", values="mlu")
    day = piv.index // spd
    # rule fixed in advance: among full test days without anomalies (static MLU <= 3 all day),
    # show the one where static routing has the most congested intervals
    full = day.value_counts()
    ok = [d for d in full.index if full[d] == spd and piv["static"][day == d].max() <= 3]
    cong = {d: (piv["static"][day == d] >= TH).sum() for d in ok}
    best = max(cong, key=cong.get)
    w = piv[day == best]
    hrs = (w.index - w.index[0]) * C.DATASETS[ds]["step_min"] / 60
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    for m in ["static", "reactive", "predictive"]:
        ax.plot(hrs, w[m].values, color=COL[m], lw=1.1, label=LAB[m])
    ax.axhline(TH, color=MUTED, lw=0.8, ls="--")
    ax.text(hrs.max(), TH, " θ", va="center", color=MUTED)
    ax.set_xlabel("Hour of the selected test day")
    ax.set_ylabel("Max. link utilisation")
    ax.set_title(f"{DSN[ds]}, medium load: test day with most congestion under static routing")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_mlu_timeseries")


def fig_ablation():
    S = pd.read_csv(RES / "routing_summary.csv")
    S = S[(S.horizon == EXP["main_horizon"]) & (S.load == "medium")]
    methods = ["static", "reactive", "predictive", "predictive_global", "oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
    for ax, ds in zip(axes, EXP["datasets"]):
        d = S[S.dataset == ds].set_index("method").reindex(methods)
        y = 100 * d["congested"].values
        lo, hi = 100 * d["congested_ci_low"].values, 100 * d["congested_ci_high"].values
        ax.bar(range(len(methods)), y, 0.7, color=[COL[m] for m in methods],
               yerr=np.vstack([y - lo, hi - y]), error_kw=dict(lw=0.7, capsize=2, ecolor=MUTED))
        for i, v in enumerate(y):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_xticks(range(len(methods)), [LAB[m].replace(" + ", "\n+ ") for m in methods], fontsize=6.5)
        ax.set_title(f"{DSN[ds]}: medium load (×1.0), h=1")
        ax.set_ylabel("Intervals with congestion (%)")
    fig.tight_layout()
    save(fig, "fig_ablation_routing")


if __name__ == "__main__":
    for f in [fig_prediction, fig_curves, fig_routing, fig_ablation, fig_importance, fig_timeseries]:
        try:
            f()
        except FileNotFoundError as e:
            print(f"skip {f.__name__}: missing {e.filename}")
