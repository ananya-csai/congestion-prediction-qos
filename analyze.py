"""Step 5 - statistics and LaTeX tables from the recorded outputs.

    python analyze.py            # filled tables from outputs/
    python analyze.py --empty    # empty table templates (same layout, '--' cells)

Statistical unit = one DAY of the test period ("block"). Consecutive
5/15-minute intervals are strongly correlated, so treating every interval
as an independent sample would exaggerate significance. For each day we
average a metric over its intervals (and over the RF seeds), then compare
methods with a paired two-sided Wilcoxon signed-rank test across days.
P-values within each configuration are Holm-corrected. 95% confidence
intervals of mean differences use a bootstrap over days (10,000 resamples).

Outputs: outputs/results/*.csv and outputs/results/tables/*.tex
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

import config as C
from features import run_tag

RES = C.OUT_DIR / "results"
TAB = RES / "tables"
EXP = json.loads((C.ROOT / "experiments.json").read_text())
LOADS = EXP["loads"]
SEEDS = EXP["seeds"]
TH, AL, MG = EXP["theta"], EXP["alpha"], EXP["margin"]
MAIN_METHODS = ["static", "reactive", "predictive", "oracle"]
NAMES = {"static": "Static SP", "reactive": "Reactive", "predictive": "Predictive (RF)",
         "oracle": "Oracle", "predictive_global": "RF + global reweight",
         "persistence": "Persistence", "logreg": "Logistic reg.", "rf": "Random Forest"}
DSN = {"abilene": "Abilene", "geant": "G\\'EANT"}
RNG = np.random.default_rng(12345)


def configs():
    out = []
    for ds in EXP["datasets"]:
        for lname, sc in LOADS.items():
            out.append((ds, EXP["main_horizon"], sc, lname))
        eh = EXP.get("extra_horizon")
        if eh:
            out.append((ds, eh["horizon"], LOADS[eh["load"]], eh["load"]))
    return out


def steps_per_day(ds):
    return 1440 // C.DATASETS[ds]["step_min"]


def boot_ci(x, n=10000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def holm(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pv[i])
        adj[i] = min(1.0, run)
    out[ok] = adj
    return out


def load_routing(ds, h, sc):
    out = C.OUT_DIR / run_tag(ds, h, TH, sc)
    frames = []
    for s in SEEDS:
        f = out / f"routing_steps_seed{s}_alpha{AL:g}_m{MG:g}.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["seed"] = s
            frames.append(d)
    return pd.concat(frames) if frames else None


# ---------------------------------------------------------------- prediction
def prediction_tables():
    rows = []
    for ds, h, sc, lname in configs():
        out = C.OUT_DIR / run_tag(ds, h, TH, sc)
        for s in SEEDS:
            f = out / f"metrics_prediction_seed{s}.csv"
            if f.exists():
                d = pd.read_csv(f)
                d["dataset"], d["horizon"], d["load"], d["scale"] = ds, h, lname, sc
                rows.append(d)
    if not rows:
        return None
    df = pd.concat(rows)
    agg = df.groupby(["dataset", "horizon", "load", "scale", "model"]).agg(
        n_seeds=("seed", "nunique"),
        precision=("precision", "mean"), precision_sd=("precision", "std"),
        recall=("recall", "mean"), recall_sd=("recall", "std"),
        f1=("f1", "mean"), f1_sd=("f1", "std"),
        roc_auc=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
        pr_auc=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
    ).reset_index()
    agg.to_csv(RES / "prediction_summary.csv", index=False)
    return agg


def prediction_block_test():
    """Per-day F1: Random Forest vs persistence (and vs logistic regression)."""
    rows = []
    for ds, h, sc, lname in configs():
        out = C.OUT_DIR / run_tag(ds, h, TH, sc)
        per_seed = []
        for s in SEEDS:
            f = out / f"test_probabilities_seed{s}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            tau = float(z["tau_rf"])
            m = pd.read_csv(out / f"metrics_prediction_seed{s}.csv").set_index("model")
            d = pd.DataFrame({"day": z["t"] // steps_per_day(ds), "y": z["y"],
                              "rf": z["rf"] >= tau,
                              "logreg": z["logreg"] >= m.loc["logreg", "tau"],
                              "persistence": z["persistence"] >= 0.5})
            g = d.groupby("day")
            f1 = pd.DataFrame({k: g.apply(lambda x, k=k: f1_score(x["y"], x[k], zero_division=np.nan)
                                          if x["y"].sum() > 0 else np.nan)
                               for k in ["rf", "logreg", "persistence"]})
            per_seed.append(f1)
        if not per_seed:
            continue
        f1 = sum(per_seed) / len(per_seed)
        f1 = f1.dropna()
        for other in ["persistence", "logreg"]:
            diff = f1["rf"] - f1[other]
            p = wilcoxon(diff).pvalue if (diff != 0).any() and len(diff) >= 6 else np.nan
            lo, hi = boot_ci(diff)
            rows.append({"dataset": ds, "horizon": h, "load": lname, "comparison": f"rf - {other}",
                         "metric": "daily F1", "n_days": len(diff), "mean_diff": diff.mean(),
                         "ci_low": lo, "ci_high": hi, "p": p})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = np.nan
        for key, g in df.groupby(["dataset", "horizon", "load"]):
            df.loc[g.index, "p_holm"] = holm(g["p"].values)
    df.to_csv(RES / "stats_prediction.csv", index=False)
    return df


# ---------------------------------------------------------------- routing
ROUTE_METRICS = {"mlu": "mean MLU", "congested": "congested-interval share",
                 "mean_delay": "mean delay (norm.)", "loss": "loss rate"}


def daily_blocks(df, ds):
    df = df.copy()
    df["day"] = df["t"] // steps_per_day(ds)
    df["congested"] = (df["n_congested"] > 0).astype(float)
    cols = ["mlu", "congested", "mean_delay", "loss", "pdr", "stretch",
            "flows_moved", "route_changed"]
    blk = df.groupby(["method", "seed", "day"])[cols].mean().reset_index()
    return blk.groupby(["method", "day"])[cols].mean().reset_index()   # average over seeds


def routing_tables():
    summ, tests = [], []
    for ds, h, sc, lname in configs():
        df = load_routing(ds, h, sc)
        if df is None:
            continue
        # per-seed summaries, then mean (and sd) across seeds
        df["congested"] = (df["n_congested"] > 0).astype(float)
        per_seed = df.groupby(["method", "seed"]).agg(
            mlu=("mlu", "mean"), congested=("congested", "mean"),
            mean_delay=("mean_delay", "mean"), loss=("loss", "mean"), pdr=("pdr", "mean"),
            stretch=("stretch", "mean"), flows_moved=("flows_moved", "mean"),
            route_changed=("route_changed", "mean")).reset_index()
        blk = daily_blocks(df, ds)
        for m, g in per_seed.groupby("method"):
            r = {"dataset": ds, "horizon": h, "load": lname, "scale": sc, "method": m,
                 "n_seeds": g["seed"].nunique(), "n_days": blk[blk.method == m]["day"].nunique()}
            for c in ["mlu", "congested", "mean_delay", "loss", "pdr", "stretch",
                      "flows_moved", "route_changed"]:
                r[c] = g[c].mean()
                r[c + "_seed_sd"] = g[c].std()
            for c in ROUTE_METRICS:
                lo, hi = boot_ci(blk[blk.method == m][c].values)
                r[c + "_ci_low"], r[c + "_ci_high"] = lo, hi
            summ.append(r)
        piv = {c: blk.pivot(index="day", columns="method", values=c) for c in ROUTE_METRICS}
        cfg_rows = []
        for base in ["static", "reactive", "oracle"]:
            for c in ROUTE_METRICS:
                diff = (piv[c]["predictive"] - piv[c][base]).dropna()
                nz = (diff.abs() > 1e-12).any()
                p = wilcoxon(diff).pvalue if nz and len(diff) >= 6 else np.nan
                lo, hi = boot_ci(diff)
                cfg_rows.append({"dataset": ds, "horizon": h, "load": lname,
                                 "comparison": f"predictive - {base}", "metric": c,
                                 "n_days": len(diff), "mean_diff": diff.mean(),
                                 "ci_low": lo, "ci_high": hi, "p": p,
                                 "predictive_better_days": int((diff < 0).sum())})
        cdf = pd.DataFrame(cfg_rows)
        cdf["p_holm"] = holm(cdf["p"].values)
        tests.append(cdf)
    S = pd.DataFrame(summ)
    T = pd.concat(tests) if tests else pd.DataFrame()
    S.to_csv(RES / "routing_summary.csv", index=False)
    T.to_csv(RES / "stats_routing.csv", index=False)
    return S, T


def feature_ablation():
    fa = EXP["feature_ablation"]
    rows = []
    for ds in EXP["datasets"]:
        out = C.OUT_DIR / run_tag(ds, fa["horizon"], TH, LOADS[fa["load"]])
        base = out / f"metrics_prediction_seed{fa['seed']}.csv"
        if base.exists():
            d = pd.read_csv(base)
            d = d[d.model == "rf"].assign(variant="all features", dataset=ds)
            rows.append(d)
        for name in fa["variants"]:
            f = out / f"metrics_prediction_seed{fa['seed']}_{name}.csv"
            if f.exists():
                d = pd.read_csv(f)
                rows.append(d[d.model == "rf"].assign(variant=name, dataset=ds))
    df = pd.concat(rows) if rows else pd.DataFrame()
    df.to_csv(RES / "ablation_features.csv", index=False)
    return df


def anomaly_sensitivity(thresh=3.0):
    """Sensitivity check: drop intervals where STATIC routing has MLU > thresh
    (demand far above any capacity; treated as measurement anomalies)."""
    rows = []
    for ds, h, sc, lname in configs():
        df = load_routing(ds, h, sc)
        if df is None:
            continue
        st = df[df.method == "static"]
        bad = set(st.loc[st.mlu > thresh, "t"])
        tot_loss = st["loss"].sum()
        for m, g in df.groupby("method"):
            keep = g[~g.t.isin(bad)]
            rows.append({"dataset": ds, "load": lname, "method": m,
                         "excluded_steps": len(bad), "total_steps": st.t.nunique(),
                         "loss_share_of_excluded_static": (st.loc[st.t.isin(bad), "loss"].sum() / tot_loss
                                                           if tot_loss > 0 else 0.0),
                         "mlu": keep.mlu.mean(), "congested": (keep.n_congested > 0).mean(),
                         "loss": keep.loss.mean(), "mean_delay": keep.mean_delay.mean()})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "sensitivity_excl_anomalies.csv", index=False)
    return df


# ---------------------------------------------------------------- LaTeX
def fmt(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def pfmt(p):
    if p is None or np.isnan(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def tex_table(name, caption, label, header, rows, colspec):
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    tex = (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
           f"\\scriptsize\\setlength{{\\tabcolsep}}{{2.5pt}}\n\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
           f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
    (TAB / f"{name}.tex").write_text(tex)


def write_tables(P, S, T, PT, A, empty=False):
    g = (lambda v, nd=3: "--") if empty else fmt
    # Table: prediction results, h=1, all loads
    rows = []
    for ds in EXP["datasets"]:
        for lname in LOADS:
            for model in ["persistence", "logreg", "rf"]:
                r = None if empty or P is None else P[(P.dataset == ds) & (P.horizon == 1) &
                                                    (P.load == lname) & (P.model == model)]
                v = (lambda c: g(r[c].item())) if r is not None and len(r) else (lambda c: "--")
                rows.append([DSN[ds], lname, NAMES[model], v("precision"), v("recall"),
                             v("f1") + ("" if empty or r is None or not len(r) or model == "persistence"
                                        or np.isnan(r['f1_sd'].item())
                                        else f" $\\pm$ {fmt(r['f1_sd'].item())}"),
                             v("roc_auc"), v("pr_auc")])
    tex_table("tab_prediction", "Congestion prediction on the test period ($h=1$, $\\theta=0.8$, seed 42). "
              "Persistence outputs only 0/1, so its ROC-AUC and PR-AUC are single-threshold "
              "values and are not directly comparable with the probability-based AUCs of the other models", "tab:pred",
              "Dataset & Load & Model & Precision & Recall & F1 & ROC-AUC & PR-AUC",
              rows, "@{}lllccccc@{}")

    # Table: routing results, h=1, all loads
    rows = []
    for ds in EXP["datasets"]:
        for lname in LOADS:
            for m in MAIN_METHODS:
                r = None if empty or S is None else S[(S.dataset == ds) & (S.horizon == 1) &
                                                    (S.load == lname) & (S.method == m)]
                ok = r is not None and len(r)
                v = (lambda c, nd=3: g(r[c].item(), nd)) if ok else (lambda c, nd=3: "--")
                pct = (lambda c: g(100 * r[c].item(), 2)) if ok else (lambda c: "--")
                rows.append([DSN[ds], lname, NAMES[m], v("mlu"), pct("congested"),
                             v("mean_delay", 2), pct("loss"), pct("pdr"),
                             v("flows_moved", 2), v("stretch")])
    tex_table("tab_routing", "Network performance on the test period ($h=1$). MLU = maximum link "
              "utilisation; Cong.\\ = \\% of intervals with $\\ge$1 congested link; Delay = "
              "normalised M/M/1 delay; Loss and PDR in \\%; Moved = flows rerouted per interval",
              "tab:route",
              "Data & Load & Method & MLU & Cong.\\,\\% & Delay & Loss\\,\\% & PDR\\,\\% & Moved & Stretch",
              rows, "@{}lllccccccc@{}")

    # Table: statistical tests, predictive vs reactive / static
    rows = []
    for ds in EXP["datasets"]:
        eh = EXP.get("extra_horizon")
        for (h, lname) in [(1, l) for l in LOADS] + ([(eh["horizon"], eh["load"])] if eh else []):
            for base in ["reactive", "static"]:
                for c in ["mlu", "congested", "loss"]:
                    r = None if empty or T is None or not len(T) else T[
                        (T.dataset == ds) & (T.horizon == h) & (T.load == lname) &
                        (T.comparison == f"predictive - {base}") & (T.metric == c)]
                    ok = r is not None and len(r)
                    sc = 100 if c in ("congested", "loss") else 1
                    rows.append([DSN[ds], lname, NAMES[base],
                                 {"mlu": "MLU", "congested": "Cong.\\ (pp)", "loss": "Loss (pp)"}[c],
                                 g(sc * r["mean_diff"].item(), 3) if ok else "--",
                                 (f"[{fmt(sc*r['ci_low'].item())}, {fmt(sc*r['ci_high'].item())}]"
                                  if ok else "--"),
                                 str(int(r["n_days"].item())) if ok else "--",
                                 pfmt(r["p_holm"].item()) if ok else "--"])
    tex_table("tab_stats", "Paired comparison of predictive routing with baselines over test days ($h=1$; "
              "difference = predictive $-$ baseline, negative is better; Cong.\\ = congested-interval share; pp = percentage points; "
              "95\\% bootstrap CI; Wilcoxon signed-rank, Holm-corrected)",
              "tab:stats", "Dataset & Load & Baseline & Metric & Mean diff. & 95\\% CI & Days & $p_{\\mathrm{Holm}}$",
              rows, "@{}llllcccc@{}")

    # Table: ablation (routing variants at medium load + feature ablation)
    rows = []
    for ds in EXP["datasets"]:
        for m in ["static", "reactive", "predictive", "predictive_global", "oracle"]:
            r = None if empty or S is None else S[(S.dataset == ds) & (S.horizon == 1) &
                                                (S.load == "medium") & (S.method == m)]
            ok = r is not None and len(r)
            rows.append([DSN[ds], NAMES[m],
                         g(r["mlu"].item()) if ok else "--",
                         g(100 * r["congested"].item(), 2) if ok else "--",
                         g(100 * r["loss"].item(), 2) if ok else "--",
                         g(r["flows_moved"].item(), 2) if ok else "--"])
    tex_table("tab_ablation_routing", "Ablation of prediction and rerouting (medium load, $h=1$)",
              "tab:abl_route", "Dataset & Variant & MLU & Cong.\\ (\\%) & Loss (\\%) & Moved",
              rows, "@{}llcccc@{}")

    rows = []
    variants = ["all features"] + list(EXP["feature_ablation"]["variants"])
    for ds in EXP["datasets"]:
        for vname in variants:
            r = None if empty or A is None or not len(A) else A[(A.dataset == ds) & (A.variant == vname)]
            ok = r is not None and len(r)
            rows.append([DSN[ds], vname.replace("_", " "),
                         g(r["f1"].item()) if ok else "--",
                         g(r["roc_auc"].item(), 4) if ok else "--",
                         g(r["pr_auc"].item()) if ok else "--"])
    tex_table("tab_ablation_features", "Feature-group ablation for the Random Forest "
              "(medium load, $h=1$, seed 42)", "tab:abl_feat",
              "Dataset & Feature set & F1 & ROC-AUC & PR-AUC", rows, "@{}llccc@{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--empty", action="store_true")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    if a.empty:
        write_tables(None, None, None, None, None, empty=True)
        print("Empty templates written to", TAB)
        return
    P = prediction_tables()
    PT = prediction_block_test()
    S, T = routing_tables()
    A = feature_ablation()
    anomaly_sensitivity()
    write_tables(P, S, T, PT, A)
    print("Results written to", RES)


if __name__ == "__main__":
    main()
