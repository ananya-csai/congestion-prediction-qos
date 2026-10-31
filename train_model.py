"""Step 3 - train congestion predictors and evaluate them.

Models
  * persistence : predicts that the link's state at t+h equals its state at t
  * logreg      : logistic regression on standardised features
  * rf          : Random Forest (main model)

Protocol
  * models are fitted on the TRAIN rows only;
  * the decision threshold tau of logreg/rf is chosen on VALIDATION rows
    (value that maximises F1);
  * TEST rows are used once, for the final numbers.

Usage:
    python train_model.py --dataset abilene [--horizon 1] [--theta 0.8] [--scale 1.0]
           [--drop-features nbr_mean,nbr_max]   (used for the ablation study)
Outputs in outputs/<tag>/: models/*.joblib, metrics_prediction.csv,
    feature_importance.csv, test_probabilities.npz
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from features import run_tag


def best_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, p >= g, zero_division=0) for g in grid]
    return float(grid[int(np.argmax(scores))])


def scores(y, p, tau):
    yhat = (p >= tau).astype(int)
    return {
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, p) if y.sum() > 0 else np.nan,
        "tau": tau,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="abilene", choices=list(C.DATASETS))
    ap.add_argument("--horizon", type=int, default=C.HORIZON)
    ap.add_argument("--theta", type=float, default=C.THETA)
    ap.add_argument("--scale", type=float, default=C.LOAD_SCALE)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--drop-features", default="",
                    help="comma-separated feature names to remove (ablation)")
    ap.add_argument("--suffix", default="", help="label for ablation outputs")
    a = ap.parse_args()

    out = C.OUT_DIR / run_tag(a.dataset, a.horizon, a.theta, a.scale)
    info = json.loads((out / "dataset_info.json").read_text())
    frame = pd.read_pickle(out / "features.pkl")
    drop = [f for f in a.drop_features.split(",") if f]
    feats = [f for f in info["feature_names"] if f not in drop]

    tr, va, te = (frame[frame["split"] == s] for s in ["train", "val", "test"])
    Xtr, ytr = tr[feats].values, tr["y"].values
    Xva, yva = va[feats].values, va["y"].values
    Xte, yte = te[feats].values, te["y"].values

    rows, probs = [], {}

    # persistence baseline: needs u_t, which is never dropped in ablation
    for name, part, y in [("val", va, yva), ("test", te, yte)]:
        p = (part["u_t"].values >= a.theta).astype(float)
        if name == "test":
            probs["persistence"] = p
            rows.append({"model": "persistence", **scores(y, p, 0.5)})

    models = {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced")),
        "rf": RandomForestClassifier(random_state=a.seed, **C.RF_PARAMS),
    }
    for name, model in models.items():
        t0 = time.time()
        model.fit(Xtr, ytr)
        fit_s = time.time() - t0
        tau = best_threshold(yva, model.predict_proba(Xva)[:, 1])
        t0 = time.time()
        p = model.predict_proba(Xte)[:, 1]
        pred_s = time.time() - t0
        probs[name] = p
        rows.append({"model": name, **scores(yte, p, tau),
                     "fit_seconds": fit_s, "predict_seconds": pred_s})
        if not drop and a.seed == C.SEED:
            (out / "models").mkdir(exist_ok=True)
            joblib.dump(model, out / "models" / f"{name}.joblib")

    res = pd.DataFrame(rows)
    res.insert(0, "seed", a.seed)
    res.insert(0, "dropped", ",".join(drop) or "none")
    sfx = f"_{a.suffix}" if a.suffix else ""
    res.to_csv(out / f"metrics_prediction_seed{a.seed}{sfx}.csv", index=False)
    print(res.round(4).to_string(index=False))

    if not drop:
        imp = pd.Series(models["rf"].feature_importances_, index=feats)
        imp.sort_values(ascending=False).to_csv(out / f"feature_importance_seed{a.seed}.csv",
                                                header=["importance"])
        np.savez_compressed(out / f"test_probabilities_seed{a.seed}.npz",
                            t=te["t"].values, link=te["link"].values, y=yte.astype(np.int8),
                            **{k: v.astype(np.float32) for k, v in probs.items()},
                            tau_rf=res.loc[res.model == "rf", "tau"].item())


if __name__ == "__main__":
    main()
