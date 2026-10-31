"""Step 2 - build the link-level dataset: features X, labels y, split.

For every link l and every time step t we compute features that use ONLY
information available at time t, and the label
    y = 1  if  U_l(t + h) >= theta   else 0.

Leakage protection:
  * the split is chronological (train -> validation -> test);
  * the last h steps of train and of validation are dropped ("purge"),
    so no training label looks into the validation/test period;
  * the outlier cap and the link capacity are fitted on training steps only.

Usage:
    python features.py --dataset abilene [--horizon 1] [--theta 0.8] [--scale 1.0]
Output: outputs/<tag>/features.pkl and outputs/<tag>/network.npz
"""
import argparse
import json

import networkx as nx
import numpy as np
import pandas as pd

import config as C
from data_loader import load_dataset
from network import (calibrate_capacity, demand_vectors, link_loads,
                     route_incidence)


def run_tag(dataset, horizon, theta, scale):
    return f"{dataset}_h{horizon}_th{theta:g}_s{scale:g}"


def split_bounds(T):
    t_train = int(C.TRAIN_FRAC * T)
    t_val = int((C.TRAIN_FRAC + C.VAL_FRAC) * T)
    return t_train, t_val


def rolling(arr, w, fn):
    """Trailing-window statistic over axis 0 (window ends at t, inclusive)."""
    out = pd.DataFrame(arr).rolling(w, min_periods=w)
    return getattr(out, fn)().values


def build(dataset, horizon=C.HORIZON, theta=C.THETA, scale=C.LOAD_SCALE):
    ds = load_dataset(dataset)
    n, G, links = ds["n"], ds["graph"], ds["links"]
    L = len(links)
    D = demand_vectors(ds["traffic"], n)            # T x n_od
    T = D.shape[0]
    t_train, t_val = split_bounds(T)

    # -- preprocessing fitted on training steps only --------------------
    cap = np.quantile(D[:t_train], C.OUTLIER_QUANTILE, axis=0)
    D = np.minimum(D, cap)

    # -- default (static) routing: unit weights = hop-count shortest path
    A0 = route_incidence(G, links, np.ones(L), n)
    lam0 = link_loads(D, A0)                         # T x L, unscaled
    capacity = calibrate_capacity(lam0[:t_train], theta, C.TARGET_CONGESTED_FRAC)
    D = D * scale                                    # stress-test scaling
    lam0 = lam0 * scale
    U = lam0 / capacity                              # utilisation, T x L

    # -- features (all use data up to and including t) -------------------
    W = C.ROLL_WINDOW
    roll_mean = rolling(U, W, "mean")
    roll_std = rolling(U, W, "std")
    roll_max = rolling(U, W, "max")

    # neighbouring links = links that share a node with link l
    nbrs = []
    for (u, v) in links:
        nbrs.append([j for j, (a, b) in enumerate(links)
                     if (a, b) != (u, v) and len({a, b} & {u, v}) > 0])
    nbr_mean = np.stack([U[:, idx].mean(axis=1) for idx in nbrs], axis=1)
    nbr_max = np.stack([U[:, idx].max(axis=1) for idx in nbrs], axis=1)

    btw = nx.edge_betweenness_centrality(G, normalized=True)
    btw_vec = np.array([btw[e] for e in links])

    steps_per_day = 1440 // ds["step_min"]
    tt = np.arange(T)
    tod = 2 * np.pi * (tt % steps_per_day) / steps_per_day
    dow = 2 * np.pi * (tt % (7 * steps_per_day)) / (7 * steps_per_day)

    t_first = max(C.N_LAGS - 1, W - 1)
    t_idx = np.arange(t_first, T - horizon)          # decision times t

    def at(arr, lag=0):
        return arr[t_idx - lag]                      # shape (len(t_idx), L)

    cols = {"u_t": at(U)}
    for k in range(1, C.N_LAGS):
        cols[f"u_lag{k}"] = at(U, k)
    cols["delta_u"] = at(U) - at(U, 1)
    cols["roll_mean"] = at(roll_mean)
    cols["roll_std"] = at(roll_std)
    cols["roll_max"] = at(roll_max)
    cols["nbr_mean"] = at(nbr_mean)
    cols["nbr_max"] = at(nbr_max)
    rep = lambda v: np.repeat(v[t_idx, None], L, axis=1)
    cols["tod_sin"], cols["tod_cos"] = rep(np.sin(tod)), rep(np.cos(tod))
    cols["dow_sin"], cols["dow_cos"] = rep(np.sin(dow)), rep(np.cos(dow))
    cols["betweenness"] = np.repeat(btw_vec[None, :], len(t_idx), axis=0)

    frame = pd.DataFrame({k: v.ravel() for k, v in cols.items()})
    frame["t"] = np.repeat(t_idx, L)
    frame["link"] = np.tile(np.arange(L), len(t_idx))
    frame["u_future"] = U[t_idx + horizon].ravel()
    frame["y"] = (frame["u_future"] >= theta).astype(np.int8)

    # -- chronological split with purge ----------------------------------
    t = frame["t"].values
    split = np.full(len(frame), "", dtype=object)
    split[(t + horizon) < t_train] = "train"
    split[(t >= t_train) & ((t + horizon) < t_val)] = "val"
    split[t >= t_val] = "test"
    frame["split"] = split
    frame = frame[frame["split"] != ""].reset_index(drop=True)

    tag = run_tag(dataset, horizon, theta, scale)
    out = C.OUT_DIR / tag
    out.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(out / "features.pkl")
    np.savez_compressed(out / "network.npz", D=D.astype(np.float32),
                        U=U.astype(np.float32), capacity=capacity,
                        links=np.array(links), t_train=t_train, t_val=t_val)
    info = {
        "dataset": dataset, "horizon": horizon, "theta": theta, "scale": scale,
        "T": int(T), "links": L, "t_train_end": int(t_train), "t_val_end": int(t_val),
        "capacity": float(capacity),
        "rows": {s: int((frame["split"] == s).sum()) for s in ["train", "val", "test"]},
        "positive_rate": {s: float(frame.loc[frame["split"] == s, "y"].mean())
                          for s in ["train", "val", "test"]},
        "feature_names": [c for c in cols],
    }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2))
    return frame, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="abilene", choices=list(C.DATASETS))
    ap.add_argument("--horizon", type=int, default=C.HORIZON)
    ap.add_argument("--theta", type=float, default=C.THETA)
    ap.add_argument("--scale", type=float, default=C.LOAD_SCALE)
    a = ap.parse_args()
    _, info = build(a.dataset, a.horizon, a.theta, a.scale)
    print(json.dumps({k: info[k] for k in
                      ["dataset", "T", "links", "rows", "positive_rate", "capacity"]}, indent=2))


if __name__ == "__main__":
    main()
