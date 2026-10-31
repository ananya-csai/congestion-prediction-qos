"""Step 4 - trace-driven routing simulation on the TEST period.

At each decision time t the controller (1) decides which links are "at risk"
for the next interval t+h, (2) moves some of the largest flows away from
those links with a capacity-aware greedy procedure (Algorithm 3 in the
paper), and (3) the ACTUAL demands of interval t+h are sent over the
chosen paths and measured.

Strategies (all except 'predictive_global' use the same greedy procedure and
differ only in the at-risk set S, risk score r and demand estimate D_hat):

  static            : no action, hop-count shortest paths
  reactive          : S = {l : U_l(t)   >= theta}  r = 1     D_hat = D(t)
  predictive        : S = {l : p_l      >= tau}    r = p_l   D_hat = D(t)
  oracle            : S = {l : U_l(t+h) >= theta}  r = 1     D_hat = D(t+h)
  predictive_global : ablation - same RF risk, but ALL paths recomputed with
                      weights 1 + alpha*p_l (no capacity check, no greedy)

Network metrics per interval (fluid model, see paper Sec. 4):
  mlu, n_congested, mean_delay (normalised M/M/1), pdr (packet delivery
  ratio), loss = 1 - pdr, flows_moved, route_changed, stretch.

Usage:
    python route_optimizer.py --dataset abilene [--seed 42] [--alpha 5] [--margin 0.1]
           [--horizon 1] [--theta 0.8] [--scale 1.0] [--max-steps 500]
"""
import argparse
import time

import networkx as nx
import numpy as np
import pandas as pd

import config as C
from data_loader import load_dataset
from features import run_tag
from network import od_pairs, route_incidence

U_CAP = 0.99
METHODS = ["static", "reactive", "predictive", "oracle", "predictive_global"]


def path_links(path, link_index):
    return [link_index[(u, v)] for u, v in zip(path[:-1], path[1:])]


def greedy_reroute(G, link_index, od, base_paths, D_hat, S, risk,
                   capacity, theta, margin, alpha):
    """Algorithm 3: move largest flows off at-risk links, capacity-aware."""
    paths = [list(p) for p in base_paths]
    n_links = len(link_index)
    load = np.zeros(n_links)
    for k, p in enumerate(paths):
        load[p] += D_hat[k]
    target = (theta - margin) * capacity
    S_set = set(S)
    risk_w = np.ones(n_links)
    for l in S:
        risk_w[l] = 1 + alpha * risk[l]
    moved = 0
    for l in sorted(S, key=lambda j: -risk[j]):
        eff = max(load[l], theta * capacity)   # predicted link assumed to reach theta
        flows = sorted([k for k, p in enumerate(paths) if l in p], key=lambda k: -D_hat[k])
        for k in flows:
            if eff <= target:
                break
            f = D_hat[k]
            if f <= 0:
                continue
            old_set = set(paths[k])

            def w(u, v, _attr, f=f, old_set=old_set, l=l):
                j = link_index[(u, v)]
                if j == l:
                    return None
                if j in S_set:
                    return risk_w[j]
                if load[j] + (0.0 if j in old_set else f) > target:
                    return None
                return 1.0

            s, d = od[k]
            try:
                new = path_links(nx.dijkstra_path(G, s, d, weight=w), link_index)
            except nx.NetworkXNoPath:
                continue
            load[paths[k]] -= f
            load[new] += f
            paths[k] = new
            eff -= f
            moved += 1
    return paths, moved


def step_metrics(paths, D_s, capacity, theta, hops_static, n_links):
    lam = np.zeros(n_links)
    for k, p in enumerate(paths):
        lam[p] += D_s[k]
    U = lam / capacity
    d_link = 1.0 / (1.0 - np.minimum(U, U_CAP))              # normalised M/M/1 delay
    loss_link = np.where(lam > capacity, 1.0 - capacity / np.maximum(lam, 1e-12), 0.0)
    keep = 1.0 - loss_link
    pdr_k = np.array([np.prod(keep[p]) for p in paths])     # delivery ratio per OD flow
    delay_k = np.array([d_link[p].sum() for p in paths])
    hops = np.array([len(p) for p in paths], dtype=float)
    total = D_s.sum()
    w = D_s / total
    delivered = float(D_s @ pdr_k)
    return {
        "mlu": float(U.max()),
        "n_congested": int((U >= theta).sum()),
        "n_overloaded": int((U >= 1.0).sum()),
        "mean_delay": float(w @ delay_k),
        "pdr": delivered / total,
        "loss": 1.0 - delivered / total,
        "throughput": delivered,                              # dataset units per interval
        "stretch": float(w @ (hops / hops_static)),
    }


def simulate(dataset, horizon, theta, scale, seed, alpha, margin, max_steps=0, verbose=True):
    out = C.OUT_DIR / run_tag(dataset, horizon, theta, scale)
    net = np.load(out / "network.npz")
    D, U, capacity = net["D"].astype(float), net["U"].astype(float), float(net["capacity"])
    pr = np.load(out / f"test_probabilities_seed{seed}.npz")
    tau = float(pr["tau_rf"])
    ds = load_dataset(dataset)
    G, n = ds["graph"], ds["n"]
    links = [tuple(int(x) for x in e) for e in net["links"]]
    L = len(links)
    link_index = {e: i for i, e in enumerate(links)}
    od = od_pairs(n)

    P = np.zeros((int(pr["t"].max()) + 1, L))
    P[pr["t"], pr["link"]] = pr["rf"]
    steps = np.unique(pr["t"])
    if max_steps:
        steps = steps[:max_steps]

    sp = dict(nx.all_pairs_shortest_path(G))
    base = [path_links(sp[s][d], link_index) for s, d in od]
    hops_static = np.array([len(p) for p in base], dtype=float)

    def global_paths(weights):
        A = route_incidence(G, links, weights, n)
        return [list(np.where(row > 0)[0]) for row in A]

    records, prev = [], {}
    t0 = time.time()
    for t in steps:
        s_int = t + horizon
        cfg = {
            "static": ([], np.zeros(L), D[t]),
            "reactive": (list(np.where(U[t] >= theta)[0]), np.ones(L), D[t]),
            "predictive": (list(np.where(P[t] >= tau)[0]), P[t], D[t]),
            "oracle": (list(np.where(U[s_int] >= theta)[0]), np.ones(L), D[s_int]),
        }
        for m in METHODS:
            if m == "predictive_global":
                S = list(np.where(P[t] >= tau)[0])
                paths = global_paths(1 + alpha * P[t]) if S else base
                moved = int(sum(p != b for p, b in zip(paths, base)))
            else:
                S, risk, D_hat = cfg[m]
                if S:
                    paths, moved = greedy_reroute(G, link_index, od, base, D_hat, S, risk,
                                                  capacity, theta, margin, alpha)
                else:
                    paths, moved = base, 0
            met = step_metrics(paths, D[s_int], capacity, theta, hops_static, L)
            key = tuple(map(tuple, paths))
            met["route_changed"] = int(m in prev and prev[m] != key)
            prev[m] = key
            records.append({"t": int(t), "method": m, "at_risk": len(S),
                            "flows_moved": moved, **met})
    df = pd.DataFrame(records)
    tag = f"seed{seed}_alpha{alpha:g}_m{margin:g}"
    df.to_csv(out / f"routing_steps_{tag}.csv", index=False)
    summ = summarise(df)
    summ.to_csv(out / f"routing_summary_{tag}.csv")
    if verbose:
        print(f"[{run_tag(dataset, horizon, theta, scale)} seed={seed}] "
              f"{len(steps)} steps x {len(METHODS)} methods in {time.time()-t0:.1f}s (tau={tau:.2f})")
        print(summ.round(4).to_string())
    return df


def summarise(df):
    return df.groupby("method").agg(
        mean_mlu=("mlu", "mean"),
        p95_mlu=("mlu", lambda x: np.quantile(x, 0.95)),
        congested_step_frac=("n_congested", lambda x: (x > 0).mean()),
        mean_congested_links=("n_congested", "mean"),
        overloaded_step_frac=("n_overloaded", lambda x: (x > 0).mean()),
        mean_delay=("mean_delay", "mean"),
        mean_pdr=("pdr", "mean"),
        mean_loss=("loss", "mean"),
        mean_stretch=("stretch", "mean"),
        mean_flows_moved=("flows_moved", "mean"),
        route_change_rate=("route_changed", "mean"),
    ).reindex(METHODS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="abilene", choices=list(C.DATASETS))
    ap.add_argument("--horizon", type=int, default=C.HORIZON)
    ap.add_argument("--theta", type=float, default=C.THETA)
    ap.add_argument("--scale", type=float, default=C.LOAD_SCALE)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--alpha", type=float, default=C.ALPHA)
    ap.add_argument("--margin", type=float, default=C.MARGIN)
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()
    simulate(a.dataset, a.horizon, a.theta, a.scale, a.seed, a.alpha, a.margin, a.max_steps)


if __name__ == "__main__":
    main()
