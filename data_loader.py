"""Step 1 - load a dataset and rebuild its topology as a NetworkX graph.

The CSV files contain:
  *_tm.csv : one row per time step, one column per origin-destination (OD)
             pair (n*n columns, row-major: column s*n+d).
  *_rm.csv : routing matrix, one row per OD pair, one column per link;
             entry 1 means the OD pair's default path uses that link.

Some columns of the routing matrix are not backbone links (ingress/egress
or "self" links used only by traffic from a node to itself). We keep only
backbone links: columns never used by a diagonal OD pair (s == s).

Run on its own to print a short summary:
    python data_loader.py --dataset abilene
"""
import argparse

import networkx as nx
import numpy as np
import pandas as pd

from config import DATA_DIR, DATASETS


def _read(path):
    df = pd.read_csv(path, header=None)
    # Files end each line with a comma -> last column is empty.
    if df.iloc[:, -1].isna().all():
        df = df.iloc[:, :-1]
    return df.values.astype(float)


def load_dataset(name):
    """Return dict with traffic (T x n x n), graph, link list and meta."""
    meta = DATASETS[name]
    n = meta["n_nodes"]
    tm = _read(DATA_DIR / f"{name}_tm.csv")
    rm = _read(DATA_DIR / f"{name}_rm.csv")
    assert tm.shape[1] == n * n, "traffic matrix width does not match n*n"
    assert rm.shape[0] == n * n, "routing matrix height does not match n*n"

    diag_rows = [s * n + s for s in range(n)]
    off_rows = [i for i in range(n * n) if i // n != i % n]
    used_by_diag = rm[diag_rows].sum(axis=0) > 0
    internal_cols = np.where(~used_by_diag)[0]

    # A backbone link u->v is identified by the OD pair (u, v) whose default
    # path consists of exactly that one backbone link.
    rm_int = rm[:, internal_cols]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for row in off_rows:
        cols = np.where(rm_int[row] > 0)[0]
        if len(cols) == 1:
            s, d = divmod(row, n)
            G.add_edge(s, d)
    if G.number_of_edges() != len(internal_cols):
        raise RuntimeError(
            f"Could only identify {G.number_of_edges()} of {len(internal_cols)} links")
    if not nx.is_strongly_connected(G):
        raise RuntimeError("Rebuilt topology is not strongly connected")

    traffic = tm.reshape(-1, n, n)
    np.clip(traffic, 0, None, out=traffic)
    for s in range(n):
        traffic[:, s, s] = 0.0  # self-traffic never crosses a backbone link

    links = sorted(G.edges())
    return {
        "name": name,
        "n": n,
        "step_min": meta["step_min"],
        "traffic": traffic,
        "graph": G,
        "links": links,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="abilene", choices=list(DATASETS))
    args = ap.parse_args()
    ds = load_dataset(args.dataset)
    T = ds["traffic"].shape[0]
    print(f"Dataset         : {ds['name']}")
    print(f"Nodes / links   : {ds['n']} / {len(ds['links'])} directed")
    print(f"Time steps      : {T} x {ds['step_min']} min = {T*ds['step_min']/1440:.1f} days")
    tot = ds["traffic"].sum(axis=(1, 2))
    print(f"Total demand    : median {np.median(tot):.3g}, max {tot.max():.3g} (dataset units)")


if __name__ == "__main__":
    main()
