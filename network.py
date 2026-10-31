"""Graph helpers: weighted shortest-path routing and link loads.

All routing methods in this project (static, reactive, predictive, oracle)
use the same function, route_incidence(), and differ only in the link
weights they pass in.
"""
import networkx as nx
import numpy as np


def od_pairs(n):
    """All ordered (s, d) pairs with s != d, in a fixed order."""
    return [(s, d) for s in range(n) for d in range(n) if s != d]


def route_incidence(G, links, weights, n):
    """Shortest paths for every OD pair under the given link weights.

    Returns A with shape (n_od, n_links): A[k, l] = 1 if OD pair k uses link l.
    Dijkstra is run once per source node (all-pairs).
    """
    link_index = {e: i for i, e in enumerate(links)}
    for (u, v), w in zip(links, weights):
        G[u][v]["w"] = float(w)
    paths = dict(nx.all_pairs_dijkstra_path(G, weight="w"))
    pairs = od_pairs(n)
    A = np.zeros((len(pairs), len(links)), dtype=np.float32)
    for k, (s, d) in enumerate(pairs):
        p = paths[s][d]
        for u, v in zip(p[:-1], p[1:]):
            A[k, link_index[(u, v)]] = 1.0
    return A


def demand_vectors(traffic, n):
    """Flatten T x n x n traffic into T x n_od, matching od_pairs order."""
    mask = ~np.eye(n, dtype=bool)
    return traffic[:, mask]


def link_loads(D, A):
    """lambda = D @ A  (works for one step or many)."""
    return D @ A


def calibrate_capacity(train_loads, theta, target_frac):
    """Uniform capacity C so that target_frac of training link-intervals
    have utilisation >= theta under default routing."""
    q = np.quantile(train_loads.ravel(), 1.0 - target_frac)
    return q / theta
