"""NetworkX-compatible algorithms backed by the Rust core.

Each function has the same signature and return value as its NetworkX
counterpart. Inputs rustnx can't handle raise ``NotImplementedError``, which
makes NetworkX fall back to its own implementation.
"""

import math

import networkx as nx

__all__ = [
    "betweenness_centrality",
    "closeness_centrality",
    "connected_components",
    "is_connected",
    "number_connected_components",
    "single_source_dijkstra_path_length",
    "single_source_shortest_path_length",
]


def _index_of(G, node, message):
    try:
        return G._index[node]
    except (KeyError, TypeError):
        raise nx.NodeNotFound(message) from None


def _check_weight(G, attr):
    """Validate a weight argument; return ``(attr, all_int)``."""
    if callable(attr):
        raise NotImplementedError("rustnx does not support callable weights")
    all_int, has_hidden = G._core.weight_info(attr)
    return attr, all_int, has_hidden


def _undirected_only(G):
    if G.is_directed():
        raise nx.NetworkXNotImplemented("not implemented for directed type")


def connected_components(G):
    _undirected_only(G)
    nodes = G._nodes
    for comp in G._core.connected_components():
        yield {nodes[i] for i in comp}


def number_connected_components(G):
    _undirected_only(G)
    return len(G._core.connected_components())


def is_connected(G):
    _undirected_only(G)
    n = len(G)
    if n == 0:
        raise nx.NetworkXPointlessConcept(
            "Connectivity is undefined for the null graph."
        )
    return len(G._core.connected_components()[0]) == n


def single_source_shortest_path_length(G, source, cutoff=None):
    s = _index_of(G, source, f"Source {source} is not in G")
    if cutoff is None:
        cutoff = math.inf
    order, levels = G._core.bfs_lengths(s, float(cutoff))
    nodes = G._nodes
    return dict(zip([nodes[i] for i in order], levels))


def single_source_dijkstra_path_length(G, source, cutoff=None, weight="weight"):
    s = _index_of(G, source, f"Node {source} not found in graph")
    weight, all_int, _ = _check_weight(G, weight)
    order, dists = G._core.dijkstra_lengths(
        s, weight, None if cutoff is None else float(cutoff)
    )
    if all_int:
        dists = map(int, dists)
    nodes = G._nodes
    result = dict(zip([nodes[i] for i in order], dists))
    result[source] = 0  # NetworkX keeps the source distance as the int 0
    return result


def _rescale(values, n, *, normalized, directed, endpoints, sampled):
    """Port of ``networkx.algorithms.centrality.betweenness._rescale``."""
    k = None if sampled is None else len(sampled)
    N = n if endpoints else n - 1
    if N < 2:
        return values
    K_source = N if k is None else k
    if k is None or endpoints:
        if normalized:
            scale = 1 / (K_source * (N - 1))
        else:
            correction = 1 if directed else 2
            scale = N / (K_source * correction)
        if scale != 1:
            values = [v * scale for v in values]
        return values
    if normalized:
        scale_source = 1 / ((K_source - 1) * (N - 1)) if K_source > 1 else math.nan
        scale_nonsource = 1 / (K_source * (N - 1))
    else:
        correction = 1 if directed else 2
        scale_source = N / ((K_source - 1) * correction) if K_source > 1 else math.nan
        scale_nonsource = N / (K_source * correction)
    sampled = set(sampled)
    return [
        v * (scale_source if i in sampled else scale_nonsource)
        for i, v in enumerate(values)
    ]


def betweenness_centrality(
    G, k=None, normalized=True, weight=None, endpoints=False, seed=None
):
    n = len(G)
    if k == n:
        k = None
    weight, _, has_hidden = _check_weight(G, weight)
    if weight is not None and has_hidden:
        raise NotImplementedError("rustnx does not support None edge weights here")
    sources = None
    if k is not None:
        # `seed` is already a `random.Random` (NetworkX's @py_random_state
        # runs before dispatch), so this samples exactly what NetworkX would.
        sources = [G._index[v] for v in seed.sample(list(G._nodes), k)]
    raw = G._core.betweenness(weight, bool(endpoints), sources)
    values = _rescale(
        raw,
        n,
        normalized=normalized,
        directed=G.is_directed(),
        endpoints=endpoints,
        sampled=sources,
    )
    return dict(zip(G._nodes, values))


def closeness_centrality(G, u=None, distance=None, wf_improved=True):
    distance, _, _ = _check_weight(G, distance)
    sources = None
    if u is not None:
        if distance is None:
            msg = f"Source {u} is not in G"
        else:
            msg = f"Node {u} not found in graph"
        sources = [_index_of(G, u, msg)]
    values = G._core.closeness(distance, bool(wf_improved), sources)
    if u is not None:
        return values[0]
    return dict(zip(G._nodes, values))
