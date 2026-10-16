"""Time every function rustnx implements against NetworkX.

    python benchmarks/all_functions.py [--only NAME ...] [--target 1.0]
                                       [--out results.jsonl] [--timeout 300]

For each function the harness finds inputs NetworkX accepts (trying graph
families in turn, and filling required arguments by name), then doubles the
input size until NetworkX takes about ``--target`` seconds (default 0.5). At each size it
times NetworkX, rustnx "cold" (converting the graph first) and rustnx "warm"
(reusing the conversion NetworkX caches on the graph), and checks that the
results agree: "same" is identical output (values and order), "close" equal
to within 1e-9 relative (the fast-float functions), "DIFFERENT" a mismatch.

Each function runs in its own subprocess with a time limit; the last size
it finished is recorded. Results go to ``--out`` as JSON lines, and a
summary table is printed.
"""

import argparse
import collections
import functools
import inspect
import itertools
import json
import math
import os
import random
import subprocess
import sys
import time
import warnings

import networkx as nx

nx.config.warnings_to_ignore.add("cache")
warnings.simplefilter("ignore")

ITER_LIMIT = 200_000  # items taken from an iterator result
MAX_NODES = 1 << 18
CACHE_DIR = os.environ.get("RUSTNX_BENCH_CACHE", "bench-cache")


# --- inputs -----------------------------------------------------------------------


def _weigh(G, seed=0):
    rng = random.Random(seed)
    kw = {"keys": True} if G.is_multigraph() else {}
    for e in G.edges(**kw):
        d = G.edges[e]
        d["weight"] = rng.randint(1, 10)
        d["capacity"] = rng.randint(1, 20)
    for v in G:
        G.nodes[v]["nattr"] = rng.randint(0, 3)
    return G


def _dag(n):
    rng = random.Random(3)
    D = nx.gnm_random_graph(n, 4 * n, seed=3, directed=True)
    rank = {v: rng.random() for v in D}
    dag = nx.DiGraph()
    dag.add_nodes_from(D)
    dag.add_edges_from((u, v) for u, v in D.edges if rank[u] < rank[v])
    return dag


def _bipartite(n):
    half = max(n // 2, 2)
    B = nx.bipartite.random_graph(half, half, min(1.0, 8 / half), seed=4)
    return B


FAMILIES = {
    # connected, scale-free: what most graph analyses look like
    "graph": lambda n: nx.barabasi_albert_graph(n, 3, seed=1),
    "digraph": lambda n: nx.gnm_random_graph(n, 4 * n, seed=2, directed=True),
    "dag": _dag,
    "tree": lambda n: nx.random_labeled_tree(n, seed=5),
    "bipartite": _bipartite,
    "grid": lambda n: nx.convert_node_labels_to_integers(
        nx.grid_2d_graph(max(int(math.sqrt(n)), 2), max(int(math.sqrt(n)), 2))
    ),
    "multigraph": lambda n: nx.MultiGraph(
        list(nx.barabasi_albert_graph(n, 3, seed=1).edges) + list(nx.barabasi_albert_graph(n, 1, seed=6).edges)
    ),
    "multidigraph": lambda n: nx.MultiDiGraph(
        list(nx.gnm_random_graph(n, 4 * n, seed=2, directed=True).edges)
        + list(nx.gnm_random_graph(n, n, seed=7, directed=True).edges)
    ),
    "tournament": lambda n: nx.tournament.random_tournament(min(n, 2000), seed=8),
    # traveling-salesman functions want complete weighted graphs
    "complete": lambda n: nx.complete_graph(min(n, 300)),
    "complete_digraph": lambda n: nx.complete_graph(min(n, 300), create_using=nx.DiGraph),
}

_family_cache = {}


def _canonical(G):
    """Unpickling gives every int above 256 a separate object at each place
    it appears, so ``G._node`` and ``G._adj`` hold equal but distinct keys;
    rustnx's graph operators check them by identity and hand such graphs to
    NetworkX. Share one object per node again (same order) so the timings
    are those of a graph built in this process."""
    canon = {v: v for v in G._node}

    def rebuild(D):
        return {canon[u]: {canon[v]: d for v, d in row.items()} if not G.is_multigraph()
                else {canon[v]: keyed for v, keyed in row.items()} for u, row in D.items()}

    if G.is_directed():
        G._succ = G._adj = rebuild(G._succ)
        G._pred = rebuild(G._pred)
    else:
        G._adj = rebuild(G._adj)
    for attr in ("adj", "succ", "pred", "edges", "nodes", "degree"):
        G.__dict__.pop(attr, None)
    return G


def family_graph(name, n):
    """The family's graph at size n, built once and kept on disk (building
    large ones takes longer than most of the functions timed on them)."""
    import pickle

    key = (name, n)
    if key not in _family_cache:
        _family_cache.clear()
        path = os.path.join(CACHE_DIR, f"{name}-{n}.pickle")
        G = None
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    G = _canonical(pickle.load(f))
            except Exception:
                G = None
        if G is None:
            G = _weigh(FAMILIES[name](n))
            if name.startswith("complete"):
                # float weights, so the traveling-salesman heuristics see no
                # ties (rustnx hands ties broken by set order to NetworkX)
                rng = random.Random(11)
                for _, _, d in G.edges(data=True):
                    d["weight"] = rng.random()
            if n >= 4096:
                os.makedirs(CACHE_DIR, exist_ok=True)
                tmp = f"{path}.{os.getpid()}"
                with open(tmp, "wb") as f:
                    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, path)
        _family_cache[key] = G
    return _family_cache[key]


def uncache(G):
    """Drop NetworkX's cached conversion (for "cold" timings) without
    copying, since a copy can order neighbors differently."""
    cache = getattr(G, "__networkx_cache__", None)
    if cache:
        cache.clear()


# Candidate values for required arguments, by name. Each is a function of
# (graph or None, size); the first combination NetworkX accepts is used.
def _nodes(G, frac):
    V = list(G)
    return V[: max(2, int(len(V) * frac))]


NODE_ARGS = {"source", "s", "u", "start", "root", "node", "node1", "n1", "root1", "_s", "x", "a"}
TARGET_ARGS = {"target", "t", "v", "node2", "n2", "root2", "_t", "y", "b"}
SET_ARGS = {"nodes", "nbunch", "S", "C", "nbunch1", "nbunch2", "T", "sources", "targets", "A"}


def graph_candidates(param, G, n):
    V = list(G)
    if param in NODE_ARGS or param == "n":
        return [V[0]]
    if param in TARGET_ARGS:
        return [V[-1], V[1]]
    if param in SET_ARGS:
        return [_nodes(G, 0.1), V[:3], set(_nodes(G, 0.5))]
    if param in ("z", "Z"):
        return [set(V[1:11])]
    if param == "center_nodes":
        return [set(V[:: max(len(V) // 20, 1)])]
    if param == "terminal_nodes":
        return [V[:: max(len(V) // 20, 1)]]
    if param == "walk_length":
        return [3]
    if param == "stretch":
        return [3]
    if param == "node_attributes":
        return [("nattr",)]
    if param == "mapping":
        k = len(V)
        return [{v: (v + k if isinstance(v, int) else (v, 1)) for v in V}]
    if param in ("row_order", "top_nodes"):
        top = [v for v, d in G.nodes(data=True) if d.get("bipartite") == 0]
        return [top] if top else []
    if param == "path":
        try:
            return [nx.shortest_path(G, V[0], V[-1])]
        except Exception:
            return [V[:2]]
    if param == "paths":
        return [[nx.shortest_path(G, V[0], V[i]) for i in range(1, min(len(V), 20)) if nx.has_path(G, V[0], V[i])]]
    if param in ("communities", "partition"):
        und = G.to_undirected() if G.is_directed() else G
        return [
            [set(c) for c in nx.community.label_propagation_communities(nx.Graph(und), backend="networkx")],
            [set(c) for c in nx.connected_components(nx.Graph(und), backend="networkx")],
        ]
    if param == "matching":
        return [nx.maximal_matching(G, backend="networkx")]
    if param == "coloring":
        return [nx.greedy_color(G, backend="networkx")]
    if param == "attribute":
        return ["nattr"]
    if param in ("k", "d", "r", "radius", "l", "h", "distance"):
        return [3, 2, 1]
    if param == "avail":
        rng = random.Random(9)
        non = []
        while len(non) < min(4 * len(V), 2000):
            a, b = rng.sample(V, 2)
            if not G.has_edge(a, b):
                non.append((a, b))
        return [non]
    if param == "edge":
        return [next(iter(G.edges))]
    if param in ("ebunch", "edges"):
        return [list(G.edges)[: max(2, G.number_of_edges() // 10)]]
    if param == "capacity":
        return ["capacity"]
    if param == "flowDict":
        try:
            return [nx.max_flow_min_cost(G, V[0], V[-1], backend="networkx")]
        except Exception:
            return []
    if param == "name":
        return ["nattr"]
    if param in ("alpha", "beta", "p", "q", "threshold"):
        return [0.5]
    if param == "data":
        return [True]
    if param == "pairs":
        return [[(V[0], V[i]) for i in range(1, min(len(V), 50))]]
    if param == "init_cycle":
        return ["greedy"]
    if param == "minimum":
        return [1]
    if param in ("max_size", "cutoff", "number_of_sets"):
        return [4]
    if param in ("sample_size", "nswap", "max_tries"):
        return [1000]
    if param == "time_delta":
        return [1]
    return []


def free_candidates(param, n):
    """Arguments of functions that take no graph (generators, sequences)."""
    if param in ("n", "n1", "n2", "N", "number_of_nodes", "r1", "r2"):
        return [n, max(int(math.sqrt(n)), 2)]
    if param in ("m", "m1", "m2"):
        return [3, 4 * n, max(int(math.sqrt(n)), 2)]
    if param in ("p", "p_in", "p_out", "p1", "p2", "q", "alpha", "beta", "gamma"):
        return [min(1.0, 8 / max(n, 1)), 0.5, 0.1]
    if param in ("k", "d", "r", "l", "h", "radius", "dim"):
        return [3, 4, 2]
    if param in ("sequence", "deg_sequence", "aseq", "bseq", "in_degree_sequence", "out_degree_sequence",
                 "in_deg_sequence", "out_deg_sequence", "in_degrees", "out_degrees", "degree_sequence",
                 "in_sequence", "out_sequence"):
        rng = random.Random(10)
        seq = [rng.randint(1, 5) for _ in range(n)]
        if sum(seq) % 2:
            seq[0] += 1
        return [seq, [3] * (n - n % 2)]
    if param == "sizes":
        return [[n // 4] * 4]
    if param == "nkk":
        return [{1: {2: n}, 2: {1: n}}]
    if param == "theta":
        return [0.5, 1.0]
    if param in ("x", "y", "z"):
        return [3]
    if param == "number_of_sets":
        return [4]
    if param == "lines":
        return [[f"{i} {i + 1}" for i in range(n)]]
    if param == "w":
        return [[1.0] * n]
    if param == "shift_list":
        return [[1, -1]]
    if param == "repeats":
        return [max(n // 2, 1)]
    if param == "offsets":
        return [[1, 2, 3]]
    if param == "distance":
        return [2]
    if param == "string":
        return ["abc" * n]
    if param == "edgelist":
        return [[(i, i + 1) for i in range(n)]]
    return []


# Functions whose inputs aren't graphs or named arguments the rules above
# know: readers and parsers (fed a file or text NetworkX wrote for a graph of
# the given size) and conversions (fed NetworkX's own export).
_tmpdir = None


def _file(n, writer, suffix, family="graph"):
    import tempfile

    global _tmpdir
    _tmpdir = _tmpdir or tempfile.mkdtemp(prefix="rustnx-bench-")
    path = os.path.join(_tmpdir, f"g{n}{suffix}")
    if not os.path.exists(path):
        writer(family_graph(family, n), path)
    return path


def _leda_lines(n):
    G = family_graph("graph", n)
    index = {v: i + 1 for i, v in enumerate(G)}
    lines = ["LEDA.GRAPH", "string", "int", "-2", str(len(G))]
    lines += [f"|{{{v}}}|" for v in G]
    lines.append(str(G.number_of_edges()))
    lines += [f"{index[u]} {index[v]} 0 |{{{d['weight']}}}|" for u, v, d in G.edges(data=True)]
    return lines


def _dot(G, path):
    from networkx.drawing import nx_pydot

    nx_pydot.write_dot(G, path)


# Fixed extra arguments, for functions whose output explodes otherwise.
EXTRA = {
    "to_nested_tuple": {"canonical_form": True},
    "rich_club_coefficient": {"normalized": False},
    "all_simple_paths": {"cutoff": 4},
    "all_simple_edge_paths": {"cutoff": 4},
}

# Starting sizes for functions NetworkX is very slow on.
START = {"held_karp_ascent": 8, "asadpour_atsp": 8, "modular_product": 8}

# Dense inputs (a NumPy or pandas matrix) are capped at these sizes.
CAPS = {"from_numpy_array": 8192, "from_pandas_adjacency": 4096, "from_nested_tuple": 4096,
        "kneser_graph": 4096}

def _log(n, base=2, low=2):
    return max(low, int(math.log(max(n, 2), base)))


def _prime_near(n):
    p = max(n, 5)
    while any(p % d == 0 for d in range(2, int(math.isqrt(p)) + 1)):
        p += 1
    return p


def _residual(n):
    G = family_graph("digraph", n)
    V = list(G)
    return {"G": G, "R": nx.flow.preflow_push(G, V[0], V[-1], backend="networkx")}


def _cd_index(n):
    G = family_graph("dag", n).copy()
    order = list(nx.topological_sort(G))
    for i, v in enumerate(order):
        G.nodes[v]["time"] = i
    node = max(G, key=G.in_degree)
    return {"G": G, "node": node, "time_delta": n}


def _directed_joint(n):
    D = family_graph("digraph", n)
    nkk = collections.defaultdict(lambda: collections.defaultdict(int))
    for u, v in D.edges:
        nkk[D.out_degree(u)][D.in_degree(v)] += 1
    return {"in_degrees": [D.in_degree(v) for v in D], "out_degrees": [D.out_degree(v) for v in D],
            "nkk": {k: dict(v) for k, v in nkk.items()}}


def _negative_cycle(n):
    G = family_graph("digraph", n).copy()
    V = list(G)
    a, b, c = V[-3:]
    for u, v in ((a, b), (b, c), (c, a)):
        G.add_edge(u, v, weight=-5)
    return {"G": G, "source": V[0]}


def _geometric(n):
    r = math.sqrt(4 / (math.pi * n))
    return {"G": nx.random_geometric_graph(n, r, seed=1), "radius": r}


def _labelled(n):
    G = family_graph("graph", n).copy()
    for i, v in enumerate(G):
        if i % 10 == 0:
            G.nodes[v]["label"] = i % 20 // 10
    return {"G": G}


def _intervals(n):
    rng = random.Random(12)
    starts = [rng.uniform(0, n) for _ in range(n)]
    return {"intervals": [(a, a + 2) for a in starts]}


def _node_weighted(n):
    G = family_graph("graph", n).copy()
    rng = random.Random(13)
    for v in G:
        G.nodes[v]["weight"] = rng.randint(1, 10)
    return {"G": G}


def _bipartite_side(n, key="nodes", graph_key="B"):
    B = family_graph("bipartite", n)
    return {graph_key: B, key: [v for v, d in B.nodes(data=True) if d["bipartite"] == 0]}


def _sbm(n):
    k = max(n // 4, 2)
    p_in, p_out = min(1.0, 8 / k), min(1.0, 1 / n)
    return {"sizes": [k] * 4, "p": [[p_in if i == j else p_out for j in range(4)] for i in range(4)]}


def _two_trees(n):
    t1 = family_graph("tree", n)
    k = len(t1)
    t2 = nx.relabel_nodes(t1, lambda v: v + k)
    return {"t1": t1, "root1": 0, "t2": t2, "root2": k}


def _prefix_paths(n):
    rng = random.Random(14)
    return {"paths": ["".join(rng.choice("abcd") for _ in range(8)) for _ in range(n)]}


def _vertex_cover(n):
    B = family_graph("bipartite", n)
    top = [v for v, d in B.nodes(data=True) if d["bipartite"] == 0]
    return {"G": B, "matching": nx.bipartite.hopcroft_karp_matching(B, top, backend="networkx"), "top_nodes": top}


MANUAL = {
    "to_vertex_cover": _vertex_cover,
    "visibility_graph": lambda n: {"series": [random.Random(17 + i).random() for i in range(n)]},
    "random_k_lift": lambda n: {"G": nx.random_regular_graph(4, n - n % 2, seed=16), "k": 4},
    "projected_graph": _bipartite_side,
    "collaboration_weighted_projected_graph": _bipartite_side,
    "overlap_weighted_projected_graph": _bipartite_side,
    "generic_weighted_projected_graph": _bipartite_side,
    "eppstein_matching": lambda n: _bipartite_side(n, "top_nodes", "G"),
    "stochastic_block_model": _sbm,
    "rooted_tree_isomorphism": _two_trees,
    "root_trees": _two_trees,
    "symmetric_difference": lambda n: {"G": family_graph("graph", n),
                                       "H": nx.gnm_random_graph(n, 3 * n, seed=15)},
    "rooted_product": lambda n: {"G": family_graph("graph", n), "H": nx.path_graph(4), "root": 0},
    "ring_of_cliques": lambda n: {"num_cliques": max(n // 8, 2), "clique_size": 8},
    "random_shell_graph": lambda n: {"constructor": [(max(n // 2, 2), n, 0.5), (max(n // 2, 2), n, 0.5)]},
    "random_clustered_graph": lambda n: {"joint_degree_sequence": [(1, 1)] * max(n - n % 6, 6)},
    "nonisomorphic_trees": lambda n: {"order": 6 + _log(n, 3, 0)},
    "navigable_small_world_graph": lambda n: {"n": max(int(math.sqrt(n)), 2)},
    "prefix_tree": _prefix_paths,
    "prefix_tree_recursive": _prefix_paths,
    "max_weight_clique": _node_weighted,
    "interval_graph": _intervals,
    "join_trees": lambda n: {"rooted_trees": [(nx.random_labeled_tree(max(n // 2, 2), seed=5), 0),
                                              (nx.random_labeled_tree(max(n // 2, 2), seed=6), 0)]},
    "harmonic_function": _labelled,
    "local_and_global_consistency": _labelled,
    "geometric_edges": _geometric,
    "graph_atlas": lambda n: {"i": 1252},
    "geometric_soft_configuration_graph": lambda n: {"beta": 1.5, "n": n, "gamma": 2.7, "mean_degree": 5},
    "gaussian_random_partition_graph": lambda n: {"n": n, "s": 20, "v": 5, "p_in": 0.25, "p_out": min(1.0, 4 / n)},
    "general_random_intersection_graph": lambda n: {"n": n, "m": max(n // 2, 1), "p": [min(1.0, 2 / n)] * max(n // 2, 1)},
    "find_negative_cycle": _negative_cycle,
    "directed_joint_degree_graph": _directed_joint,
    "joint_degree_graph": lambda n: {"joint_degrees": nx.degree_mixing_dict(family_graph("graph", n), normalized=False)},
    "cd_index": _cd_index,
    "degree_sequence_tree": lambda n: {"deg_sequence": [d for _, d in family_graph("tree", n).degree()]},
    "chordal_cycle_graph": lambda n: {"p": _prime_near(n)},
    "bipartite_read_edgelist": lambda n: {"path": _file(n, nx.bipartite.write_edgelist, ".bedges", "bipartite")},
    "bipartite_parse_edgelist": lambda n: {"lines": list(nx.bipartite.generate_edgelist(family_graph("bipartite", n)))},
    "build_flow_dict": _residual,
    # generators whose size grows exponentially (or polynomially) in their arguments
    "binomial_tree": lambda n: {"n": _log(n)},
    "hypercube_graph": lambda n: {"n": _log(n)},
    "random_cograph": lambda n: {"n": _log(n)},
    "mycielski_graph": lambda n: {"n": _log(n)},
    "dorogovtsev_goltsev_mendes_graph": lambda n: {"n": _log(n, 3)},
    "balanced_tree": lambda n: {"r": 2, "h": _log(n)},
    "full_rary_tree": lambda n: {"r": 3, "n": n},
    "kneser_graph": lambda n: {"n": int((6 * n) ** (1 / 3)) + 3, "k": 3},
    "paley_graph": lambda n: {"p": _prime_near(int(math.sqrt(n) * 8))},
    "grid_graph": lambda n: {"dim": [max(int(math.sqrt(n)), 2)] * 2},
    "complete_multipartite_graph": lambda n: {"_args": [max(n // 64, 1)] * 8},
    "adjacency_graph": lambda n: {"data": nx.adjacency_data(family_graph("graph", n))},
    "cytoscape_graph": lambda n: {"data": nx.cytoscape_data(family_graph("graph", n))},
    "node_link_graph": lambda n: {"data": nx.node_link_data(family_graph("graph", n), edges="edges")},
    "tree_graph": lambda n: {"data": nx.tree_data(nx.bfs_tree(family_graph("tree", n), 0), 0)},
    "read_adjlist": lambda n: {"path": _file(n, nx.write_adjlist, ".adj")},
    "read_multiline_adjlist": lambda n: {"path": _file(n, nx.write_multiline_adjlist, ".madj")},
    "read_edgelist": lambda n: {"path": _file(n, nx.write_edgelist, ".edges")},
    "read_weighted_edgelist": lambda n: {"path": _file(n, nx.write_weighted_edgelist, ".wedges")},
    "read_gexf": lambda n: {"path": _file(n, nx.write_gexf, ".gexf")},
    "read_graphml": lambda n: {"path": _file(n, nx.write_graphml, ".graphml")},
    "read_gml": lambda n: {"path": _file(n, nx.write_gml, ".gml")},
    "read_pajek": lambda n: {"path": _file(n, nx.write_pajek, ".net")},
    "read_graph6": lambda n: {"path": _file(n, nx.write_graph6, ".g6")},
    "read_sparse6": lambda n: {"path": _file(n, nx.write_sparse6, ".s6")},
    "read_leda": lambda n: {"path": _file(n, lambda G, p: open(p, "w").write("\n".join(_leda_lines(n))), ".leda")},
    "pydot_read_dot": lambda n: {"path": _file(n, _dot, ".dot")},
    "parse_adjlist": lambda n: {"lines": list(nx.generate_adjlist(family_graph("graph", n)))},
    "parse_multiline_adjlist": lambda n: {"lines": _Lines(list(nx.generate_multiline_adjlist(family_graph("graph", n))))},
    "parse_edgelist": lambda n: {"lines": list(nx.generate_edgelist(family_graph("graph", n)))},
    "parse_gml": lambda n: {"lines": list(nx.generate_gml(family_graph("graph", n)))},
    "parse_graphml": lambda n: {"graphml_string": "\n".join(nx.generate_graphml(family_graph("graph", n)))},
    "parse_pajek": lambda n: {"lines": list(nx.generate_pajek(family_graph("graph", n)))},
    "parse_leda": lambda n: {"lines": _leda_lines(n)},
    "from_graph6_bytes": lambda n: {"bytes_in": nx.to_graph6_bytes(family_graph("graph", n), header=False).strip()},
    "from_sparse6_bytes": lambda n: {"string": nx.to_sparse6_bytes(family_graph("graph", n), header=False).strip()},
    "from_dict_of_dicts": lambda n: {"d": nx.to_dict_of_dicts(family_graph("graph", n))},
    "from_dict_of_lists": lambda n: {"d": nx.to_dict_of_lists(family_graph("graph", n))},
    "from_edgelist": lambda n: {"edgelist": list(family_graph("graph", n).edges)},
    "from_numpy_array": lambda n: {"A": nx.to_numpy_array(family_graph("graph", min(n, 8192)))},
    "from_scipy_sparse_array": lambda n: {"A": nx.to_scipy_sparse_array(family_graph("graph", n))},
    "from_pandas_edgelist": lambda n: {"df": nx.to_pandas_edgelist(family_graph("graph", n)), "edge_attr": True},
    "from_pandas_adjacency": lambda n: {"df": nx.to_pandas_adjacency(family_graph("graph", min(n, 4096)))},
    "from_biadjacency_matrix": lambda n: {"A": nx.bipartite.biadjacency_matrix(
        family_graph("bipartite", n), [v for v, d in family_graph("bipartite", n).nodes(data=True) if d["bipartite"] == 0])},
    "from_prufer_sequence": lambda n: {"sequence": nx.to_prufer_sequence(family_graph("tree", n))},
    "from_nested_tuple": lambda n: {"sequence": nx.to_nested_tuple(family_graph("tree", min(n, 4096)), 0)},
}


# --- comparing results --------------------------------------------------------------


def norm(x, depth=0):
    if isinstance(x, nx.Graph):
        kw = {"keys": True} if x.is_multigraph() else {}
        return ("graph", type(x).__name__, norm(list(x.nodes(data=True))), norm(list(x.edges(data=True, **kw))))
    if isinstance(x, dict):
        return ("dict", [(norm(k), norm(v)) for k, v in x.items()])
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, [norm(v) for v in x])
    if isinstance(x, (set, frozenset)):
        return (type(x).__name__, [norm(v) for v in x])
    if isinstance(x, float):
        return ("float", x)
    if hasattr(x, "tocsr") and hasattr(x, "shape"):
        # Sparse: compare the structure without densifying.
        c = x.tocsr()
        c.sort_indices()
        return ("sparse", x.shape, ("bytes", c.indptr.tobytes()), ("bytes", c.indices.tobytes()),
                ("float-array", c.data.astype(float).tolist()))
    if hasattr(x, "tolist") and hasattr(x, "shape"):
        if getattr(x, "size", 0) > 4_000_000:
            return ("array", x.shape, str(x.dtype), ("bytes", x.tobytes()))
        return ("array", norm(x.tolist()))
    r = repr(x)
    if " at 0x" in r:
        # a default repr carries a memory address, different in any two runs
        # (preflow_push's CurrentEdge helpers, say)
        return ("object", type(x).__name__)
    return (type(x).__name__, r)


def close(a, b):
    if a == b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, tuple) and len(a) == 2 and a[0] == "float":
        x, y = a[1], b[1]
        return (math.isnan(x) and math.isnan(y)) or abs(x - y) <= 1e-9 * max(1.0, abs(x), abs(y))
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(close(p, q) for p, q in zip(a, b))
    return False


# Items taken from iterator results that never end (or nearly never).
LIMITS = {"shortest_simple_paths": 1000, "all_topological_sorts": 20_000}
_limit = [ITER_LIMIT]


def materialize(r):
    if isinstance(r, (nx.Graph, dict, list, tuple, set, frozenset, str, int, float, bool)) or r is None:
        return r
    if hasattr(r, "__iter__") and hasattr(r, "__next__"):
        out = []
        for item in itertools.islice(r, _limit[0]):
            out.append(item if not hasattr(item, "__next__") else list(itertools.islice(item, ITER_LIMIT)))
        return out
    return r


# --- one function -----------------------------------------------------------------


def registered(name):
    return nx.utils.backends._registered_algorithms[name]


def public(name):
    """The function users call (with the decorators NetworkX puts outside
    its dispatcher, such as the one turning ``seed`` into a generator)."""
    import importlib

    d = registered(name)
    module = importlib.import_module(d.orig_func.__module__)
    return (getattr(module, name, None) or getattr(module, d.orig_func.__name__, None)
            or getattr(nx, name, None) or d)


@functools.lru_cache(maxsize=None)
def _positional_only(func):
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return ()
    return tuple(p.name for p in sig.parameters.values() if p.kind is p.POSITIONAL_ONLY)


class _Lines:
    """Lines handed out as an iterator, restarted for every call."""

    def __init__(self, items):
        self.items = items
        self._it = iter(items)

    def _reset(self):
        self._it = iter(self.items)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)


def call(func, args, kwargs, backend):
    for v in kwargs.values():
        if isinstance(v, _Lines):
            v._reset()
    if "_args" in kwargs:
        kwargs = dict(kwargs)
        args = tuple(kwargs.pop("_args")) + tuple(args)
    pos = [p for p in _positional_only(func) if p in kwargs]
    if pos:
        # positional-only parameters (minimal_branching's G, say)
        kwargs = dict(kwargs)
        args = tuple(kwargs.pop(p) for p in pos) + tuple(args)
    # Some functions draw from the global generator (lukes_partitioning's
    # root, for one); both backends start from the same state.
    random.seed(12345)
    t = time.perf_counter()
    r = materialize(func(*args, backend=backend, **kwargs))
    return time.perf_counter() - t, r


def find_inputs(name, func, n):
    """(family, builder) where builder(n) gives (args, kwargs, graph)."""
    if name in MANUAL:
        def make(n2):
            kwargs = dict(MANUAL[name](n2))
            if "seed" in inspect.signature(registered(name).orig_func).parameters:
                kwargs["seed"] = 1
            return kwargs, None
        try:
            make(n)
            return "manual", make
        except Exception:
            return None, None
    d = registered(name)
    sig = inspect.signature(d.orig_func)
    params = [p for p in sig.parameters.values() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    # Only the required graph inputs (flow functions also list optional
    # `residual` / `auxiliary` graphs).
    graph_params = [p.name for p in params if p.name in d.graphs and p.default is p.empty]
    seed = "seed" in sig.parameters
    fam, make = _find(name, func, n, params, graph_params, d, seed, optional=False)
    if make is None:
        fam, make = _find(name, func, n, params, graph_params, d, seed, optional=True)
    return fam, make


def _find(name, func, n, params, graph_params, d, seed, optional):
    required = [p for p in params if p.name not in d.graphs and p.name != "seed" and (
        p.default is p.empty or (optional and p.name in NODE_ARGS | TARGET_ARGS | SET_ARGS))]

    def builders():
        if graph_params:
            for fam in FAMILIES:
                def build(n, fam=fam):
                    G = family_graph(fam, n)
                    return G, [graph_candidates(p.name, G, n) for p in required]
                yield fam, build
        else:
            def build(n):
                return None, [free_candidates(p.name, n) for p in required]
            yield "none", build

    for fam, build in builders():
        G, cands = build(n)
        if any(not c for c in cands):
            continue
        combos = list(itertools.product(*(range(len(c)) for c in cands)))
        if not graph_params and len(combos) > 1:
            combos = _by_growth(func, build, required, combos, seed, name, n)
        for combo in combos:
            def make(n2, combo=combo, build=build):
                G2, cands2 = build(n2)
                kwargs = {p.name: cands2[i][j] if j < len(cands2[i]) else cands2[i][0]
                          for i, (p, j) in enumerate(zip(required, combo))}
                for gi, gp in enumerate(graph_params):
                    if gi > 0 and gp not in getattr(d, "list_graphs", ()):
                        # a second graph (union, full_join, ...): a copy with
                        # shifted labels, so the node sets are disjoint
                        k = len(G2)
                        kwargs[gp] = nx.relabel_nodes(G2, lambda v: v + k if isinstance(v, int) else (v, 1))
                        continue
                    if gp in getattr(d, "list_graphs", ()):
                        # graph lists (compose_all, union_all, ...): the graph
                        # and a copy with shifted labels
                        k = len(G2)
                        shifted = nx.relabel_nodes(G2, lambda v: v + k if isinstance(v, int) else (v, 1))
                        kwargs[gp] = [G2, shifted]
                    else:
                        kwargs[gp] = G2
                if seed:
                    kwargs["seed"] = 1
                kwargs.update(EXTRA.get(name, {}))
                return kwargs, G2
            kwargs, G = make(n)
            try:
                call(func, (), kwargs, "networkx")
            except Exception:
                continue
            try:
                call(func, (), make(n)[0], "rustnx")
            except NotImplementedError:
                continue
            except Exception:
                pass
            return fam, make
    return None, None


def _by_growth(func, build, required, combos, seed, name, n):
    """Generator argument combinations, those whose output grows most nearly
    linearly with the size argument first (so a fixed-size choice, such as
    ``barbell_graph(3, 3)``, isn't timed)."""

    def nodes(combo, size):
        _, cands = build(size)
        kwargs = {p.name: cands[i][j] if j < len(cands[i]) else cands[i][0]
                  for i, (p, j) in enumerate(zip(required, combo))}
        if seed:
            kwargs["seed"] = 1
        kwargs.update(EXTRA.get(name, {}))
        r = func(**kwargs, backend="networkx")
        return len(r) if isinstance(r, nx.Graph) else None

    scored = []
    for combo in combos:
        try:
            a, b = nodes(combo, n), nodes(combo, 4 * n)
        except Exception:
            continue
        if a is None or b is None:
            scored.append((10.0, combo))
            continue
        ratio = b / max(a, 1)
        scored.append((abs(math.log(max(ratio, 1e-9)) - math.log(4)), combo))
    scored.sort(key=lambda t: t[0])
    return [c for _, c in scored] or combos


def run_one(name, target):
    func = public(name)
    _limit[0] = LIMITS.get(name, ITER_LIMIT)
    n = START.get(name, 64)
    last_size = None
    prev = None
    fam, make = find_inputs(name, func, n)
    if make is None:
        print(json.dumps({"name": name, "status": "no-inputs"}), flush=True)
        return
    while True:
        try:
            kwargs, G = make(n)
        except Exception as exc:
            print(json.dumps({"name": name, "status": "build-error", "error": repr(exc)[:200], "n": n}), flush=True)
            return
        try:
            t_nx, r_nx = call(func, (), kwargs, "networkx")
        except Exception as exc:
            if prev is not None:
                # NetworkX fails at this size (a recursion limit, say): keep
                # the last size it handled.
                prev["note"] = f"NetworkX raises at the next size: {repr(exc)[:120]}"
                print(json.dumps(prev), flush=True)
            else:
                print(json.dumps({"name": name, "status": "nx-error", "error": repr(exc)[:200], "n": n}), flush=True)
            return
        size = len(G) if G is not None else n
        # Capped families stop growing; so does the search.
        final = t_nx >= target or n >= CAPS.get(name, MAX_NODES) or size == last_size
        last_size = size
        rec = {"name": name, "family": fam, "n": size, "nx": t_nx}
        if G is not None:
            rec["edges"] = G.number_of_edges()
        try:
            cold_kwargs = kwargs
            if G is not None:
                uncache(G)
            rec["cold"], r_rx = call(func, (), cold_kwargs, "rustnx")
            warm = [call(func, (), cold_kwargs, "rustnx")[0] for _ in range(3 if final else 1)]
            rec["warm"] = min(warm)
            a, b = norm(r_rx), norm(r_nx)
            rec["match"] = "same" if a == b else ("close" if close(a, b) else "DIFFERENT")
            rec["status"] = "ok"
        except NotImplementedError as exc:
            rec["status"] = "declined"
            rec["error"] = str(exc.__cause__ or exc)[:200]
        except Exception as exc:
            rec["status"] = "rx-error"
            rec["error"] = repr(exc)[:200]
        if final and rec.get("status") == "ok" and t_nx < 5 * target:
            # A second NetworkX run at the final size for a steadier number.
            try:
                rec["nx"] = min(t_nx, call(func, (), kwargs, "networkx")[0])
            except Exception:
                pass
        if rec["status"] == "declined" and prev is not None:
            # rustnx hands larger inputs to NetworkX (a tree too deep, say):
            # keep the last size it ran.
            prev["note"] = f"rustnx hands {size:,} nodes to NetworkX: {rec.get('error', '')[:100]}"
            print(json.dumps(prev), flush=True)
            return
        print(json.dumps(rec), flush=True)
        if final or rec["status"] != "ok":
            return
        prev = rec
        # Aim at the target assuming roughly linear growth, by 2x to 16x.
        n *= 1 << max(1, min(4, int(math.log2(max(target / max(t_nx, 1e-6), 2)))))


# --- driver -----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--one")
    ap.add_argument("--target", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--out", default="all_functions.jsonl")
    args = ap.parse_args()
    if args.one:
        import rustnx  # noqa: F401
        run_one(args.one, args.target)
        return
    from rustnx import _info

    names = sorted(n for n in _info.FUNCTIONS if n in nx.utils.backends._registered_algorithms)
    if args.only:
        names = [n for n in names if n in args.only]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(line)["name"] for line in f if line.strip()}
    with open(args.out, "a") as out:
        for i, name in enumerate(names):
            if name in done:
                continue
            cmd = [sys.executable, __file__, "--one", name, "--target", str(args.target)]
            last = None
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
                lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
                last = json.loads(lines[-1]) if lines else {"name": name, "status": "crash", "error": p.stderr[-300:]}
            except subprocess.TimeoutExpired as exc:
                lines = [l for l in (exc.stdout or b"").decode().splitlines() if l.startswith("{")]
                last = json.loads(lines[-1]) if lines else {"name": name, "status": "timeout"}
                last.setdefault("note", "timed out at the next size")
            out.write(json.dumps(last) + "\n")
            out.flush()
            s = last.get("status")
            sp = f"{last['nx'] / last['warm']:.1f}x" if s == "ok" and last.get("warm") else ""
            print(f"[{i + 1}/{len(names)}] {name}: {s} {last.get('family', '')} n={last.get('n', '')} {sp} {last.get('match', '')}", flush=True)


if __name__ == "__main__":
    main()
