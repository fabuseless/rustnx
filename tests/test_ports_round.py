"""Ports round: functions that needed real Rust ports. Results must be
exactly NetworkX's."""

import contextlib
import io
import random
import warnings

import networkx as nx
import pytest

from rustnx import algorithms


def typed(x):
    if isinstance(x, dict):
        return ("dict", [(typed(k), typed(v)) for k, v in x.items()])
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, [typed(v) for v in x])
    return (type(x).__name__, repr(x))


def snapshot(G):
    adj = [(typed(u), [(typed(v), typed(d)) for v, d in G._adj[u].items()]) for u in G]
    return type(G).__name__, typed(G.graph), [(typed(n), typed(d)) for n, d in G._node.items()], adj


# --- DOT (pydot) ----------------------------------------------------------------


def _dot_outcome(func, text):
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return "ok", snapshot(func(io.StringIO(text)))
    except NotImplementedError:
        return ("declined",)
    except Exception as exc:
        return type(exc).__name__, str(exc)


def dot_same(text):
    """rustnx reads text as pydot + NetworkX do, or declines (leaving
    the file where it was); True if rustnx read it."""
    pytest.importorskip("pydot")
    if algorithms._dot_parser_info() is None:
        pytest.skip("not pydot 4.0.1")
    ref = _dot_outcome(lambda f: nx.nx_pydot.read_dot(f, backend="networkx"), text)
    f = io.StringIO(text)
    ours = _dot_outcome(algorithms.pydot_read_dot, text)
    if ours[0] == "declined":
        return False
    assert ours == ref
    return True


DOT_DOCUMENTS = r'''
digraph { c [label="multi\\\nline"] }
graph {}
digraph G { a -> b -> c; b -> d [color=red, w=1.5]; }
strict digraph { a -> b; a -> b }
graph "my g" { a -- b; "a" -- "b" [x="1"]; a:p1 -- b:p2:n; "q":r -- s }
DiGraph G { node [shape=box]; edge [color=blue]; graph [rankdir=LR]; a; b [label="B"]; a -> b }
digraph { rankdir = LR; size="7,7"; a -> {b c d} -> e; subgraph cluster0 { x; y -> z } ; {p q} -> r }
digraph { a [label=<<b>bold</b>>]; b [label="x" + "y" + "z"]; c [label="multi\
line"] }
digraph { /* comment */ a -> b // trailing
# hash comment
 c -> d }
digraph { a [x=1e5, y=-3.5, z=.5, w=1.2.3]; b [k=-x] }
digraph { nodeA; edge -> a; graphics; Node [x=1]; node; "node" [y=2] }
digraph { a [] [b=1][c] ; b; ; }
digraph { subgraph s1 { node [k=1]; a } -> b }
digraph { a -> b [key=1] }
digraph { a [name=1] }
digraph { a -> b [src=1] }
digraph { a -- b -> c }
digraph { a:b:c:d }
digraph { "a\"b" -> "c\\" }
digraph { é -> ü; ſtrict; }
digraph { 1.5 -> 2 }
strictly digraph { a }
digraph { a } digraph { b }
digraph { a } garbage here
digraph { a -> }
graph { a -- b; a -- b; b -- a }
strict graph { a -- b [w=1]; b -- a [w=2] }
digraph { a [label=x,] }
digraph { a=b; c=d }
digraph{a->b}
digraph { a b c }
  digraph { "" -> a }
digraph { a [x=y z=w] }
digraph { a [x = "1" + "2"] }
'''.strip().split("\n")


@pytest.mark.parametrize("index", range(len(DOT_DOCUMENTS)))
def test_read_dot_documents(index):
    dot_same(DOT_DOCUMENTS[index].replace("\\n", "\n"))


def test_read_dot_reads_common_documents():
    read = sum(dot_same(doc.replace("\\n", "\n")) for doc in DOT_DOCUMENTS)
    assert read >= 25


@pytest.mark.parametrize("cls", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_read_dot_written_by_networkx(cls):
    pytest.importorskip("pydot")
    rnd = random.Random(1)
    G = cls(nx.gnm_random_graph(25, 60, seed=2, directed=cls().is_directed()))
    for u, v, d in G.edges(data=True):
        d["weight"] = rnd.choice([1, 2.5, "x y", 'q"uote'])
    for n in G:
        G.nodes[n]["label"] = f"n{n}"
    G.graph["rankdir"] = "LR"
    buf = io.StringIO()
    nx.nx_pydot.write_dot(G, buf)
    assert dot_same(buf.getvalue())


@pytest.mark.parametrize("seed", range(20))
def test_read_dot_random_documents(seed):
    toks = ["a", "b", "c", "1", "-2.5", '"q r"', '"x"+"y"', "<h<i>>", "->", "--", "[", "]", "=",
            ",", ";", "{", "}", "subgraph", "node", "edge", "graph", ":", "p", "/*c*/", "#c\n",
            "//c\n", "strict", " ", "\n", "Node", "x1.y", "k=1"]
    rnd = random.Random(seed)
    for _ in range(100):
        head = rnd.choice(["digraph", "graph", "strict digraph", "Graph G", 'digraph "n"'])
        body = " ".join(rnd.choice(toks) for _ in range(rnd.randint(0, 20)))
        dot_same(f"{head} {{ {body} }}")


def test_read_dot_file_position_kept_on_decline(tmp_path):
    pytest.importorskip("pydot")
    path = tmp_path / "g.dot"
    path.write_text("digraph { a -> }")
    with open(path) as f:
        with pytest.raises(NotImplementedError):
            algorithms.pydot_read_dot(f)
        assert f.tell() == 0


# --- Randomized measures (random.Random / NumPy RandomState replays) --------------


def _run(func, G, make_seed, **kwargs):
    seed = make_seed()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = func(G, seed=seed, **kwargs)
    except Exception as exc:
        return type(exc).__name__, str(exc)
    if isinstance(result, nx.Graph):
        result = snapshot(result)
    after = seed.random() if isinstance(seed, random.Random) else seed.random_sample()
    return "ok", result, after


def _small_graphs(s):
    rnd = random.Random(s)
    G = nx.connected_watts_strogatz_graph(rnd.randint(4, 30), 4, rnd.random(), seed=s)
    if s % 3 == 0:
        G = nx.relabel_nodes(G, {v: f"n{v}" for v in G})
    if s % 4 == 0:
        u, v = list(G)[:2]
        G.add_edge(u, v, weight=2.5)
        G.nodes[u]["c"] = 1
        G.graph["g"] = 1
    if s % 5 == 0:
        G = nx.gnm_random_graph(rnd.randint(4, 20), rnd.randint(2, 30), seed=s)
    return G


@pytest.mark.parametrize("s", range(25))
def test_reference_graphs_and_approximate_clustering(s):
    G = _small_graphs(s)
    for name, options in [
        ("random_reference", [{}, {"niter": 3}, {"connectivity": False}, {"niter": 0}]),
        ("lattice_reference", [{}, {"niter": 2}, {"connectivity": False}]),
    ]:
        func = getattr(nx, name)
        for kwargs in options:
            ref = _run(lambda G, **k: func(G, backend="networkx", **k), G, lambda: random.Random(s), **kwargs)
            ours = _run(lambda G, **k: func(G, backend="rustnx", **k), G, lambda: random.Random(s), **kwargs)
            assert ours == ref
    for kwargs in ({}, {"trials": 3000}):
        func = nx.approximation.average_clustering
        ref = _run(lambda G, **k: func(G, backend="networkx", **k), G, lambda: random.Random(s), **kwargs)
        ours = _run(lambda G, **k: func(G, backend="rustnx", **k), G, lambda: random.Random(s), **kwargs)
        assert ours == ref


@pytest.mark.parametrize("name", ["sigma", "omega"])
@pytest.mark.parametrize("s", range(4))
def test_small_world_coefficients(name, s):
    G = nx.connected_watts_strogatz_graph(30 + 5 * s, 4, 0.2, seed=s)
    func = getattr(nx, name)
    ref = _run(lambda G, **k: func(G, backend="networkx", **k), G, lambda: random.Random(s), niter=2, nrand=2)
    ours = _run(lambda G, **k: func(G, backend="rustnx", **k), G, lambda: random.Random(s), niter=2, nrand=2)
    assert ours == ref


@pytest.mark.parametrize("s", range(12))
def test_generate_random_paths(s):
    import itertools

    np = pytest.importorskip("numpy")
    G = _small_graphs(s)
    rnd = random.Random(s)
    if s % 2:
        for u, v in G.edges():
            G[u][v]["weight"] = rnd.choice([1, 2, 0.5, 3.25])
    options = [{}, {"path_length": 0}, {"path_length": 9}, {"weight": None}]
    if nx.__version__ >= "3.5":
        options.append({"source": list(G)[1]})
    for kwargs in options:
        for take in (None, 3):
            out = []
            for backend in ("networkx", "rustnx"):
                rs = np.random.RandomState(s)
                index_map = {}
                try:
                    gen = nx.generate_random_paths(G, 30, index_map=index_map, seed=rs, backend=backend, **kwargs)
                    first = next(gen)
                    touched = rs.random_sample()  # the generator used between paths
                    rest = list(gen) if take is None else list(itertools.islice(gen, take))
                    out.append(([typed(p) for p in [first] + rest], typed(index_map), touched, rs.random_sample()))
                except ValueError as exc:  # isolated nodes: NumPy refuses NaN probabilities
                    out.append(("ValueError", str(exc)))
                except NotImplementedError:  # rustnx leaves isolated nodes to NetworkX
                    out.append(("declined",))
            if ("declined",) in out:
                continue
            assert out[0] == out[1]


@pytest.mark.parametrize("name", ["panther_similarity", "panther_vector_similarity"])
@pytest.mark.parametrize("s", range(3))
def test_panther(name, s):
    import inspect

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    if not hasattr(nx, name) or "seed" not in inspect.signature(getattr(nx, name)).parameters:
        pytest.skip("this NetworkX has no seeded " + name)
    G = nx.connected_watts_strogatz_graph(40 + 10 * s, 6, 0.2, seed=s)
    func = getattr(nx, name)
    out = []
    for backend in ("networkx", "rustnx"):
        rs = np.random.RandomState(s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = func(G, 0, seed=rs, backend=backend)
        out.append((typed(result), rs.random_sample()))
    assert out[0] == out[1]


def _complete(n, seed, directed=False):
    G = nx.complete_graph(n, create_using=nx.DiGraph if directed else nx.Graph)
    r = random.Random(seed)
    pos = {v: (r.random(), r.random()) for v in G}
    for u, v in G.edges():
        G[u][v]["weight"] = round(((pos[u][0] - pos[v][0]) ** 2 + (pos[u][1] - pos[v][1]) ** 2) ** 0.5 * 100, 3)
    return G


@pytest.mark.parametrize("s", range(6))
def test_traveling_salesman(s):
    pytest.importorskip("scipy")
    G = _complete(10 + 5 * s, s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert (nx.approximation.christofides(G, backend="rustnx")
                == nx.approximation.christofides(G, backend="networkx"))
        for kwargs in ({}, {"cycle": False}, {"nodes": list(G)[: len(G) // 2]}):
            assert (nx.approximation.traveling_salesman_problem(G, backend="rustnx", **kwargs)
                    == nx.approximation.traveling_salesman_problem(G, backend="networkx", **kwargs))
        H = nx.connected_watts_strogatz_graph(30, 4, 0.3, seed=s)
        for u, v in H.edges():
            H[u][v]["weight"] = (u * 7 + v * 3) % 11 + 1
        assert (nx.approximation.traveling_salesman_problem(H, backend="rustnx")
                == nx.approximation.traveling_salesman_problem(H, backend="networkx"))


# --- Louvain ------------------------------------------------------------------


def _louvain_graph(s):
    rnd = random.Random(s)
    n = rnd.randint(1, 70)
    kind = s % 6
    if kind == 0:
        G = nx.connected_watts_strogatz_graph(max(n, 5), 4, 0.3, seed=s)
    elif kind == 1:
        G = nx.gnm_random_graph(n, rnd.randint(0, 3 * n), seed=s)
    elif kind == 2:
        G = nx.gnm_random_graph(n, rnd.randint(0, 3 * n), seed=s, directed=True)
    elif kind == 3:
        G = nx.planted_partition_graph(4, max(n // 4, 2), 0.7, 0.05, seed=s)
    elif kind == 4:
        G = nx.relabel_nodes(nx.karate_club_graph(), {v: f"n{v}" for v in range(34)})
    else:
        G = nx.gnp_random_graph(n, 0.1, seed=s, directed=True)
    if s % 3 == 0:
        for u, v in G.edges():
            G[u][v]["weight"] = rnd.choice([1, 2, 0.5, 3.25, 7])
    if s % 7 == 0 and len(G):
        G.add_edge(list(G)[0], list(G)[0], weight=2)
    return G


def _louvain_outcome(func, G, s, **kwargs):
    seed = random.Random(s)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = func(G, seed=seed, **kwargs)
            if not isinstance(out, list):
                out = list(out)
    except NotImplementedError:
        return ("declined",)
    except Exception as exc:
        return type(exc).__name__, str(exc)
    # Each set's iteration order too: the sets are built as NetworkX's.
    sets = [[typed(list(c)) for c in part] for part in out] if out and isinstance(out[0], list) else [typed(list(c)) for c in out]
    return "ok", sets, seed.random()


@pytest.mark.parametrize("s", range(40))
def test_louvain(s):
    G = _louvain_graph(s)
    for kwargs in ({}, {"resolution": 0.5}, {"resolution": 2}, {"threshold": 0.01}, {"weight": None}, {"weight": "w"}):
        for name in ("louvain_partitions", "louvain_communities"):
            func = getattr(nx.community, name)
            ref = _louvain_outcome(lambda G, **k: func(G, backend="networkx", **k), G, s, **kwargs)
            ours = _louvain_outcome(lambda G, **k: func(G, backend="rustnx", **k), G, s, **kwargs)
            assert ours == ref


def test_louvain_is_used():
    assert algorithms._louvain_variant() is not None
    G = nx.karate_club_graph()
    # Raises NotImplementedError if rustnx declines (backend given explicitly).
    assert list(nx.community.louvain_partitions(G, seed=random.Random(1), backend="rustnx"))


# --- lukes, approximate k-components, random spanning trees, ATSP ---------------


def _run(func, *args, **kwargs):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return "ok", func(*args, **kwargs)
    except Exception as exc:
        return type(exc).__name__, str(exc)


def _tree(n, seed, directed):
    T = nx.random_labeled_tree(n, seed=seed) if hasattr(nx, "random_labeled_tree") else nx.random_tree(n, seed=seed)
    rnd = random.Random(seed)
    if rnd.random() < 0.3:
        T = nx.relabel_nodes(T, {v: (f"s{v}" if v % 2 else (v, v)) for v in T})
    if directed:
        T = nx.bfs_tree(T, next(iter(T)))
    for u, v in T.edges:
        T.edges[u, v]["w"] = rnd.choice([rnd.random(), rnd.randint(1, 5)])
    for v in T:
        T.nodes[v]["nw"] = rnd.randint(1, 3)
    return T


@pytest.mark.parametrize("s", range(20))
def test_lukes_partitioning(s):
    for directed in (False, True):
        T = _tree(1 + s % 25, s, directed)
        for kw in ({"max_size": 4}, {"max_size": 6, "node_weight": "nw", "edge_weight": "w"},
                   {"max_size": 3, "edge_weight": "w"}, {"max_size": 0}):
            def run(backend):
                random.seed(s)
                out = _run(nx.community.lukes_partitioning, T, backend=backend, **kw)
                if out[0] == "ok" and out[1] is not None:
                    out = "ok", [list(c) for c in out[1]]
                return out, random.random()
            assert run("rustnx") == run("networkx")


def test_fast_descendants_iterate_the_same():
    from rustnx import _hybrid

    for G in (nx.gnp_random_graph(40, 0.08, seed=1, directed=True), nx.gnp_random_graph(30, 0.1, seed=2),
              nx.relabel_nodes(nx.path_graph(9, create_using=nx.DiGraph), {i: str(i) for i in range(9)})):
        for v in G:
            assert list(_hybrid._descendants(G, v)) == list(nx.descendants(G, v, backend="networkx"))
    with pytest.raises(nx.NetworkXError):
        _hybrid._descendants(nx.path_graph(3, create_using=nx.DiGraph), 7)


@pytest.mark.parametrize("s", range(6))
def test_approximate_k_components(s):
    from networkx.algorithms import approximation as ap

    G = [nx.connected_watts_strogatz_graph(30 + 10 * s, 4 + s % 3, 0.3, seed=s), nx.gnm_random_graph(40, 160, seed=s),
         nx.karate_club_graph()][s % 3]

    def run(backend):
        out = _run(ap.k_components, G, backend=backend)
        if out[0] == "ok":
            out = "ok", [(k, [list(c) for c in v]) for k, v in out[1].items()]
        return out

    assert run("rustnx") == run("networkx")


def test_approximate_k_components_uses_rust():
    from rustnx import _hybrid

    assert "approximate_local_node_connectivity" in _hybrid.DISPATCH_HELPERS
    assert _hybrid.scratch() is None


def _spanning_outcome(G, backend, **kw):
    out = _run(nx.random_spanning_tree, G, backend=backend, **kw)
    if out[0] == "ok":
        T = out[1]
        out = "ok", list(T.nodes), list(T.edges), T.graph
    return out


@pytest.mark.parametrize("s", range(60))
def test_random_spanning_tree(s):
    rnd = random.Random(s)
    n = rnd.randint(0, 14)
    G = nx.gnm_random_graph(n, rnd.randint(0, n * (n - 1) // 2), seed=s)
    if rnd.random() < 0.3:
        G = nx.relabel_nodes(G, {v: (f"s{v}" if v % 2 else (v, 1)) for v in G})
    for u, v in G.edges:
        G.edges[u, v]["w"] = rnd.randint(1, 6)
    for kw in ({}, {"weight": "w"}):
        ours = _spanning_outcome(G, "rustnx", seed=s, **kw)
        if ours[0] == "NotImplementedError":
            assert not algorithms._random_spanning_tree_recognized()
            continue
        assert ours == _spanning_outcome(G, "networkx", seed=s, **kw)
    random.seed(s)
    ours = _spanning_outcome(G, "rustnx"), random.random()
    random.seed(s)
    assert ours == (_spanning_outcome(G, "networkx"), random.random()) or ours[0][0] == "NotImplementedError"


def test_random_spanning_tree_declines():
    G = nx.cycle_graph(5)
    G.add_edge(0, 0)
    H = nx.cycle_graph(5)
    nx.set_edge_attributes(H, 0.5, "w")
    for graph, kw in ((G, {}), (H, {"weight": "w"}), (nx.cycle_graph(5), {"multiplicative": False, "weight": "w"})):
        with pytest.raises(NotImplementedError):
            nx.random_spanning_tree(graph, seed=1, backend="rustnx", **kw)


@pytest.mark.parametrize("s", range(8))
def test_asadpour_and_held_karp(s):
    from networkx.algorithms import approximation as ap

    rnd = random.Random(s)
    n = rnd.randint(3, 6)
    A = nx.complete_graph(n, create_using=nx.DiGraph)
    for u, v in A.edges:
        A.edges[u, v]["weight"] = rnd.choice([rnd.randint(1, 100), rnd.random() * 50])

    def run(func, backend, **kw):
        out = _run(func, A, backend=backend, **kw)
        if out[0] == "ok" and isinstance(out[1], tuple):
            out = "ok", (out[1][0], snapshot(out[1][1]) if hasattr(out[1][1], "edges") else out[1][1])
        return out

    for func, kw in ((ap.asadpour_atsp, {"seed": s}), (ap.traveling_salesman.held_karp_ascent, {})):
        ref = run(func, "networkx", **kw)
        # NetworkX keeps the minimum 1-arborescences in a set of graphs,
        # which iterates in memory-address order; compare only where its
        # own runs agree.
        if ref != run(func, "networkx", **kw):
            continue
        assert run(func, "rustnx", **kw) == ref


def _walk(G, backend, n, **kw):
    import itertools

    try:
        it = nx.random_walk(G, backend=backend, **kw)
    except Exception as exc:
        return "call", type(exc).__name__, str(exc)
    out = []
    try:
        out.extend(itertools.islice(it, n))
        return "ok", out
    except Exception as exc:
        return type(exc).__name__, str(exc), out


@pytest.mark.parametrize("s", range(30))
def test_random_walk(s):
    if not algorithms._random_walk_recognized():
        pytest.skip("this NetworkX has no random_walk rustnx follows")
    rnd = random.Random(s)
    n = rnd.randint(1, 12)
    G = nx.gnp_random_graph(n, rnd.random() * 0.5, seed=s, directed=rnd.random() < 0.5)
    for u, v in G.edges:
        r = rnd.random()
        G.edges[u, v]["w"] = 0 if r < 0.2 else (-1 if r < 0.25 else (rnd.randint(1, 4) if r < 0.6 else rnd.random()))
    for kw in ({}, {"weight": "w"}, {"weight": "missing"}):
        for start in (0, n + 5):
            assert _walk(G, "rustnx", 300, start=start, seed=s, **kw) == _walk(G, "networkx", 300, start=start, seed=s, **kw)
    random.seed(s)
    ours = _walk(G, "rustnx", 50, start=0), random.random()
    random.seed(s)
    assert ours == (_walk(G, "networkx", 50, start=0), random.random())


@pytest.mark.parametrize("s", range(6))
def test_random_walk_follows_graph_changes(s):
    import itertools

    if not algorithms._random_walk_recognized():
        pytest.skip("this NetworkX has no random_walk rustnx follows")
    runs = []
    for backend in ("rustnx", "networkx"):
        G = nx.cycle_graph(8)
        out = []
        for i, x in enumerate(itertools.islice(nx.random_walk(G, start=0, seed=random.Random(s), backend=backend), 200)):
            out.append(x)
            if i % 17 == 5:
                G.add_edge(x, (x * 3 + i) % 11)
            if i % 29 == 7 and G.has_edge(x, (x + 1) % 8):
                G.remove_edge(x, (x + 1) % 8)
            if i == 150:
                G.__networkx_cache__ = None
            if i in (160, 170):
                G.add_edge(x, 20 + i)
        runs.append(out)
    assert runs[0] == runs[1]
