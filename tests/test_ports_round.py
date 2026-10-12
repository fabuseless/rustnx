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
