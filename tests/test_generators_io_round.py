"""Generators (todo item 49) and I/O (item 50) added in the generators/I/O
round: each must give exactly NetworkX's graph, including node and
adjacency order, key types and graph attributes, and advance the random
generator exactly as NetworkX does."""

import random
import warnings

import networkx as nx
import pytest

np = pytest.importorskip("numpy")


def snapshot(G):
    def key(x):
        return (type(x).__name__, x)

    multi = G.is_multigraph()
    edges = list(G.edges(keys=True, data=True) if multi else G.edges(data=True))
    adj = [(key(u), [key(v) for v in G._adj[u]]) for u in G]
    pred = [(key(u), [key(v) for v in G._pred[u]]) for u in G] if G.is_directed() else None
    return type(G).__name__, G.graph, list(G.nodes(data=True)), edges, adj, pred


def outcome(func, *args, **kwargs):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = func(*args, **kwargs)
    except Exception as exc:  # compare errors too
        return type(exc).__name__, str(exc)
    if isinstance(result, list):
        return "ok", [snapshot(g) for g in result]
    return "ok", snapshot(result) if isinstance(result, nx.Graph) else result


def same(make_call):
    """Both backends from identical fresh arguments; rustnx may decline."""
    ref = outcome(*make_call("networkx"))
    ours = outcome(*make_call("rustnx"))
    if ours[0] == "NotImplementedError":
        return False
    assert ours == ref
    return True


def seeds(seed):
    out = [lambda: seed]
    if nx.__version__ >= "3.5":
        out.append(lambda: np.random.RandomState(seed))
    else:
        out.append(lambda: random.Random(seed))
    return out


@pytest.mark.parametrize("seed", range(15))
def test_random_k_out_graph(seed):
    ran = 0
    for n, k, alpha, self_loops in [(20, 3, 1.0, True), (20, 3, 0.5, False), (15, 4, 2, True),
                                    (30, 2, 0.0, True), (10, 1, 1, False), (1, 1, 1.0, False),
                                    (12, 12, 0.25, True), (5, 2, -1.0, True)]:
        for make_seed in seeds(seed):
            ran += same(lambda b: (lambda: nx.random_k_out_graph(
                n, k, alpha, self_loops=self_loops, seed=make_seed(), backend=b),))
    assert ran >= 8


def test_random_k_out_graph_advances_the_generator():
    if nx.__version__ >= "3.5":
        a, b = np.random.RandomState(5), np.random.RandomState(5)
        draw = lambda s: s.random_sample()  # noqa: E731
    else:
        a, b = random.Random(5), random.Random(5)
        draw = lambda s: s.random()  # noqa: E731
    nx.random_k_out_graph(30, 3, 1.0, seed=a, backend="networkx")
    nx.random_k_out_graph(30, 3, 1.0, seed=b, backend="rustnx")
    assert draw(a) == draw(b)


@pytest.mark.parametrize("i", list(range(0, 1253, 31)) + [1252, -1, 1253])
def test_graph_atlas(i):
    assert same(lambda b: (lambda: nx.graph_atlas(i, backend=b),))


def test_graph_atlas_g():
    assert same(lambda b: (lambda: nx.graph_atlas_g(backend=b),))


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("n", [0, 1, 2, 7, 30])
def test_random_unlabeled_trees(seed, n):
    for func, extra in [(nx.random_unlabeled_tree, {"number_of_trees": 3}),
                        (nx.random_unlabeled_rooted_tree, {"number_of_trees": 3}),
                        (nx.random_unlabeled_rooted_forest, {"number_of_forests": 3, "q": 4})]:
        for kwargs in ({}, extra):
            assert same(lambda b: (lambda: func(n, seed=random.Random(seed), backend=b, **kwargs),))


def test_tree_counts_match_networkx():
    from networkx.generators import trees

    from rustnx import algorithms

    counts = [0, 1]
    trees._num_rooted_trees(200, counts)
    assert counts == [algorithms._num_rooted_trees_cached(i, [0, 1]) for i in range(201)]
    for q in (1, 3, 50, 200):
        forests = [1]
        trees._num_rooted_forests(200, q, forests)
        assert forests == [algorithms._num_rooted_forests_cached(i, q, [1]) for i in range(201)]


# --- GraphML ------------------------------------------------------------------


def typed(x):
    """``x`` with the type of every value and key, so 1, 1.0 and True differ."""
    if isinstance(x, dict):
        return ("dict", [(typed(k), typed(v)) for k, v in x.items()])
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, [typed(v) for v in x])
    return (type(x).__name__, repr(x))


def typed_snapshot(G):
    """Everything about ``G``: node and adjacency order, every key and value
    with its type, and which adjacency entries share one dict."""
    adj = [(typed(u), [(typed(v), typed(d)) for v, d in G._adj[u].items()]) for u in G]
    pred = None
    shared = None
    if G.is_directed():
        pred = [(typed(u), [(typed(v), typed(d)) for v, d in G._pred[u].items()]) for u in G]
        shared = all(G._succ[u][v] is G._pred[v][u] for u in G for v in G._succ[u])
    else:
        shared = all(G._adj[u][v] is G._adj[v][u] for u in G for v in G._adj[u])
    nodes = [(typed(n), typed(d)) for n, d in G._node.items()]
    return type(G).__name__, typed(G.graph), nodes, adj, pred, shared


def typed_outcome(func, *args, **kwargs):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = func(*args, **kwargs)
    except NotImplementedError:
        return "declined", None
    except Exception as exc:
        return type(exc).__name__, str(exc)
    return "ok", typed_snapshot(result)


def graphml_same(data, **kwargs):
    """rustnx's read_graphml and parse_graphml on ``data`` (bytes) give
    NetworkX's graph or decline (leaving the file where it was); True if
    rustnx read it."""
    import io

    from rustnx import algorithms

    ref = typed_outcome(nx.read_graphml, io.BytesIO(data), backend="networkx", **kwargs)
    f = io.BytesIO(data)
    ours = typed_outcome(algorithms.read_graphml, f, **kwargs)
    if ours[0] == "declined":
        assert f.tell() == 0
        return False
    assert ours == ref
    text = data.decode("utf-8")
    ref = typed_outcome(nx.parse_graphml, text, backend="networkx", **kwargs)
    ours = typed_outcome(algorithms.parse_graphml, text, **kwargs)
    assert ours[0] == "declined" or ours == ref
    return True


GRAPHML_READ_OPTIONS = [
    {},
    {"node_type": int},
    {"node_type": float},
    {"node_type": lambda s: ("n", s)},
    {"force_multigraph": True},
    {"edge_key_type": str},
    {"edge_key_type": float},
]


def _graphml_test_graphs():
    rnd = random.Random(1)
    graphs = [nx.karate_club_graph()]
    for cls in (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph):
        base = nx.gnm_random_graph(30, 80, seed=3, directed=cls().is_directed())
        H = cls()
        H.graph.update(name="x", num=3, fl=1.5, flag=True)
        for n in base:
            H.add_node(n, w=rnd.random(), c=rnd.choice("ab"), i=rnd.randint(0, 9),
                       b=rnd.random() < 0.5)
        for u, v in base.edges():
            H.add_edge(u, v, weight=rnd.random(), lab="e<&>\"'", k=rnd.randint(-5, 5))
            if H.is_multigraph() and rnd.random() < 0.3:
                H.add_edge(u, v, weight=2.0)
        graphs.append(H)
        graphs.append(nx.relabel_nodes(H, {n: f"n {n}\t&" for n in H}))
        E = cls()
        E.add_nodes_from([3, 1, 2])
        graphs.append(E)
        S = cls()
        S.add_edges_from([(1, 1), (1, 2), (2, 1), (2, 3)])
        graphs.append(S)
    return graphs


@pytest.mark.parametrize("index", range(17))
def test_read_graphml_written_by_networkx(index):
    import io

    G = _graphml_test_graphs()[index]
    read = 0
    for write_kwargs in ({}, {"named_key_ids": True}, {"edge_id_from_attribute": "k"},
                         {"infer_numeric_types": True}):
        buf = io.BytesIO()
        try:
            nx.write_graphml(G, buf, **write_kwargs)
        except (TypeError, KeyError):
            continue
        for kwargs in GRAPHML_READ_OPTIONS:
            read += graphml_same(buf.getvalue(), **kwargs)
    assert read


_NS = 'xmlns="http://graphml.graphdrawing.org/xmlns"'
_KEYS = (
    '<key id="d0" for="node" attr.name="color" attr.type="string"><default>yellow</default></key>'
    '<key id="d1" for="edge" attr.name="weight" attr.type="double"/>'
    '<key id="d2" for="node" attr.name="ok" attr.type="boolean"><default>TRUE</default></key>'
    '<key id="d3" for="graph" attr.name="n" attr.type="int"/>'
    '<key id="d4" for="all" attr.name="x" attr.type="long"/>'
    '<key id="k" for="edge" attr.name="key" attr.type="string"/>'
)


def _doc(body, keys=_KEYS, head='<?xml version="1.0" encoding="UTF-8"?>\n',
         graph='edgedefault="undirected"'):
    return f"{head}<graphml {_NS}>{keys}<graph {graph}>{body}</graph></graphml>"


GRAPHML_DOCUMENTS = [
    _doc('<node id="a"/><node id="b"/><edge source="a" target="b"/>'),
    _doc('<node id="a"><data key="d0">red</data></node><node id="b"/>'
         '<edge source="a" target="b"><data key="d1">1.5</data></edge>'),
    _doc('<node id="a"><data key="d0"></data></node><node id="b"><data key="d0"/></node>'),
    _doc('<node id="a"><data key="d0">  </data></node>'),
    _doc('<node id="a"><data key="d0"><![CDATA[x<y]]></data></node>'),
    _doc('<node id="a"><data key="d0"><![CDATA[]]></data></node>'),
    _doc('<node id="a"><data key="d0">a&amp;b&#65;&#x42;&lt;</data></node>'),
    _doc('<node id="a"><data key="d0">l1\r\nl2\rl3\n</data></node>'),
    _doc('<node id="a"><data key="d0">ab<!--c-->cd<?pi x?>ef</data></node>'),
    _doc('<node id="a&#10;b\tc&#9;d\r\ne"/>'),
    _doc('<node id="a"><data key="d2">False</data></node><node id="b"><data key="d2">1</data></node>'),
    _doc('<node id="c"><data key="d2">yes</data></node>'),
    _doc('<node id="a"><data key="d4"> 12 </data></node><node id="b"><data key="d4">1_2</data></node>'),
    _doc('<node id="a"><data key="d4">١٢</data></node>'),
    _doc('<node id="a"><data key="d1">nan</data></node><node id="b"><data key="d1">-1e400</data></node>'),
    _doc('<node id="a"><data key="d1"> 1_0.5 </data></node><node id="b"><data key="d1">.5e-3</data></node>'),
    _doc('<node id="a"><data key="d4">123456789012345678901234567890</data></node>'),
    _doc('<node id="a"><data key="zz">1</data></node>'),
    _doc('<data key="d3">5</data><node id="a"/><data key="d0">g</data>'),
    _doc('<node/><edge/>'),
    _doc('<node id="a"/><edge source="a" target="b" directed="true"/>'),
    _doc('<node id="a"/><edge source="a" target="b" directed="false"/>', graph='edgedefault="directed"'),
    _doc('<edge source="a" target="b" id="7"/><edge source="a" target="b" id="x"/>'
         '<edge source="b" target="a"/>'),
    _doc('<edge source="a" target="b" id="7"/><edge source="a" target="b" id="7"/>'),
    _doc('<edge source="a" target="b" id="7"/><edge source="b" target="a" id="8"/>',
         graph='edgedefault="directed"'),
    _doc('<edge source="a" target="b"><data key="k">kk</data></edge>'),
    _doc('<edge source="a" target="b"><data key="k">kk</data></edge>'
         '<edge source="a" target="b"><data key="k">kk</data><data key="d1">2</data></edge>'),
    _doc('<edge source="a" target="b" id=""/><edge source="b" target="c" id="1.0"/>'),
    _doc('<node id="1"/><node id="01"/><node id="1.0"/><edge source="1" target="01" id="0x5"/>'
         '<edge source="-0" target="0"/><edge source="nan" target="nan"/>'),
    _doc('<edge source="1" target="2"/><edge source="2" target="3" id="e"/><edge source="3" target="1"/>'
         '<edge source="3" target="3"/><node id="2"/>'),
    _doc('<node id="a"/>', keys='<key id="q" for="node" attr.name="v"/>'),
    _doc('<node id="a"/>', keys='<key id="q" for="node" attr.type="int"/>'),
    _doc('<node id="a"/>', keys='<key id="q" for="node" attr.name="v" attr.type="int"><default>x</default></key>'),
    _doc('<node id="a"/>', keys='<key id="q" for="node" attr.name="v" attr.type="int"><default/></key>'),
    _doc('<node id="a"/>', keys='<key id="q" for="node" attr.name="v" attr.type="string"><default/></key>'),
    _doc('<node id="a"/>', keys='<key id="q" for="node" attr.name="v" attr.type="foo"/>'),
    _doc('<node id="a"/>', keys='<key id="q" for="node" attr.name="v" attr.type="string"/>'
                                '<key id="q" for="edge" attr.name="w" attr.type="int"/>'),
    _doc('<node id="a"/>', keys='<key id="q" for="graphml" attr.name="v" attr.type="string">'
                                '<default>1</default></key>'),
    _doc('<node id="a"/><node id="a"><data key="d0">z</data></node>'),
    _doc('<node id="a"><port name="p"/></node>'),
    _doc('<node id="a"/><hyperedge/>'),
    _doc('<node id="a"><graph><node id="b"/></graph></node>'),
    _doc('<node id="a" yfiles.foldertype="group"><graph><node id="b"/></graph></node>'),
    _doc('<node id="a"><foo/><data key="d0">r</data></node>'),
    _doc('<node id="a"/>', graph='edgedefault="directed"'),
    _doc('<node id="a"/>', graph=''),
    _doc('<node id="a"/>', head=''),
    _doc('<node id="a"/>', head='<?xml version="1.0" encoding="ISO-8859-1"?>'),
    _doc('<node id="a"/>', head='﻿'),
    _doc('<node id="é"/>'),
    _doc('<node id="a"/><x:node xmlns:x="http://graphml.graphdrawing.org/xmlns" id="b"/>'),
    _doc('<node id="a"/><node xmlns="other" id="b"/>'),
    '<graphml><graph><node id="a"/></graph></graphml>',
    f'<graphml {_NS}></graphml>',
    f'<graphml {_NS}><graph/><graph/></graphml>',
    _doc('<node id="a"/>') + '<!-- trailing -->',
    _doc('<node id="a">'),
    _doc('<node id="a" id="b"/>'),
    _doc('<node id="a"/>&'),
    _doc('<node id="a<"/>'),
    _doc('<node id="&foo;"/>'),
    _doc('<node id="a"/><y:x/>'),
    _doc('<node id="a"/>', head='<!DOCTYPE g [<!ENTITY e "x">]>'),
]


@pytest.mark.parametrize("index", range(len(GRAPHML_DOCUMENTS)))
def test_read_graphml_documents(index):
    for kwargs in GRAPHML_READ_OPTIONS:
        graphml_same(GRAPHML_DOCUMENTS[index].encode("utf-8"), **kwargs)


def test_read_graphml_reads_common_documents():
    """The usual cases are read by rustnx, not declined."""
    for index in (0, 1, 2, 6, 7, 9, 22, 25, 28, 29):
        assert graphml_same(GRAPHML_DOCUMENTS[index].encode("utf-8")), index


def test_read_graphml_dispatch(tmp_path):
    import gzip

    G = _graphml_test_graphs()[1]
    path = tmp_path / "g.graphml"
    nx.write_graphml(G, path)
    gz = tmp_path / "g.graphml.gz"
    gz.write_bytes(gzip.compress(path.read_bytes()))
    for p in (path, str(path), gz):
        ours = nx.read_graphml(p, backend="rustnx")
        assert typed_snapshot(ours) == typed_snapshot(nx.read_graphml(p, backend="networkx"))
    with open(path, "rb") as f:
        f.read(10)  # not at the start: NetworkX reads from here and fails
        assert typed_outcome(nx.read_graphml, f, backend="networkx")[0] == "ParseError"
        f.seek(10)
        with pytest.raises(NotImplementedError):
            nx.read_graphml(f, backend="rustnx")
        assert f.tell() == 10


# --- GEXF ---------------------------------------------------------------------


def gexf_same(data, **kwargs):
    """rustnx's read_gexf on ``data`` (bytes) gives NetworkX's graph or
    declines (leaving the file where it was); True if rustnx read it."""
    import io

    from rustnx import algorithms

    ref = typed_outcome(nx.read_gexf, io.BytesIO(data), backend="networkx", **kwargs)
    f = io.BytesIO(data)
    ours = typed_outcome(algorithms.read_gexf, f, **kwargs)
    if ours[0] == "declined":
        assert f.tell() == 0
        return False
    assert ours == ref
    return True


def _gexf_versions():
    from networkx.readwrite import gexf

    return list(gexf.GEXF.versions)


def _gexf_read_options():
    return [{}, {"node_type": int}, {"node_type": float}, {"relabel": True}] + [
        {"version": v} for v in _gexf_versions()
    ]


def _gexf_test_graphs():
    rnd = random.Random(1)
    graphs = [nx.karate_club_graph(), nx.les_miserables_graph()]
    for cls in (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph):
        base = nx.gnm_random_graph(30, 80, seed=3, directed=cls().is_directed())
        H = cls()
        H.graph.update(name="x", mode="static")
        for n in base:
            H.add_node(n, w=rnd.random(), c=rnd.choice("ab"), i=rnd.randint(0, 9),
                       b=rnd.random() < 0.5, label=f"L{n}")
            if rnd.random() < 0.5:
                H.nodes[n]["viz"] = {
                    "color": {"r": 1, "g": 2, "b": 3, "a": 0.5}, "size": 2.5,
                    "position": {"x": 1.0, "y": 2.0, "z": 0.0}, "shape": "disc",
                }
        for u, v in base.edges():
            H.add_edge(u, v, weight=rnd.random(), lab="e<&>\"'", k=rnd.randint(-5, 5))
            if H.is_multigraph() and rnd.random() < 0.3:
                H.add_edge(u, v, weight=2.0)
        graphs.append(H)
        graphs.append(nx.relabel_nodes(H, {n: f"n {n}" for n in H}))
        E = cls()
        E.add_nodes_from([3, 1, 2])
        graphs.append(E)
        S = cls()
        S.add_edges_from([(1, 1), (1, 2), (2, 1)])
        graphs.append(S)
        D = cls()
        D.add_node(1, start=1, end=5)
        D.add_edge(1, 2, start=2, end=3, w=[(1.0, 1, 2), (2.0, 2, 3)])
        graphs.append(D)
    return graphs


@pytest.mark.parametrize("index", range(22))
def test_read_gexf_written_by_networkx(index):
    import io

    G = _gexf_test_graphs()[index]
    read = 0
    for version in _gexf_versions():
        buf = io.BytesIO()
        try:
            nx.write_gexf(G, buf, version=version)
        except (TypeError, ValueError, KeyError):
            continue
        for kwargs in _gexf_read_options():
            read += gexf_same(buf.getvalue(), **kwargs)
    assert read


_V12 = 'xmlns="http://www.gexf.net/1.2draft" xmlns:viz="http://www.gexf.net/1.2draft/viz"'
_V11 = 'xmlns="http://www.gexf.net/1.1draft" xmlns:viz="http://www.gexf.net/1.1draft/viz"'
_V13 = 'xmlns="http://gexf.net/1.3" xmlns:viz="http://gexf.net/1.3/viz"'
_NODE_ATTRS = (
    '<attributes class="node"><attribute id="0" title="age" type="integer"><default>7</default>'
    '</attribute><attribute id="1" title="ok" type="boolean"/><attribute id="2" title="s" '
    'type="string"/><attribute id="3" title="f" type="float"/><attribute id="4" title="n" '
    'type="int"/></attributes>'
)
_EDGE_ATTRS = ('<attributes class="edge" mode="dynamic"><attribute id="0" title="w" '
               'type="double"/></attributes>')


def _gexf(body, ns=_V12, graph='defaultedgetype="undirected"',
          head='<?xml version="1.0" encoding="UTF-8"?>', meta=""):
    return f'{head}<gexf {ns} version="1.2">{meta}<graph {graph}>{body}</graph></gexf>'


GEXF_DOCUMENTS = [
    _gexf('<nodes><node id="a" label="A"/><node id="b"/></nodes><edges><edge source="a" target="b"/></edges>'),
    _gexf(_NODE_ATTRS + '<nodes><node id="a"><attvalues><attvalue for="0" value="3"/><attvalue '
          'for="1" value="true"/><attvalue for="2" value="x"/><attvalue for="3" value="1e3"/>'
          '</attvalues></node></nodes>'),
    _gexf(_NODE_ATTRS + '<nodes><node id="a"><attvalues><attvalue for="4" value="3"/></attvalues></node></nodes>'),
    _gexf(_NODE_ATTRS + '<nodes><node id="a"><attvalues><attvalue for="1" value="yes"/></attvalues></node></nodes>'),
    _gexf(_NODE_ATTRS + '<nodes><node id="a"><attvalues><attvalue for="9" value="1"/></attvalues></node></nodes>'),
    _gexf(_NODE_ATTRS + '<nodes><node id="a"><attvalues><attvalue for="2"/><attvalue for="0" '
          'value=" 5 "/></attvalues></node></nodes>'),
    _gexf(_EDGE_ATTRS + '<nodes><node id="a"/></nodes><edges><edge source="a" target="b"><attvalues>'
          '<attvalue for="0" value="1" start="1" end="2"/><attvalue for="0" value="2" start="2" '
          'end="3"/></attvalues></edge></edges>',
          graph='defaultedgetype="undirected" mode="dynamic" timeformat="double"'),
    _gexf(_EDGE_ATTRS + '<edges><edge source="a" target="b"><attvalues><attvalue for="0" value="1" '
          'start="1" end="2"/></attvalues></edge></edges>', graph='timeformat="date"'),
    _gexf(_EDGE_ATTRS + '<edges><edge source="a" target="b"><attvalues><attvalue for="0" value="1" '
          'start="1"/></attvalues></edge></edges>'),
    _gexf('<nodes><node id="a" start="1" end="2"><spells><spell start="1" end="2"/><spell start="3"/>'
          '</spells></node></nodes>', graph='timeformat="integer" start="0" end="9" name="nm"'),
    _gexf('<nodes><node id="a" start="1"/></nodes>'),
    _gexf('<nodes><node id="a"><slices><slice start="1" end="2"/><slice/></slices></node></nodes>', ns=_V11),
    _gexf('<nodes><node id="a"><viz:color r="1" g="2" b="3"/><viz:size value="2"/><viz:thickness '
          'value="1.5"/><viz:shape shape="image" uri="u"/><viz:position x="1" z="-2.5"/></node></nodes>'),
    _gexf('<nodes><node id="a"><viz:color r="1" g="2" b="3" a="0.5"/></node></nodes>', ns=_V11),
    _gexf('<nodes><node id="a"><viz:color r="1" g="2"/></node></nodes>'),
    _gexf('<nodes><node id="a"><viz:shape/></node><node id="b"><viz:color r=" 1" g="2" b="3"/></node></nodes>'),
    _gexf('<nodes><node id="a"><parents><parent for="x"/><parent/></parents></node></nodes>'),
    _gexf('<nodes><node id="a" pid="p"/><node id="b"><nodes><node id="c"/><node id="d" pid="q"><nodes>'
          '<node id="e"/></nodes></node></nodes></node><node id="c" label="again"/></nodes>'),
    _gexf('<nodes><node id="1"/><node id="2"/></nodes><edges><edge source="1" target="2" type="mutual" '
          'id="e"/><edge source="2" target="3" type="mutual"/></edges>'),
    _gexf('<nodes><node id="1"/><node id="2"/></nodes><edges><edge source="1" target="2" type="mutual" '
          'id="e"/><edge source="2" target="3" type="mutual"/></edges>', graph='defaultedgetype="directed"'),
    _gexf('<edges><edge source="2" target="1" weight="3"/><edge source="1" target="2" type="mutual"/></edges>',
          graph='defaultedgetype="directed"'),
    _gexf('<edges><edge source="2" target="1" id="k"/><edge source="1" target="2" type="mutual" id="k" '
          'label="L"/></edges>', graph='defaultedgetype="directed"'),
    _gexf('<edges><edge source="1" target="2" type="directed"/></edges>'),
    _gexf('<edges><edge source="1" target="2" type="undirected"/></edges>', graph='defaultedgetype="directed"'),
    _gexf('<edges><edge source="1" target="2" id="a"/><edge source="2" target="1" id="b" weight="2"/>'
          '<edge source="1" target="1"/><edge source="1" target="1"/></edges>'),
    _gexf('<attributes class="edge"><attribute id="k" title="networkx_key" type="long"/></attributes>'
          '<edges><edge source="1" target="2" id="a"><attvalues><attvalue for="k" value="5"/></attvalues>'
          '</edge><edge source="1" target="2" id="b"><attvalues><attvalue for="k" value="5"/></attvalues>'
          '</edge></edges>'),
    _gexf('<attributes class="edge"><attribute id="k" title="key" type="long"/></attributes><edges>'
          '<edge source="1" target="2"><attvalues><attvalue for="k" value="5"/></attvalues></edge></edges>'),
    _gexf('<attributes class="node"><attribute id="k" title="node_for_adding" type="long"/></attributes>'
          '<nodes><node id="1"><attvalues><attvalue for="k" value="5"/></attvalues></node></nodes>'),
    _gexf('<attributes class="edge"><attribute id="weight" title="weight" type="string"/><attribute '
          'id="x" title="weight2" type="string"><default>d</default></attribute></attributes><edges>'
          '<edge source="1" target="2"><attvalues><attvalue for="weight" value="5"/></attvalues></edge></edges>'),
    _gexf('<attributes class="other"/><nodes><node id="a"/></nodes>'),
    _gexf('<attributes class="node"><attribute id="0" title="t"/></attributes><attributes class="node">'
          '<attribute id="0" title="u" type="string"><default>z</default></attribute></attributes><nodes>'
          '<node id="a"><attvalues><attvalue for="0" value="v"/></attvalues></node></nodes>'),
    _gexf('<attributes class="node"><attribute id="0" title="b" type="boolean"><default>True</default>'
          '</attribute><attribute id="1" title="c" type="boolean"><default>TRUE</default></attribute></attributes>'),
    _gexf('<nodes><node/></nodes>'),
    _gexf('<edges><edge source="1"/></edges>'),
    _gexf('<nodes><node id="a"/></nodes>', ns=_V13),
    _gexf('<nodes><node id="a"/></nodes>', ns=_V11),
    _gexf('<nodes><node id="a"/></nodes>', ns='xmlns="other"'),
    _gexf('<nodes><node id="a"/></nodes>', meta='<meta><description>d &amp; e</description><keywords/></meta>'),
    _gexf('<nodes><node id="a"/></nodes>', meta='<meta><creator>c</creator></meta>'),
    _gexf('<nodes><node id="01"/><node id="1"/><node id=" 2 "/></nodes><edges><edge source="1" target="2"/></edges>'),
    _gexf('<nodes><node id="a"/></nodes>', head='<?xml version="1.0" encoding="ISO-8859-1"?>'),
    _gexf('<nodes><node id="a"/><node id="b"/></nodes>') + "<x/>",
    _gexf('<nodes><node id="a" id="b"/></nodes>'),
    _gexf('<nodes><node id="a"/></nodes><edges/><nodes><node id="z"/></nodes>'),
    _gexf('<nodes><node id="é" label="&#233;&lt;"/></nodes>'),
]


@pytest.mark.parametrize("index", range(len(GEXF_DOCUMENTS)))
def test_read_gexf_documents(index):
    for kwargs in _gexf_read_options():
        gexf_same(GEXF_DOCUMENTS[index].encode("utf-8"), **kwargs)


def test_read_gexf_reads_common_documents():
    """The usual cases are read by rustnx, not declined."""
    for index in (0, 1, 6, 12, 16, 17, 18, 19, 24, 25, 37):
        assert gexf_same(GEXF_DOCUMENTS[index].encode("utf-8")), index


def test_read_gexf_dispatch(tmp_path):
    G = _gexf_test_graphs()[2]
    path = tmp_path / "g.gexf"
    nx.write_gexf(G, path)
    for p in (path, str(path)):
        ours = nx.read_gexf(p, backend="rustnx")
        assert typed_snapshot(ours) == typed_snapshot(nx.read_gexf(p, backend="networkx"))


# --- pandas -------------------------------------------------------------------


def frame_snapshot(df):
    columns = list(df.columns)
    return (
        [typed(c) for c in columns],
        [str(t) for t in df.dtypes],
        [typed(list(df.iloc[:, i])) for i in range(len(columns))],
    )


def pandas_outcome(func, *args, **kwargs):
    pd = pytest.importorskip("pandas")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = func(*args, **kwargs)
    except NotImplementedError:
        return "declined", None
    except Exception as exc:
        return type(exc).__name__, str(exc)
    if isinstance(result, pd.DataFrame):
        return "ok", frame_snapshot(result)
    return "ok", typed_snapshot(result)


def _frames():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(1)
    n = 200
    return [
        pd.DataFrame({"source": rng.integers(0, 30, n), "target": rng.integers(0, 30, n),
                      "weight": rng.random(n), "c": rng.choice(["a", "b"], n)}),
        pd.DataFrame({"source": [f"n{i}" for i in rng.integers(0, 30, n)],
                      "target": [f"n{i}" for i in rng.integers(0, 30, n)], "w": rng.integers(0, 5, n)}),
        pd.DataFrame({"source": [1.0, float("nan"), 2.0, float("nan")], "target": [2.0, 3.0, float("nan"), 1.0],
                      "x": [1, 2, 3, 4]}),
        pd.DataFrame({"source": pd.array([1, 2, None, 3], dtype="Int64"), "target": [2, 3, 4, 1],
                      "d": pd.to_datetime(["2020-01-01"] * 4)}),
        pd.DataFrame({"source": pd.Categorical(["a", "b", "a", "c"]), "target": ["b", "c", "c", "a"],
                      "y": [None, 1.5, "s", (1, 2)]}),
        pd.DataFrame({"source": [1, 1, 2, 2], "target": [2, 2, 1, 2], "a": [1, 2, 3, 4],
                      "b": [True, False, True, None]}),
        pd.DataFrame({"source": [None, 1], "target": [1, 2]}),
        pd.DataFrame({"s": [1, 2], "t": [2, 3], "source": [9, 9]}),
        pd.DataFrame(columns=["source", "target"]),
    ]


_FROM_EDGELIST_OPTIONS = [
    {}, {"edge_attr": True}, {"edge_attr": "weight"}, {"edge_attr": ["weight", "c"]},
    {"edge_attr": ("a",)}, {"edge_attr": []}, {"create_using": nx.DiGraph},
    {"create_using": nx.DiGraph, "edge_attr": True},
    {"source": "s", "target": "t", "edge_attr": True}, {"edge_attr": "missing"},
    {"create_using": nx.MultiGraph, "edge_attr": True},
]


@pytest.mark.parametrize("index", range(9))
def test_pandas_edgelists(index):
    from rustnx import algorithms

    df = _frames()[index]
    for kwargs in _FROM_EDGELIST_OPTIONS:
        ref = pandas_outcome(nx.from_pandas_edgelist, df, backend="networkx", **kwargs)
        ours = pandas_outcome(algorithms.from_pandas_edgelist, df, **kwargs)
        assert ours[0] == "declined" or ours == ref
        if ref[0] != "ok":
            continue
        G = nx.from_pandas_edgelist(df, backend="networkx", **kwargs)
        for to_kwargs in ({}, {"source": "weight"}, {"dtype": object}, {"source": "u", "target": "v"},
                          {"nodelist": list(G)[:3]}):
            ref = pandas_outcome(nx.to_pandas_edgelist, G, backend="networkx", **to_kwargs)
            ours = pandas_outcome(nx.to_pandas_edgelist, G, backend="rustnx", **to_kwargs)
            assert ours[0] == "declined" or ours == ref


def test_pandas_edgelists_read_common_frames():
    from rustnx import algorithms

    df = _frames()[0]
    for kwargs in ({}, {"edge_attr": True}, {"create_using": nx.DiGraph, "edge_attr": ["weight"]}):
        ours = pandas_outcome(algorithms.from_pandas_edgelist, df, **kwargs)
        assert ours[0] == "ok"
        assert ours == pandas_outcome(nx.from_pandas_edgelist, df, backend="networkx", **kwargs)
    G = nx.from_pandas_edgelist(df, edge_attr=True)
    from rustnx.interface import convert_from_nx

    R = convert_from_nx(G, preserve_edge_attrs=True, name="to_pandas_edgelist")
    assert pandas_outcome(algorithms.to_pandas_edgelist, R)[0] == "ok"


def test_from_pandas_adjacency():
    import inspect

    pd = pytest.importorskip("pandas")
    from rustnx import algorithms

    rng = np.random.default_rng(2)
    M = (rng.random((20, 20)) < 0.2) * rng.integers(1, 5, (20, 20))
    names = [f"n{i}" for i in range(20)]
    frames = [
        pd.DataFrame(M), pd.DataFrame(M * 1.5, index=names, columns=names),
        pd.DataFrame(M.astype(bool)), pd.DataFrame(M, index=range(20)[::-1], columns=range(20)),
        pd.DataFrame(M[:3, :4]), pd.DataFrame(np.array([[0, np.nan], [1, 0]])),
        pd.DataFrame(np.array([[0, 1], [1, 0]], dtype=object)),
        pd.DataFrame(M[:5, :5], index=[0.5, 1, 2, 3, 4], columns=[0.5, 1, 2, 3, 4]),
    ]
    options = [{}, {"create_using": nx.DiGraph}, {"create_using": nx.MultiGraph}]
    if "nonedge" in inspect.signature(nx.from_pandas_adjacency.orig_func).parameters:
        options.append({"nonedge": 1})
    read = 0
    for df in frames:
        for kwargs in options:
            ref = pandas_outcome(nx.from_pandas_adjacency, df, backend="networkx", **kwargs)
            ours = pandas_outcome(algorithms.from_pandas_adjacency, df, **kwargs)
            if ours[0] != "declined":
                assert ours == ref
                read += 1
    assert read >= 10
