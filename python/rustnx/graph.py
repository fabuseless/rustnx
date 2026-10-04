"""Graphs stored in Rust: snapshots of NetworkX graphs, and native graphs."""

import sys

import networkx as nx

from . import _core

__all__ = ["DiGraph", "Graph", "RustnxGraph", "from_networkx", "to_networkx"]

# Key in the source graph's ``__networkx_cache__``. NetworkX clears that cache
# on every change made through its API, so while the token is still there
# the graph is unchanged since this snapshot was taken.
_SNAPSHOT_KEY = "rustnx-snapshot"


class RustnxGraph:
    """A read-only snapshot of a NetworkX graph stored in Rust.

    Nodes keep their original Python objects; the Rust core works on integer
    indices ``0..n-1`` in the same order as ``list(G)``.
    """

    __networkx_backend__ = "rustnx"

    # Converted from a MultiGraph/MultiDiGraph: neighbors appear once and edge
    # weights are the minimum over parallel edges, as NetworkX's shortest
    # path code sees them. Only functions with the same semantics accept it.
    _multigraph = False

    def __init__(self, core, nodes, index, graph_attrs, source, weight_attrs=(), multigraph=False):
        self._multigraph = multigraph
        self._core = core
        self._nodes = nodes
        self._index = index
        self.graph = graph_attrs
        # Original graph, used to convert back to NetworkX (see `convert_to_nx`).
        self._source = source
        self._weight_attrs = list(weight_attrs)
        self._snapshot_token = _snapshot_token(source)
        self.__networkx_cache__ = {}

    def _source_unchanged(self):
        cache = getattr(self._source, "__networkx_cache__", None)
        return (
            self._snapshot_token is not None
            and cache is not None
            and cache.get(_SNAPSHOT_KEY) is self._snapshot_token
        )

    def _ensure_exact_pred(self):
        """Load NetworkX's exact in-edge order (see ``CoreGraph.pred``)."""
        if self._core.has_exact_pred():  # always true for native graphs
            return
        if not self._source_unchanged():
            raise NotImplementedError("the graph changed since it was converted")
        self._core.load_exact_pred(
            self._nodes, self._index, self._source._pred, self._weight_attrs, self._multigraph
        )

    def _ensure_weight(self, attr):
        """Make sure edge attribute ``attr`` is converted (missing values
        count as 1, NetworkX's default), converting it now if needed.

        NetworkX may hand over a cached conversion made for a different
        call, e.g. one that kept all attributes on the NetworkX side.
        """
        if self._core.is_native() or any(name == attr for name, _ in self._weight_attrs):
            return
        if not isinstance(attr, str):
            raise NotImplementedError("rustnx only supports string edge attribute names")
        if not self._source_unchanged():
            raise NotImplementedError("the graph changed since it was converted")
        attrs = dict(self._weight_attrs)
        attrs[attr] = 1
        fresh = from_networkx(self._source, attrs, multigraph_ok=self._multigraph)
        self._core = fresh._core
        self._weight_attrs = fresh._weight_attrs

    def is_directed(self):
        return self._core.directed

    # --- Read-only queries --------------------------------------------------
    # A small, NetworkX-like subset. For the full NetworkX API, use
    # ``to_networkx()``; NetworkX functions accept rustnx graphs directly.

    def nodes(self):
        """Nodes, in order."""
        return list(self._nodes)

    @property
    def edges(self):
        """Edges in ``networkx.Graph.edges`` order: ``G.edges(data=False)``.

        With ``data=True``, ``(u, v, attrs)`` triples. Native graphs give the
        attributes they store; snapshots read them from the source graph.
        Iterating ``G.edges`` itself gives the ``(u, v)`` pairs, as with
        NetworkX's edge view (NetworkX's test harness does that).
        """
        return _EdgeView(self)

    def _edge_list(self, data=False):
        us, vs, ids = self._core.edges_in_order()
        nodes = self._nodes
        if not data:
            return [(nodes[u], nodes[v]) for u, v in zip(us, vs)]
        if ids is None:
            return list(self._source.edges(data=True))
        attrs = self._edge_attr_dicts()
        return [(nodes[u], nodes[v], attrs[e]) for u, v, e in zip(us, vs, ids)]

    def has_node(self, node):
        return node in self

    def has_edge(self, u, v):
        if u not in self or v not in self:
            return False
        return self._core.has_edge(self._index[u], self._index[v])

    def neighbors(self, node):
        """Neighbors (successors for directed graphs), in NetworkX order."""
        nodes = self._nodes
        return iter([nodes[i] for i in self._core.neighbors(self._node_index(node))])

    def successors(self, node):
        if not self.is_directed():
            raise nx.NetworkXError("successors is only defined for directed graphs")
        return self.neighbors(node)

    def predecessors(self, node):
        if not self.is_directed():
            raise nx.NetworkXError("predecessors is only defined for directed graphs")
        nodes = self._nodes
        return iter([nodes[i] for i in self._core.predecessors(self._node_index(node))])

    def degree(self, node=None):
        """Degree of ``node``, or ``{node: degree}`` for all nodes.

        As in NetworkX, a self-loop counts twice, and a directed graph's
        degree is in-degree plus out-degree.
        """
        degrees = self._core.degrees()
        if node is not None:
            return degrees[self._node_index(node)]
        return dict(zip(self._nodes, degrees))

    def to_networkx(self):
        """The equivalent ``networkx.Graph`` or ``networkx.DiGraph``."""
        return to_networkx(self)

    def _node_index(self, node):
        try:
            return self._index[node]
        except (KeyError, TypeError):
            raise nx.NetworkXError(f"The node {node} is not in the graph.") from None

    def _edge_attr_dicts(self):
        """Per native edge, its attribute dict."""
        src, _, attrs = self._core.native_edges()
        dicts = [{} for _ in src]
        for name, values, kinds in attrs:
            for e, (value, kind) in enumerate(zip(values, kinds)):
                if kind:
                    dicts[e][name] = _decode(value, kind)
        return dicts

    def is_multigraph(self):
        return self._multigraph

    def number_of_nodes(self):
        return len(self._nodes)

    def number_of_edges(self):
        return self._core.number_of_edges()

    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    def __contains__(self, node):
        try:
            return node in self._index
        except TypeError:
            return False

    def __repr__(self):
        kind = "directed" if self.is_directed() else "undirected"
        return (
            f"<{type(self).__name__} ({kind}) with {len(self)} nodes "
            f"and {self.number_of_edges()} edges>"
        )


class _EdgeView:
    """``RustnxGraph.edges``: callable like a method, iterable like a view."""

    __slots__ = ("_graph",)

    def __init__(self, graph):
        self._graph = graph

    def __call__(self, data=False):
        return self._graph._edge_list(data)

    def __iter__(self):
        return iter(self._graph._edge_list())

    def __len__(self):
        return self._graph.number_of_edges()


def _decode(value, kind):
    """Inverse of the attribute kinds stored by ``native.rs``."""
    if kind == 1:
        return int(value)
    if kind == 3:
        return None
    if kind == 4:
        return bool(value)
    return value


class _RangeIndex:
    """``{i: i for i in range(n)}`` without building the dict.

    Lookups follow dict semantics: any key equal to (and hashing like) an int
    in range finds it, as ``True`` or ``1.0`` would find ``1`` in a dict.
    """

    __slots__ = ("n",)

    def __init__(self, n):
        self.n = n

    def _position(self, key):
        if type(key) is not int:
            try:
                i = int(key)
            except (TypeError, ValueError, OverflowError):
                return None
            if i != key or hash(i) != hash(key):
                return None
            key = i
        return key if 0 <= key < self.n else None

    def __getitem__(self, key):
        i = self._position(key)
        if i is None:
            raise KeyError(key)
        return i

    def __contains__(self, key):
        return self._position(key) is not None

    def get(self, key, default=None):
        i = self._position(key)
        return default if i is None else i

    def __len__(self):
        return self.n

    def __reduce__(self):
        return (_RangeIndex, (self.n,))


class Graph(RustnxGraph):
    """An undirected graph built and stored in Rust.

    Build it once, then run algorithms on it: there's no conversion step, and
    it uses far less memory than ``networkx.Graph``. It is read-only. NetworkX
    functions accept it directly; call ``rustnx.enable()`` first so functions
    rustnx doesn't implement convert it to NetworkX automatically.

    ``edges`` holds ``(u, v)``, ``(u, v, weight)`` or ``(u, v, {attr: value})``
    tuples, with numeric (or ``None``) attribute values. Results are exactly
    those of a ``networkx.Graph`` built from the same edges in the same order.

    >>> G = rustnx.Graph([("a", "b", 2.5), ("b", "c", 1)])
    >>> nx.single_source_dijkstra_path_length(G, "a")
    {'a': 0, 'b': 2.5, 'c': 3.5}
    """

    _directed = False

    def __init__(self, edges=(), nodes=None, *, weight="weight", **attr):
        core, node_list, index = _core.build_native(
            edges, nodes, self._directed, weight
        )
        super().__init__(core, node_list, index, dict(attr), None)

    @classmethod
    def from_arrays(cls, src, dst, weights=None, *, num_nodes=None, weight="weight", **attr):
        """Build from integer arrays of edge endpoints (nodes ``0..n-1``).

        The fast way to load a large graph: ``src``, ``dst`` and ``weights``
        can be NumPy arrays or sequences, and no Python object is created per
        edge. ``num_nodes`` defaults to the largest node id plus one.
        """
        core = _core.build_native_arrays(
            _int64_bytes(src),
            _int64_bytes(dst),
            None if weights is None else _float64_bytes(weights),
            num_nodes,
            cls._directed,
            weight,
        )
        G = cls.__new__(cls)
        n = len(core)
        RustnxGraph.__init__(G, core, range(n), _RangeIndex(n), dict(attr), None)
        return G


class DiGraph(Graph):
    """A directed graph built and stored in Rust. See ``Graph``."""

    _directed = True


def _int64_bytes(values):
    try:
        import numpy as np
    except ImportError:
        import array

        arr = array.array("q", values)
        if sys.byteorder == "big":
            arr.byteswap()
        return arr.tobytes()
    arr = np.asarray(values)
    if arr.dtype.kind not in "iub" and arr.size:
        raise TypeError(f"node ids must be integers, got dtype {arr.dtype}")
    if arr.size and (arr.min() < 0 or arr.max() >= 2**32 - 1):
        raise ValueError("node ids must be in 0 .. 2**32 - 2")
    return np.ascontiguousarray(arr, dtype="<i8").tobytes()


def _float64_bytes(values):
    try:
        import numpy as np
    except ImportError:
        import array

        arr = array.array("d", values)
        if sys.byteorder == "big":
            arr.byteswap()
        return arr.tobytes()
    return np.ascontiguousarray(values, dtype="<f8").tobytes()


# Methods through which NetworkX algorithms read a graph's structure. rustnx
# reads `G._adj` directly, so a subclass overriding any of these (such as
# NetworkX's own `_AntiGraph`, which presents the complement graph) would be
# converted wrongly.
_STRUCTURE_METHODS = (
    "__iter__", "__contains__", "__len__", "__getitem__", "adj", "succ", "pred",
    "nodes", "edges", "degree", "neighbors", "successors", "predecessors",
    "adjacency", "has_edge", "has_node", "number_of_nodes", "nbunch_iter",
)


def _overrides_structure(G):
    if G.is_multigraph():
        base = nx.MultiDiGraph if G.is_directed() else nx.MultiGraph
    else:
        base = nx.DiGraph if G.is_directed() else nx.Graph
    cls = type(G)
    if cls is base:
        return False
    return any(
        getattr(cls, name, None) is not getattr(base, name, None)
        for name in _STRUCTURE_METHODS
    )


def from_networkx(G, weights=(), *, multigraph_ok=False):
    """Convert a NetworkX ``Graph`` or ``DiGraph``.

    ``weights`` is an iterable of edge attribute names, or a mapping of
    attribute name to the default used when an edge lacks it (default ``1``).
    ``multigraph_ok`` also accepts multigraphs, collapsing parallel edges to
    the minimum weight (only some functions accept the result).
    """
    multigraph = G.is_multigraph()
    if multigraph and not multigraph_ok:
        raise NotImplementedError("rustnx does not support multigraphs here")
    if _overrides_structure(G):
        raise NotImplementedError(
            f"{type(G).__name__} overrides how its structure is read"
        )
    if not isinstance(weights, dict):
        weights = dict.fromkeys(weights, 1)
    if any(not isinstance(attr, str) for attr in weights):
        raise NotImplementedError("rustnx only supports string edge attribute names")
    nodes = list(G)
    index = {node: i for i, node in enumerate(nodes)}
    weight_attrs = list(weights.items())
    core = _core.build_graph(nodes, index, G._adj, G.is_directed(), weight_attrs, multigraph)
    return RustnxGraph(core, nodes, index, G.graph, G, weight_attrs, multigraph)


def _snapshot_token(G):
    """A token that stays in ``G.__networkx_cache__`` until G changes."""
    cache = getattr(G, "__networkx_cache__", None)
    # Views keep their own cache, which changes to the viewed graph don't clear.
    if cache is None or getattr(G, "_graph", None) is not None:
        return None
    return cache.setdefault(_SNAPSHOT_KEY, object())


def to_networkx(G):
    if isinstance(G, RustnxGraph):
        if G._source is not None:
            return G._source
        if not G._core.is_native():
            raise NotImplementedError("this RustnxGraph has no source graph")
        return _native_to_networkx(G)
    if isinstance(G, nx.Graph):
        return G
    raise TypeError(f"expected a RustnxGraph, got {type(G).__name__}")


def _native_to_networkx(G):
    H = nx.DiGraph() if G.is_directed() else nx.Graph()
    H.graph.update(G.graph)
    nodes = G._nodes
    H.add_nodes_from(nodes)
    src, dst, _ = G._core.native_edges()
    attrs = G._edge_attr_dicts()
    # Insertion order, so H's adjacency order matches G's exactly.
    H.add_edges_from(
        (nodes[u], nodes[v], d) for u, v, d in zip(src, dst, attrs)
    )
    return H
