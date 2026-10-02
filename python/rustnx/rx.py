"""A rustworkx-compatible API on rustnx's Rust core.

    import rustnx.rx as rx      # instead of: import rustworkx as rx

``PyGraph`` and ``PyDiGraph`` follow rustworkx's index-based model, including
the details code can depend on:

- Node and edge indices of removed items are reused, most recently removed
  first; removing a node removes its outgoing edges, then its incoming ones,
  each newest first (as petgraph's ``StableGraph`` does).
- Graphs are multigraphs by default. With ``multigraph=False``, adding an
  existing edge replaces its payload and returns its index.
- Neighbors are visited in petgraph's order (outgoing edges newest first,
  then incoming), so order-dependent results such as
  ``strongly_connected_components`` and ``topological_sort`` match rustworkx.

Results are plain ``list``/``dict``/``set`` objects rather than rustworkx's
custom ``NodeIndices``/``*Mapping`` types; they compare equal to them.
``neighbors()`` returns each neighbor once in an unspecified order, as in
rustworkx. ``edge_cost_fn``/``weight_fn`` are called once per edge.
"""

import math

from . import _core

__all__ = [
    "DAGHasCycle",
    "DAGWouldCycle",
    "FailedToConverge",
    "InvalidNode",
    "NegativeCycle",
    "NoEdgeBetweenNodes",
    "NullGraph",
    "PyDiGraph",
    "PyGraph",
    "all_pairs_dijkstra_path_lengths",
    "betweenness_centrality",
    "closeness_centrality",
    "connected_components",
    "digraph_betweenness_centrality",
    "digraph_closeness_centrality",
    "digraph_dijkstra_shortest_path_lengths",
    "dijkstra_shortest_path_lengths",
    "graph_betweenness_centrality",
    "graph_closeness_centrality",
    "graph_dijkstra_shortest_path_lengths",
    "is_connected",
    "is_directed_acyclic_graph",
    "is_strongly_connected",
    "is_weakly_connected",
    "networkx_converter",
    "number_connected_components",
    "number_strongly_connected_components",
    "number_weakly_connected_components",
    "pagerank",
    "strongly_connected_components",
    "topological_sort",
    "weakly_connected_components",
]


class InvalidNode(Exception):
    """The provided node is invalid."""


class DAGWouldCycle(Exception):
    """Performing this operation would result in trying to add a cycle to a DAG."""


class NoEdgeBetweenNodes(Exception):
    """There is no edge present between the provided nodes."""


class DAGHasCycle(Exception):
    """The specified Directed Graph has a cycle and can't be treated as a DAG."""


class NullGraph(Exception):
    """Invalid operation on a null graph"""


class NegativeCycle(Exception):
    """Negative Cycle found on shortest-path algorithm"""


class FailedToConverge(Exception):
    """Failed to Converge on a solution"""


_HOLE = object()  # marks a removed node slot


class _Graph:
    _directed = False

    def __init__(self, multigraph=True, attrs=None):
        self.multigraph = multigraph
        self.attrs = attrs
        self.clear()

    # --- Storage ----------------------------------------------------------
    # _nodes[i]: payload or _HOLE; _edges[e]: [src, dst, payload] or None.
    # _out[i] / _in[i]: edge ids touching node i, oldest first (petgraph
    # walks them newest first). _order: live edge ids in insertion order.

    def clear(self):
        self._nodes = []
        self._node_free = []
        self._edges = []
        self._edge_free = []
        self._out = []
        self._in = []
        self._order = {}
        self._layout = None

    def _changed(self):
        self._layout = None

    def _is_node(self, i):
        return (
            isinstance(i, int)
            and 0 <= i < len(self._nodes)
            and self._nodes[i] is not _HOLE
        )

    def _check_node(self, i):
        if not self._is_node(i):
            raise IndexError("One of the endpoints of the edge does not exist in graph")

    def _find_edge(self, a, b):
        """petgraph ``find_edge``: first match walking a's edges newest first."""
        if not self._is_node(a):
            return None
        edges = self._edges
        for e in reversed(self._out[a]):
            if edges[e][1] == b:
                return e
        if not self._directed:
            for e in reversed(self._in[a]):
                if edges[e][0] == b:
                    return e
        return None

    def _add_edge(self, a, b, payload):
        if not self.multigraph:
            e = self._find_edge(a, b)
            if e is not None:
                self._edges[e][2] = payload
                return e
        if self._edge_free:
            e = self._edge_free.pop()
            self._edges[e] = [a, b, payload]
        else:
            e = len(self._edges)
            self._edges.append([a, b, payload])
        self._out[a].append(e)
        self._in[b].append(e)
        self._order[e] = None
        self._changed()
        return e

    def _raw_add_edge(self, a, b, payload):
        """Add an edge without the multigraph check (petgraph ``add_edge``)."""
        multigraph, self.multigraph = self.multigraph, True
        try:
            return self._add_edge(a, b, payload)
        finally:
            self.multigraph = multigraph

    def _remove_edge(self, e):
        edge = self._edges[e] if 0 <= e < len(self._edges) else None
        if edge is None:
            return
        a, b, _ = edge
        self._out[a].remove(e)
        self._in[b].remove(e)
        self._edges[e] = None
        self._edge_free.append(e)
        del self._order[e]
        self._changed()

    # --- Nodes ---------------------------------------------------------------

    def add_node(self, obj):
        if self._node_free:
            i = self._node_free.pop()
            self._nodes[i] = obj
        else:
            i = len(self._nodes)
            self._nodes.append(obj)
            self._out.append([])
            self._in.append([])
        self._changed()
        return i

    def add_nodes_from(self, obj_list):
        return [self.add_node(obj) for obj in obj_list]

    def remove_node(self, node):
        if not self._is_node(node):
            return
        for e in reversed(list(self._out[node])):
            self._remove_edge(e)
        for e in reversed(list(self._in[node])):
            self._remove_edge(e)
        self._nodes[node] = _HOLE
        self._node_free.append(node)
        self._changed()

    def remove_nodes_from(self, index_list):
        for node in index_list:
            self.remove_node(node)

    def has_node(self, node):
        return self._is_node(node)

    def num_nodes(self):
        return len(self._nodes) - len(self._node_free)

    def node_indices(self):
        return [i for i, x in enumerate(self._nodes) if x is not _HOLE]

    node_indexes = node_indices

    def nodes(self):
        return [x for x in self._nodes if x is not _HOLE]

    def get_node_data(self, node):
        if not self._is_node(node):
            raise IndexError("No node found for index")
        return self._nodes[node]

    def __getitem__(self, node):
        if not self._is_node(node):
            raise IndexError("No node found for index")
        return self._nodes[node]

    def __setitem__(self, node, obj):
        if not self._is_node(node):
            raise IndexError("No node found for index")
        self._nodes[node] = obj

    def __len__(self):
        return self.num_nodes()

    # --- Edges ---------------------------------------------------------------

    def add_edge(self, node_a, node_b, edge):
        self._check_node(node_a)
        self._check_node(node_b)
        return self._add_edge(node_a, node_b, edge)

    def add_edges_from(self, obj_list):
        return [self.add_edge(a, b, w) for a, b, w in obj_list]

    def add_edges_from_no_data(self, obj_list):
        return [self.add_edge(a, b, None) for a, b in obj_list]

    def extend_from_edge_list(self, edge_list):
        self._extend_edges((a, b, None) for a, b in edge_list)

    def extend_from_weighted_edge_list(self, edge_list):
        self._extend_edges(edge_list)

    def _extend_edges(self, triples):
        nodes, out, inc, edges, order = self._nodes, self._out, self._in, self._edges, self._order
        for a, b, w in triples:
            top = a if a > b else b
            if top >= len(nodes) - len(self._node_free):
                self._extend_to(top)
            if (
                self.multigraph
                and not self._edge_free
                and type(a) is int
                and type(b) is int
                and 0 <= a < len(nodes)
                and 0 <= b < len(nodes)
                and nodes[a] is not _HOLE
                and nodes[b] is not _HOLE
            ):
                # Fast path: the common case of appending to a multigraph.
                e = len(edges)
                edges.append([a, b, w])
                out[a].append(e)
                inc[b].append(e)
                order[e] = None
            else:
                self._check_node(a)
                self._check_node(b)
                self._add_edge(a, b, w)
        self._changed()

    def _extend_to(self, index):
        # rustworkx compares against the node *count* (not the index bound).
        while index >= self.num_nodes():
            self.add_node(None)

    def remove_edge(self, node_a, node_b):
        e = self._find_edge(node_a, node_b)
        if e is None:
            raise NoEdgeBetweenNodes("No edge found between nodes")
        self._remove_edge(e)

    def remove_edge_from_index(self, edge):
        self._remove_edge(edge)

    def remove_edges_from(self, index_list):
        for a, b in index_list:
            self.remove_edge(a, b)

    def num_edges(self):
        return len(self._order)

    def edge_indices(self):
        return [e for e, x in enumerate(self._edges) if x is not None]

    def edges(self):
        return [x[2] for x in self._edges if x is not None]

    def edge_list(self):
        return [(x[0], x[1]) for x in self._edges if x is not None]

    def weighted_edge_list(self):
        return [(x[0], x[1], x[2]) for x in self._edges if x is not None]

    def edge_index_map(self):
        return {e: (x[0], x[1], x[2]) for e, x in enumerate(self._edges) if x is not None}

    def has_edge(self, node_a, node_b):
        return self._find_edge(node_a, node_b) is not None

    def get_edge_data(self, node_a, node_b):
        e = self._find_edge(node_a, node_b)
        if e is None:
            raise NoEdgeBetweenNodes("No edge found between nodes")
        return self._edges[e][2]

    def get_all_edge_data(self, node_a, node_b):
        data = [self._edges[e][2] for e in self._incident(node_a) if self._other(e, node_a) == node_b]
        if not data:
            raise NoEdgeBetweenNodes("No edge found between nodes")
        return data

    def get_edge_data_by_index(self, edge_index):
        if not (0 <= edge_index < len(self._edges)) or self._edges[edge_index] is None:
            raise IndexError(f"Provided edge index {edge_index} is not present in the graph")
        return self._edges[edge_index][2]

    def get_edge_endpoints_by_index(self, edge_index):
        if not (0 <= edge_index < len(self._edges)) or self._edges[edge_index] is None:
            raise IndexError(f"Provided edge index {edge_index} is not present in the graph")
        return tuple(self._edges[edge_index][:2])

    def update_edge(self, source, target, edge):
        e = self._find_edge(source, target)
        if e is None:
            raise NoEdgeBetweenNodes("No edge found between nodes")
        self._edges[e][2] = edge

    def update_edge_by_index(self, edge_index, edge):
        if not (0 <= edge_index < len(self._edges)) or self._edges[edge_index] is None:
            raise IndexError(f"Provided edge index {edge_index} is not present in the graph")
        self._edges[edge_index][2] = edge

    def clear_edges(self):
        self._edges = []
        self._edge_free = []
        self._out = [[] for _ in self._nodes]
        self._in = [[] for _ in self._nodes]
        self._order = {}
        self._changed()

    def has_parallel_edges(self):
        if not self.multigraph:
            return False
        seen = set()
        for a, b, _ in (x for x in self._edges if x is not None):
            key = (a, b) if self._directed or a <= b else (b, a)
            if key in seen:
                return True
            seen.add(key)
        return False

    def find_node_by_weight(self, obj):
        for i, x in enumerate(self._nodes):
            if x is not _HOLE and x == obj:
                return i
        return None

    def filter_nodes(self, filter_function):
        return [i for i in self.node_indices() if filter_function(self._nodes[i])]

    def filter_edges(self, filter_function):
        return [e for e in self.edge_indices() if filter_function(self._edges[e][2])]

    def edge_indices_from_endpoints(self, node_a, node_b):
        if not self._is_node(node_a):
            return []
        if self._directed:
            ids = reversed(self._out[node_a])
        else:
            ids = self._incident(node_a)
        return [e for e in ids if self._other(e, node_a) == node_b]

    def _incident(self, node):
        """Edge ids in petgraph ``edges(node)`` order."""
        out = list(reversed(self._out[node]))
        if self._directed:
            return out
        edges = self._edges
        return out + [e for e in reversed(self._in[node]) if edges[e][0] != node]

    def _other(self, e, node):
        a, b, _ = self._edges[e]
        return b if a == node else a

    def neighbors(self, node):
        if not self._is_node(node):
            return []
        return list(dict.fromkeys(self._other(e, node) for e in self._incident(node)))

    def degree(self, node):
        if not self._is_node(node):
            return 0
        edges = self._edges
        return sum(2 if edges[e][0] == edges[e][1] else 1 for e in self._incident(node))

    def incident_edges(self, node):
        if not self._is_node(node):
            return []
        return self._incident(node)

    def adj(self, node):
        if not self._is_node(node):
            return {}
        out = {}
        for e in self._incident(node):
            out[self._other(e, node)] = self._edges[e][2]
        return out

    def copy(self):
        new = type(self).__new__(type(self))
        new.__dict__.update(self.__dict__)
        new._nodes = list(self._nodes)
        new._node_free = list(self._node_free)
        new._edges = [None if x is None else list(x) for x in self._edges]
        new._edge_free = list(self._edge_free)
        new._out = [list(x) for x in self._out]
        new._in = [list(x) for x in self._in]
        new._order = dict(self._order)
        new._layout = None
        return new

    def __repr__(self):
        return f"{type(self).__name__}<{self.num_nodes()} nodes, {self.num_edges()} edges>"

    # --- Bridge to the Rust core ------------------------------------------------

    def _compact(self):
        """(live node ids, position of each slot, edge ids, src, dst), cached."""
        if self._layout is None:
            live = self.node_indices()
            pos = [-1] * len(self._nodes)
            for k, i in enumerate(live):
                pos[i] = k
            order = list(self._order)
            edges = self._edges
            src = [pos[edges[e][0]] for e in order]
            dst = [pos[edges[e][1]] for e in order]
            self._layout = (live, pos, order, src, dst, None)
        return self._layout

    def _core(self, costs=None):
        live, _, _, src, dst, cached = self._compact()
        if costs is None:
            if cached is None:
                cached = _core.build_rx(len(live), self._directed, src, dst)
                self._layout = self._layout[:5] + (cached,)
            return cached
        return _core.build_rx(len(live), self._directed, src, dst, costs)

    def _costs(self, fn, default=1.0):
        """One float per edge (insertion order) from ``fn(payload)``."""
        _, _, order, _, _, _ = self._compact()
        edges = self._edges
        if fn is None:
            return [default] * len(order)
        return [float(fn(edges[e][2])) for e in order]


class PyGraph(_Graph):
    """An undirected graph, API-compatible with ``rustworkx.PyGraph``."""

    def out_edges(self, node):
        if not self._is_node(node):
            return []
        return [(node, self._other(e, node), self._edges[e][2]) for e in self._incident(node)]

    def in_edges(self, node):
        if not self._is_node(node):
            return []
        return [(self._other(e, node), node, self._edges[e][2]) for e in self._incident(node)]

    def out_edge_indices(self, node):
        return self._incident(node) if self._is_node(node) else []

    # rustworkx's PyGraph returns the same edges for both.
    in_edge_indices = out_edge_indices

    def incident_edge_index_map(self, node):
        if not self._is_node(node):
            return {}
        return {
            e: (node, self._other(e, node), self._edges[e][2]) for e in self._incident(node)
        }

    def to_directed(self):
        g = PyDiGraph(multigraph=self.multigraph, attrs=self.attrs)
        g._nodes = list(self._nodes)
        g._node_free = list(self._node_free)
        g._out = [[] for _ in self._nodes]
        g._in = [[] for _ in self._nodes]
        for e in self._order:
            a, b, w = self._edges[e]
            g._add_edge(a, b, w)
            if a != b:
                g._add_edge(b, a, w)
        return g


class PyDiGraph(_Graph):
    """A directed graph, API-compatible with ``rustworkx.PyDiGraph``."""

    _directed = True

    def __init__(self, check_cycle=False, multigraph=True, attrs=None):
        if check_cycle:
            raise NotImplementedError("rustnx.rx.PyDiGraph does not support check_cycle")
        self.check_cycle = False
        super().__init__(multigraph=multigraph, attrs=attrs)

    def successor_indices(self, node):
        if not self._is_node(node):
            return []
        return list(dict.fromkeys(self._edges[e][1] for e in reversed(self._out[node])))

    def predecessor_indices(self, node):
        if not self._is_node(node):
            return []
        return list(dict.fromkeys(self._edges[e][0] for e in reversed(self._in[node])))

    def successors(self, node):
        return [self._nodes[i] for i in self.successor_indices(node)]

    def predecessors(self, node):
        return [self._nodes[i] for i in self.predecessor_indices(node)]

    def out_edges(self, node):
        if not self._is_node(node):
            return []
        return [tuple(self._edges[e]) for e in reversed(self._out[node])]

    def in_edges(self, node):
        if not self._is_node(node):
            return []
        return [tuple(self._edges[e]) for e in reversed(self._in[node])]

    def out_degree(self, node):
        return len(self._out[node]) if self._is_node(node) else 0

    def in_degree(self, node):
        return len(self._in[node]) if self._is_node(node) else 0

    def neighbors(self, node):
        return self.successor_indices(node)

    def neighbors_undirected(self, node):
        if not self._is_node(node):
            return []
        nbrs = [self._edges[e][1] for e in reversed(self._out[node])]
        nbrs += [self._edges[e][0] for e in reversed(self._in[node])]
        return list(dict.fromkeys(nbrs))

    def out_edge_indices(self, node):
        return list(reversed(self._out[node])) if self._is_node(node) else []

    def in_edge_indices(self, node):
        return list(reversed(self._in[node])) if self._is_node(node) else []

    def incident_edge_index_map(self, node, all_edges=False):
        if not self._is_node(node):
            return {}
        ids = list(reversed(self._out[node]))
        if all_edges:
            ids += list(reversed(self._in[node]))
        return {e: tuple(self._edges[e]) for e in ids}

    def add_child(self, parent, obj, edge):
        self._check_node(parent)
        child = self.add_node(obj)
        self._raw_add_edge(parent, child, edge)
        return child

    def add_parent(self, child, obj, edge):
        self._check_node(child)
        parent = self.add_node(obj)
        self._raw_add_edge(parent, child, edge)
        return parent

    def reverse(self):
        """Reverse every edge in place (indices are kept)."""
        for e in self.edge_indices():
            a, b, w = self._edges[e]
            self._remove_edge(e)
            self._raw_add_edge(b, a, w)

    def to_undirected(self, multigraph=True, weight_combo_fn=None):
        g = PyGraph(multigraph=multigraph)
        node_map = {i: g.add_node(self._nodes[i]) for i in self.node_indices()}
        for e in self.edge_indices():
            a, b, w = self._edges[e]
            a, b = node_map[a], node_map[b]
            if multigraph:
                g._raw_add_edge(a, b, w)
                continue
            existing = g._find_edge(a, b)
            if existing is None:
                g._raw_add_edge(a, b, w)
            elif weight_combo_fn is not None:
                g._edges[existing][2] = weight_combo_fn(g._edges[existing][2], w)
            else:
                g._edges[existing][2] = w
        return g

    def adj(self, node):
        if not self._is_node(node):
            return {}
        out = {}
        for e in reversed(self._out[node]):
            out[self._edges[e][1]] = self._edges[e][2]
        for e in reversed(self._in[node]):
            out.setdefault(self._edges[e][0], self._edges[e][2])
        return out


# --- Algorithms -------------------------------------------------------------------


def _require(graph, cls):
    if not isinstance(graph, cls):
        raise TypeError(f"argument 'graph': expected {cls.__name__}, got {type(graph).__name__}")


def _by_index(graph, values):
    live = graph._compact()[0]
    return dict(zip(live, values))


def _sets(graph, comps):
    live = graph._compact()[0]
    return [{live[i] for i in comp} for comp in comps]


def connected_components(graph):
    _require(graph, PyGraph)
    return _sets(graph, graph._core().connected_components())


def number_connected_components(graph):
    _require(graph, PyGraph)
    return len(graph._core().connected_components())


def is_connected(graph):
    _require(graph, PyGraph)
    if graph.num_nodes() == 0:
        raise NullGraph("Invalid operation on a NullGraph")
    return len(graph._core().connected_components()[0]) == graph.num_nodes()


def weakly_connected_components(graph):
    _require(graph, PyDiGraph)
    return _sets(graph, graph._core().weakly_connected_components())


def number_weakly_connected_components(graph):
    _require(graph, PyDiGraph)
    return len(graph._core().weakly_connected_components())


def is_weakly_connected(graph):
    _require(graph, PyDiGraph)
    if graph.num_nodes() == 0:
        raise NullGraph("Invalid operation on a NullGraph")
    return len(graph._core().weakly_connected_components()[0]) == graph.num_nodes()


def strongly_connected_components(graph):
    _require(graph, PyDiGraph)
    live = graph._compact()[0]
    return [[live[i] for i in comp] for comp in graph._core().rx_scc()]


def number_strongly_connected_components(graph):
    _require(graph, PyDiGraph)
    return len(graph._core().rx_scc())


def is_strongly_connected(graph):
    _require(graph, PyDiGraph)
    if graph.num_nodes() == 0:
        raise NullGraph("Invalid operation on a NullGraph")
    return len(graph._core().rx_scc()) == 1


def topological_sort(graph):
    _require(graph, PyDiGraph)
    order = graph._core().rx_toposort()
    if order is None:
        raise DAGHasCycle("Sort encountered a cycle")
    live = graph._compact()[0]
    return [live[i] for i in order]


def is_directed_acyclic_graph(graph):
    _require(graph, PyDiGraph)
    return graph._core().rx_toposort() is not None


def _rescale(values, n, normalized, directed, endpoints):
    """rustworkx-core ``_rescale``."""
    scale = None
    if normalized:
        if endpoints:
            if n >= 2:
                scale = 1.0 / (n * (n - 1))
        elif n > 2:
            scale = 1.0 / ((n - 1) * (n - 2))
    elif not directed:
        scale = 0.5
    if scale is None:
        return values
    return [v * scale for v in values]


def _betweenness(graph, normalized, endpoints):
    raw = graph._core().betweenness(None, bool(endpoints), None)
    values = _rescale(raw, graph.num_nodes(), normalized, graph._directed, endpoints)
    return _by_index(graph, values)


def graph_betweenness_centrality(graph, normalized=True, endpoints=False, parallel_threshold=50):
    _require(graph, PyGraph)
    return _betweenness(graph, normalized, endpoints)


def digraph_betweenness_centrality(graph, normalized=True, endpoints=False, parallel_threshold=50):
    _require(graph, PyDiGraph)
    return _betweenness(graph, normalized, endpoints)


def betweenness_centrality(graph, normalized=True, endpoints=False, parallel_threshold=50):
    if isinstance(graph, PyDiGraph):
        return digraph_betweenness_centrality(graph, normalized, endpoints, parallel_threshold)
    return graph_betweenness_centrality(graph, normalized, endpoints, parallel_threshold)


def _closeness(graph, wf_improved):
    core = graph._core()
    n = graph.num_nodes()
    # BFS on the reversed graph, as rustworkx (and NetworkX) do.
    stats = core.bfs_stats(None, graph._directed)
    values = []
    for reached, total, _ in stats:
        if reached == 1:
            values.append(0.0)
            continue
        c = (reached - 1) / total
        if wf_improved:
            c = c * (reached - 1) / (n - 1)  # rustworkx's operation order
        values.append(c)
    return _by_index(graph, values)


def graph_closeness_centrality(graph, wf_improved=True, parallel_threshold=50):
    _require(graph, PyGraph)
    return _closeness(graph, wf_improved)


def digraph_closeness_centrality(graph, wf_improved=True, parallel_threshold=50):
    _require(graph, PyDiGraph)
    return _closeness(graph, wf_improved)


def closeness_centrality(graph, wf_improved=True, parallel_threshold=50):
    if isinstance(graph, PyDiGraph):
        return digraph_closeness_centrality(graph, wf_improved, parallel_threshold)
    return graph_closeness_centrality(graph, wf_improved, parallel_threshold)


def pagerank(
    graph,
    alpha=0.85,
    weight_fn=None,
    nstart=None,
    personalization=None,
    tol=1.0e-6,
    max_iter=100,
    dangling=None,
):
    _require(graph, PyDiGraph)
    if graph.num_nodes() == 0:
        return {}
    live = graph._compact()[0]

    def vector(mapping):
        if mapping is None:
            return None
        return [float(mapping.get(i, 0.0)) for i in live]

    core = graph._core(graph._costs(weight_fn))
    scores = core.pagerank(
        float(alpha),
        vector(personalization),
        int(max_iter),
        float(tol),
        vector(nstart),
        _core_weight_name(),
        vector(dangling),
        True,  # rustworkx returns the iterate before the converging step
    )
    if scores is None:
        raise FailedToConverge(
            f"Function failed to converge on a solution in {max_iter} iterations"
        )
    return dict(zip(live, scores))


def _core_weight_name():
    return "__rx_cost__"


def _dijkstra(graph, node, edge_cost_fn, goal):
    if not graph._is_node(node):
        raise IndexError(f'Node source index "{node}" out of graph bound')
    live, pos, _, _, _, _ = graph._compact()
    core = graph._core(graph._costs(edge_cost_fn))
    goal_pos = None
    if goal is not None:
        goal_pos = pos[goal] if graph._is_node(goal) else None
    dists = core.rx_dijkstra(pos[node], goal_pos)
    if goal is not None:
        if goal_pos is None or math.isnan(dists[goal_pos]):
            return {}
        return {goal: dists[goal_pos]}
    return {live[k]: d for k, d in enumerate(dists) if live[k] != node and not math.isnan(d)}


def graph_dijkstra_shortest_path_lengths(graph, node, edge_cost_fn, goal=None):
    _require(graph, PyGraph)
    return _dijkstra(graph, node, edge_cost_fn, goal)


def digraph_dijkstra_shortest_path_lengths(graph, node, edge_cost_fn, goal=None):
    _require(graph, PyDiGraph)
    return _dijkstra(graph, node, edge_cost_fn, goal)


def dijkstra_shortest_path_lengths(graph, node, edge_cost_fn, goal=None):
    if isinstance(graph, PyDiGraph):
        return digraph_dijkstra_shortest_path_lengths(graph, node, edge_cost_fn, goal)
    return graph_dijkstra_shortest_path_lengths(graph, node, edge_cost_fn, goal)


def all_pairs_dijkstra_path_lengths(graph, edge_cost_fn):
    live = graph._compact()[0]
    core = graph._core(graph._costs(edge_cost_fn))
    out = {}
    for k, source in enumerate(live):
        dists = core.rx_dijkstra(k, None)
        out[source] = {
            live[j]: d for j, d in enumerate(dists) if j != k and not math.isnan(d)
        }
    return out


def networkx_converter(graph, keep_attributes=False):
    """Convert a NetworkX graph, as ``rustworkx.networkx_converter`` does."""
    if graph.is_directed():
        new_graph = PyDiGraph(multigraph=graph.is_multigraph())
    else:
        new_graph = PyGraph(multigraph=graph.is_multigraph())
    nodes = list(graph.nodes)
    node_indices = dict(zip(nodes, new_graph.add_nodes_from(nodes)))
    new_graph.add_edges_from(
        [(node_indices[u], node_indices[v], d) for u, v, d in graph.edges(data=True)]
    )
    if keep_attributes:
        for node, index in node_indices.items():
            attributes = graph.nodes[node]
            attributes["__networkx_node__"] = node
            new_graph[index] = attributes
    return new_graph
