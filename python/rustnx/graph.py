"""The graph object NetworkX hands to rustnx algorithms."""

import networkx as nx

from . import _core

__all__ = ["RustnxGraph", "from_networkx"]


class RustnxGraph:
    """A read-only snapshot of a NetworkX graph stored in Rust.

    Nodes keep their original Python objects; the Rust core works on integer
    indices ``0..n-1`` in the same order as ``list(G)``.
    """

    __networkx_backend__ = "rustnx"

    def __init__(self, core, nodes, index, graph_attrs, source):
        self._core = core
        self._nodes = nodes
        self._index = index
        self.graph = graph_attrs
        # Original graph, used to convert back to NetworkX (see `convert_to_nx`).
        self._source = source
        self.__networkx_cache__ = {}

    def is_directed(self):
        return self._core.directed

    def is_multigraph(self):
        return False

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
            f"<RustnxGraph ({kind}) with {len(self)} nodes "
            f"and {self.number_of_edges()} edges>"
        )


def from_networkx(G, weights=()):
    """Convert a NetworkX ``Graph`` or ``DiGraph``.

    ``weights`` is an iterable of edge attribute names, or a mapping of
    attribute name to the default used when an edge lacks it (default ``1``).
    """
    if G.is_multigraph():
        raise NotImplementedError("rustnx does not support multigraphs yet")
    if not isinstance(weights, dict):
        weights = dict.fromkeys(weights, 1)
    if any(not isinstance(attr, str) for attr in weights):
        raise NotImplementedError("rustnx only supports string edge attribute names")
    nodes = list(G)
    index = {node: i for i, node in enumerate(nodes)}
    core = _core.build_graph(
        nodes,
        index,
        G._adj,
        G._pred if G.is_directed() else None,
        list(weights.items()),
    )
    return RustnxGraph(core, nodes, index, G.graph, G)


def to_networkx(G):
    if isinstance(G, RustnxGraph):
        if G._source is None:
            raise NotImplementedError("this RustnxGraph has no source graph")
        return G._source
    if isinstance(G, nx.Graph):
        return G
    raise TypeError(f"expected a RustnxGraph, got {type(G).__name__}")
