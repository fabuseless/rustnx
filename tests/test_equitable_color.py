"""equitable_color against NetworkX: the same coloring, in the same order,
and the same errors. Its Case II (the one place a set's iteration order
reaches the result) is practically unreachable from equitable_color, so
NetworkX's own hand-built procedure_P states are replayed too."""

import copy
import importlib.util
import os
import random

import networkx as nx
import networkx.algorithms.coloring.equitable_coloring as eqc
import pytest

from rustnx import _core
from test_against_networkx import exact_outcome


def sample_graph(seed):
    rng = random.Random(seed)
    n = rng.randint(0, 50)
    kind = seed % 5
    if kind == 0:
        G = nx.gnp_random_graph(n, rng.random(), seed=seed)
    elif kind == 1:
        G = nx.bipartite.random_graph(n // 2, n - n // 2, rng.choice([0.5, 0.9]), seed=seed)
    elif kind == 2:
        # Complete multipartite graphs reach procedure_P's harder branch.
        G = nx.complete_multipartite_graph(*[rng.randint(1, 6) for _ in range(rng.randint(1, 6))])
    elif kind == 3:
        G = nx.complete_multipartite_graph(*[rng.randint(1, 5) for _ in range(rng.randint(2, 5))])
        for _ in range(4):
            if len(G) > 1:
                u, v = rng.sample(list(G), 2)
                G.remove_edge(u, v) if G.has_edge(u, v) else G.add_edge(u, v)
    else:
        G = nx.empty_graph(n)
    if seed % 7 == 0 and len(G):
        u = rng.choice(list(G))
        G.add_edge(u, u)
    # Shuffled insertion order and non-int labels: NetworkX numbers the
    # nodes in G.nodes order.
    order = list(G)
    rng.shuffle(order)
    label = (lambda v: v) if seed % 3 == 0 else (lambda v: f"n{v}") if seed % 3 == 1 else (lambda v: (v, "x"))
    H = nx.Graph()
    H.add_nodes_from(label(v) for v in order)
    H.add_edges_from((label(u), label(v)) for u, v in G.edges)
    return H


@pytest.mark.parametrize("seed", range(150))
def test_equitable_color(seed):
    G = sample_graph(seed)
    maxd = max(dict(G.degree).values(), default=0)
    for k in {maxd + 1, maxd + 2, maxd + 7, len(G) + 3}:
        result = exact_outcome(nx.equitable_color, G, k)
        assert result[0] == "ok"
        # rustnx answered itself (it hands nothing here to NetworkX).
        nx.equitable_color(G, k, backend="rustnx")
    exact_outcome(nx.equitable_color, G, maxd)  # too few colors: NetworkX's error


def test_equitable_color_inputs_left_to_networkx():
    G = nx.cycle_graph(6)
    exact_outcome(nx.equitable_color, G, 3.0)
    exact_outcome(nx.equitable_color, nx.DiGraph(G), 4)
    exact_outcome(nx.equitable_color, nx.MultiGraph(G), 4)
    exact_outcome(nx.equitable_color, nx.Graph(), 0)
    exact_outcome(nx.equitable_color, nx.Graph(), 3)


def _hand_built_cases():
    """(state before, state after) of each top-level procedure_P call in
    NetworkX's own tests."""
    orig = eqc.procedure_P
    cases = []
    depth = [0]

    def capture(V_minus, V_plus, N, H, F, C, L, excluded_colors=None):
        if depth[0] == 0:
            cases.append([copy.deepcopy(L), dict(F), {c: list(v) for c, v in C.items()}, V_minus, V_plus])
        depth[0] += 1
        try:
            return orig(V_minus, V_plus, N, H, F, C, L, excluded_colors)
        finally:
            depth[0] -= 1
            if depth[0] == 0:
                cases[-1] += [dict(F), {c: list(v) for c, v in C.items()}]

    path = os.path.join(os.path.dirname(eqc.__file__), "tests", "test_coloring.py")
    spec = importlib.util.spec_from_file_location("_nx_test_coloring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    eqc.procedure_P = capture
    try:
        tests = module.TestColoring()
        for name in ("test_case_V_plus_not_in_A_cal", "test_cast_no_solo", "test_hard_prob", "test_hardest_prob"):
            getattr(tests, name)()
    finally:
        eqc.procedure_P = orig
    return cases


def test_procedure_p_hand_built_states():
    cases = _hand_built_cases()
    assert len(cases) == 4
    for L, F, C, v_minus, v_plus, F_after, C_after in cases:
        nodes = list(L)
        pos = {v: i for i, v in enumerate(nodes)}
        colors = list(C)
        cpos = {c: i for i, c in enumerate(colors)}
        f, c = _core.CoreGraph.equitable_procedure_p(
            [[pos[w] for w in L[v]] for v in nodes],
            [cpos[F[v]] for v in nodes],
            [[pos[w] for w in C[col]] for col in colors],
            [hash(col) for col in colors],
            cpos[v_minus],
            cpos[v_plus],
        )
        assert {v: colors[f[i]] for i, v in enumerate(nodes)} == F_after
        assert {col: [nodes[x] for x in c[cpos[col]]] for col in colors} == C_after
