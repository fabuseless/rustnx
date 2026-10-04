"""rustnx.Graph / rustnx.DiGraph against networkx graphs built from the same edges.

A native graph must behave exactly like a ``networkx.Graph`` built with
``add_edge`` calls in the same order: same nodes, same adjacency order, same
merged attributes for duplicate edges, and so the same algorithm results.
"""

import math
import pickle
from itertools import islice
import random
import warnings

import networkx as nx
from networkx.algorithms.coloring.equitable_coloring import is_coloring
import pytest

import rustnx


def random_edges(seed, directed):
    rng = random.Random(seed)
    n = rng.randint(1, 30)
    labels = list(range(n))
    if rng.random() < 0.5:
        labels = [f"n{i}" if i % 2 else (i, "t") for i in labels]
    edges = []
    for _ in range(rng.randint(0, 3 * n)):
        u, v = rng.choice(labels), rng.choice(labels)
        form = rng.random()
        if form < 0.3:
            edges.append((u, v))
        elif form < 0.6:
            edges.append((u, v, rng.choice([1, 2, 3, 0.5, 2.25])))
        else:
            d = {}
            if rng.random() < 0.8:
                d["weight"] = rng.choice([1, 2, 4, 1.5])
            if rng.random() < 0.3:
                d["cap"] = rng.randint(0, 9)
            edges.append((u, v, d))
    # Duplicates and reversed duplicates exercise NetworkX's merge rules.
    for _ in range(rng.randint(0, 5)):
        if edges:
            e = rng.choice(edges)
            edges.append((e[1], e[0]) + tuple(e[2:]) if rng.random() < 0.5 else e)
    extra_nodes = [f"lonely{i}" for i in range(rng.randint(0, 2))]
    return edges, extra_nodes


def to_nx(edges, nodes, directed):
    H = nx.DiGraph() if directed else nx.Graph()
    H.add_nodes_from(nodes)
    for e in edges:
        if len(e) == 2:
            H.add_edge(*e)
        elif isinstance(e[2], dict):
            H.add_edge(e[0], e[1], **e[2])
        else:
            H.add_edge(e[0], e[1], weight=e[2])
    return H


def build(seed, directed):
    edges, nodes = random_edges(seed, directed)
    cls = rustnx.DiGraph if directed else rustnx.Graph
    return cls(edges, nodes=nodes), to_nx(edges, nodes, directed)


def outcome(func):
    try:
        result = func()
        if hasattr(result, "__next__"):
            result = list(result)
            result = [
                (k, list(v.items())) if isinstance(v, dict) else (k, v)
                for k, v in result
            ] if result and isinstance(result[0], tuple) else result
        return ("ok", result)
    except Exception as exc:
        if isinstance(exc, nx.PowerIterationFailedConvergence):
            return (type(exc), str(exc))  # its args hold the exception itself
        return (type(exc), exc.args)


def same(a, b):
    """Equal, treating floats within 1e-12 as equal (parallel sums)."""
    if a == b:
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) or (
            math.isnan(a) and math.isnan(b)
        )
    return False


SEEDS = range(50)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_structure_matches_networkx(seed, directed):
    G, H = build(seed, directed)
    assert G.nodes() == list(H)
    assert G.edges() == list(H.edges())
    assert G.edges(data=True) == list(H.edges(data=True))
    assert G.number_of_edges() == H.number_of_edges()
    assert G.degree() == dict(H.degree())
    for v in H:
        assert list(G.neighbors(v)) == list(H.neighbors(v))
        if directed:
            assert list(G.predecessors(v)) == list(H.predecessors(v))
    back = G.to_networkx()
    assert list(back) == list(H)
    assert list(back.edges(data=True)) == list(H.edges(data=True))
    for v in H:  # adjacency order too
        assert list(back.adj[v]) == list(H.adj[v])


def target_lengths(G, target, backend):
    # NetworkX 3.4 returns an iterator, with a FutureWarning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return dict(nx.single_target_shortest_path_length(G, target, backend=backend))


def algorithm_calls(H, directed):
    nodes = list(H)
    src = nodes[0] if nodes else None
    calls = {
        "betweenness": lambda G, b: nx.betweenness_centrality(G, backend=b),
        "betweenness_w": lambda G, b: nx.betweenness_centrality(G, weight="weight", backend=b),
        "edge_betweenness": lambda G, b: list(nx.edge_betweenness_centrality(G, backend=b).items()),
        "edge_betweenness_w": lambda G, b: list(
            nx.edge_betweenness_centrality(G, weight="weight", backend=b).items()
        ),
        "closeness": lambda G, b: nx.closeness_centrality(G, backend=b),
        "closeness_w": lambda G, b: nx.closeness_centrality(G, distance="weight", backend=b),
        "pagerank": lambda G, b: nx.pagerank(G, backend=b),
        "pagerank_cap": lambda G, b: nx.pagerank(G, weight="cap", backend=b),
        "pagerank_missing_attr": lambda G, b: nx.pagerank(G, weight="nope", backend=b),
        "all_pairs": lambda G, b: nx.all_pairs_shortest_path_length(G, backend=b),
        "all_pairs_w": lambda G, b: nx.all_pairs_dijkstra_path_length(G, backend=b),
        "all_paths": lambda G, b: nx.all_pairs_shortest_path(G, backend=b),
        "all_paths_w": lambda G, b: nx.all_pairs_dijkstra_path(G, backend=b),
        "diameter": lambda G, b: nx.diameter(G, backend=b),
        "harmonic": lambda G, b: list(nx.harmonic_centrality(G, backend=b).items()),
        "eigenvector": lambda G, b: list(nx.eigenvector_centrality(G, max_iter=1000, backend=b).items()),
        "katz": lambda G, b: list(nx.katz_centrality(G, alpha=0.02, backend=b).items()),
        "is_bipartite": lambda G, b: nx.is_bipartite(G, backend=b),
        "greedy_color": lambda G, b: list(nx.greedy_color(G, backend=b).items()),
        "core_number": lambda G, b: list(nx.core_number(G, backend=b).items()),
        "k_core": lambda G, b: sorted(map(str, nx.k_core(G, backend=b).edges)),
        "clustering": lambda G, b: list(nx.clustering(G, backend=b).items()),
        "avg_clustering": lambda G, b: nx.average_clustering(G, backend=b),
        "transitivity": lambda G, b: nx.transitivity(G, backend=b),
        "avg_spl_w": lambda G, b: nx.average_shortest_path_length(G, weight="weight", backend=b),
        "degree_centrality": lambda G, b: list(nx.degree_centrality(G, backend=b).items()),
        "is_tree": lambda G, b: nx.is_tree(G, backend=b),
        "is_forest": lambda G, b: nx.is_forest(G, backend=b),
        "dfs_postorder": lambda G, b: list(nx.dfs_postorder_nodes(G, backend=b)),
        "dfs_successors": lambda G, b: list(nx.dfs_successors(G, backend=b).items()),
        "k_shell": lambda G, b: sorted(map(str, nx.k_shell(G, backend=b).edges)),
        "k_truss": lambda G, b: list(nx.k_truss(G, 3, backend=b).edges),
        "onion_layers": lambda G, b: list(nx.onion_layers(G, backend=b).items()),
        "square_clustering": lambda G, b: list(nx.square_clustering(G, backend=b).items()),
        "generalized_degree": lambda G, b: [
            (k, list(v.items())) for k, v in nx.generalized_degree(G, backend=b).items()
        ],
        "harmonic_diameter": lambda G, b: nx.harmonic_diameter(G, backend=b),
        "barycenter": lambda G, b: nx.barycenter(G, backend=b),
        "is_distance_regular": lambda G, b: nx.is_distance_regular(G, backend=b),
        "dsatur": lambda G, b: list(nx.greedy_color(G, "DSATUR", backend=b).items()),
        "is_coloring": lambda G, b: is_coloring(G, {v: i % 2 for i, v in enumerate(G)}, backend=b),
        "isolates": lambda G, b: list(nx.isolates(G, backend=b)),
        "is_regular": lambda G, b: nx.is_regular(G, backend=b),
        "is_eulerian": lambda G, b: nx.is_eulerian(G, backend=b),
        "find_cycle": lambda G, b: nx.find_cycle(G, orientation="ignore", backend=b),
        "kruskal_mst_edges": lambda G, b: list(
            nx.algorithms.tree.mst.kruskal_mst_edges(G, True, data=False, backend=b)
        ),
        "maximum_branching": lambda G, b: list(nx.maximum_branching(G, backend=b).edges(data=True)),
        "greedy_branching": lambda G, b: list(nx.tree.greedy_branching(G, backend=b).edges(data=True)),
        "branching_weight": lambda G, b: nx.tree.branching_weight(G, backend=b),
    }
    if src is not None:
        calls["bfs"] = lambda G, b: nx.single_source_shortest_path_length(G, src, backend=b)
        calls["dijkstra"] = lambda G, b: nx.single_source_dijkstra_path_length(G, src, backend=b)
        calls["paths"] = lambda G, b: list(nx.single_source_shortest_path(G, src, backend=b).items())
        calls["paths_w"] = lambda G, b: list(nx.single_source_dijkstra_path(G, src, backend=b).items())
        calls["target_paths"] = lambda G, b: list(nx.shortest_path(G, target=src, backend=b).items())
        calls["target_paths_w"] = lambda G, b: list(
            nx.shortest_path(G, target=src, weight="weight", backend=b).items()
        )
        dst = list(H)[-1]
        calls["pair"] = lambda G, b: nx.shortest_path(G, src, dst, backend=b)
        calls["pair_w"] = lambda G, b: nx.dijkstra_path(G, src, dst, backend=b)
        calls["has_path"] = lambda G, b: nx.has_path(G, dst, src, backend=b)
        calls["descendants"] = lambda G, b: nx.descendants(G, src, backend=b)
        calls["bidir_dijkstra"] = lambda G, b: nx.bidirectional_dijkstra(G, src, dst, backend=b)
        calls["bfs_edges"] = lambda G, b: list(nx.bfs_edges(G, src, backend=b))
        calls["dfs_edges"] = lambda G, b: list(nx.dfs_edges(G, src, backend=b))
        calls["dfs_preorder"] = lambda G, b: list(nx.dfs_preorder_nodes(G, backend=b))
        calls["bfs_tree"] = lambda G, b: list(nx.bfs_tree(G, src, backend=b).edges)
        calls["all_sp"] = lambda G, b: list(nx.all_shortest_paths(G, src, dst, backend=b))
        calls["all_sp_w"] = lambda G, b: list(nx.all_shortest_paths(G, src, dst, weight="weight", backend=b))
        calls["ancestors"] = lambda G, b: nx.ancestors(G, src, backend=b)
        calls["bfs_layers"] = lambda G, b: list(nx.bfs_layers(G, src, backend=b))
        calls["bfs_successors"] = lambda G, b: list(nx.bfs_successors(G, src, backend=b))
        calls["at_distance"] = lambda G, b: nx.descendants_at_distance(G, src, 2, backend=b)
        calls["multi_source"] = lambda G, b: nx.multi_source_dijkstra(G, [src, dst], backend=b)
        calls["dijkstra_pred"] = lambda G, b: nx.dijkstra_predecessor_and_distance(G, src, backend=b)
        calls["predecessor"] = lambda G, b: nx.predecessor(G, src, backend=b)
        calls["ss_all_sp"] = lambda G, b: list(nx.single_source_all_shortest_paths(G, src, weight="weight", backend=b))
        calls["bellman_ford"] = lambda G, b: nx.single_source_bellman_ford(G, src, backend=b)
        calls["bf_pred"] = lambda G, b: nx.bellman_ford_predecessor_and_distance(G, src, backend=b)
        calls["neg_cycle"] = lambda G, b: nx.negative_edge_cycle(G, backend=b)
        calls["astar"] = lambda G, b: nx.astar_path_length(G, src, dst, backend=b)
        calls["target_length"] = lambda G, b: target_lengths(G, src, b)
        calls["bfs_labeled"] = lambda G, b: list(nx.bfs_labeled_edges(G, src, backend=b))
        calls["dfs_labeled"] = lambda G, b: list(nx.dfs_labeled_edges(G, src, backend=b))
        calls["generic_bfs"] = lambda G, b: list(nx.generic_bfs_edges(G, src, backend=b))
        calls["edge_bfs"] = lambda G, b: list(nx.edge_bfs(G, src, backend=b))
        calls["edge_dfs"] = lambda G, b: list(nx.edge_dfs(G, src, backend=b))
        few = nodes[:3]
        calls["subset_bc"] = lambda G, b: nx.betweenness_centrality_subset(G, few, nodes, weight="weight", backend=b)
        calls["subset_ebc"] = lambda G, b: list(
            nx.edge_betweenness_centrality_subset(G, nodes, few, backend=b).items()
        )
        calls["group_closeness"] = lambda G, b: nx.group_closeness_centrality(G, few, weight="weight", backend=b)
        calls["group_degree"] = lambda G, b: nx.group_degree_centrality(G, few, backend=b)
        calls["group_bc"] = lambda G, b: nx.group_betweenness_centrality(G, few, backend=b)
        calls["local_reaching"] = lambda G, b: nx.local_reaching_centrality(G, src, backend=b)
        calls["local_reaching_w"] = lambda G, b: nx.local_reaching_centrality(G, src, weight="weight", backend=b)
        calls["goldberg_radzik"] = lambda G, b: nx.goldberg_radzik(G, src, backend=b)
        calls["simple_paths"] = lambda G, b: list(islice(nx.all_simple_paths(G, src, dst, cutoff=4, backend=b), 100))
        calls["shortest_simple"] = lambda G, b: list(
            islice(nx.shortest_simple_paths(G, src, dst, weight="weight", backend=b), 20)
        )
        calls["is_simple_path"] = lambda G, b: nx.is_simple_path(G, nodes[:3], backend=b)
    calls["load"] = lambda G, b: list(nx.load_centrality(G, backend=b).items())
    calls["load_w"] = lambda G, b: list(nx.load_centrality(G, weight="weight", backend=b).items())
    calls["edge_load"] = lambda G, b: list(nx.edge_load_centrality(G, backend=b).items())
    calls["percolation"] = lambda G, b: list(nx.percolation_centrality(G, backend=b).items())
    calls["voterank"] = lambda G, b: nx.voterank(G, backend=b)
    calls["floyd_warshall"] = lambda G, b: [
        (k, list(v.items())) for k, v in nx.floyd_warshall(G, backend=b).items()
    ]
    calls["floyd_warshall_numpy"] = lambda G, b: nx.floyd_warshall_numpy(G, backend=b).tolist()
    calls["johnson"] = lambda G, b: nx.johnson(G, backend=b)
    calls["is_isomorphic"] = lambda G, b: nx.is_isomorphic(G, G, backend=b)
    calls["vf2pp_is_isomorphic"] = lambda G, b: nx.vf2pp_is_isomorphic(G, G, node_label="x", backend=b)
    calls["faster_could_be"] = lambda G, b: nx.faster_could_be_isomorphic(G, G, backend=b)
    calls["could_be"] = lambda G, b: nx.could_be_isomorphic(G, G, backend=b)

    def quiet(func):  # Weisfeiler-Lehman hashes warn about changes in 3.5
        def run(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return func(*args, **kwargs)
        return run

    calls["wl_hash"] = lambda G, b: quiet(nx.weisfeiler_lehman_graph_hash)(G, backend=b)
    calls["wl_subgraph_hashes"] = lambda G, b: list(
        quiet(nx.weisfeiler_lehman_subgraph_hashes)(G, iterations=2, backend=b).items()
    )
    if not directed:
        calls["tree_isomorphism"] = lambda G, b: nx.isomorphism.tree_isomorphism(G, G, backend=b)
        if src is not None:
            calls["rooted_tree_isomorphism"] = lambda G, b: nx.isomorphism.rooted_tree_isomorphism(
                G, src, G, src, backend=b
            )
    if directed:
        calls["scc"] = lambda G, b: list(nx.strongly_connected_components(G, backend=b))
        calls["wcc"] = lambda G, b: list(nx.weakly_connected_components(G, backend=b))
        calls["topo"] = lambda G, b: list(nx.topological_sort(G, backend=b))
        calls["dag"] = lambda G, b: nx.is_directed_acyclic_graph(G, backend=b)
        calls["attracting"] = lambda G, b: list(nx.attracting_components(G, backend=b))
        calls["in_degree_centrality"] = lambda G, b: list(nx.in_degree_centrality(G, backend=b).items())
        calls["kosaraju"] = lambda G, b: list(nx.kosaraju_strongly_connected_components(G, backend=b))
        calls["condensation"] = lambda G, b: (
            lambda C: (list(C.nodes(data=True)), list(C.edges), C.graph)
        )(nx.condensation(G, backend=b))
        calls["semiconnected"] = lambda G, b: nx.is_semiconnected(G, backend=b)
        calls["has_cycle"] = lambda G, b: nx.dag.has_cycle(G, backend=b)
        calls["v_structures"] = lambda G, b: list(nx.dag.v_structures(G, backend=b))
        calls["longest_path"] = lambda G, b: nx.dag_longest_path(G, backend=b)
        calls["longest_path_length"] = lambda G, b: nx.dag_longest_path_length(G, backend=b)
        calls["reduction"] = lambda G, b: list(nx.transitive_reduction(G, backend=b).edges)
        calls["closure"] = lambda G, b: list(nx.transitive_closure(G, backend=b).edges(data=True))
        calls["edge_bfs_ignore"] = lambda G, b: list(nx.edge_bfs(G, orientation="ignore", backend=b))
        calls["edge_dfs_reverse"] = lambda G, b: list(nx.edge_dfs(G, orientation="reverse", backend=b))
        calls["is_arborescence"] = lambda G, b: nx.is_arborescence(G, backend=b)
        calls["antichains"] = lambda G, b: list(islice(nx.antichains(G, backend=b), 200))
        calls["all_pairs_lca"] = lambda G, b: list(nx.all_pairs_lowest_common_ancestor(G, backend=b))
        calls["tree_lca"] = lambda G, b: list(nx.tree_all_pairs_lowest_common_ancestor(G, backend=b))
        calls["triadic_census"] = lambda G, b: list(nx.triadic_census(G, backend=b).items())
        calls["d_separator"] = lambda G, b: nx.is_d_separator(G, set(nodes[:2]), set(nodes[-2:]), set(), backend=b)
        if src is not None:
            calls["idom"] = lambda G, b: list(nx.immediate_dominators(G, src, backend=b).items())
            calls["frontiers"] = lambda G, b: [
                (k, list(v)) for k, v in nx.dominance_frontiers(G, src, backend=b).items()
            ]
    else:
        calls["cc"] = lambda G, b: list(nx.connected_components(G, backend=b))
        calls["mst"] = lambda G, b: [(u, v) for u, v in nx.minimum_spanning_edges(G, data=False, backend=b)]
        calls["label_prop"] = lambda G, b: [sorted(map(str, c)) for c in nx.community.label_propagation_communities(G, backend=b)]
        calls["triangles"] = lambda G, b: list(nx.triangles(G, backend=b).items())
        calls["articulation"] = lambda G, b: list(nx.articulation_points(G, backend=b))
        calls["biconnected"] = lambda G, b: list(nx.biconnected_component_edges(G, backend=b))
        calls["is_biconnected"] = lambda G, b: nx.is_biconnected(G, backend=b)
        calls["dispersion"] = lambda G, b: list(nx.dispersion(G, backend=b).items())
        calls["bridges"] = lambda G, b: list(nx.bridges(G, backend=b))
        calls["chains"] = lambda G, b: list(nx.chain_decomposition(G, backend=b))
        calls["cycle_basis"] = lambda G, b: nx.cycle_basis(G, backend=b)
        calls["girth"] = lambda G, b: nx.girth(G, backend=b)
        calls["local_bridges"] = lambda G, b: list(nx.local_bridges(G, backend=b))
        calls["minimum_cycle_basis"] = lambda G, b: nx.minimum_cycle_basis(G, weight="weight", backend=b)
        calls["prim"] = lambda G, b: list(
            nx.algorithms.tree.mst.prim_mst_edges(G, True, data=False, backend=b)
        )
        calls["boruvka"] = lambda G, b: list(
            nx.algorithms.tree.mst.boruvka_mst_edges(G, data=False, backend=b)
        )
        calls["partition_spanning_tree"] = lambda G, b: list(nx.partition_spanning_tree(G, backend=b).edges)
        calls["max_weight_matching"] = lambda G, b: list(nx.max_weight_matching(G, backend=b))
        calls["min_weight_matching"] = lambda G, b: list(nx.min_weight_matching(G, backend=b))
        calls["maximal_matching"] = lambda G, b: list(nx.maximal_matching(G, backend=b))
        calls["node_boundary"] = lambda G, b: list(nx.node_boundary(G, nodes[::2], backend=b))
        calls["edge_boundary"] = lambda G, b: list(nx.edge_boundary(G, nodes[::2], backend=b))
        calls["all_cliques"] = lambda G, b: list(nx.enumerate_all_cliques(G, backend=b))
        calls["clique_number"] = lambda G, b: list(nx.node_clique_number(G, nodes, backend=b).items())
        calls["max_weight_clique"] = lambda G, b: nx.max_weight_clique(G, weight=None, backend=b)
        calls["dominating"] = lambda G, b: nx.is_dominating_set(G, nodes[::3], backend=b)
        if src is not None:
            calls["node_cc"] = lambda G, b: nx.node_connected_component(G, src, backend=b)
        calls["is_chordal"] = lambda G, b: nx.is_chordal(G, backend=b)
        calls["treewidth"] = lambda G, b: nx.chordal_graph_treewidth(G, backend=b)
        calls["to_chordal"] = lambda G, b: (
            lambda H, alpha: (list(H.edges), list(alpha.items()))
        )(*nx.complete_to_chordal_graph(G, backend=b))
        calls["at_free"] = lambda G, b: nx.is_at_free(G, backend=b)
    calls["is_planar"] = lambda G, b: nx.is_planar(G, backend=b)
    calls["planarity"] = lambda G, b: (
        lambda ok, E: (ok, None if E is None else list(E.edges(data=True)))
    )(*nx.check_planarity(G, True, backend=b))
    if directed:
        calls["score_sequence"] = lambda G, b: nx.tournament.score_sequence(G, backend=b)
        if len(nodes) <= 10:  # quintic time in NetworkX 3.4 and 3.5
            calls["tournament_sc"] = lambda G, b: nx.tournament.is_strongly_connected(G, backend=b)
        if src is not None:
            calls["is_reachable"] = lambda G, b: nx.tournament.is_reachable(G, src, nodes[-1], backend=b)
    # Batch 15: communities, efficiency and structural holes.
    halves = [set(nodes[::2]), set(nodes[1::2])]
    calls["modularity"] = lambda G, b: nx.community.modularity(G, halves, backend=b)
    calls["partition_quality"] = lambda G, b: nx.community.partition_quality(G, halves, backend=b)
    calls["greedy_modularity"] = lambda G, b: [
        list(c) for c in nx.community.greedy_modularity_communities(G, weight="weight", backend=b)
    ]
    calls["asyn_lpa"] = lambda G, b: [
        list(c) for c in nx.community.asyn_lpa_communities(G, seed=1, backend=b)
    ]
    calls["closeness_vitality"] = lambda G, b: list(nx.closeness_vitality(G, weight="weight", backend=b).items())
    if directed:
        calls["flow_hierarchy"] = lambda G, b: nx.flow_hierarchy(G, backend=b)
    else:
        calls["global_efficiency"] = lambda G, b: nx.global_efficiency(G, backend=b)
        calls["local_efficiency"] = lambda G, b: nx.local_efficiency(G, backend=b)
        calls["gutman_index"] = lambda G, b: nx.gutman_index(G, weight="weight", backend=b)
        calls["edge_betweenness_partition"] = lambda G, b: [
            list(c) for c in nx.community.edge_betweenness_partition(G, 2, backend=b)
        ] if len(nodes) >= 2 else None
    return calls


@pytest.fixture
def enabled():
    old = nx.config.backend_priority.algos, nx.config.fallback_to_nx
    rustnx.enable()
    yield
    nx.config.backend_priority.algos, nx.config.fallback_to_nx = old


def check_all(G, H, directed):
    for name, call in algorithm_calls(H, directed).items():
        ours = outcome(lambda: call(G, "rustnx"))
        if ours[0] is NotImplementedError:
            # rustnx deliberately hands this case to NetworkX (e.g. empty
            # graphs); through normal dispatch the native graph is converted.
            ours = outcome(lambda: call(G, None))
        ref = outcome(lambda: call(H, "networkx"))
        assert same(ours, ref), name


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_algorithms_match_networkx(seed, directed, enabled):
    G, H = build(seed, directed)
    check_all(G, H, directed)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_from_arrays(seed, directed, enabled):
    rng = random.Random(seed)
    n = rng.randint(1, 40)
    m = rng.randint(0, 3 * n)
    src = [rng.randrange(n) for _ in range(m)]
    dst = [rng.randrange(n) for _ in range(m)]
    w = [rng.choice([0.5, 1.0, 2.0]) for _ in range(m)]
    cls = rustnx.DiGraph if directed else rustnx.Graph
    G = cls.from_arrays(src, dst, w, num_nodes=n)
    H = nx.DiGraph() if directed else nx.Graph()
    H.add_nodes_from(range(n))
    H.add_weighted_edges_from(zip(src, dst, w))
    assert G.nodes() == list(H)
    assert G.edges(data=True) == list(H.edges(data=True))
    check_all(G, H, directed)


def test_from_numpy_arrays():
    np = pytest.importorskip("numpy")
    src = np.array([0, 1, 2], dtype=np.int32)
    dst = np.array([1, 2, 0], dtype=np.uint16)
    G = rustnx.DiGraph.from_arrays(src, dst, np.array([1, 2, 3], dtype=np.float32))
    assert G.edges(data=True) == [(0, 1, {"weight": 1.0}), (1, 2, {"weight": 2.0}), (2, 0, {"weight": 3.0})]
    with pytest.raises(TypeError, match="integers"):
        rustnx.Graph.from_arrays(np.array([0.5]), np.array([1]))
    with pytest.raises(ValueError, match="num_nodes"):
        rustnx.Graph.from_arrays([0, 5], [1, 2], num_nodes=3)


def test_range_index_lookups():
    G = rustnx.Graph.from_arrays([0, 1], [1, 2])
    assert 1 in G and True in G and 1.0 in G
    assert 3 not in G and -1 not in G and "1" not in G and 1.5 not in G
    assert nx.single_source_shortest_path_length(G, True) == {True: 0, 0: 1, 2: 1}


def test_invalid_input():
    with pytest.raises(TypeError, match="numeric"):
        rustnx.Graph([(0, 1, {"color": "red"})])
    with pytest.raises(ValueError, match="NaN"):
        rustnx.Graph([(0, 1, float("nan"))])
    with pytest.raises(ValueError, match="2 or 3"):
        rustnx.Graph([(0, 1, 2, 3)])
    with pytest.raises(ValueError, match="None cannot be a node"):
        rustnx.Graph([(None, 1)])
    with pytest.raises(TypeError, match="strings"):
        rustnx.Graph([(0, 1, {5: 1})])


def test_unimplemented_functions_fall_back_with_enable():
    G = rustnx.DiGraph([(0, 1), (1, 2)])
    old = nx.config.backend_priority.algos, nx.config.fallback_to_nx
    try:
        nx.config.backend_priority.algos = []
        nx.config.fallback_to_nx = False
        with pytest.raises(NotImplementedError):
            nx.trophic_levels(G)
        rustnx.enable()
        assert nx.config.backend_priority.algos[0] == "rustnx"
        assert nx.trophic_levels(G) == {0: 1.0, 1: 2.0, 2: 3.0}
    finally:
        nx.config.backend_priority.algos, nx.config.fallback_to_nx = old


def test_pickle_native():
    G = rustnx.DiGraph([("a", "b", 2), ("b", "c", {"weight": 1.5, "cap": 3})], kind="test")
    H = pickle.loads(pickle.dumps(G))
    assert H.graph == {"kind": "test"}
    assert H.edges(data=True) == G.edges(data=True)
    assert nx.closeness_centrality(H, distance="weight") == nx.closeness_centrality(
        G.to_networkx(), distance="weight", backend="networkx"
    )
    A = rustnx.Graph.from_arrays([0, 1], [1, 2])
    assert pickle.loads(pickle.dumps(A)).edges() == A.edges()


def test_graph_attributes_and_repr():
    G = rustnx.Graph([(1, 2)], name="g")
    assert G.graph == {"name": "g"}
    assert repr(G) == "<Graph (undirected) with 2 nodes and 1 edges>"
    assert G.has_edge(2, 1) and not G.has_edge(1, 3)
    with pytest.raises(nx.NetworkXError):
        G.successors(1)
