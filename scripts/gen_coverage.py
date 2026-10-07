"""Generate docs/COVERAGE.md: every NetworkX function a backend can implement,
and whether rustnx implements it.

    python scripts/gen_coverage.py

The list comes from the installed NetworkX (run it with the newest release),
grouped into the areas used by the todo list in CLAUDE.md. Unlike
docs/API.md this is not checked by a test, because it depends on the
NetworkX version.
"""

import collections
import pathlib

import networkx as nx
from networkx.utils.backends import _registered_algorithms

from rustnx import algorithms  # the installed (or develop-mode) package

ROOT = pathlib.Path(__file__).resolve().parent.parent

OUTPUT = ROOT / "docs" / "COVERAGE.md"

# (todo item, area name, NetworkX modules). A module is the part of the
# function's module path after "networkx.", e.g. "algorithms.centrality".
AREAS = [
    (28, "Shortest paths", ["algorithms.shortest_paths"]),
    (29, "Centrality", ["algorithms.centrality"]),
    (30, "Components", ["algorithms.components"]),
    (31, "Directed acyclic graphs", ["algorithms.dag"]),
    (32, "Traversal", ["algorithms.traversal"]),
    (33, "Distance measures", ["algorithms.distance_measures", "algorithms.distance_regular"]),
    (34, "Clustering and cores", ["algorithms.cluster", "algorithms.core"]),
    (35, "Trees, branchings and arborescences", ["algorithms.tree"]),
    (36, "Link analysis", ["algorithms.link_analysis"]),
    (37, "Bipartite graphs", ["algorithms.bipartite"]),
    (38, "Connectivity, flows and cuts", [
        "algorithms.connectivity", "algorithms.flow", "algorithms.cuts",
    ]),
    (39, "Approximation algorithms", ["algorithms.approximation"]),
    (40, "Communities", ["algorithms.community"]),
    (41, "Coloring", ["algorithms.coloring"]),
    (42, "Isomorphism and graph hashing", [
        "algorithms.isomorphism", "algorithms.graph_hashing",
    ]),
    (43, "Matching, cliques, covers and independent sets", [
        "algorithms.matching", "algorithms.clique", "algorithms.covering",
        "algorithms.dominating", "algorithms.mis",
    ]),
    (44, "Cycles, Euler tours and simple paths", [
        "algorithms.cycles", "algorithms.euler", "algorithms.simple_paths",
        "algorithms.chains",
    ]),
    (45, "Structural tests and graph classes", [
        "algorithms.planarity", "algorithms.chordal", "algorithms.perfect_graph",
        "algorithms.asteroidal", "algorithms.graphical", "algorithms.regular",
        "algorithms.tournament", "algorithms.d_separation", "algorithms.bridges",
        "algorithms.isolate", "algorithms.boundary", "algorithms.triads",
        "algorithms.moral", "algorithms.dominance",
        "algorithms.lowest_common_ancestors",
    ]),
    (46, "Graph measures", [
        "algorithms.assortativity", "algorithms.link_prediction",
        "algorithms.similarity", "algorithms.structuralholes",
        "algorithms.efficiency_measures", "algorithms.smallworld",
        "algorithms.richclub", "algorithms.reciprocity", "algorithms.wiener",
        "algorithms.communicability_alg", "algorithms.vitality",
        "algorithms.walks", "algorithms.polynomials", "algorithms.smetric",
        "algorithms.non_randomness", "algorithms.hierarchy",
        "algorithms.voronoi", "algorithms.broadcasting",
        "algorithms.time_dependent",
    ]),
    (47, "Graph operations and transforms", [
        "algorithms.operators", "algorithms.minors", "algorithms.swap",
        "algorithms.sparsifiers", "algorithms.summarization",
        "algorithms.hybrid", "algorithms.node_classification",
    ]),
    (48, "Linear algebra", ["linalg"]),
    (49, "Graph generators", ["generators"]),
    (50, "Reading, writing, conversion and drawing", [
        "readwrite", "convert", "convert_matrix", "relabel", "drawing", "classes",
    ]),
]

# Missing functions with a recorded reason (todo item numbers).
BLOCKED = {
    "simple_cycles": 14,
    "louvain_communities": 15,
}


# Functions that stay in NetworkX on purpose, with the reason: rustnx can't
# make them faster (their time is in LAPACK, SciPy or random sampling, not
# in anything rustnx could do in Rust), or they are O(1) per call, so
# dispatch would only add overhead. An area is complete when every function
# is done or listed here.
SMALL_GRAPH = (
    "a fixed graph of at most 77 nodes that NetworkX builds in under 0.35 ms (measured); rustnx's dispatch alone costs about 20 microseconds, and its small-input rule keeps such calls in NetworkX anyway"
)

MUTATES = (
    "it changes the caller's NetworkX graph in place, and NetworkX never hands such calls to a backend that converts the graph; converting alone (0.11 s for 500,000 edges, measured) is half of set_edge_attributes' 0.21 s"
)

CONSTRUCTOR = (
    "the hook behind every nx.Graph() / DiGraph() / MultiGraph() / MultiDiGraph(): creating a graph takes 3.4 microseconds and dispatching a call about 9 (measured), and rustnx's results are NetworkX graphs anyway"
)

STAYS = {
    'bull_graph': SMALL_GRAPH,
    'chvatal_graph': SMALL_GRAPH,
    'cubical_graph': SMALL_GRAPH,
    'davis_southern_women_graph': SMALL_GRAPH,
    'desargues_graph': SMALL_GRAPH,
    'diamond_graph': SMALL_GRAPH,
    'dodecahedral_graph': SMALL_GRAPH,
    'florentine_families_graph': SMALL_GRAPH,
    'frucht_graph': SMALL_GRAPH,
    'heawood_graph': SMALL_GRAPH,
    'hoffman_singleton_graph': SMALL_GRAPH,
    'house_graph': SMALL_GRAPH,
    'house_x_graph': SMALL_GRAPH,
    'icosahedral_graph': SMALL_GRAPH,
    'karate_club_graph': SMALL_GRAPH,
    'krackhardt_kite_graph': SMALL_GRAPH,
    'les_miserables_graph': SMALL_GRAPH,
    'moebius_kantor_graph': SMALL_GRAPH,
    'octahedral_graph': SMALL_GRAPH,
    'pappus_graph': SMALL_GRAPH,
    'petersen_graph': SMALL_GRAPH,
    'sedgewick_maze_graph': SMALL_GRAPH,
    'shrikhande_graph': SMALL_GRAPH,
    'tetrahedral_graph': SMALL_GRAPH,
    'triad_graph': SMALL_GRAPH,
    'trivial_graph': SMALL_GRAPH,
    'null_graph': SMALL_GRAPH,
    'truncated_cube_graph': SMALL_GRAPH,
    'truncated_tetrahedron_graph': SMALL_GRAPH,
    'tutte_graph': SMALL_GRAPH,
    'number_of_nonisomorphic_trees': 'a count NetworkX computes in microseconds (2 microseconds for n=20, measured); dispatch would only add time',
    'is_valid_joint_degree': 'a few checks over the given dict (4 microseconds for a small one, measured); converting and dispatching cost as much',
    'random_kernel_graph': "calls the caller's Python kernel function and SciPy's root finder for every step; those calls are the cost",
    'spectral_graph_forge': 'its time is in LAPACK (a dense eigendecomposition) and NumPy',
    'random_regular_expander_graph': "its time is in SciPy's ARPACK eigensolver, whose random start also changes how many graphs it tries; with rustnx's generator inside it measured 0.5x to 1.5x",
    'is_regular_expander': "its time is in SciPy's ARPACK eigensolver; with the matrix built in Rust it measured 1.0x to 1.4x",
    'adjacency_spectrum': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'laplacian_spectrum': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'normalized_laplacian_spectrum': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'modularity_spectrum': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'bethe_hessian_spectrum': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'magnetic_spectrum': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'modularity_matrix': 'dense NumPy arithmetic on an n x n matrix dominates; measured no faster with the matrix built in Rust',
    'directed_modularity_matrix': 'dense NumPy arithmetic on an n x n matrix dominates; measured no faster with the matrix built in Rust',
    'directed_laplacian_matrix': 'dense NumPy arithmetic on an n x n matrix dominates; measured no faster with the matrix built in Rust',
    'directed_combinatorial_laplacian_matrix': 'dense NumPy arithmetic on an n x n matrix dominates; measured no faster with the matrix built in Rust',
    'algebraic_connectivity': "its time is in SciPy's sparse eigensolvers (TraceMIN, LOBPCG, ARPACK); measured no faster with the matrix built in Rust",
    'fiedler_vector': "its time is in SciPy's sparse eigensolvers (TraceMIN, LOBPCG, ARPACK); measured no faster with the matrix built in Rust",
    'spectral_ordering': "its time is in SciPy's sparse eigensolvers (TraceMIN, LOBPCG, ARPACK); measured no faster with the matrix built in Rust",
    'spectral_bisection': "its time is in SciPy's sparse eigensolvers (TraceMIN, LOBPCG, ARPACK); measured no faster with the matrix built in Rust",
    'katz_centrality_numpy': 'dense NumPy arithmetic on an n x n matrix dominates; measured no faster with the matrix built in Rust',
    'laplacian_centrality': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'subgraph_centrality': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'subgraph_centrality_exp': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'estrada_index': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'communicability_betweenness_centrality': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'second_order_centrality': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'current_flow_closeness_centrality': "its time is in SciPy's sparse LU solves; measured no faster with the matrix built in Rust",
    'current_flow_betweenness_centrality': "its time is in SciPy's sparse LU solves; measured no faster with the matrix built in Rust",
    'edge_current_flow_betweenness_centrality': "its time is in SciPy's sparse LU solves; measured no faster with the matrix built in Rust",
    'current_flow_betweenness_centrality_subset': "its time is in SciPy's sparse LU solves; measured no faster with the matrix built in Rust",
    'edge_current_flow_betweenness_centrality_subset': "its time is in SciPy's sparse LU solves; measured no faster with the matrix built in Rust",
    'approximate_current_flow_betweenness_centrality': "its time is in SciPy's sparse LU solves; measured no faster with the matrix built in Rust",
    'flow_matrix_row': "its time is in SciPy's sparse LU solves; measured no faster with the matrix built in Rust",
    'trophic_levels': 'dense NumPy arithmetic on an n x n matrix dominates; measured no faster with the matrix built in Rust',
    'trophic_differences': 'dense NumPy arithmetic on an n x n matrix dominates; measured no faster with the matrix built in Rust',
    'trophic_incoherence_parameter': 'dense NumPy arithmetic on an n x n matrix dominates; measured no faster with the matrix built in Rust',
    'incremental_closeness_centrality': "updates closeness while the caller edits the graph; converting the changed graph on every call costs more than NetworkX's incremental update",
    'resistance_distance': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'effective_graph_resistance': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'kemeny_constant': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'number_of_spanning_trees': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'google_matrix': 'dense NumPy arithmetic on an n x n matrix dominates; measured no faster with the matrix built in Rust',
    'birank': 'a few sparse matrix-vector products; measured no faster with the matrix built in Rust',
    'minimum_weight_full_matching': "its time is in SciPy's min_weight_full_bipartite_matching; measured no faster with the matrix built in Rust",
    'spectral_bipartivity': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'degrees': "returns NetworkX's own degree views; there is nothing to compute",
    'colliders': "NetworkX's C-level loop over predecessor pairs beats building the tuples from Rust (measured 3x slower in rustnx)",
    'reconstruct_path': 'walks a predecessor dict for the length of one path; dispatch would only add time',
    'bfs_beam_edges': "calls the caller's Python `value` function for every neighbor; those calls are the cost",
    'is_edge_cover': 'O(1) or one C-level set operation per call; dispatch would only add time',
    'is_isolate': 'O(1) or one C-level set operation per call; dispatch would only add time',
    'is_triad': 'O(1) or one C-level set operation per call; dispatch would only add time',
    'triad_type': 'O(1) or one C-level set operation per call; dispatch would only add time',
    'all_triads': 'builds a NetworkX subgraph copy for every node triple; those copies are the cost',
    'triads_by_type': 'builds a NetworkX subgraph copy for every node triple; those copies are the cost',
    'communicability': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'communicability_exp': 'its time is in LAPACK (dense eigenvalues, inverses or matrix exponentials); measured no faster with the matrix built in Rust',
    'mutual_weight': 'O(degree) per call; dispatch would only add time',
    'normalized_mutual_weight': 'O(degree) per call; dispatch would only add time',
    'is_empty': 'stops at the first node with a neighbor (1.3 microseconds on a 500,000-edge graph, measured); converting the graph would cost far more',
    'to_edgelist': 'returns a lazy edge view (1.8 microseconds, measured); nothing to compute',
    'set_node_attributes': MUTATES,
    'set_edge_attributes': MUTATES,
    'remove_node_attributes': MUTATES,
    'remove_edge_attributes': MUTATES,
    'graph__new__': CONSTRUCTOR,
    'digraph__new__': CONSTRUCTOR,
    'multigraph__new__': CONSTRUCTOR,
    'multidigraph__new__': CONSTRUCTOR,
    'forceatlas2_layout': "its time is NumPy's dense O(n^2) force arithmetic in every iteration (the adjacency matrix rustnx could build is 0.1% of it, measured), and its floats follow NumPy's summation order",
    'to_pandas_adjacency': 'its time is in pandas building the dense DataFrame; with the matrix built in Rust it measured 0.8x to 1.0x',
    'from_pydot': "it reads pydot's Python objects through pydot's own methods (0.055 s for 8,000 edges, measured), which rustnx would have to call the same way",
}


def module_of(func):
    path = getattr(func, "orig_func", func).__module__.split(".")[1:]
    return ".".join(path[:2])


def area_of(module):
    for item, name, modules in AREAS:
        for prefix in modules:
            if module == prefix or module.startswith(prefix + "."):
                return item, name
    raise SystemExit(f"module {module} has no area in AREAS")


def main():
    implemented = set(algorithms.__all__)
    by_area = collections.defaultdict(list)
    for name, func in _registered_algorithms.items():
        by_area[area_of(module_of(func))].append(name)
    total = sum(len(v) for v in by_area.values())
    done = sum(name in implemented for v in by_area.values() for name in v)

    lines = [
        "# NetworkX coverage",
        "",
        "Every function NetworkX lets a backend implement, and whether rustnx",
        "does. Functions rustnx doesn't implement still work: NetworkX runs",
        "them itself. Generated by `scripts/gen_coverage.py` from NetworkX",
        f"{nx.__version__}; the item numbers refer to the todo list in `CLAUDE.md`.",
        "",
        f"rustnx implements {done} of {total} functions.",
        "",
        f"{sum(n in STAYS and n not in implemented for v in by_area.values() for n in v)} more"
        " stay in NetworkX on purpose, each with its reason below: rustnx",
        "can't make them faster.",
        "",
        "| Item | Area | Functions | rustnx | Stays in NetworkX | Left |",
        "|------|------|-----------|--------|-------------------|------|",
    ]
    for item, name, _ in AREAS:
        names = by_area.get((item, name), [])
        count = sum(n in implemented for n in names)
        stays = sum(n in STAYS and n not in implemented for n in names)
        left = len(names) - count - stays
        lines.append(
            f"| {item} | [{name}](#{_anchor(item, name)}) | {len(names)} | {count} | {stays} | {left} |"
        )
    for item, name, _ in AREAS:
        names = sorted(by_area.get((item, name), []))
        lines += ["", f"## {item}. {name}", ""]
        if not names:
            lines.append("No functions in this NetworkX release.")
            continue
        lines += ["| Function | Status |", "|----------|--------|"]
        for n in names:
            if n in implemented:
                status = "Done"
            elif n in STAYS:
                status = f"Stays in NetworkX: {STAYS[n]}"
            elif n in BLOCKED:
                status = f"Blocked (item {BLOCKED[n]})"
            else:
                status = "To do"
            lines.append(f"| `{n}` | {status} |")
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {done} of {total} functions")


def _anchor(item, name):
    text = f"{item}. {name}".lower()
    return "".join(c for c in text if c.isalnum() or c in " -").replace(" ", "-")


if __name__ == "__main__":
    main()
