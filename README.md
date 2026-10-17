# rustnx

**NetworkX, but fast.** rustnx is a Rust-powered backend for
[NetworkX](https://networkx.org). You keep writing normal NetworkX code, and
supported algorithms run in Rust instead of Python: across the 665 functions
timed, the median is 10× faster, and the heavy ones (centrality,
distances, clustering) are 50–600× faster. On seven real-world analysis
pipelines the whole script ran 1.8–495× faster ([details](docs/BENCHMARKS.md)).

Install with `pip install rustnx` (Python 3.10 or newer, including
free-threaded 3.14t; prebuilt for Linux, macOS and Windows).

```python
import networkx as nx
import rustnx

rustnx.enable()                           # or see "Turning it on" below

G = nx.barabasi_albert_graph(4000, 4, seed=1)
nx.betweenness_centrality(G)              # runs in Rust: 0.4s instead of 65s
```

Nothing else changes, with one caveat: `betweenness_centrality`,
`edge_betweenness_centrality`, `pagerank` and weighted `clustering` /
`average_clustering` use fast float arithmetic by default, so their floats
can differ from NetworkX's in the last bits (about
1e-16 to 1e-15 relative). Set `nx.config.backends.rustnx.exact_floats = True`
for NetworkX's exact numbers; see [Fast and exact
floats](#fast-and-exact-floats). Anything rustnx doesn't support, such as other
functions, multigraphs in functions that treat parallel edges specially, or callable weights, keeps running in NetworkX, so
turning it on never breaks working code.

## When rustnx helps, and when it doesn't

A NetworkX graph is a dict of dicts in Python, so rustnx first copies it into
a compact Rust layout. NetworkX caches the copy on the graph, and later calls
reuse it until the graph changes.

| 200,000 nodes / 1M edges | Converting to rustnx | Just walking the graph in pure Python |
|---|---|---|
| Undirected, no weights | 0.23 s | 0.09 s |
| Undirected, with weights | 0.59 s | 0.45 s |
| Directed, with weights | 0.32 s | 0.21 s |

Conversion costs one to three times as much as reading the graph once in
Python, which is the floor for anything that starts from a NetworkX graph. So:

- **Heavy algorithms win right away.** Betweenness, closeness, PageRank,
  distance measures, clustering and the centralities take seconds to minutes
  in NetworkX and a fraction of a second here, conversion included.
- **Cheap algorithms win when repeated.** A single BFS or Dijkstra on a big
  graph costs about the same as one conversion, so the first call is not
  much faster; the following calls on the same graph are.
- **Small graphs stay in NetworkX.** Below 500 nodes, linear-time functions
  run in NetworkX automatically, since converting would cost more than it
  saves. You can still force rustnx with `backend="rustnx"`.
- **Big Python results limit the gain.** Functions that return a path for
  every node (`single_source_shortest_path`, `all_pairs_shortest_path`,
  `all_shortest_paths`) or a new graph (`k_core`, `bfs_tree`) spend most of
  their time building Python objects, so they speed up 1–10× rather than 50×.
- **Quick checks and single queries stay in NetworkX.** `is_regular`,
  `is_eulerian`, `has_path`, `find_cycle` and the like often finish in
  microseconds, at the first counter-example or after a few steps, so
  converting the graph would cost more than NetworkX's whole answer. They
  run in NetworkX until the graph has been converted for something else;
  after that rustnx answers in about the same time (NetworkX's dispatch
  overhead, around 20 µs, is most of either).
- **Some inputs run in NetworkX:** unsupported functions or parameters,
  callable weights, multigraphs in functions that treat parallel edges
  specially, graphs whose weights mix ints and floats in functions that
  return lengths, and graph subclasses that override how they are read. The
  results are still correct; they're just not faster.

To skip conversion entirely, build the graph in Rust (next section).

## Native graphs: skip NetworkX entirely

For big graphs, build the graph in Rust directly. There's no conversion
step, and it uses a fraction of the memory:

```python
import networkx as nx
import rustnx

rustnx.enable()  # use rustnx where it can; everything else falls back to NetworkX

G = rustnx.DiGraph([("a", "b", 2.5), ("b", "c", 1), ("c", "a", {"weight": 4})])
nx.pagerank(G)                       # runs in Rust
nx.is_tree(G)                        # not in rustnx: converted to NetworkX automatically

G = rustnx.Graph.from_arrays(src, dst, weights)   # NumPy arrays; nodes 0..n-1
```

| 1M-edge directed graph | `networkx.DiGraph` | `rustnx.DiGraph(edges)` | `rustnx.DiGraph.from_arrays` |
|---|---|---|---|
| Build time | 3.53 s | 1.52 s | 0.30 s |
| Memory (growth of the process) | 357 MiB | 97 MiB | 122 MiB |
| `pagerank`, first call with rustnx enabled | 0.81 s (converts first) | 0.074 s | 0.062 s |

- **The same results as NetworkX.** Nodes come in order of first
  appearance, each node's neighbors in insertion order, and duplicate edges
  merge as `add_edge` would. So every algorithm returns what it would on a
  `networkx.Graph` built from the same edges.
- **Read-only.** It has `nodes()`, `edges(data=True)`, `neighbors`,
  `successors`, `predecessors`, `degree`, `has_edge`, `has_node` and `len`.
  `G.to_networkx()` gives the full NetworkX API.
- **Edge attributes must be numeric** (or `None`). Graphs can be pickled.
- **Call `rustnx.enable()` first.** Without it, NetworkX raises
  `NotImplementedError` when a rustnx graph reaches a function rustnx doesn't
  implement, instead of converting it.

## rustworkx-compatible API

The same Rust core also serves rustworkx's API:

```python
import rustnx.rx as rx      # instead of: import rustworkx as rx

g = rx.PyDiGraph()
g.extend_from_weighted_edge_list([(0, 1, 2.0), (1, 2, 1.0), (2, 0, 4.0)])
rx.strongly_connected_components(g)
rx.dijkstra_shortest_path_lengths(g, 0, float)
```

- **It behaves the same, not just the same names.** `PyGraph` and
  `PyDiGraph` follow rustworkx's index model. Indices of removed nodes and
  edges are reused, most recently removed first. Graphs are multigraphs by
  default, and `multigraph=False` merges duplicate edges as rustworkx does.
  Neighbors are visited in petgraph's order, so order-dependent results
  (`strongly_connected_components`, `topological_sort`) match rustworkx
  exactly. Exceptions use rustworkx's names (`NullGraph`, `DAGHasCycle`,
  `NoEdgeBetweenNodes`, `FailedToConverge` and so on).
- **It's tested against the real rustworkx.** `tests/test_rx_api.py` applies
  random sequences of adds and removals to both libraries and compares every
  query and algorithm.
- **Supported:** the core graph-building, editing and query methods, plus
  `betweenness_centrality`, `closeness_centrality`, `pagerank`,
  `dijkstra_shortest_path_lengths`, `all_pairs_dijkstra_path_lengths`, the
  connected, strongly and weakly connected component functions,
  `topological_sort`, `is_directed_acyclic_graph` and `networkx_converter`.
  Not yet: subgraphs, `compose`, contraction, matrix and file I/O, and
  `check_cycle=True`.
- **Speed:** the algorithms are as fast as rustworkx's or faster
  (betweenness 1.6×, closeness 38×, strong components 1.2×, rustworkx
  0.18.1). Building graphs is slower: the graph is stored in Python, so
  adding 500k edges takes 1.1 s, against rustworkx's 0.04 s.

## Supported algorithms

The full list, with every parameter rustnx handles and what falls back to
NetworkX, is in [docs/API.md](docs/API.md).

| Function | Notes |
|---|---|
| `betweenness_centrality` | Unweighted and weighted, `normalized`, `endpoints`, and `k` sampling (picks the same nodes as NetworkX for a given `seed`). Parallel. |
| `edge_betweenness_centrality` | Unweighted and weighted, `normalized`, and `k` sampling (same nodes as NetworkX for a given `seed`). Parallel. Rescaled by the installed NetworkX's own code. |
| `closeness_centrality` | Unweighted and `distance=`, `wf_improved`, single node `u=`. Parallel. Results are **bit-for-bit identical** to NetworkX. |
| `single_source_shortest_path_length` | Same nodes and same dict order as NetworkX, with `cutoff`. |
| `single_source_dijkstra_path_length` | Same order as NetworkX, with `cutoff`. Integer weights give integer distances. Raises the same error on negative cycles. |
| `connected_components`, `number_connected_components`, `is_connected` | Components come out in the same order as NetworkX. |
| `pagerank` | All options: `alpha`, `personalization`, `nstart`, `dangling`, `weight`, `tol`, `max_iter`. Raises `PowerIterationFailedConvergence` like NetworkX. |
| `strongly_connected_components`, `number_strongly_connected_components`, `is_strongly_connected` | Same components in the same order as NetworkX. |
| `weakly_connected_components`, `number_weakly_connected_components`, `is_weakly_connected` | Same components in the same order as NetworkX. |
| `topological_sort`, `topological_generations`, `is_directed_acyclic_graph` | Same order as NetworkX. If the graph changes mid-iteration, it raises the same errors as NetworkX. |
| `eccentricity`, `diameter`, `radius`, `center`, `periphery` | Unweighted: bit-parallel BFS. Weighted: parallel Dijkstra. Same errors as NetworkX (disconnected graphs, negative weights). `usebounds=True`, `e=`/`sp=` and trees in `center` run in NetworkX. |
| `average_shortest_path_length`, `wiener_index` | As above; weighted sums are added in NetworkX's order, so float results match exactly. |
| `all_pairs_shortest_path_length`, `all_pairs_dijkstra_path_length` | Parallel, in batches; same order as NetworkX. |
| `shortest_path`, `shortest_path_length`, `single_source_shortest_path`, `single_target_shortest_path`, `bidirectional_shortest_path`, `has_path` | The same paths as NetworkX, ties included, in the same dict order. `shortest_path` with no source and no target runs in NetworkX. |
| `dijkstra_path`, `dijkstra_path_length`, `single_source_dijkstra`, `single_source_dijkstra_path` | The same paths as NetworkX, ties included. The paths dict follows the installed NetworkX's order, which changed in 3.6. If weights mix ints and floats, functions that return lengths run in NetworkX (whether a length is an int depends on the path). |
| `all_pairs_shortest_path`, `all_pairs_dijkstra_path`, `all_pairs_dijkstra` | Parallel, in batches. |
| `descendants`, `ancestors` | Same sets and errors as NetworkX. |
| `triangles`, `clustering`, `average_clustering`, `transitivity` | Directed and undirected, with `nodes=`. Parallel. Unweighted results are **bit-for-bit identical** to NetworkX. Weighted clustering uses [fast floats](#fast-and-exact-floats) (about 70x faster); with exact floats it runs in NetworkX. |
| `normalized_laplacian_matrix`, `bethe_hessian_matrix`, `tournament_matrix`, `eigenvector_centrality_numpy`, `hits` | NetworkX's own SciPy code, with the sparse matrix it builds supplied by rustnx: identical inputs, so identical results, 2x to 21x faster. `eigenvector_centrality_numpy` and `hits` (before NetworkX 3.7) use ARPACK, which starts from a random vector, so their last bits vary from run to run in NetworkX itself; rustnx's are as close to NetworkX as NetworkX is to itself. |
| `k_factor`, `junction_tree`, `find_induced_nodes` | NetworkX's own code with its expensive inner call (`max_weight_matching`, chordal completion and cliques, `is_chordal`) run in Rust. **Bit-for-bit identical**; 1.2x to 54x faster. |
| `simple_cycles`, `chordless_cycles` | NetworkX's own code picks components, start nodes and stems (all by set order); the cycle searches inside run in Rust on the same neighbor lists. The same cycles in the same order, lazily, multigraphs included; 2x to 5x faster. |
| `eulerize`, `local_and_global_consistency`, `harmonic_function` | NetworkX's own code with its matching or adjacency matrix from rustnx. **Bit-for-bit identical**; 2x to 5x faster. |
| `attr_matrix`, `attr_sparse_matrix`, `magnetic_laplacian_matrix` | Built in Rust, **bit-for-bit identical**, 6x to 15x faster. Attribute matrices with `node_attr` fall back. |
| `bidirectional_dijkstra` | Same path and distance as NetworkX, ties included. Also used by weighted `shortest_path(G, source, target)`. |
| `harmonic_centrality` | Unweighted and `distance=`, `sources=`. Parallel. **Bit-for-bit identical** to NetworkX. A small `nbunch` with many `sources` runs in NetworkX. |
| `eigenvector_centrality`, `katz_centrality` | All options except Katz's `nstart` and per-node `beta`. **Bit-for-bit identical** to NetworkX, including when they stop. |
| `core_number`, `k_core` | Same values and errors as NetworkX. `k_core` builds its subgraph in NetworkX, so the speedup is only in the core numbers. |
| `is_bipartite` | Directed and undirected. |
| `bfs_edges`, `dfs_edges`, `dfs_preorder_nodes` | Same order as NetworkX, with `depth_limit` and `reverse`. `sort_neighbors` runs in NetworkX. |
| `bfs_tree`, `dfs_tree` | Same trees as NetworkX. |
| `minimum_spanning_edges`, `maximum_spanning_edges`, `minimum_spanning_tree`, `maximum_spanning_tree` | Kruskal's algorithm (the default), with the same edges in the same order, ties included; yields the graph's own edge data dicts. Prim and Borůvka also run in Rust, with the same start nodes and tie-breaking as NetworkX. |
| `all_shortest_paths` | Unweighted and Dijkstra. Paths are generated lazily in Rust, in NetworkX's order (which differs before 3.7 when zero-weight cycles exist). |
| `greedy_color`, `label_propagation_communities` | `greedy_color` with the `largest_first`, DSATUR, connected-sequential and random-sequential strategies; `smallest_last`, `independent_set`, callable strategies and `interchange` run in NetworkX. |
| `degree_centrality`, `in_degree_centrality`, `out_degree_centrality` | Same values as NetworkX. |
| `node_connected_component`, `articulation_points`, `biconnected_components`, `biconnected_component_edges`, `is_biconnected` | Same order as NetworkX, including the order inside each returned set. |
| `attracting_components`, `number_attracting_components`, `is_attracting_component` | Same components in the same order as NetworkX. |
| `bfs_predecessors`, `bfs_successors`, `bfs_layers`, `descendants_at_distance` | Same order as NetworkX, including each version's first layer in `bfs_layers` and the 3.7 deprecation warning of `bfs_predecessors`. |
| `dfs_postorder_nodes`, `dfs_predecessors`, `dfs_successors` | Same order as NetworkX, with `depth_limit`. `sort_neighbors` runs in NetworkX. |
| `is_tree`, `is_forest` | Same results and errors as NetworkX. |
| `multi_source_dijkstra`, `multi_source_dijkstra_path`, `multi_source_dijkstra_path_length`, `dijkstra_predecessor_and_distance`, `predecessor`, `single_target_shortest_path_length`, `single_source_all_shortest_paths`, `all_pairs_all_shortest_paths` | Same paths, predecessor lists and order as each NetworkX version (the 3.4 iterator of `single_target_shortest_path_length` included). |
| `bellman_ford_path`, `bellman_ford_path_length`, `single_source_bellman_ford`, `single_source_bellman_ford_path`, `single_source_bellman_ford_path_length`, `bellman_ford_predecessor_and_distance`, `all_pairs_bellman_ford_path`, `all_pairs_bellman_ford_path_length`, `negative_edge_cycle`, `find_negative_cycle` | NetworkX's queue-based Bellman-Ford, with the same negative-cycle detection and the same cycle found. `None` weights run in NetworkX. |
| `astar_path`, `astar_path_length` | Without a `heuristic`; a heuristic runs in NetworkX. |
| `dag_longest_path`, `dag_longest_path_length`, `lexicographical_topological_sort`, `all_topological_sorts`, `nx.dag.has_cycle`, `is_aperiodic`, `nx.dag.v_structures`, `nx.dag.root_to_leaf_paths`, `dag_to_branching` | Same order as NetworkX. `topo_order` and `key` arguments run in NetworkX. |
| `transitive_closure`, `transitive_closure_dag`, `transitive_reduction`, `condensation` | Same graphs as NetworkX, including node and edge order and the set order inside `condensation`'s members. |
| `generic_bfs_edges`, `bfs_labeled_edges`, `dfs_labeled_edges`, `edge_bfs`, `edge_dfs` | Same order as NetworkX. Callable arguments run in NetworkX. |
| `kosaraju_strongly_connected_components`, `is_semiconnected` | Same components in the same order as each NetworkX version. |
| `betweenness_centrality_subset`, `edge_betweenness_centrality_subset`, `load_centrality` (`newman_betweenness_centrality`), `edge_load_centrality`, `percolation_centrality` | Parallel, and **bit-for-bit identical** to NetworkX. Load centrality needs all-int or all-str node labels. |
| `group_betweenness_centrality`, `group_closeness_centrality`, `group_degree_centrality`, `group_in_degree_centrality`, `group_out_degree_centrality`, `prominent_group` | Follow each NetworkX version's algorithm (group betweenness changed in 3.7). `prominent_group` needs pandas, like NetworkX. |
| `local_reaching_centrality`, `global_reaching_centrality`, `voterank`, `dispersion` | **Bit-for-bit identical.** `dispersion` runs in Rust for undirected graphs without self-loops. |
| `k_shell`, `k_crust`, `k_corona`, `k_truss`, `onion_layers` | Same subgraphs and errors as NetworkX; the subgraphs are built from the original graph. |
| `square_clustering`, `generalized_degree`, `all_triangles` | **Bit-for-bit identical**, following the installed NetworkX's formula (square clustering changed in 3.5). |
| `centroid` (`barycenter` before 3.7), `harmonic_diameter`, `is_distance_regular`, `intersection_array`, `is_strongly_regular` | Same results and errors as each NetworkX version. |
| `is_coloring`, `is_equitable` | Same results as NetworkX. |
| `equitable_color` | NetworkX's port of Kierstead, Kostochka, Mydlarz and Szemerédi followed step by step, its one set-order-dependent step (Case II) replayed with CPython's set order: the same coloring, in the same order. 35x to 170x faster. Directed graphs and multigraphs run in NetworkX. |
| `is_arborescence`, `is_branching`, `to_prufer_sequence`, `kruskal_mst_edges` | Same results as NetworkX. |
| `bridges`, `has_bridges`, `local_bridges`, `chain_decomposition`, `isolates`, `number_of_isolates`, `is_regular`, `is_k_regular`, `is_tournament`, `immediate_dominators`, `dominance_frontiers` | Same order as each NetworkX version (dominators changed in 3.7). `bridges` with a `root` runs in NetworkX. |
| `is_eulerian`, `has_eulerian_path`, `is_semieulerian`, `eulerian_circuit`, `eulerian_path`, `cycle_basis`, `find_cycle`, `girth` | Same circuits, cycles and order as NetworkX. |
| `floyd_warshall`, `floyd_warshall_predecessor_and_distance`, `floyd_warshall_tree`, `floyd_warshall_numpy` | Parallel. The same dicts (row order, int/float types) or NumPy array as each NetworkX version (3.7 changed missing weights and added negative-cycle checks). Mixed int/float weights run in NetworkX. |
| `johnson`, `goldberg_radzik` | Same paths, predecessors and distances, ties included; `goldberg_radzik` follows each version's scan order. |
| `all_simple_paths`, `all_simple_edge_paths`, `shortest_simple_paths`, `is_simple_path` | Paths generated lazily in Rust, in NetworkX's order (Yen's algorithm with NetworkX's tie-breaking for `shortest_simple_paths`). |
| `antichains`, `nx.dag.antichain_width`, `minimum_cycle_basis` | Same results and order as NetworkX; antichains are generated lazily. |
| `maximum_branching`, `minimum_branching`, `minimal_branching`, `maximum_spanning_arborescence`, `minimum_spanning_arborescence`, `greedy_branching`, `branching_weight` | Edmonds' algorithm as NetworkX runs it: the same edges in the same order. The minimum and arborescence variants change G's weights exactly as NetworkX does. |
| `prim_mst_edges`, `boruvka_mst_edges`, `partition_spanning_tree`, `from_prufer_sequence`, `from_nested_tuple`, `to_nested_tuple`, `nx.tree.centroid` | Same edges, graphs and tuples as NetworkX. `to_nested_tuple` runs in Rust with `canonical_form=True`. |
| `lowest_common_ancestor`, `all_pairs_lowest_common_ancestor`, `tree_all_pairs_lowest_common_ancestor` | Same answers and order as NetworkX. |
| `is_planar`, `check_planarity`, `check_planarity_recursive`, `get_counterexample`, `get_counterexample_recursive` | The same `PlanarEmbedding` (half-edge order and data) and counterexample as NetworkX. |
| `is_chordal`, `chordal_graph_treewidth`, `complete_to_chordal_graph`, `is_at_free`, `is_perfect_graph` | Same results as NetworkX. Graphs with self-loops run in NetworkX. |
| `nx.tournament.is_reachable`, `nx.tournament.is_strongly_connected`, `nx.tournament.score_sequence` | Same results as NetworkX. |
| `triadic_census`, `is_d_separator`, `is_minimal_d_separator`, `find_minimal_d_separator`, `node_boundary`, `edge_boundary` | Same results, and the same set order, as NetworkX. |
| `is_graphical`, `is_digraphical`, `is_multigraphical`, `is_pseudographical`, `is_valid_degree_sequence_erdos_gallai`, `is_valid_degree_sequence_havel_hakimi` | Degree-sequence tests. They take no graph, so they run in rustnx when it is first in `backend_priority` or with `backend="rustnx"`. |
| `max_weight_matching`, `min_weight_matching`, `maximal_matching`, `is_matching`, `is_maximal_matching`, `is_perfect_matching`, `min_edge_cover` | A Rust port of NetworkX's blossom algorithm: the same matching, set order and edge orientation. |
| `is_dominating_set`, `connected_dominating_set`, `is_connected_dominating_set`, `enumerate_all_cliques`, `node_clique_number`, `max_weight_clique` | Same results and order as NetworkX. |
| `is_isomorphic`, `vf2pp_is_isomorphic`, `vf2pp_subgraph_is_isomorphic`, `vf2pp_is_monomorphic`, `could_be_isomorphic`, `fast_could_be_isomorphic`, `faster_could_be_isomorphic` | Yes/no isomorphism tests with an exact matcher in Rust. Node and edge match functions run in NetworkX. |
| `tree_isomorphism`, `rooted_tree_isomorphism`, `root_trees`, `weisfeiler_lehman_graph_hash`, `weisfeiler_lehman_subgraph_hashes` | Same mappings and hashes as each NetworkX version. |
| `nx.bipartite`: `color`, `sets`, `is_bipartite_node_set`, `hopcroft_karp_matching`, `to_vertex_cover`, `closeness_centrality`, `node_redundancy`, `butterflies` | Same results and order as NetworkX. |
| `maximum_flow`, `maximum_flow_value`, `minimum_cut`, `minimum_cut_value`, `edmonds_karp`, `shortest_augmenting_path`, `dinitz`, `boykov_kolmogorov`, `preflow_push`, `build_residual_network`, `build_flow_dict`, `gomory_hu_tree` | NetworkX's algorithms ported step by step: the same flows, values (int or float), residual networks and minimum cut sets, with each version's set order. |
| `network_simplex`, `min_cost_flow`, `min_cost_flow_cost`, `max_flow_min_cost`, `cost_of_flow` | Same flows and costs as each NetworkX version. `capacity_scaling` runs in NetworkX. |
| `cut_size`, `volume`, `normalized_cut_size`, `conductance`, `edge_expansion`, `mixing_expansion`, `node_expansion`, `boundary_expansion` | Bit-for-bit identical. |
| `node_connectivity`, `edge_connectivity`, `local_node_connectivity`, `local_edge_connectivity`, `average_node_connectivity`, `all_pairs_node_connectivity`, `minimum_node_cut`, `minimum_edge_cut`, `minimum_st_node_cut`, `minimum_st_edge_cut`, `node_disjoint_paths`, `edge_disjoint_paths`, `stoer_wagner` | Same values, cut sets (and their iteration order) and paths as each NetworkX version. Pairs run in parallel. |
| `bridge_components`, `k_edge_components`, `k_edge_subgraphs`, `is_k_edge_connected`, `is_locally_k_edge_connected`, `k_edge_augmentation`, `one_edge_augmentation`, `bridge_augmentation` and their unconstrained variants | k-edge components for k up to 2; augmentation without `avail`. Other cases run in NetworkX. |
| `degree_assortativity_coefficient`, `degree_pearson_correlation_coefficient`, `attribute_assortativity_coefficient`, `numeric_assortativity_coefficient`, mixing dicts and matrices, `node_degree_xy`, `node_attribute_xy`, `average_degree_connectivity`, `average_neighbor_degree` | Mixing data built in Rust; the final NumPy step is NetworkX's own code, so results are identical. |
| `jaccard_coefficient`, `adamic_adar_index`, `resource_allocation_index`, `preferential_attachment`, `cn_soundarajan_hopcroft`, `ra_index_soundarajan_hopcroft`, `within_inter_cluster`, `common_neighbor_centrality` | Streamed in batches; float sums follow NetworkX's set order exactly. |
| `reciprocity`, `overall_reciprocity`, `rich_club_coefficient`, `s_metric`, `number_of_walks` | `rich_club_coefficient` with `normalized=False`; the normalized form runs in NetworkX. |
| `modularity`, `overlapping_modularity`, `partition_quality`, `is_partition`, `is_cover`, `intra_community_edges`, `inter_community_edges`, `inter_community_non_edges` | Bit-for-bit identical. |
| `greedy_modularity_communities`, `naive_greedy_modularity_communities`, `girvan_newman`, `edge_betweenness_partition`, `asyn_lpa_communities`, `fast_label_propagation_communities`, `asyn_fluidc` | Same communities, in the same order, as NetworkX; the seeded methods replay Python's random number generator, so a given seed gives NetworkX's result. |
| `efficiency`, `global_efficiency`, `local_efficiency`, `closeness_vitality`, `gutman_index`, `schultz_index`, `hyper_wiener_index`, `flow_hierarchy`, `voronoi_cells` | Bit-for-bit identical. |
| `nx.approximation`: `min_weighted_vertex_cover`, `min_weighted_dominating_set`, `min_edge_dominating_set`, `min_maximal_matching`, `greedy_tsp`, `simulated_annealing_tsp`, `threshold_accepting_tsp`, `treewidth_min_fill_in`, `treewidth_decomp`, `diameter`, `one_exchange`, `randomized_partitioning`, `steiner_tree`, `densest_subgraph` | Same results as each NetworkX version, including the random draws for a given seed. Cases that depend on set order run in NetworkX. |
| `complement`, `power`, `difference`, `symmetric_difference`, `is_kl_connected`, `kl_connected_subgraph` | Same graphs (node and adjacency order) as NetworkX. |
| `union`, `union_all`, `compose`, `compose_all`, `disjoint_union`, `disjoint_union_all`, `full_join`, `intersection`, `intersection_all`, `reverse`, `moral_graph`, `line_graph`, `ego_graph` | Same graphs as NetworkX: node and adjacency order, and attribute dicts copied, shared or deep-copied as NetworkX does. 1.4x to 5x faster. Multigraphs, subclasses and views run in NetworkX. |
| `cartesian_product`, `tensor_product`, `strong_product`, `lexicographic_product`, `rooted_product`, `corona_product` | Same graphs, with attribute tuples in NetworkX's key order. 2x to 7x faster. |
| `constraint`, `effective_size`, `local_constraint`, `tree_broadcast_center`, `tree_broadcast_time` | **Bit-for-bit identical**, with float sums in NetworkX's set order. `constraint` and `effective_size` with `nodes=None` use SciPy in NetworkX 3.5+ and run there. |
| `nx.bipartite`: `density`, `degree_centrality`, `projected_graph`, `weighted_projected_graph` | Same results and order as NetworkX. |
| `to_scipy_sparse_array`, `adjacency_matrix`, `laplacian_matrix`, `incidence_matrix`, `to_numpy_array`, `nx.bipartite.biadjacency_matrix` | The same arrays as NetworkX (dtype, format, index arrays and data). 11x to 98x faster (`incidence_matrix` the most). |
| `to_dict_of_lists`, `number_of_selfloops`, `is_weighted`, `is_negatively_weighted`, `get_node_attributes`, `get_edge_attributes`, `relabel_nodes`, `convert_node_labels_to_integers` | Same results as NetworkX. `relabel_nodes(copy=False)` changes the graph in place and runs in NetworkX. |
| Graph builders: `from_dict_of_lists`, `from_dict_of_dicts`, `from_edgelist`, `from_numpy_array`, `from_scipy_sparse_array`, `nx.bipartite.from_biadjacency_matrix` | The same NetworkX graph, 1.2x to 2x faster. |
| Deterministic generators: `empty_graph`, `complete_graph`, `cycle_graph`, `path_graph`, `star_graph`, `wheel_graph`, `ladder_graph`, `circular_ladder_graph`, `lollipop_graph`, `barbell_graph`, `tadpole_graph`, `full_rary_tree`, `balanced_tree`, `binomial_tree`, `complete_bipartite_graph`, `complete_multipartite_graph`, `turan_graph`, `grid_2d_graph`, `grid_graph`, `hypercube_graph`, `hexagonal_lattice_graph`, `triangular_lattice_graph`, `circulant_graph`, `caveman_graph`, `connected_caveman_graph`, `ring_of_cliques`, `windmill_graph`, `sudoku_graph`, `LCF_graph`, `generalized_petersen_graph`, `dorogovtsev_goltsev_mendes_graph`, `mycielski_graph`, `paley_graph`, `kneser_graph` | Built in Rust straight into a NetworkX graph identical to NetworkX's (node and row order, attributes, multigraph keys), with `create_using`. Up to 56x faster (`windmill_graph`); the plainest ones (`complete_graph`, `barbell_graph`, `complete_multipartite_graph`, `turan_graph`), where filling NetworkX's dicts is all the work, are 1.6x to 2.1x faster, and `empty_graph` is slower (0.4x). Periodic hexagonal and triangular lattices run in NetworkX. |
| Random generators: `gnp_random_graph`, `fast_gnp_random_graph`, `gnm_random_graph`, `dense_gnm_random_graph`, `barabasi_albert_graph`, `dual_barabasi_albert_graph`, `extended_barabasi_albert_graph`, `watts_strogatz_graph`, `newman_watts_strogatz_graph`, `connected_watts_strogatz_graph`, `powerlaw_cluster_graph`, `random_regular_graph`, `gn_graph`, `gnr_graph`, `gnc_graph`, `random_uniform_k_out_graph`, `random_lobster_graph` (`random_lobster` before 3.6), `random_tournament`, `stochastic_block_model`, `random_partition_graph`, `planted_partition_graph`, `random_geometric_graph`, `waxman_graph`, `nx.bipartite.random_graph`, `nx.bipartite.gnmk_random_graph` | The same graph for the same `seed`, and `random.Random` left in the same state, by replaying NetworkX's draws on a copy of CPython's generator. 1.3x to 280x faster; `planted_partition_graph` takes about 0.1 ms on the benchmark's input, against NetworkX's 0.06 ms (0.6x). NumPy random generators run in NetworkX. |
| Readers and parsers: `read_edgelist`, `parse_edgelist`, `read_weighted_edgelist`, `read_adjlist`, `parse_adjlist`, `read_multiline_adjlist`, `parse_multiline_adjlist`, `read_leda`, `parse_leda`, `read_pajek`, `parse_pajek`, `read_gml`, `parse_gml`, `read_graph6`, `from_graph6_bytes`, `read_sparse6`, `from_sparse6_bytes`, `node_link_graph`, `adjacency_graph`, `cytoscape_graph`, `tree_graph`, `nx.bipartite.read_edgelist`, `nx.bipartite.parse_edgelist` | Parsed in Rust into the same NetworkX graph (class, node and adjacency order, attribute types, multigraph keys). Files and paths (also `.gz` and `.bz2`) in UTF-8, ASCII or Latin-1; anything else runs in NetworkX. 1.5x to 64x faster. |
| `read_graphml`, `parse_graphml`, `read_gexf` | XML parsed in Rust (quick-xml) and NetworkX's reader followed step by step: the same graph class, node and row order, value types, edge ids and keys. yFiles data, ports, hyperedges, nested GraphML graphs, a DOCTYPE or a non-UTF-8 encoding go to NetworkX, and malformed files raise NetworkX's error. About 7x (GraphML) and 5x (GEXF) faster. |
| `from_pandas_edgelist`, `to_pandas_edgelist`, `from_pandas_adjacency` | `Graph` and `DiGraph` built from, or edge columns gathered out of, NetworkX's dicts in Rust; the same frames and graphs. 1.3x to 3x faster. Need pandas, like NetworkX. |
| Degree-sequence and tree generators: `configuration_model`, `directed_configuration_model`, `expected_degree_graph`, `havel_hakimi_graph`, `directed_havel_hakimi_graph`, `degree_sequence_tree`, `random_degree_sequence_graph`, `random_powerlaw_tree`, `random_powerlaw_tree_sequence`, `random_labeled_tree`, `random_labeled_rooted_tree`, `random_labeled_rooted_forest`, `random_cograph`, `random_clustered_graph`, `joint_degree_graph`, `directed_joint_degree_graph`, `is_valid_directed_joint_degree`, `nx.bipartite.configuration_model`, `nx.bipartite.havel_hakimi_graph`, `reverse_havel_hakimi_graph`, `alternating_havel_hakimi_graph` | The same graph for the same `seed`, with `random.Random` left in the same state (including CPython's set and `set.pop()` order where NetworkX depends on it). 1.6x to 140x faster; `random_degree_sequence_graph` about 270x. |
| Growth, geometric and community generators: `duplication_divergence_graph`, `partial_duplication_graph`, `scale_free_graph`, `random_shell_graph`, `relaxed_caveman_graph`, `gaussian_random_partition_graph`, `navigable_small_world_graph`, `geographical_threshold_graph`, `soft_random_geometric_graph`, `thresholded_random_geometric_graph`, `geometric_soft_configuration_graph`, `geometric_edges`, `uniform_random_intersection_graph`, `k_random_intersection_graph`, `general_random_intersection_graph`, `random_k_lift`, `maybe_regular_expander_graph` (`maybe_regular_expander` before 3.6), `random_internet_as_graph`, `nx.bipartite.preferential_attachment_graph` | The same graph for the same `seed` (NetworkX's `random.Random` draws, NumPy's legacy `RandomState` for the expanders, and Python's set order replayed). 1.2x to 3,600x faster; `relaxed_caveman_graph` takes about 80 µs on the benchmark's input, against NetworkX's 50 µs (0.6x). |
| Generators and transformations: `margulis_gabber_galil_graph`, `chordal_cycle_graph`, `hkn_harary_graph`, `hnm_harary_graph`, `prefix_tree`, `prefix_tree_recursive`, `interval_graph`, `visibility_graph`, `nonisomorphic_trees`, `join_trees`, `mycielskian`, `stochastic_graph`, `contracted_nodes`, `contracted_edge`, `quotient_graph`, `modular_product`, `inverse_line_graph` | Same graphs as NetworkX, with attribute dicts copied or shared as NetworkX does. 2x to 1,000x faster. `quotient_graph` with custom relations or float weights runs in NetworkX. |
| Cliques, communities and structure: `find_cliques`, `find_cliques_recursive`, `make_max_clique_graph`, `make_clique_bipartite`, `k_clique_communities`, `kernighan_lin_bisection`, `dominating_set`, `maximal_independent_set`, `chordal_graph_cliques`, `find_asteroidal_triple`, `nx.tournament.hamiltonian_path`, `build_auxiliary_node_connectivity`, `build_auxiliary_edge_connectivity` | Same results and order as NetworkX (set order replayed; seeded ones replay `random.Random`). `find_cliques` stays lazy. 2x to 850x faster. |
| `nx.approximation`: `max_clique`, `maximum_independent_set`, `clique_removal`, `large_clique_size`, `ramsey_R2`, `treewidth_min_degree`, `metric_closure`, `local_node_connectivity`, `node_connectivity`, `all_pairs_node_connectivity` | Same results as NetworkX. The recursive ones run without recursion, so inputs deep enough to hit Python's recursion limit get an answer where NetworkX raises `RecursionError`. |
| `nx.bipartite`: `clustering` (`latapy_clustering`), `average_clustering`, `robins_alexander_clustering`, `betweenness_centrality`, `generic_weighted_projected_graph`, `overlap_weighted_projected_graph`, `collaboration_weighted_projected_graph`, `min_edge_cover`, `eppstein_matching`, `maximal_extendability`, `modularity` (3.7+); and `cd_index` | Same results as NetworkX. 3x to 235x faster. |

| `nx.nx_pydot.read_dot` | Parsed in Rust with pydot 4.0.1's grammar, then NetworkX's own `from_pydot`; the same graph. Other pydot versions go to pydot. About 340x faster. |
| `louvain_communities`, `louvain_partitions` | Louvain in Rust with Python's own arithmetic and set operations, for both NetworkX variants (3.4 to 3.6, and 3.7's): the same communities for the same `seed`. About 11x faster. |
| Randomized measures: `random_reference`, `lattice_reference`, `sigma`, `omega`, `nx.approximation.average_clustering`, `generate_random_paths`, `panther_similarity`, `panther_vector_similarity`, `random_spanning_tree`, `random_walk` | The same results for the same `seed` (Python's `random.Random` and NumPy's legacy `RandomState` replayed, or called in NetworkX's order). 1.2x to 75x faster. |
| `christofides`, `traveling_salesman_problem`, `asadpour_atsp`, `held_karp_ascent`, `nx.approximation.k_components` | NetworkX's own code with its inner work (spanning trees, matchings, arborescences, connectivity) in rustnx. 8x to 19x faster (`asadpour_atsp` and `held_karp_ascent` about 30x on the inputs NetworkX finishes). |
| `snap_aggregation`, `create_component_structure`, `capacity_scaling`, `spanner`, `k_edge_augmentation` (k >= 3 or `avail`) and its helpers | The same results in the same order (Python's set orders, `id()` tie-breaks and random draws replayed where NetworkX depends on them). 1.3x to 160x faster. |
| `lukes_partitioning` | The same partition, with each set in the same order (the root draw, set orders and Python's `sum()` replayed). 100x to 1,000x faster (650x at 512 nodes). |
| `vf2pp_isomorphism`, `vf2pp_all_isomorphisms` and (NetworkX 3.7) the subgraph and monomorphism mappings | The same mappings in the same order (the matching order and candidate set orders replayed). 10x to 25x faster on 200- to 500-node regular graphs; 3x to 5x on small ones. |
The rest of NetworkX's functions run in NetworkX. [docs/COVERAGE.md](docs/COVERAGE.md)
lists every function NetworkX lets a backend implement and which ones rustnx
does.

NetworkX picks the backend for functions that return a graph (generators,
readers, graph builders and graph operations) from
`nx.config.backend_priority.generators`, and for everything else from
`nx.config.backend_priority.algos`. `rustnx.enable()` sets both. Small
inputs (under about 100 nodes, edges or lines) stay in NetworkX, which is
faster there; NetworkX's own routing still adds a few microseconds to each
such call.

### Turning it on

```python
import rustnx
rustnx.enable()
```

or, without changing code,

```
NETWORKX_BACKEND_PRIORITY_ALGOS=rustnx NETWORKX_BACKEND_PRIORITY_GENERATORS=rustnx python my_script.py
```

`nx.config.backend_priority = ["rustnx"]` and `NETWORKX_BACKEND_PRIORITY=rustnx`
set only the `algos` list, so generators, readers and graph operations keep
running in NetworkX.

### Multigraphs

`MultiGraph` and `MultiDiGraph` run in Rust for the functions whose NetworkX
code sees a multigraph only through its neighbors and, for weights, the
minimum over parallel edges: components, traversals, the shortest path
family, betweenness, closeness and harmonic centrality, the distance measures,
biconnected and attracting components, `is_bipartite`, and most of the
functions added in the coverage batches (`docs/API.md` lists which). Other functions run in
NetworkX. `pagerank` runs multigraphs in Rust with fast floats (the
default), and in NetworkX with exact floats (next section).

### Fast and exact floats

> **Read this if you compare rustnx's numbers with NetworkX's.** Five
> functions, `betweenness_centrality`, `edge_betweenness_centrality`,
> `pagerank`, and `clustering` and `average_clustering` with weights, use
> **fast floats** by default. Their results can differ from
> NetworkX's in the last bits of each float, around 1e-16 to 1e-15
> relative. Everything else about them is identical: the same keys in the
> same order, the same errors. Every other rustnx function always returns
> exactly what NetworkX returns, floats included. For NetworkX's exact
> numbers, turn on **exact floats** (below).

Why they differ: these functions add many floats together. Float addition
is not associative: `(a + b) + c` can differ from `a + (b + c)` in the last
bit. Fast floats add things up in whatever order is fastest (parallel
blocks, or Rust instead of SciPy), while exact floats add them up in
NetworkX's order, or hand identical arrays to NetworkX's own SciPy code.
Neither result is more accurate than the other.

When to turn exact floats on:

- you compare results with `==`, hash them, or test them against stored
  NetworkX output;
- you sort or rank by score and need ties broken exactly as NetworkX would
  (two scores that tie in NetworkX might not tie in fast mode, and vice
  versa);
- you need results reproducible against a pure-NetworkX run.

Fast floats are themselves deterministic: the same input gives the same
result on any machine and any thread count.

| Function | Fast floats (default) | Exact floats | Fast vs exact (4 cores) |
|---|---|---|---|
| `betweenness_centrality` | Parallel blocks of sources, block sums added at the end | Parallel, each source's share added in NetworkX's order | 0.52 s vs 0.56 s (5,000 nodes; NetworkX takes 164 s) |
| `edge_betweenness_centrality` | As above | As above | 0.57 s vs 0.64 s (5,000 nodes; NetworkX takes 181 s) |
| `pagerank` | Power iteration in Rust; multigraphs too | Rust builds the sparse matrix, NetworkX's own SciPy code does the arithmetic; multigraphs run in NetworkX | 0.028 s vs 0.132 s (200,000 nodes; NetworkX takes 1.5 s) |
| `clustering`, `average_clustering` (weighted) | Cube roots summed in Rust, in parallel | Runs in NetworkX (unweighted is exact in Rust either way) | 0.046 s vs NetworkX's 3.1 s (20,000 nodes, 200,000 edges) |

#### Settings

There is one global setting and one per function. **A per-function
setting always wins over the global one.**

```python
import networkx as nx
import rustnx

cfg = nx.config.backends.rustnx

cfg.exact_floats = True                              # every function: exact
cfg.exact_floats_overrides["pagerank"] = False       # ...except pagerank: fast

cfg.exact_floats = False                             # every function: fast (default)
cfg.exact_floats_overrides = {"betweenness_centrality": True}   # ...except this one

with nx.config.backends.rustnx(exact_floats=True):  # just for a block
    nx.pagerank(G)

rustnx.enable(exact_floats=True, exact_floats_overrides={"pagerank": False})
```

From the environment, before Python starts:

```
RUSTNX_EXACT_FLOATS=1                                         # global: exact
RUSTNX_EXACT_FLOATS_OVERRIDES=pagerank=0,betweenness_centrality=1   # per function
```

`rustnx.enable(exact_floats_overrides=...)` raises `ValueError` for a name
without a float setting, so typos don't go unnoticed.

#### Checking which setting applies

```python
>>> print(rustnx.explain_floats())
rustnx float settings (every other function is always exact):
  betweenness_centrality: exact; set by exact_floats_overrides['betweenness_centrality']
  edge_betweenness_centrality: fast (may differ by about 1e-15 relative); set by the global exact_floats setting
  clustering: fast (may differ by about 1e-15 relative, weighted only); set by the global exact_floats setting
  average_clustering: fast (may differ by about 1e-15 relative, weighted only); set by the global exact_floats setting
  pagerank: fast (may differ by about 1e-16 relative); set by the global exact_floats setting
```

`rustnx.float_settings()` returns the same as a dict, plus any override
names it doesn't recognize (`"unknown_overrides"`).

To see it as your code runs, set `cfg.verbose = True` (or
`RUSTNX_VERBOSE=1`). The first call of each of these functions under each
setting then logs, at INFO on the `rustnx` logger (printed to stderr if you
haven't configured logging):

```
rustnx pagerank: fast floats (set by the global exact_floats setting): power iteration in Rust; multigraphs run in Rust too. Floats may differ from NetworkX's in the last bits (about 1e-16 relative). For NetworkX's exact numbers set nx.config.backends.rustnx.exact_floats_overrides['pagerank'] = True, or nx.config.backends.rustnx.exact_floats = True for every function.
```

Every call is also logged at DEBUG, so `logging.getLogger("rustnx").setLevel
(logging.DEBUG)` shows all of them.

With exact floats on, `pagerank` on a multigraph runs in NetworkX. Called
with `backend="rustnx"`, it raises `NotImplementedError`, and the error's
cause says which setting to change.

## Benchmarks

### Real-world pipelines

`python benchmarks/real_world.py` runs seven analysis scripts on
[SNAP](https://snap.stanford.edu/data/) datasets, each step once, as a
script would. rustnx's time includes converting the graph on its first step.
Every step's result was checked against NetworkX's (identical, or within
1e-9 for the fast-float functions). Per-step times are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| Pipeline | Nodes / edges | Steps | NetworkX | rustnx | Speedup |
|---|---|---|---|---|---|
| Social network (Facebook) | 4,039 / 88,234 | 13 | 163 s | 0.92 s | **176×** |
| Email network (Enron) | 36,692 / 183,831 | 9 | 149 s | 1.57 s | **95×** |
| Collaboration network (astrophysics) | 17,903 / 196,972 | 7 | 629 s | 1.27 s | **495×** |
| Trust network (Epinions) | 75,879 / 508,837 | 8 | 10.2 s | 1.33 s | 7.7× |
| Citation network (HEP-PH) | 34,546 / 421,534 | 6 | 7.15 s | 0.82 s | 8.8× |
| Road routing (Pennsylvania) | 1,088,092 / 1,541,898 | 6 | 92.7 s | 14.0 s | 6.6× |
| Web graph (Google) | 875,713 / 5,105,039 | 4 | 18.5 s | 10.4 s | 1.8× |

The analyses heavy on all-pairs work (betweenness, closeness, distances,
clustering) gain 100–500×. The directed and road pipelines are mostly
linear-time steps on big graphs, where converting the graph is a large part
of rustnx's time: the first step on the 5M-edge web graph spends about 3 s
converting, `weakly_connected_components` there reads the predecessor
order from `G._pred` (so its sets iterate exactly as NetworkX's) and gains
nothing, and `minimum_spanning_tree` on the 1.1M-node road network (2.9×)
spends most of its time allocating the result graph's 3 million Python
dicts.

### Every function

`python benchmarks/all_functions.py` times all 668 functions rustnx
implements, each on an input grown until NetworkX takes about half a second
(up to 262,144 nodes). 666 were timed (NetworkX can't finish `asadpour_atsp`
or `held_karp_ascent` at any size tried), and all 666 returned NetworkX's
results: 661 identical, 5 within 1e-9. On a repeat call the median function
is **10×** faster: 334 functions are over 10× faster, 72 over 100×, and 13
are slower, all of them queries NetworkX answers in microseconds or
functions whose time goes into building Python results (see "When rustnx
helps" above). The full table, grouped by
NetworkX module, is in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### Headline functions

`python benchmarks/bench.py` on a 4-core machine (NetworkX 3.7, Python 3.12):

| Function | Graph | NetworkX | rustnx | Speedup |
|---|---|---|---|---|
| `betweenness_centrality` | 4,000 / 16k | 65.5 s | 0.40 s | **165×** |
| `betweenness_centrality` (weighted) | 4,000 / 16k | 147.9 s | 1.17 s | **127×** |
| `edge_betweenness_centrality` | 4,000 / 16k | 78.4 s | 0.37 s | **214×** |
| `edge_betweenness_centrality` (weighted) | 4,000 / 16k | 161.5 s | 1.26 s | **128×** |
| `closeness_centrality` | 4,000 / 16k | 6.77 s | 0.0086 s | **787×** |
| `single_source_dijkstra_path_length` | 160,000 / 319k | 0.36 s | 0.052 s | 6.9× |
| `single_source_shortest_path_length` | 160,000 / 319k | 0.078 s | 0.019 s | 4.0× |
| `connected_components` | 200,000 / 300k | 0.21 s | 0.035 s | 6.0× |
| `pagerank` | 200,000 / 1M (directed) | 1.26 s | 0.027 s | **47×** |
| `pagerank` (weighted) | 200,000 / 1M (directed) | 1.45 s | 0.025 s | **57×** |
| `strongly_connected_components` | 200,000 / 1M (directed) | 0.56 s | 0.044 s | 13× |
| `topological_sort` | 200,000 / 500k (DAG) | 0.32 s | 0.022 s | 15× |
| `weakly_connected_components` | 200,000 / 1M (directed) | 0.38 s | 0.032 s | 12× |
| `diameter` | 5,000 / 15k | 11.1 s | 0.015 s | **715×** |
| `average_shortest_path_length` | 5,000 / 15k | 11.8 s | 0.015 s | **801×** |
| `wiener_index` | 5,000 / 15k | 11.7 s | 0.017 s | **707×** |
| `eccentricity` (weighted) | 5,000 / 15k | 63.2 s | 1.66 s | **38×** |
| `all_pairs_dijkstra_path_length` | 5,000 / 15k | 64.2 s | 7.28 s | 8.8× |
| `all_pairs_shortest_path_length` | 5,000 / 15k | 11.7 s | 4.72 s | 2.5× (building 25M Python dict entries dominates) |
| `dijkstra_path` (50 pairs) | 200,000 / 1M | 101.1 s | 3.21 s | **31×** |
| `single_source_dijkstra_path` | 200,000 / 1M | 3.66 s | 0.39 s | 9.3× |
| `single_source_shortest_path` | 200,000 / 1M | 0.42 s | 0.16 s | 2.6× (building the path lists dominates) |
| `clustering` | 100,000 / 500k (Barabási–Albert) | 5.02 s | 0.033 s | **153×** |
| `transitivity` | 100,000 / 500k (Barabási–Albert) | 5.23 s | 0.036 s | **146×** |
| `triangles` | 100,000 / 500k (Barabási–Albert) | 1.03 s | 0.041 s | 25× |
| `katz_centrality` | 200,000 / 1M (directed) | 14.7 s | 0.071 s | **206×** |
| `core_number` | 200,000 / 1M | 1.61 s | 0.037 s | **44×** |
| `eigenvector_centrality` | 200,000 / 1M | 8.12 s | 0.18 s | **45×** |
| `harmonic_centrality` | 3,000 / 12k | 5.07 s | 0.10 s | **51×** |
| `is_bipartite` | 199,809 / 399k (grid) | 0.36 s | 0.0018 s | **200×** |
| `bfs_edges` | 200,000 / 1M | 0.45 s | 0.12 s | 3.6× |
| `dfs_edges` | 200,000 / 1M | 1.74 s | 0.13 s | 13× |
| `bidirectional_dijkstra` (20 pairs) | 200,000 / 1M | 0.22 s | 0.024 s | 9.1× |
| `k_core` | 200,000 / 1M | 7.21 s | 6.30 s | 1.1× (copying the subgraph in NetworkX dominates) |
| `label_propagation_communities` | 100,000 / 500k (Barabási–Albert) | 3.02 s | 0.063 s | **48×** |
| `greedy_color` | 100,000 / 500k (Barabási–Albert) | 0.25 s | 0.013 s | 18× |
| `minimum_spanning_tree` | 200,000 / 1M | 6.16 s | 0.55 s | 11× (building the result graph dominates) |
| `bfs_tree` | 200,000 / 1M | 2.04 s | 0.48 s | 4.2× (building the tree in NetworkX dominates) |
| `all_shortest_paths` | 3,600 / 7k (grid, 2.7M paths) | 9.76 s | 9.64 s | 1.0× (building the path lists dominates) |

The rustnx column is a repeat call. The first call on a graph also converts
it to rustnx's format (about 0.2–0.3 s for a 1M-edge directed graph), and
NetworkX caches that conversion on the graph. For the heavy algorithms
conversion is negligible. For the linear-time ones, the first call is still
faster than NetworkX, but by less.

### Compared with other Rust backends

`python benchmarks/compare.py` runs the same calls through each installed
backend and checks every result against NetworkX (same machine, NetworkX
3.7, rustworkx 0.18.1 via nx-rustworkx 0.2.1, franken-networkx 0.2.3).
Times are repeat calls; the first call also includes conversion.

| Function (graph) | rustnx | nx-rustworkx | FrankenNetworkX |
|---|---|---|---|
| `betweenness_centrality` (2k nodes) | 0.116 s | 0.158 s | **0.095 s** |
| `closeness_centrality` (2k nodes) | **0.0035 s** | 0.152 s | fails on NetworkX 3.7 (`TypeError`) |
| `pagerank` (100k nodes, 500k edges) | **0.0185 s** | 0.203 s, *differs from NetworkX* | 0.027 s |
| `single_source_dijkstra_path_length` (90k nodes) | **0.030 s** | 0.119 s | 0.075 s |
| `connected_components` (200k nodes) | **0.047 s** | 0.62 s | 0.081 s |
| `strongly_connected_components` (100k nodes) | **0.020 s** | 0.048 s, *different order* | 0.052 s |
| `topological_sort` (100k nodes) | **0.0105 s** | 0.024 s, *different order* | 0.151 s |

rustnx is the fastest on first calls for all seven, because converting the
graph takes it several times less than the others (for `pagerank`, 0.18 s
against about 1 s for both). FrankenNetworkX is the fastest on repeat calls
of `betweenness_centrality`. rustnx matches NetworkX's output, including
order, for all seven.

On small graphs (under 500 nodes), the linear-time functions stay in
NetworkX automatically, because the dispatch overhead outweighs the work.
The centrality functions are faster in Rust at every size.

## Correctness

Results must match NetworkX, or the speed is worthless. Three layers
check that:

1. **`tests/`**: about 25,500 tests comparing rustnx with NetworkX, run on
   directed and undirected graphs, int, float and missing weights,
   self-loops, and shuffled or non-integer node labels. They check values,
   dict ordering and error messages.
2. **NetworkX's own test suite**, with every supported call routed through
   rustnx:

   ```bash
   NETWORKX_TEST_BACKEND=rustnx NETWORKX_FALLBACK_TO_NX=True \
       pytest --pyargs networkx
   ```

   Result: no failures from rustnx on NetworkX 3.4.2, 3.5 and 3.7. On 3.7,
   9,173 tests passed, with 417,724 calls to 642 functions handled by
   rustnx. (NetworkX 3.4's pydot tests fail with pydot 4 with or without
   rustnx.) CI runs both layers on NetworkX 3.4, 3.5 and the latest release.
3. **`benchmarks/all_functions.py`** also checks every function's result
   against NetworkX's on large inputs (see [Benchmarks](#benchmarks)).

rustnx checks every call against the installed NetworkX's own signature.
If a newer NetworkX adds a parameter, rustnx ignores it while it is left at
its default. If the caller actually uses it, rustnx hands the call back to
NetworkX.

Floats match bit for bit too, except in the five functions that use fast
floats by default (see [Fast and exact floats](#fast-and-exact-floats));
turn on exact floats and they match too. Both modes give the same result on
any machine and thread count.

## How it works

```
nx.betweenness_centrality(G)
        │  NetworkX dispatch (backend_priority = ["rustnx"])
        ▼
rustnx.interface      convert G once: nodes → 0..n-1, adjacency → CSR arrays
        │             (NetworkX caches this on G for later calls)
        ▼
rustnx._core (Rust)   algorithm on flat arrays, GIL released, parallel via rayon
        │
        ▼
dict keyed by your original nodes, in NetworkX's order
```

Neighbor order in the Rust arrays follows NetworkX's adjacency dicts exactly.
That's why traversal order, tie-breaking and result ordering all match.

## Releasing

The `Wheels` workflow builds packages for Linux (x86-64, ARM), macOS (Intel,
Apple Silicon) and Windows, plus a source package. It then installs each
wheel on that platform and runs the test suite against it. The workflow runs
on every pull request.

To publish a release:

1. One-time setup: on PyPI, add a
   [trusted publisher](https://docs.pypi.org/trusted-publishers/) for
   `fabuseless/rustnx`, workflow `wheels.yml`, environment `pypi`. Then create
   an environment named `pypi` in the repo's GitHub settings. No API token is
   needed.
2. Set the version in `pyproject.toml` and `Cargo.toml`, and move the
   changelog's *Unreleased* entries under the new version.
3. Push a tag such as `v0.1.0a1`. If every build and test passes, the
   workflow uploads the packages to PyPI.

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install maturin networkx pytest numpy scipy
maturin develop --release
pytest                                  # rustnx comparison tests
python benchmarks/bench.py --quick      # benchmarks
```

Layout:

- `src/`: Rust core. `graph.rs` handles conversion and CSR storage;
  `algorithms/` holds the traversals and centrality.
- `python/rustnx/`: the NetworkX backend. `interface.py` is the entry point;
  `algorithms.py` holds the NetworkX-compatible wrappers.
- `tests/`: comparisons against NetworkX.
- `benchmarks/`: speed comparisons.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the rules and how to add an
algorithm, and [SECURITY.md](SECURITY.md) for reporting security problems.

## Roadmap

- More algorithms: see the todo table in [CLAUDE.md](CLAUDE.md).
- Multigraph support for the remaining functions (`pagerank`, degree-based ones).

## License

BSD 3-Clause, the same license as NetworkX. See [LICENSE](LICENSE). Release
notes are in [CHANGELOG.md](CHANGELOG.md).

rustnx depends on NetworkX (it is a NetworkX backend) and doesn't bundle it.
Many of its algorithms are ported step by step from NetworkX, so that it
returns exactly what NetworkX returns, and a few short expressions are taken
from NetworkX's source. `LICENSE` includes NetworkX's copyright notice and
license for those portions.
