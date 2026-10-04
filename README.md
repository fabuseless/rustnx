# rustnx

**NetworkX, but fast.** rustnx is a Rust-powered backend for
[NetworkX](https://networkx.org). You keep writing normal NetworkX code, and
supported algorithms run in Rust instead of Python, often 50–100× faster.

Install with `pip install rustnx` (Python 3.10 or newer, including
free-threaded 3.14t; prebuilt for Linux, macOS and Windows).

```python
import networkx as nx

nx.config.backend_priority = ["rustnx"]   # or: NETWORKX_BACKEND_PRIORITY=rustnx

G = nx.barabasi_albert_graph(4000, 4, seed=1)
nx.betweenness_centrality(G)              # runs in Rust: 0.4s instead of 41s
```

Nothing else changes. Anything rustnx doesn't support, such as other
functions, multigraphs in functions that treat parallel edges specially, or callable weights, keeps running in NetworkX, so
turning it on never breaks working code.

## When rustnx helps, and when it doesn't

A NetworkX graph is a dict of dicts in Python, so rustnx first copies it into
a compact Rust layout. NetworkX caches the copy on the graph, and later calls
reuse it until the graph changes.

| 200,000 nodes / 1M edges | Converting to rustnx | Just walking the graph in pure Python |
|---|---|---|
| Undirected, no weights | 0.22 s | 0.13 s |
| Undirected, with weights | 0.58 s | 0.60 s |
| Directed, with weights | 0.29 s | 0.29 s |

Conversion costs about as much as reading the graph once in Python, which is
the floor for anything that starts from a NetworkX graph. So:

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
  their time building Python objects, so they speed up 2–8× rather than 50×.
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
| Build time | 10.0 s | 2.7 s | 0.48 s |
| Memory | 323 MiB | ~65 MiB | ~47 MiB |
| `pagerank`, first call | 4.1 s | 0.04 s | 0.04 s |

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
  (betweenness 1.8×, closeness 23×, strong components 2.5×). Building graphs
  is slower: the graph is stored in Python, so adding 500k edges takes 0.8 s,
  against rustworkx's 0.04 s.

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
| `triangles`, `clustering`, `average_clustering`, `transitivity` | Unweighted, directed and undirected, with `nodes=`. Parallel. Results are **bit-for-bit identical** to NetworkX. Weighted clustering runs in NetworkX. |
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

The rest of NetworkX's functions run in NetworkX. [docs/COVERAGE.md](docs/COVERAGE.md)
lists every function NetworkX lets a backend implement and which ones rustnx
does.

### Multigraphs

`MultiGraph` and `MultiDiGraph` run in Rust for the functions whose NetworkX
code sees a multigraph only through its neighbors and, for weights, the
minimum over parallel edges: components, traversals, the shortest path
family, betweenness, closeness and harmonic centrality, the distance measures,
biconnected and attracting components, `is_bipartite`, and most of the
functions added in the coverage batches (`docs/API.md` lists which). Other functions (for example `pagerank`, which sums
parallel weights, or degree-based ones) run in NetworkX.

## Benchmarks

`python benchmarks/bench.py` on a 4-core machine (NetworkX 3.6.1):

| Function | Graph | NetworkX | rustnx | Speedup |
|---|---|---|---|---|
| `betweenness_centrality` | 4,000 nodes / 16k edges | 40.9 s | 0.38 s | **107×** |
| `betweenness_centrality` (weighted) | 4,000 / 16k | 115.3 s | 1.24 s | **93×** |
| `edge_betweenness_centrality` | 4,000 / 16k | 42.3 s | 0.46 s | **93×** |
| `edge_betweenness_centrality` (weighted) | 4,000 / 16k | 106.8 s | 1.33 s | **80×** |
| `closeness_centrality` | 4,000 / 16k | 5.45 s | 0.0097 s | **562×** |
| `single_source_dijkstra_path_length` | 160,000 / 319k | 0.38 s | 0.063 s | 6× |
| `single_source_shortest_path_length` | 160,000 / 319k | 0.11 s | 0.020 s | 5× |
| `connected_components` | 200,000 / 300k | 0.23 s | 0.040 s | 6× |
| `pagerank` | 200,000 / 1M (directed) | 5.76 s | 0.024 s | **243×** |
| `pagerank` (weighted) | 200,000 / 1M (directed) | 3.81 s | 0.024 s | **161×** |
| `strongly_connected_components` | 200,000 / 1M (directed) | 1.96 s | 0.052 s | **38×** |
| `topological_sort` | 200,000 / 499k (DAG) | 0.40 s | 0.023 s | 17× |
| `weakly_connected_components` | 200,000 / 1M (directed) | 0.39 s | 0.036 s | 11× |
| `diameter` | 5,000 / 15k | 11.2 s | 0.014 s | **807×** |
| `average_shortest_path_length` | 5,000 / 15k | 10.8 s | 0.014 s | **783×** |
| `wiener_index` | 5,000 / 15k | 10.0 s | 0.012 s | **829×** |
| `eccentricity` (weighted) | 5,000 / 15k | 57.0 s | 1.41 s | 40× |
| `all_pairs_dijkstra_path_length` | 5,000 / 15k | 60.2 s | 4.90 s | 12× |
| `all_pairs_shortest_path_length` | 5,000 / 15k | 9.6 s | 2.56 s | 4× (building 25M Python dict entries dominates) |
| `dijkstra_path` (50 pairs) | 200,000 / 1M | 117 s | 3.3 s | **35×** |
| `single_source_dijkstra_path` | 200,000 / 1M | 4.56 s | 0.83 s | 5.5× |
| `single_source_shortest_path` | 200,000 / 1M | 1.10 s | 0.57 s | 1.9× (building the path lists dominates) |
| `clustering` | 100,000 / 500k (Barabási–Albert) | 5.09 s | 0.092 s | **55×** |
| `transitivity` | 100,000 / 500k (Barabási–Albert) | 5.12 s | 0.088 s | **58×** |
| `triangles` | 100,000 / 500k (Barabási–Albert) | 1.52 s | 0.091 s | 17× |
| `katz_centrality` | 200,000 / 1M (directed) | 13.3 s | 0.073 s | **180×** |
| `core_number` | 200,000 / 1M | 2.88 s | 0.042 s | **69×** |
| `eigenvector_centrality` | 200,000 / 1M | 10.3 s | 0.18 s | **57×** |
| `harmonic_centrality` | 3,000 / 12k | 4.56 s | 0.14 s | **33×** |
| `is_bipartite` | 200,000 / 400k (grid) | 0.33 s | 0.012 s | 27× |
| `bfs_edges` | 200,000 / 1M | 2.46 s | 0.13 s | 20× |
| `dfs_edges` | 200,000 / 1M | 2.49 s | 0.15 s | 17× |
| `bidirectional_dijkstra` (20 pairs) | 200,000 / 1M | 0.24 s | 0.023 s | 11× |
| `k_core` | 200,000 / 1M | 10.2 s | 7.4 s | 1.4× (copying the subgraph in NetworkX dominates) |
| `label_propagation_communities` | 100,000 / 500k (Barabási–Albert) | 3.01 s | 0.069 s | **43×** |
| `greedy_color` | 100,000 / 500k (Barabási–Albert) | 0.28 s | 0.014 s | 21× |
| `minimum_spanning_tree` | 200,000 / 1M | 7.73 s | 1.01 s | 7.7× |
| `bfs_tree` | 200,000 / 1M | 3.67 s | 1.08 s | 3.4× (building the tree in NetworkX dominates) |
| `all_shortest_paths` | 3,600 / 7k (grid, 2.7M paths) | 6.63 s | 2.80 s | 2.4× (building the path lists dominates) |

The rustnx column is a repeat call. The first call on a graph also converts
it to rustnx's format (about 0.15–0.25 s for a 1M-edge directed graph), and
NetworkX caches that conversion on the graph. For the heavy algorithms
conversion is negligible. For the linear-time ones, the first call is still
faster than NetworkX, but by less.

### Compared with other Rust backends

`python benchmarks/compare.py` runs the same calls through each installed
backend and checks every result against NetworkX (same machine, NetworkX
3.6.1, rustworkx 0.18.1 via nx-rustworkx 0.2.1, franken-networkx 0.2.1).
Times are repeat calls; the first call also includes conversion.

| Function (graph) | rustnx | nx-rustworkx | FrankenNetworkX |
|---|---|---|---|
| `betweenness_centrality` (2k nodes) | **0.099 s** | 0.161 s | 8.23 s (no speedup) |
| `closeness_centrality` (2k nodes) | **0.0030 s** | 0.104 s | 0.0031 s |
| `pagerank` (100k nodes, 500k edges) | **0.014 s** | 0.212 s, *differs from NetworkX* | 0.063 s |
| `single_source_dijkstra_path_length` (90k nodes) | **0.025 s** | 0.094 s | 0.123 s |
| `connected_components` (200k nodes) | 0.040 s | 0.072 s | **0.036 s** |
| `strongly_connected_components` (100k nodes) | **0.020 s** | 0.041 s, *different order* | 0.054 s |
| `topological_sort` (100k nodes) | **0.010 s** | 0.019 s, *different order* | 0.040 s |

rustnx is also the fastest on first calls for all seven, because its
conversion is about 5–20× quicker than nx-rustworkx's and 25–55× quicker than
FrankenNetworkX's. It matches NetworkX's output,
including order, for all seven.

On small graphs (under 500 nodes), the linear-time functions stay in
NetworkX automatically, because the dispatch overhead outweighs the work.
The centrality functions are faster in Rust at every size.

## Correctness

Results must match NetworkX, or the speed is worthless. Two test layers
check that:

1. **`tests/`**: about 1,400 randomized comparisons against NetworkX, run on
   directed and undirected graphs, int, float and missing weights,
   self-loops, and shuffled or non-integer node labels. They check values,
   dict ordering and error messages.
2. **NetworkX's own test suite**, with every supported call routed through
   rustnx:

   ```bash
   NETWORKX_TEST_BACKEND=rustnx NETWORKX_FALLBACK_TO_NX=True \
       pytest --pyargs networkx
   ```

   Result: 0 failures on NetworkX 3.4.2, 3.5 and 3.7 (9,066 tests passed on
   3.7), with over 300,000 calls handled by rustnx. CI runs both layers on
   NetworkX 3.4, 3.5 and the latest release.

rustnx checks every call against the installed NetworkX's own signature.
If a newer NetworkX adds a parameter, rustnx ignores it while it is left at
its default. If the caller actually uses it, rustnx hands the call back to
NetworkX.

Betweenness sums per-source contributions in parallel. It matches NetworkX
to about 1e-15 relative error rather than bit-for-bit, and it gives the same
result on any machine regardless of thread count.

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
