"""Backend metadata for ``networkx.backend_info``.

This module must stay importable without importing NetworkX, since NetworkX
reads it while it is still being imported.
"""

FUNCTIONS = {
    "all_pairs_dijkstra": {},
    "all_pairs_dijkstra_path": {},
    "all_pairs_dijkstra_path_length": {},
    "all_pairs_shortest_path": {},
    "all_pairs_shortest_path_length": {},
    "ancestors": {},
    "average_clustering": {},
    "average_shortest_path_length": {},
    "betweenness_centrality": {},
    "bidirectional_shortest_path": {},
    "center": {},
    "closeness_centrality": {},
    "clustering": {},
    "connected_components": {},
    "descendants": {},
    "diameter": {},
    "dijkstra_path": {},
    "dijkstra_path_length": {},
    "eccentricity": {},
    "has_path": {},
    "is_connected": {},
    "is_directed_acyclic_graph": {},
    "is_strongly_connected": {},
    "is_weakly_connected": {},
    "number_connected_components": {},
    "number_strongly_connected_components": {},
    "number_weakly_connected_components": {},
    "pagerank": {},
    "periphery": {},
    "radius": {},
    "shortest_path": {},
    "shortest_path_length": {},
    "single_source_dijkstra": {},
    "single_source_dijkstra_path": {},
    "single_source_dijkstra_path_length": {},
    "single_source_shortest_path": {},
    "single_source_shortest_path_length": {},
    "single_target_shortest_path": {},
    "strongly_connected_components": {},
    "topological_generations": {},
    "topological_sort": {},
    "transitivity": {},
    "triangles": {},
    "weakly_connected_components": {},
    "wiener_index": {},
}


def get_info():
    return {
        "backend_name": "rustnx",
        "project": "rustnx",
        "package": "rustnx",
        "short_summary": "Rust-accelerated algorithms, no code changes needed.",
        "functions": FUNCTIONS,
        "default_config": {},
    }
