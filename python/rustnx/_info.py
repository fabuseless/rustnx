"""Backend metadata for ``networkx.backend_info``.

This module must stay importable without importing NetworkX, since NetworkX
reads it while it is still being imported.
"""

FUNCTIONS = {
    "all_pairs_dijkstra_path_length": {},
    "all_pairs_shortest_path_length": {},
    "average_shortest_path_length": {},
    "betweenness_centrality": {},
    "center": {},
    "closeness_centrality": {},
    "connected_components": {},
    "diameter": {},
    "eccentricity": {},
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
    "single_source_dijkstra_path_length": {},
    "single_source_shortest_path_length": {},
    "strongly_connected_components": {},
    "topological_generations": {},
    "topological_sort": {},
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
