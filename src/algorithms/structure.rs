//! Structural measures: core numbers and bipartiteness.

use crate::graph::Csr;

/// Whether any node has an edge to itself.
pub fn has_self_loops(succ: &Csr, n: usize) -> bool {
    (0..n).any(|v| succ.neighbors(v).contains(&(v as u32)))
}

/// `nx.core_number` (Batagelj–Zaversnik). For directed graphs `pred` gives
/// the in-neighbors and neighbors are counted with multiplicity, as
/// `nx.all_neighbors` does. Core numbers are unique, so any correct
/// peeling order gives NetworkX's values. Self-loops must be rejected by the
/// caller.
pub fn core_number(succ: &Csr, pred: Option<&Csr>, n: usize) -> Vec<u32> {
    let nbrs = |v: usize| {
        succ.neighbors(v)
            .iter()
            .chain(pred.map_or(&[][..], |p| p.neighbors(v)))
            .copied()
    };
    let mut core: Vec<u32> = (0..n).map(|v| nbrs(v).count() as u32).collect();
    let max_deg = core.iter().copied().max().unwrap_or(0) as usize;
    // Bucket sort nodes by degree.
    let mut bin = vec![0usize; max_deg + 2];
    for &d in &core {
        bin[d as usize + 1] += 1;
    }
    for d in 0..=max_deg {
        bin[d + 1] += bin[d];
    }
    let mut pos = vec![0usize; n];
    let mut order = vec![0u32; n];
    let mut next = bin.clone();
    for v in 0..n {
        let d = core[v] as usize;
        pos[v] = next[d];
        order[next[d]] = v as u32;
        next[d] += 1;
    }
    // `bin[d]` = first position of degree-d nodes.
    for i in 0..n {
        let v = order[i] as usize;
        for u in nbrs(v) {
            let u = u as usize;
            if core[u] > core[v] {
                let du = core[u] as usize;
                let pu = pos[u];
                let pw = bin[du];
                let w = order[pw] as usize;
                if u != w {
                    order.swap(pu, pw);
                    pos[u] = pw;
                    pos[w] = pu;
                }
                bin[du] += 1;
                core[u] -= 1;
            }
        }
    }
    core
}

/// Whether the graph (ignoring direction) can be 2-colored. A self-loop
/// makes it non-bipartite, as in `nx.bipartite.color`.
pub fn is_bipartite(succ: &Csr, pred: Option<&Csr>, n: usize) -> bool {
    const UNSEEN: u8 = 2;
    let mut color = vec![UNSEEN; n];
    let mut stack = Vec::new();
    for s in 0..n {
        if color[s] != UNSEEN {
            continue;
        }
        color[s] = 1;
        stack.push(s as u32);
        while let Some(v) = stack.pop() {
            let v = v as usize;
            let c = 1 - color[v];
            let rows = succ
                .neighbors(v)
                .iter()
                .chain(pred.map_or(&[][..], |p| p.neighbors(v)));
            for &w in rows {
                let w = w as usize;
                if color[w] == UNSEEN {
                    color[w] = c;
                    stack.push(w as u32);
                } else if color[w] == color[v] {
                    return false;
                }
            }
        }
    }
    true
}

/// `nx.greedy_color` with the default `largest_first` strategy: nodes in
/// processing order (degree descending, ties in node order) and the color of
/// each node position. `degree` is NetworkX's `G.degree` (self-loops count
/// twice); colors avoid those of already-colored successors.
pub fn greedy_color(succ: &Csr, n: usize, degree: &[usize]) -> (Vec<u32>, Vec<u32>) {
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| degree[b as usize].cmp(&degree[a as usize])); // stable
    const NONE: u32 = u32::MAX;
    let mut color = vec![NONE; n];
    let mut used: Vec<u32> = Vec::new(); // mark[c] == stamp means color c is taken
    for (stamp, &u) in order.iter().enumerate() {
        let stamp = stamp as u32 + 1;
        for &v in succ.neighbors(u as usize) {
            let c = color[v as usize];
            if c != NONE {
                let c = c as usize;
                if c >= used.len() {
                    used.resize(c + 1, 0);
                }
                used[c] = stamp;
            }
        }
        let mut c = 0;
        while c < used.len() && used[c] == stamp {
            c += 1;
        }
        color[u as usize] = c as u32;
    }
    (order, color)
}

/// The most frequent labels among `v`'s neighbors, counted with repetition
/// as `Counter(labeling[q] for q in G[v])`, into `best`.
fn most_frequent(
    adj: &Csr,
    v: usize,
    label: &[u32],
    count: &mut [u32],
    touched: &mut Vec<u32>,
    best: &mut Vec<u32>,
) {
    best.clear();
    let mut max = 0;
    for &q in adj.neighbors(v) {
        let l = label[q as usize] as usize;
        if count[l] == 0 {
            touched.push(l as u32);
        }
        count[l] += 1;
        max = max.max(count[l]);
    }
    for &l in touched.iter() {
        if count[l as usize] == max {
            best.push(l);
        }
        count[l as usize] = 0;
    }
    touched.clear();
}

/// `label_propagation_communities`' semi-synchronous updates, visiting nodes
/// in `order` (color classes in NetworkX's iteration order). Returns each
/// node's final label (labels start as node positions).
pub fn label_propagation(adj: &Csr, n: usize, order: &[u32]) -> Vec<u32> {
    let mut label: Vec<u32> = (0..n as u32).collect();
    let mut count = vec![0u32; n];
    let mut touched = Vec::new();
    let mut best = Vec::new();
    loop {
        let mut complete = true;
        for v in 0..n {
            if adj.neighbors(v).is_empty() {
                continue;
            }
            most_frequent(adj, v, &label, &mut count, &mut touched, &mut best);
            if !best.contains(&label[v]) {
                complete = false;
                break;
            }
        }
        if complete {
            return label;
        }
        for &v in order {
            let v = v as usize;
            if adj.neighbors(v).is_empty() {
                continue; // `{labeling[node]}`: nothing changes
            }
            most_frequent(adj, v, &label, &mut count, &mut touched, &mut best);
            if best.len() == 1 {
                label[v] = best[0];
            } else if !best.contains(&label[v]) {
                label[v] = *best.iter().max().expect("non-empty");
            }
        }
    }
}

/// Kruskal's spanning forest as `nx.kruskal_mst_edges` builds it: edges
/// `(u, v, weight)` in `G.edges` order, stably sorted by weight (descending
/// for a maximum tree), kept when they join two components. Returns the
/// indices of the kept edges in yield order.
pub fn kruskal(n: usize, edges: &[(u32, u32, f64)], maximum: bool) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..edges.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        let (wa, wb) = (edges[a as usize].2, edges[b as usize].2);
        let ord = wa.partial_cmp(&wb).expect("no NaN weights");
        if maximum {
            ord.reverse()
        } else {
            ord
        }
    });
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            let p = parent[x as usize];
            parent[x as usize] = parent[p as usize];
            x = p;
        }
        x
    }
    let mut kept = Vec::new();
    for i in idx {
        let (u, v, _) = edges[i as usize];
        let (ru, rv) = (find(&mut parent, u), find(&mut parent, v));
        if ru != rv {
            parent[ru as usize] = rv;
            kept.push(i);
            if kept.len() + 1 == n {
                break;
            }
        }
    }
    kept
}

/// What `_biconnected_dfs` produces: articulation points (in yield order,
/// repeats removed as `articulation_points` does) or the edge lists of the
/// biconnected components (concatenated, with each component's end).
pub enum Biconnected {
    Articulation(Vec<u32>),
    Components(Vec<(u32, u32)>, Vec<u32>),
}

/// Port of NetworkX's `_biconnected_dfs`, in its exact visit order.
pub fn biconnected(adj: &Csr, n: usize, components: bool) -> Biconnected {
    const UNSEEN: u32 = u32::MAX;
    let mut discovery = vec![UNSEEN; n];
    let mut low = vec![0u32; n];
    // Position in `edge_stack` of each tree edge, by child (a tree edge
    // (grandparent, parent) is looked up by `parent`).
    let mut tree_edge = vec![0usize; n];
    let mut seen_articulation = vec![false; n];
    let mut articulation = Vec::new();
    let mut out_edges = Vec::new();
    let mut ends = Vec::new();
    let mut edge_stack: Vec<(u32, u32)> = Vec::new();
    // Stack of (grandparent, parent, next neighbor position).
    let mut stack: Vec<(u32, u32, usize)> = Vec::new();
    let mut emit = |v: u32, seen: &mut Vec<bool>| {
        if !seen[v as usize] {
            seen[v as usize] = true;
            articulation.push(v);
        }
    };
    for start in 0..n {
        if discovery[start] != UNSEEN {
            continue;
        }
        let mut count = 1u32; // len(discovery) within this search
        discovery[start] = 0;
        low[start] = 0;
        let mut root_children = 0;
        edge_stack.clear();
        let s = start as u32;
        stack.push((s, s, adj.range(start).start));
        while let Some(&mut (grandparent, parent, ref mut pos)) = stack.last_mut() {
            let p = parent as usize;
            if *pos < adj.range(p).end {
                let child = adj.targets[*pos];
                *pos += 1;
                if grandparent == child {
                    continue;
                }
                let c = child as usize;
                if discovery[c] != UNSEEN {
                    if discovery[c] <= discovery[p] {
                        // back edge
                        low[p] = low[p].min(discovery[c]);
                        if components {
                            edge_stack.push((parent, child));
                        }
                    }
                } else {
                    discovery[c] = count;
                    low[c] = count;
                    count += 1;
                    stack.push((parent, child, adj.range(c).start));
                    if components {
                        tree_edge[c] = edge_stack.len();
                        edge_stack.push((parent, child));
                    }
                }
                continue;
            }
            stack.pop();
            let g = grandparent as usize;
            if stack.len() > 1 {
                if low[p] >= discovery[g] {
                    if components {
                        let ind = tree_edge[p];
                        out_edges.extend(edge_stack.drain(ind..));
                        ends.push(out_edges.len() as u32);
                    } else {
                        emit(grandparent, &mut seen_articulation);
                    }
                }
                low[g] = low[p].min(low[g]);
            } else if !stack.is_empty() {
                root_children += 1;
                if components {
                    let ind = tree_edge[p];
                    out_edges.extend(edge_stack.drain(ind..));
                    ends.push(out_edges.len() as u32);
                }
            }
        }
        if !components && root_children > 1 {
            emit(s, &mut seen_articulation);
        }
    }
    if components {
        Biconnected::Components(out_edges, ends)
    } else {
        Biconnected::Articulation(articulation)
    }
}
