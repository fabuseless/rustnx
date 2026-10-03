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
