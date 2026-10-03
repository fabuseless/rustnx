//! Directed-graph algorithms: strong/weak components and topological order.

use crate::graph::Csr;

/// Strongly connected components in NetworkX's output order.
///
/// Port of the iterative Tarjan variant in NetworkX 3.7
/// (`strongly_connected_components`). With `early_exit` it stops, as 3.7
/// does, once the whole graph turns out to be one component; that only
/// changes the order of that component's nodes (the order NetworkX fills
/// its set in). NetworkX 3.4 to 3.6 have no early exit.
pub fn strongly_connected_components(succ: &Csr, n: usize, early_exit: bool) -> Vec<Vec<u32>> {
    const UNSET: u32 = 0;
    let mut comps = Vec::new();
    if n == 0 {
        return comps;
    }
    let mut lowlink = vec![UNSET; n];
    let mut found = vec![false; n];
    // Parallel DFS stacks: node, next edge position, "is SCC root" flag.
    let mut dfs_v: Vec<usize> = Vec::new();
    let mut dfs_pos: Vec<usize> = Vec::new();
    let mut dfs_lead: Vec<bool> = Vec::new();
    let mut comp_stack: Vec<usize> = Vec::new();
    let mut index: u32 = 0;

    for source in 0..n {
        if lowlink[source] != UNSET {
            continue;
        }
        index += 1;
        let root_low = index;
        lowlink[source] = index;
        dfs_v.push(source);
        dfs_pos.push(succ.offsets[source]);
        dfs_lead.push(true);

        while let Some(&v) = dfs_v.last() {
            let end = succ.offsets[v + 1];
            let top = dfs_v.len() - 1;
            let mut descended = false;
            while dfs_pos[top] < end {
                let w = succ.targets[dfs_pos[top]] as usize;
                dfs_pos[top] += 1;
                if lowlink[w] == UNSET {
                    // Tree arc: previsit w and descend.
                    index += 1;
                    lowlink[w] = index;
                    dfs_v.push(w);
                    dfs_pos.push(succ.offsets[w]);
                    dfs_lead.push(true);
                    descended = true;
                    break;
                }
                // Back/cross arc.
                if !found[w] && lowlink[v] > lowlink[w] {
                    dfs_lead[top] = false;
                    lowlink[v] = lowlink[w];
                    if early_exit && lowlink[v] == root_low && index as usize == n {
                        // Every node is preordered and v links to the root:
                        // the whole graph is one SCC.
                        let mut scc: Vec<u32> = comp_stack.iter().map(|&u| u as u32).collect();
                        scc.extend(dfs_v.iter().map(|&u| u as u32));
                        comps.push(scc);
                        return comps;
                    }
                }
            }
            if descended {
                continue;
            }
            // All successors processed: postvisit v.
            dfs_v.pop();
            dfs_pos.pop();
            if dfs_lead.pop().unwrap_or(false) {
                let v_low = lowlink[v];
                let mut scc = vec![v as u32];
                while let Some(&u) = comp_stack.last() {
                    if lowlink[u] < v_low {
                        break;
                    }
                    scc.push(u as u32);
                    comp_stack.pop();
                }
                for &u in &scc {
                    found[u as usize] = true;
                }
                comps.push(scc);
            } else {
                comp_stack.push(v);
                let parent = *dfs_v.last().expect("non-root has a parent");
                if lowlink[parent] > lowlink[v] {
                    *dfs_lead.last_mut().expect("parent frame") = false;
                    lowlink[parent] = lowlink[v];
                }
            }
        }
    }
    comps
}

/// Weakly connected components (edge direction ignored), in NetworkX order.
pub fn weakly_connected_components(succ: &Csr, pred: &Csr, n: usize) -> Vec<Vec<u32>> {
    let mut seen = vec![false; n];
    let mut comps = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut comp = vec![start as u32];
        let mut i = 0;
        while i < comp.len() {
            let v = comp[i] as usize;
            for &w in succ.neighbors(v).iter().chain(pred.neighbors(v)) {
                if !seen[w as usize] {
                    seen[w as usize] = true;
                    comp.push(w);
                }
            }
            i += 1;
        }
        comps.push(comp);
    }
    comps
}

/// `nx.topological_generations`: the generations produced before any cycle
/// blocks progress, and whether a cycle was found.
pub fn topological_generations(succ: &Csr, pred: &Csr, n: usize) -> (Vec<Vec<u32>>, bool) {
    let mut indegree: Vec<usize> = (0..n).map(|v| pred.neighbors(v).len()).collect();
    let mut remaining = indegree.iter().filter(|&&d| d > 0).count();
    let mut zero: Vec<u32> = (0..n as u32)
        .filter(|&v| indegree[v as usize] == 0)
        .collect();
    let mut generations = Vec::new();
    while !zero.is_empty() {
        let this_generation = std::mem::take(&mut zero);
        for &node in &this_generation {
            for &child in succ.neighbors(node as usize) {
                let d = &mut indegree[child as usize];
                *d -= 1;
                if *d == 0 {
                    zero.push(child);
                    remaining -= 1;
                }
            }
        }
        generations.push(this_generation);
    }
    (generations, remaining > 0)
}
