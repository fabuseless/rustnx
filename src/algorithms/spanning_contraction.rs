//! The matrices `random_spanning_tree` takes determinants of. For each edge
//! NetworkX contracts the accepted edges `U` (merging classes in the order
//! it iterates the Python set `U`, which decides each class's
//! representative and so the order of the Laplacian's rows), drops the
//! rejected edges, and takes the reduced Laplacian of the result, and of
//! the result with the edge itself contracted. The caller hands these to
//! the same `np.linalg.det`.

use super::pyset::PySet;

pub struct Contraction {
    n: usize,
    ei: Vec<u32>,
    ej: Vec<u32>,
    w: Vec<i64>,
    /// `hash((u, v))` of each edge, for `U`'s order.
    hashes: Vec<i64>,
    alive: Vec<bool>,
    accepted: PySet,
}

/// A reduced Laplacian (`L[1:, 1:]`, row-major) of size `k - 1`, or None
/// where the contracted graph is disconnected (NetworkX's 0).
pub type Reduced = Option<(usize, Vec<f64>)>;

const NONE: u32 = u32::MAX;

/// `find_node(merged_nodes, x)` with its path compression.
fn find(parent: &mut [u32], x: u32) -> u32 {
    let mut root = x;
    while parent[root as usize] != NONE {
        root = parent[root as usize];
    }
    let mut v = x;
    while parent[v as usize] != NONE {
        let next = parent[v as usize];
        parent[v as usize] = root;
        v = next;
    }
    // The root itself isn't a key; the nodes on the path now point at it.
    root
}

impl Contraction {
    pub fn new(n: usize, ei: Vec<u32>, ej: Vec<u32>, w: Vec<i64>, hashes: Vec<i64>) -> Self {
        let m = ei.len();
        Contraction {
            n,
            ei,
            ej,
            w,
            hashes,
            alive: vec![true; m],
            accepted: PySet::default(),
        }
    }

    pub fn edges(&self) -> usize {
        self.ei.len()
    }

    pub fn accept(&mut self, e: u32) {
        self.accepted.add(e, &self.hashes);
    }

    pub fn reject(&mut self, e: u32) {
        self.alive[e as usize] = false;
    }

    /// The reduced Laplacian of the contracted graph, and (when `e`'s ends
    /// are in different classes joined by a live edge) of that graph with
    /// `e` contracted.
    pub fn step(&self, e: u32) -> (Reduced, Option<Reduced>) {
        let n = self.n;
        let mut parent = vec![NONE; n];
        for f in self.accepted.iter() {
            let (x, y) = (self.ei[f as usize], self.ej[f as usize]);
            let xr = find(&mut parent, x);
            let yr = find(&mut parent, y);
            if xr != yr {
                parent[yr as usize] = xr;
            }
        }
        let cls: Vec<u32> = (0..n as u32).map(|v| find(&mut parent, v)).collect();
        let keep: Vec<u32> = (0..n as u32).filter(|&v| cls[v as usize] == v).collect();
        let total = self.reduced(&cls, &keep);
        let (u, v) = (self.ei[e as usize], self.ej[e as usize]);
        let (ru, rv) = (cls[u as usize], cls[v as usize]);
        let joined = ru != rv
            && (0..self.ei.len()).any(|f| {
                self.alive[f] && {
                    let (a, b) = (cls[self.ei[f] as usize], cls[self.ej[f] as usize]);
                    (a == ru && b == rv) || (a == rv && b == ru)
                }
            });
        if !joined {
            return (total, None);
        }
        let cls_e: Vec<u32> = cls.iter().map(|&c| if c == rv { ru } else { c }).collect();
        let keep_e: Vec<u32> = keep.iter().copied().filter(|&c| c != rv).collect();
        (total, Some(self.reduced(&cls_e, &keep_e)))
    }

    /// `total_weight(cls, alive, keep)`'s matrix.
    fn reduced(&self, cls: &[u32], keep: &[u32]) -> Reduced {
        let k = keep.len();
        let mut pos = vec![u32::MAX; self.n];
        for (i, &c) in keep.iter().enumerate() {
            pos[c as usize] = i as u32;
        }
        let mut edges = Vec::new();
        for f in 0..self.ei.len() {
            if !self.alive[f] {
                continue;
            }
            let a = pos[cls[self.ei[f] as usize] as usize];
            let b = pos[cls[self.ej[f] as usize] as usize];
            if a != b {
                edges.push((a as usize, b as usize, self.w[f]));
            }
        }
        if k > 1 {
            let mut root: Vec<usize> = (0..k).collect();
            fn top(root: &mut [usize], mut x: usize) -> usize {
                while root[x] != x {
                    root[x] = root[root[x]];
                    x = root[x];
                }
                x
            }
            let mut parts = k;
            for &(a, b, _) in &edges {
                let (ra, rb) = (top(&mut root, a), top(&mut root, b));
                if ra != rb {
                    root[rb] = ra;
                    parts -= 1;
                }
            }
            if parts != 1 {
                return None;
            }
        }
        // Integer entries, so the order of the additions doesn't matter.
        let mut lap = vec![0i64; k * k];
        for &(a, b, w) in &edges {
            lap[a * k + b] -= w;
            lap[b * k + a] -= w;
            lap[a * k + a] += w;
            lap[b * k + b] += w;
        }
        let r = k.saturating_sub(1);
        let mut out = Vec::with_capacity(r * r);
        for i in 1..k {
            for j in 1..k {
                out.push(lap[i * k + j] as f64);
            }
        }
        Some((r, out))
    }
}
