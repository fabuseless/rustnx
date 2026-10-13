//! NetworkX's `spanner` (Baswana-Sen), replayed: edge weights are
//! NetworkX's tuples `(weight, id(u), id(v))` (or `(id(u), id(v))`), the
//! random draws come from Python's Mersenne Twister in NetworkX's order
//! (one per center of `set(clustering.values())`), and the sets whose
//! order shows (the centers, the edges to add) follow CPython's set order.
//! The result is the sequence of `_add_edge_to_spanner` calls.

use std::cmp::Ordering;
use std::collections::HashMap;

use super::operators::tuple_hash;
use super::pyrandom::Mt19937;
use super::pyset::PySet;

/// An edge's weight tuple; `w` is 0 for every edge when unweighted.
#[derive(Clone, Copy)]
pub struct WeightKey {
    pub w: f64,
    pub a: u64,
    pub b: u64,
}

impl WeightKey {
    fn lt(&self, o: &WeightKey) -> bool {
        match self.w.partial_cmp(&o.w) {
            Some(Ordering::Less) => true,
            Some(Ordering::Greater) => false,
            _ => (self.a, self.b) < (o.a, o.b),
        }
    }
}

pub struct Input<'a> {
    /// Each node's neighbors (node, edge) in adjacency order.
    pub adj: &'a [Vec<(u32, u32)>],
    pub keys: &'a [WeightKey],
    pub hashes: &'a [i64],
    pub k: u64,
    pub sample_prob: f64,
    pub size_limit: f64,
}

/// A dict from node to center, in insertion order.
#[derive(Clone)]
struct Clustering {
    center: Vec<Option<u32>>,
    order: Vec<u32>,
}

impl Clustering {
    fn new(n: usize) -> Self {
        Clustering {
            center: vec![None; n],
            order: Vec::new(),
        }
    }
    fn set(&mut self, v: u32, c: u32) {
        if self.center[v as usize].is_none() {
            self.order.push(v);
        }
        self.center[v as usize] = Some(c);
    }
    fn get(&self, v: u32) -> Option<u32> {
        self.center[v as usize]
    }
}

/// `_lightest_edge_dicts`: per neighboring center (in first-seen order),
/// the neighbor and edge with the lightest weight.
fn lightest(
    inp: &Input,
    alive: &[bool],
    clustering: &Clustering,
    v: u32,
) -> Option<Vec<(u32, u32, u32)>> {
    let mut out: Vec<(u32, u32, u32)> = Vec::new();
    let mut pos: HashMap<u32, usize> = HashMap::new();
    for &(nbr, e) in &inp.adj[v as usize] {
        if !alive[e as usize] {
            continue;
        }
        let c = clustering.get(nbr)?;
        match pos.get(&c) {
            None => {
                pos.insert(c, out.len());
                out.push((c, nbr, e));
            }
            Some(&i) => {
                if inp.keys[e as usize].lt(&inp.keys[out[i].2 as usize]) {
                    out[i] = (c, nbr, e);
                }
            }
        }
    }
    Some(out)
}

/// The `_add_edge_to_spanner(H, residual, u, v)` calls as `(u, v, edge)`,
/// or None where NetworkX would raise (rustnx lets it).
pub fn run(inp: &Input, rng: &mut Mt19937) -> Option<Vec<(u32, u32, u32)>> {
    let n = inp.adj.len();
    let m = inp.keys.len();
    let mut alive = vec![true; m];
    let mut node_alive = vec![true; n];
    let mut clustering = Clustering::new(n);
    for v in 0..n as u32 {
        clustering.set(v, v);
    }
    let mut ops = Vec::new();
    // Tuple keys for the sets of (u, v) pairs.
    let mut pair_id: HashMap<(u32, u32), u32> = HashMap::new();
    let mut pairs: Vec<(u32, u32, u32)> = Vec::new();
    let mut pair_hashes: Vec<i64> = Vec::new();
    let mut i = 0u64;
    while i + 1 < inp.k {
        let mut centers = PySet::default();
        for &v in &clustering.order {
            centers.add(clustering.get(v)?, inp.hashes);
        }
        let mut sampled = vec![false; n];
        for c in centers.iter() {
            if rng.random() < inp.sample_prob {
                sampled[c as usize] = true;
            }
        }
        let mut to_add = PySet::default();
        let mut to_remove: Vec<u32> = Vec::new();
        let mut new_clustering = Clustering::new(n);
        let mut add = |u: u32, v: u32, e: u32, to_add: &mut PySet| {
            let id = *pair_id.entry((u, v)).or_insert_with(|| {
                pairs.push((u, v, e));
                pair_hashes.push(tuple_hash(&[
                    inp.hashes[u as usize],
                    inp.hashes[v as usize],
                ]));
                (pairs.len() - 1) as u32
            });
            to_add.add(id, &pair_hashes);
        };
        for v in 0..n as u32 {
            if !node_alive[v as usize] {
                continue;
            }
            if sampled[clustering.get(v)? as usize] {
                continue;
            }
            let light = lightest(inp, &alive, &clustering, v)?;
            let closest = light
                .iter()
                .filter(|&&(c, _, _)| sampled[c as usize])
                .min_by(|x, y| {
                    if inp.keys[x.2 as usize].lt(&inp.keys[y.2 as usize]) {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                })
                .copied();
            match closest {
                None => {
                    for &(_, nbr, e) in &light {
                        add(v, nbr, e, &mut to_add);
                    }
                    for &(_, e) in &inp.adj[v as usize] {
                        if alive[e as usize] {
                            to_remove.push(e);
                        }
                    }
                }
                Some((cc, cn, ce)) => {
                    add(v, cn, ce, &mut to_add);
                    new_clustering.set(v, cc);
                    let cw = inp.keys[ce as usize];
                    for &(_, nbr, e) in &light {
                        if inp.keys[e as usize].lt(&cw) {
                            add(v, nbr, e, &mut to_add);
                        }
                    }
                    for &(nbr, e) in &inp.adj[v as usize] {
                        if !alive[e as usize] {
                            continue;
                        }
                        let nc = clustering.get(nbr)?;
                        let nw = light.iter().find(|x| x.0 == nc)?.2;
                        if nc == cc || inp.keys[nw as usize].lt(&cw) {
                            to_remove.push(e);
                        }
                    }
                }
            }
        }
        if to_add.len() as f64 > inp.size_limit {
            continue;
        }
        i += 1;
        for id in to_add.iter() {
            let (u, v, e) = pairs[id as usize];
            ops.push((u, v, e));
        }
        for e in to_remove {
            alive[e as usize] = false;
        }
        for &v in &clustering.order {
            let c = clustering.get(v)?;
            if sampled[c as usize] {
                new_clustering.set(v, c);
            }
        }
        clustering = new_clustering;
        for u in 0..n as u32 {
            if !node_alive[u as usize] {
                continue;
            }
            for &(v, e) in &inp.adj[u as usize] {
                if alive[e as usize] && clustering.get(u)? == clustering.get(v)? {
                    alive[e as usize] = false;
                }
            }
        }
        for (v, live) in node_alive.iter_mut().enumerate() {
            if *live && clustering.get(v as u32).is_none() {
                *live = false;
                for &(_, e) in &inp.adj[v] {
                    alive[e as usize] = false;
                }
            }
        }
    }
    for v in 0..n as u32 {
        if !node_alive[v as usize] {
            continue;
        }
        for (_, nbr, e) in lightest(inp, &alive, &clustering, v)? {
            ops.push((v, nbr, e));
        }
    }
    Some(ops)
}
