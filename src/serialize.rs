//! Pickle support for `CoreGraph`.
//!
//! NetworkX caches converted graphs in `G.__networkx_cache__`, so a
//! `CoreGraph` ends up inside any NetworkX graph that rustnx has worked on.
//! Pickling or deep-copying that graph (e.g. to send it to a
//! `multiprocessing` worker) must therefore work.

use std::collections::HashMap;
use std::sync::OnceLock;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::graph::{CoreGraph, Csr, ExactPred, Weights};

const MAGIC: &[u8; 4] = b"RNX1";

#[derive(Default)]
struct Writer(Vec<u8>);

impl Writer {
    fn u64(&mut self, x: u64) {
        self.0.extend_from_slice(&x.to_le_bytes());
    }
    fn flag(&mut self, b: bool) {
        self.0.push(b as u8);
    }
    fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.0.extend_from_slice(s.as_bytes());
    }
    fn f64s(&mut self, v: &[f64]) {
        self.u64(v.len() as u64);
        for x in v {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
    }
    fn csr(&mut self, c: &Csr) {
        self.u64(c.offsets.len() as u64);
        for &o in &c.offsets {
            self.u64(o as u64);
        }
        self.u64(c.targets.len() as u64);
        for &t in &c.targets {
            self.0.extend_from_slice(&t.to_le_bytes());
        }
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

fn malformed() -> PyErr {
    PyValueError::new_err("malformed rustnx graph data")
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> PyResult<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or_else(malformed)?;
        let out = self.data.get(self.pos..end).ok_or_else(malformed)?;
        self.pos = end;
        Ok(out)
    }
    fn u64(&mut self) -> PyResult<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().map_err(|_| malformed())?,
        ))
    }
    fn len(&mut self, item_size: usize) -> PyResult<usize> {
        let n = usize::try_from(self.u64()?).map_err(|_| malformed())?;
        // Reject lengths that can't fit in the remaining bytes.
        if n.checked_mul(item_size).ok_or_else(malformed)? > self.data.len() - self.pos {
            return Err(malformed());
        }
        Ok(n)
    }
    fn flag(&mut self) -> PyResult<bool> {
        match self.take(1)?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(malformed()),
        }
    }
    fn str(&mut self) -> PyResult<String> {
        let n = self.len(1)?;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| malformed())
    }
    fn f64s(&mut self) -> PyResult<Vec<f64>> {
        let n = self.len(8)?;
        let bytes = self.take(n * 8)?;
        Ok(bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&b| f64::from_le_bytes(b))
            .collect())
    }
    fn csr(&mut self, n: usize) -> PyResult<Csr> {
        let no = self.len(8)?;
        let offsets = (0..no)
            .map(|_| self.u64().map(|o| o as usize))
            .collect::<PyResult<Vec<_>>>()?;
        let nt = self.len(4)?;
        let targets: Vec<u32> = self
            .take(nt * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| u32::from_le_bytes(b))
            .collect();
        let valid = offsets.len() == n + 1
            && offsets.first() == Some(&0)
            && offsets.last() == Some(&targets.len())
            && offsets.windows(2).all(|w| w[0] <= w[1])
            && targets.iter().all(|&t| (t as usize) < n);
        if !valid {
            return Err(malformed());
        }
        Ok(Csr { offsets, targets })
    }
}

pub fn to_bytes(g: &CoreGraph) -> Vec<u8> {
    let mut w = Writer::default();
    w.0.extend_from_slice(MAGIC);
    w.u64(g.n as u64);
    w.flag(g.directed);
    w.csr(&g.succ);
    w.flag(g.pred.is_some());
    if let Some(pred) = &g.pred {
        w.csr(pred);
    }
    let mut names: Vec<&String> = g.weights.keys().collect();
    names.sort();
    w.u64(names.len() as u64);
    for name in names {
        let wt = &g.weights[name];
        w.str(name);
        w.f64s(&wt.succ);
        w.flag(wt.pred.is_some());
        if let Some(pred) = &wt.pred {
            w.f64s(pred);
        }
        w.flag(wt.all_int);
        w.flag(wt.has_hidden);
    }
    let exact = g.exact_pred.get();
    w.flag(exact.is_some());
    if let Some(exact) = exact {
        w.csr(&exact.csr);
        let mut names: Vec<&String> = exact.weights.keys().collect();
        names.sort();
        w.u64(names.len() as u64);
        for name in names {
            w.str(name);
            w.f64s(&exact.weights[name]);
        }
    }
    w.0
}

pub fn from_bytes(data: &[u8]) -> PyResult<CoreGraph> {
    let mut r = Reader { data, pos: 0 };
    if r.take(4)? != MAGIC {
        return Err(malformed());
    }
    let n = usize::try_from(r.u64()?).map_err(|_| malformed())?;
    let directed = r.flag()?;
    let succ = r.csr(n)?;
    let m = succ.targets.len();
    let pred = if r.flag()? { Some(r.csr(n)?) } else { None };
    if pred.is_some() != directed || pred.as_ref().is_some_and(|p| p.targets.len() != m) {
        return Err(malformed());
    }
    let mut weights = HashMap::new();
    for _ in 0..r.len(1)? {
        let name = r.str()?;
        let succ_w = r.f64s()?;
        let pred_w = if r.flag()? { Some(r.f64s()?) } else { None };
        if succ_w.len() != m || pred_w.as_ref().is_some_and(|p| p.len() != m) {
            return Err(malformed());
        }
        let all_int = r.flag()?;
        let has_hidden = r.flag()?;
        weights.insert(
            name,
            Weights {
                succ: succ_w,
                pred: pred_w,
                all_int,
                has_hidden,
            },
        );
    }
    let exact_pred = OnceLock::new();
    if r.flag()? {
        let csr = r.csr(n)?;
        let mut ws = HashMap::new();
        for _ in 0..r.len(1)? {
            let name = r.str()?;
            let vals = r.f64s()?;
            if vals.len() != csr.targets.len() {
                return Err(malformed());
            }
            ws.insert(name, vals);
        }
        let _ = exact_pred.set(ExactPred { csr, weights: ws });
    }
    if r.pos != data.len() {
        return Err(malformed());
    }
    Ok(CoreGraph {
        n,
        directed,
        succ,
        pred,
        exact_pred,
        weights,
    })
}

#[pyfunction]
pub fn _core_graph_from_bytes(data: &Bound<'_, PyBytes>) -> PyResult<CoreGraph> {
    from_bytes(data.as_bytes())
}
