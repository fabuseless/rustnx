//! GEXF reading for `read_gexf`.
//!
//! The document is parsed into a small element tree holding what
//! NetworkX's `GEXFReader` looks at (each element's namespace, name,
//! unprefixed attributes, children, and the text of `default`,
//! `description` and `keywords` elements, as `xml.etree.ElementTree`
//! gives them). The graph is then built following `GEXFReader` step by
//! step. Documents with a DOCTYPE, a non-UTF-8 encoding or entity
//! references other than XML's own make the parse return `None`, and
//! NetworkX reads them itself.

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use quick_xml::events::{BytesStart, Event};
use quick_xml::name::ResolveResult;
use quick_xml::reader::NsReader;

use super::graphml::{normalize_eol, plain_float, plain_int, resolve_ref};
use super::structure_more::PairHasher;

/// No element (a missing child or sibling).
const NONE: u32 = u32::MAX;

pub struct Element {
    /// Index into `Dom::namespaces`, or `None` without a namespace.
    ns: Option<usize>,
    local: Cow<'static, str>,
    /// This element's range of `Dom::attrs`.
    attrs: (u32, u32),
    first_child: u32,
    last_child: u32,
    next_sibling: u32,
    /// ElementTree's `.text` (text before the first child), kept for the
    /// elements whose text NetworkX reads.
    text: Option<String>,
}

/// The document's elements, with all attributes in one array; attribute
/// values borrow from the input unless they needed unescaping.
pub struct Dom<'i> {
    namespaces: Vec<String>,
    elements: Vec<Element>,
    attrs: Vec<(Cow<'static, str>, Cow<'i, str>)>,
}

/// The element and attribute names NetworkX looks at, stored without
/// allocating.
const KNOWN: &[&str] = &[
    "a",
    "attribute",
    "attributes",
    "attvalue",
    "attvalues",
    "b",
    "class",
    "color",
    "default",
    "defaultedgetype",
    "description",
    "edge",
    "edges",
    "end",
    "for",
    "g",
    "graph",
    "id",
    "keywords",
    "label",
    "meta",
    "mode",
    "name",
    "node",
    "nodes",
    "parent",
    "parents",
    "pid",
    "position",
    "r",
    "shape",
    "size",
    "slice",
    "slices",
    "source",
    "spell",
    "spells",
    "start",
    "target",
    "thickness",
    "timeformat",
    "title",
    "type",
    "uri",
    "value",
    "weight",
    "x",
    "y",
    "z",
];

fn name(bytes: &[u8]) -> Option<Cow<'static, str>> {
    let s = std::str::from_utf8(bytes).ok()?;
    Some(match KNOWN.binary_search(&s) {
        Ok(i) => Cow::Borrowed(KNOWN[i]),
        Err(_) => Cow::Owned(s.to_owned()),
    })
}

/// Elements whose text NetworkX reads.
fn keeps_text(local: &[u8]) -> bool {
    matches!(local, b"default" | b"description" | b"keywords")
}

/// Appends `e`'s unprefixed attributes to `out`.
fn read_attrs<'i>(
    input: &'i [u8],
    e: &BytesStart<'_>,
    out: &mut Vec<(Cow<'static, str>, Cow<'i, str>)>,
) -> Option<()> {
    let start = out.len();
    for a in e.attributes() {
        let a = a.ok()?;
        if a.key.prefix().is_some() {
            continue;
        }
        let key = name(a.key.into_inner())?;
        if out[start..].iter().any(|(k, _)| *k == key) {
            return None; // duplicate attribute: expat rejects it
        }
        let raw = std::str::from_utf8(&a.value).ok()?;
        let value = if raw
            .bytes()
            .any(|b| matches!(b, b'&' | b'\t' | b'\n' | b'\r'))
        {
            let raw = raw.replace("\r\n", " ").replace(['\t', '\n', '\r'], " ");
            Cow::Owned(quick_xml::escape::unescape(&raw).ok()?.into_owned())
        } else {
            // The value as a slice of the input, when it is one.
            let offset = (raw.as_ptr() as usize).wrapping_sub(input.as_ptr() as usize);
            match input.get(offset..offset.wrapping_add(raw.len())) {
                Some(slice) if slice.as_ptr() == raw.as_ptr() => {
                    Cow::Borrowed(std::str::from_utf8(slice).ok()?)
                }
                _ => Cow::Owned(raw.to_owned()),
            }
        };
        out.push((key, value));
    }
    Some(())
}

/// Parses a GEXF document; `None` to let NetworkX read it.
pub fn parse(bytes: &[u8]) -> Option<Dom<'_>> {
    let mut reader = NsReader::from_reader(bytes);
    reader.config_mut().trim_text(false);
    reader.config_mut().check_end_names = true;
    let mut dom = Dom {
        namespaces: Vec::new(),
        elements: Vec::new(),
        attrs: Vec::new(),
    };
    let mut ns_index: HashMap<Vec<u8>, usize> = HashMap::new();
    // Open elements, and whether each still collects text (it keeps its
    // text and has no child yet).
    let mut stack: Vec<(usize, bool)> = Vec::new();
    loop {
        let (ns, event) = reader.read_resolved_event().ok()?;
        match event {
            Event::Decl(d) => {
                if let Some(enc) = d.encoding() {
                    let enc = enc.ok()?;
                    let enc = std::str::from_utf8(&enc).ok()?.to_ascii_lowercase();
                    if enc != "utf-8" && enc != "utf8" {
                        return None;
                    }
                }
            }
            Event::DocType(_) => return None, // may declare entities
            Event::Start(ref e) | Event::Empty(ref e) => {
                let ns = match ns {
                    ResolveResult::Bound(n) => {
                        let n = n.as_ref();
                        Some(match ns_index.get(n) {
                            Some(&i) => i,
                            None => {
                                dom.namespaces.push(std::str::from_utf8(n).ok()?.to_owned());
                                ns_index.insert(n.to_vec(), dom.namespaces.len() - 1);
                                dom.namespaces.len() - 1
                            }
                        })
                    }
                    ResolveResult::Unbound => None,
                    ResolveResult::Unknown(_) => return None, // expat: unbound prefix
                };
                let local = e.local_name();
                let local = local.as_ref();
                let index = dom.elements.len();
                if index >= NONE as usize {
                    return None;
                }
                match stack.last_mut() {
                    Some((parent, collecting)) => {
                        *collecting = false;
                        let p = &mut dom.elements[*parent];
                        let last = p.last_child;
                        p.last_child = index as u32;
                        if last == NONE {
                            p.first_child = index as u32;
                        } else {
                            dom.elements[last as usize].next_sibling = index as u32;
                        }
                    }
                    None if index > 0 => return None, // a second root
                    None => {}
                }
                let keep = keeps_text(local);
                let first = dom.attrs.len() as u32;
                read_attrs(bytes, e, &mut dom.attrs)?;
                dom.elements.push(Element {
                    ns,
                    local: name(local)?,
                    attrs: (first, dom.attrs.len() as u32),
                    first_child: NONE,
                    last_child: NONE,
                    next_sibling: NONE,
                    text: None,
                });
                if matches!(event, Event::Start(_)) {
                    stack.push((index, keep));
                }
            }
            Event::End(_) => {
                stack.pop()?;
            }
            Event::Text(t) => {
                if let Some(&(i, true)) = stack.last() {
                    let s = t.decode().ok()?;
                    let text = dom.elements[i].text.get_or_insert_with(String::new);
                    text.push_str(&normalize_eol(&s));
                }
            }
            Event::CData(t) => {
                if let Some(&(i, true)) = stack.last() {
                    let s = std::str::from_utf8(&t).ok()?;
                    let text = dom.elements[i].text.get_or_insert_with(String::new);
                    text.push_str(&normalize_eol(s));
                }
            }
            Event::GeneralRef(r) => {
                let s = resolve_ref(&r)?;
                if let Some(&(i, true)) = stack.last() {
                    dom.elements[i]
                        .text
                        .get_or_insert_with(String::new)
                        .push_str(&s);
                }
            }
            Event::Comment(_) | Event::PI(_) => {}
            Event::Eof => break,
        }
    }
    if !stack.is_empty() || dom.elements.is_empty() {
        return None;
    }
    for e in &mut dom.elements {
        // ElementTree gives None for empty text.
        if e.text.as_deref() == Some("") {
            e.text = None;
        }
    }
    Some(dom)
}

// --- Building NetworkX's graph -------------------------------------------------

/// A GEXF version as NetworkX's `GEXF.versions` describes it.
pub struct Version {
    pub ns_gexf: String,
    pub ns_viz: String,
    pub version: String,
}

/// An attribute declared in `<attributes>`: `{"title", "type", "mode"}`.
#[derive(Clone)]
struct AttrInfo {
    title: Option<String>,
    ty: Option<String>,
    dynamic: bool,
}

/// One keydict of the multigraph NetworkX builds, as `(key, datadict)` in
/// insertion order (a Python dict only if the result is a multigraph).
struct Pair<'py> {
    entries: Vec<(Bound<'py, PyAny>, Bound<'py, PyDict>)>,
}

impl<'py> Pair<'py> {
    /// The entry whose key equals `key`, as a dict lookup finds it.
    fn position(&self, key: &Bound<'py, PyAny>) -> PyResult<Option<usize>> {
        for (i, (k, _)) in self.entries.iter().enumerate() {
            if k.is(key) || k.eq(key)? {
                return Ok(Some(i));
            }
        }
        Ok(None)
    }
}

struct Reader<'a, 'py> {
    py: Python<'py>,
    dom: &'a Dom<'a>,
    ns: Option<usize>,
    ns_viz: Option<usize>,
    version_1_1: bool,
    timeformat: Option<String>,
    python_type: &'a Bound<'py, PyDict>,
    convert_bool: &'a Bound<'py, PyDict>,
    int_nodes: bool,
    directed: bool,
    // Nodes: values by text, and G._node order.
    by_text: HashMap<&'a str, usize>,
    by_value: Bound<'py, PyDict>,
    values: Vec<Bound<'py, PyAny>>,
    data: Vec<Bound<'py, PyDict>>,
    pos: Vec<Option<usize>>,
    order: Vec<usize>,
    // The multigraph: rows in insertion order, keydicts by node pair.
    succ: Vec<Vec<(usize, usize)>>,
    pred: Vec<Vec<(usize, usize)>>,
    pairs: Vec<Pair<'py>>,
    pair_index: HashMap<u64, usize, BuildHasherDefault<PairHasher>>,
    simple_graph: bool,
}

/// Keyword names `G.add_node(n, **data)` / `G.add_edge(u, v, key=..., **data)`
/// bind to parameters (NetworkX raises TypeError).
const NODE_RESERVED: &[&str] = &["self", "node_for_adding"];
const EDGE_RESERVED: &[&str] = &["self", "u_for_edge", "v_for_edge", "key"];

/// An error that makes rustnx decline (NetworkX raises its own).
fn decline() -> PyErr {
    pyo3::exceptions::PyValueError::new_err("rustnx declines")
}

impl<'a, 'py> Reader<'a, 'py> {
    fn el(&self, i: usize) -> &'a Element {
        &self.dom.elements[i]
    }

    fn get(&self, i: usize, name: &str) -> Option<&'a str> {
        let (a, b) = self.el(i).attrs;
        let dom: &'a Dom<'a> = self.dom;
        dom.attrs[a as usize..b as usize]
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_ref())
    }

    /// `element.find("{ns}local")`: the first such child.
    fn find(&self, i: usize, ns: Option<usize>, local: &'a str) -> Option<usize> {
        self.findall(i, ns, local).next()
    }

    /// `element.findall("{ns}local")`.
    fn findall(
        &self,
        i: usize,
        ns: Option<usize>,
        local: &'a str,
    ) -> impl Iterator<Item = usize> + 'a {
        let dom = self.dom;
        let mut c = if ns.is_some() {
            dom.elements[i].first_child
        } else {
            NONE
        };
        std::iter::from_fn(move || {
            while c != NONE {
                let e = &dom.elements[c as usize];
                let here = c as usize;
                c = e.next_sibling;
                if e.ns == ns && e.local == local {
                    return Some(here);
                }
            }
            None
        })
    }

    fn opt_str(&self, s: Option<&str>) -> Bound<'py, PyAny> {
        match s {
            Some(s) => PyString::new(self.py, s).into_any(),
            None => self.py.None().into_bound(self.py),
        }
    }

    /// `int(text)`.
    fn int(&self, text: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(i) = text.and_then(plain_int) {
            return Ok(i.into_pyobject(self.py)?.into_any());
        }
        self.py.get_type::<PyInt>().call1((self.opt_str(text),))
    }

    /// `float(value)` for an attribute string.
    fn float(&self, text: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(f) = text.and_then(plain_float) {
            return Ok(PyFloat::new(self.py, f).into_any());
        }
        self.py.get_type::<PyFloat>().call1((self.opt_str(text),))
    }

    /// `self.python_type[ty](value)`.
    fn convert(&self, ty: Option<&str>, value: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        let Some(t) = self.python_type.get_item(self.opt_str(ty))? else {
            return Err(decline()); // KeyError
        };
        if t.is(py.get_type::<PyInt>()) {
            return self.int(value);
        }
        if t.is(py.get_type::<PyFloat>()) {
            return self.float(value);
        }
        if t.is(py.get_type::<PyString>()) {
            return Ok(PyString::new(py, value.unwrap_or("None")).into_any());
        }
        t.call1((self.opt_str(value),))
    }

    /// `self.convert_bool[value]`.
    fn boolean(&self, value: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        self.convert_bool
            .get_item(self.opt_str(value))?
            .ok_or_else(decline)
    }

    /// `find_gexf_attributes`.
    fn attributes(
        &self,
        a: usize,
        attrs: &mut Vec<(Option<&'a str>, AttrInfo)>,
        defaults: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        let dynamic = self.get(a, "mode") == Some("dynamic");
        for k in self.findall(a, self.ns, "attribute") {
            let id = self.get(k, "id");
            let title = self.get(k, "title");
            let ty = self.get(k, "type");
            let info = AttrInfo {
                title: title.map(str::to_owned),
                ty: ty.map(str::to_owned),
                dynamic,
            };
            set_attr(attrs, id, info);
            if let Some(d) = self.find(k, self.ns, "default") {
                let text = self.el(d).text.as_deref();
                let value = if ty == Some("boolean") {
                    self.boolean(text)?
                } else {
                    self.convert(ty, text)?
                };
                defaults.set_item(self.opt_str(title), value)?;
            }
        }
        Ok(())
    }

    /// `decode_attr_elements`; `reserved` are the titles `**data` can't
    /// pass on.
    fn decode(
        &self,
        keys: &[(Option<&'a str>, AttrInfo)],
        obj: usize,
        reserved: &[&str],
    ) -> PyResult<Bound<'py, PyDict>> {
        let py = self.py;
        let out = PyDict::new(py);
        let Some(values) = self.find(obj, self.ns, "attvalues") else {
            return Ok(out);
        };
        for a in self.findall(values, self.ns, "attvalue") {
            let key = self.get(a, "for");
            let Some(info) = keys.iter().rev().find(|(id, _)| *id == key).map(|(_, i)| i) else {
                return Err(decline()); // "No attribute defined for=..."
            };
            let ty = info.ty.as_deref();
            let raw = self.get(a, "value");
            let value = if ty == Some("boolean") {
                self.boolean(raw)?
            } else {
                self.convert(ty, raw)?
            };
            // `**data` needs string keys that aren't parameter names.
            let Some(title) = info.title.as_deref().filter(|t| !reserved.contains(t)) else {
                return Err(decline());
            };
            let title = PyString::new(py, title);
            if info.dynamic {
                let tf = self.timeformat.as_deref();
                let start = self.convert(tf, self.get(a, "start"))?;
                let end = self.convert(tf, self.get(a, "end"))?;
                let entry = PyTuple::new(py, [value, start, end])?;
                match out.get_item(&title)? {
                    Some(list) => {
                        // attr[title].append(...): a list from an earlier
                        // dynamic value, or NetworkX raises.
                        let list = list.cast_into::<PyList>().map_err(|_| decline())?;
                        list.append(entry)?;
                    }
                    None => out.set_item(&title, PyList::new(py, [entry])?)?,
                }
            } else {
                out.set_item(&title, value)?;
            }
        }
        Ok(out)
    }

    /// `add_slices` (GEXF 1.1) or `add_spells`.
    fn add_time(&self, data: &Bound<'py, PyDict>, xml: usize) -> PyResult<()> {
        let py = self.py;
        if self.version_1_1 {
            if let Some(slices) = self.find(xml, self.ns, "slices") {
                let list = PyList::empty(py);
                data.set_item("slices", &list)?;
                for s in self.findall(slices, self.ns, "slice") {
                    let pair = [
                        self.opt_str(self.get(s, "start")),
                        self.opt_str(self.get(s, "end")),
                    ];
                    list.append(PyTuple::new(py, pair)?)?;
                }
            }
        } else if let Some(spells) = self.find(xml, self.ns, "spells") {
            let list = PyList::empty(py);
            data.set_item("spells", &list)?;
            let tf = self.timeformat.as_deref();
            for s in self.findall(spells, self.ns, "spell") {
                let start = self.convert(tf, self.get(s, "start"))?;
                let end = self.convert(tf, self.get(s, "end"))?;
                list.append(PyTuple::new(py, [start, end])?)?;
            }
        }
        Ok(())
    }

    /// `add_start_end`.
    fn add_start_end(&self, data: &Bound<'py, PyDict>, xml: usize) -> PyResult<()> {
        let tf = self.timeformat.as_deref();
        if let Some(s) = self.get(xml, "start") {
            data.set_item("start", self.convert(tf, Some(s))?)?;
        }
        if let Some(e) = self.get(xml, "end") {
            data.set_item("end", self.convert(tf, Some(e))?)?;
        }
        Ok(())
    }

    /// `add_viz`.
    fn add_viz(&self, data: &Bound<'py, PyDict>, xml: usize) -> PyResult<()> {
        let py = self.py;
        let viz = PyDict::new(py);
        if let Some(c) = self.find(xml, self.ns_viz, "color") {
            let color = PyDict::new(py);
            color.set_item("r", self.int(self.get(c, "r"))?)?;
            color.set_item("g", self.int(self.get(c, "g"))?)?;
            color.set_item("b", self.int(self.get(c, "b"))?)?;
            if !self.version_1_1 {
                let a = match self.get(c, "a") {
                    Some(a) => self.float(Some(a))?,
                    None => PyFloat::new(py, 1.0).into_any(),
                };
                color.set_item("a", a)?;
            }
            viz.set_item("color", color)?;
        }
        if let Some(s) = self.find(xml, self.ns_viz, "size") {
            viz.set_item("size", self.float(self.get(s, "value"))?)?;
        }
        if let Some(t) = self.find(xml, self.ns_viz, "thickness") {
            viz.set_item("thickness", self.float(self.get(t, "value"))?)?;
        }
        if let Some(s) = self.find(xml, self.ns_viz, "shape") {
            let shape = self.get(s, "shape");
            let shape = if shape == Some("image") {
                self.get(s, "uri")
            } else {
                shape
            };
            viz.set_item("shape", self.opt_str(shape))?;
        }
        if let Some(p) = self.find(xml, self.ns_viz, "position") {
            let position = PyDict::new(py);
            for axis in ["x", "y", "z"] {
                let v = match self.get(p, axis) {
                    Some(v) => self.float(Some(v))?,
                    None => PyFloat::new(py, 0.0).into_any(),
                };
                position.set_item(axis, v)?;
            }
            viz.set_item("position", position)?;
        }
        if !viz.is_empty() {
            data.set_item("viz", viz)?;
        }
        Ok(())
    }

    /// The node `node_type(text)` (the text itself without a node type).
    fn node(&mut self, text: Option<&'a str>) -> PyResult<usize> {
        let py = self.py;
        let Some(text) = text else {
            return Err(decline()); // None (cannot be a node) or int(None)
        };
        if let Some(&i) = self.by_text.get(text) {
            return Ok(i);
        }
        let i = if self.int_nodes {
            let value = self.int(Some(text))?;
            match self.by_value.get_item(&value)? {
                Some(i) => i.extract::<usize>()?,
                None => {
                    self.by_value.set_item(&value, self.values.len())?;
                    self.push(value)
                }
            }
        } else {
            self.push(PyString::new(py, text).into_any())
        };
        self.by_text.insert(text, i);
        Ok(i)
    }

    fn push(&mut self, value: Bound<'py, PyAny>) -> usize {
        self.values.push(value);
        self.data.push(PyDict::new(self.py));
        self.pos.push(None);
        self.succ.push(Vec::new());
        self.pred.push(Vec::new());
        self.values.len() - 1
    }

    /// Adds node `i` to `G._node` if it isn't there.
    fn place(&mut self, i: usize) {
        if self.pos[i].is_none() {
            self.pos[i] = Some(self.order.len());
            self.order.push(i);
        }
    }

    /// `add_node`, recursively for subnodes.
    fn add_node(
        &mut self,
        node_attr: &[(Option<&'a str>, AttrInfo)],
        xml: usize,
        pid: Option<usize>,
    ) -> PyResult<()> {
        let py = self.py;
        let data = self.decode(node_attr, xml, NODE_RESERVED)?;
        if let Some(parents) = self.find(xml, self.ns, "parents") {
            let list = PyList::empty(py);
            data.set_item("parents", &list)?;
            for p in self.findall(parents, self.ns, "parent") {
                list.append(self.opt_str(self.get(p, "for")))?;
            }
        }
        self.add_time(&data, xml)?;
        self.add_viz(&data, xml)?;
        self.add_start_end(&data, xml)?;
        let id = self.node(self.get(xml, "id"))?;
        data.set_item(intern!(py, "label"), self.opt_str(self.get(xml, "label")))?;
        // node_xml.get("pid", node_pid): the attribute text, or the
        // parent's (converted) id.
        let pid_value = match (self.get(xml, "pid"), pid) {
            (Some(p), _) => Some(PyString::new(py, p).into_any()),
            (None, Some(p)) => Some(self.values[p].clone()),
            (None, None) => None,
        };
        if let Some(p) = pid_value {
            data.set_item("pid", p)?;
        }
        if let Some(sub) = self.find(xml, self.ns, "nodes") {
            let subs: Vec<usize> = self.findall(sub, self.ns, "node").collect();
            for s in subs {
                self.add_node(node_attr, s, Some(id))?;
            }
        }
        self.place(id);
        self.data[id].update(data.as_mapping())?;
        Ok(())
    }

    /// `MultiGraph.add_edge(u, v, key=key, **data)`.
    /// `data` becomes the new edge's dict (the caller passes a dict of its
    /// own).
    fn multi_add_edge(
        &mut self,
        u: usize,
        v: usize,
        key: Bound<'py, PyAny>,
        data: Bound<'py, PyDict>,
    ) -> PyResult<()> {
        let py = self.py;
        self.place(u);
        self.place(v);
        let pair = pair_key(self.directed, u, v);
        let p = match self.pair_index.get(&pair) {
            Some(&p) => p,
            None => {
                let p = self.pairs.len();
                self.pairs.push(Pair {
                    entries: Vec::new(),
                });
                self.pair_index.insert(pair, p);
                self.succ[u].push((v, p));
                if self.directed {
                    self.pred[v].push((u, p));
                } else if u != v {
                    self.succ[v].push((u, p));
                }
                p
            }
        };
        let pair = &mut self.pairs[p];
        let key = if key.is_none() {
            // new_edge_key
            let mut k = pair.entries.len();
            while pair.position(&k.into_pyobject(py)?.into_any())?.is_some() {
                k += 1;
            }
            k.into_pyobject(py)?.into_any()
        } else {
            key
        };
        match pair.position(&key)? {
            Some(i) => pair.entries[i].1.update(data.as_mapping())?,
            None => pair.entries.push((key, data)),
        }
        Ok(())
    }

    /// `add_edge`.
    fn add_edge(&mut self, edge_attr: &[(Option<&'a str>, AttrInfo)], xml: usize) -> PyResult<()> {
        let py = self.py;
        let direction = self.get(xml, "type");
        match (self.directed, direction) {
            (true, Some("undirected")) | (false, Some("directed")) => return Err(decline()),
            _ => {}
        }
        let (source, target) = (self.get(xml, "source"), self.get(xml, "target"));
        // node_type(source), node_type(target); with no node type, the
        // texts (None fails when the edge is added).
        let u = self.node(source)?;
        let v = self.node(target)?;
        let data = self.decode(edge_attr, xml, EDGE_RESERVED)?;
        self.add_start_end(&data, xml)?;
        self.add_time(&data, xml)?;
        let mut key = self.opt_str(self.get(xml, "id"));
        if !key.is_none() {
            data.set_item(intern!(py, "id"), &key)?;
        }
        // data.pop("networkx_key", None)
        if !data.is_empty() {
            let name = intern!(py, "networkx_key");
            if let Some(k) = data.get_item(name)? {
                data.del_item(name)?;
                if !k.is_none() {
                    key = k;
                }
            }
        }
        if let Some(w) = self.get(xml, "weight") {
            data.set_item(intern!(py, "weight"), self.float(Some(w))?)?;
        }
        if let Some(l) = self.get(xml, "label") {
            data.set_item(intern!(py, "label"), l)?;
        }
        let pair = pair_key(self.directed, u, v);
        if self.pair_index.contains_key(&pair) {
            self.simple_graph = false;
        }
        if direction == Some("mutual") {
            let copy = data.copy()?;
            self.multi_add_edge(u, v, key.clone(), data)?;
            self.multi_add_edge(v, u, key, copy)?;
        } else {
            self.multi_add_edge(u, v, key, data)?;
        }
        Ok(())
    }
}

/// A node pair as a map key (both orders the same when undirected).
fn pair_key(directed: bool, u: usize, v: usize) -> u64 {
    let (a, b) = if directed || u <= v { (u, v) } else { (v, u) };
    ((a as u64) << 32) | b as u64
}

/// `attrs[attr_id] = info` on a list kept in dict order.
fn set_attr<'a>(attrs: &mut Vec<(Option<&'a str>, AttrInfo)>, id: Option<&'a str>, info: AttrInfo) {
    match attrs.iter_mut().find(|(k, _)| *k == id) {
        Some(slot) => slot.1 = info,
        None => attrs.push((id, info)),
    }
}

/// NetworkX's `read_gexf` graph (before `relabel`), built into one of
/// `classes = (Graph, DiGraph, MultiGraph, MultiDiGraph)`; `None` lets
/// NetworkX read the document. `versions` lists the GEXF versions to try,
/// in NetworkX's order (the requested one first); `read_meta` is whether
/// this NetworkX reads `<meta>`; `python_type` and `convert_bool` are its
/// reader's tables; `int_nodes` is `node_type=int` (otherwise None).
#[allow(clippy::too_many_arguments)]
pub fn build<'py>(
    py: Python<'py>,
    dom: &Dom<'_>,
    versions: &[Version],
    read_meta: bool,
    python_type: &Bound<'py, PyDict>,
    convert_bool: &Bound<'py, PyDict>,
    int_nodes: bool,
    classes: &Bound<'py, PyTuple>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match build_inner(
        py,
        dom,
        versions,
        read_meta,
        python_type,
        convert_bool,
        int_nodes,
        classes,
    ) {
        Ok(g) => Ok(g),
        Err(_) => Ok(None),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_inner<'py>(
    py: Python<'py>,
    dom: &Dom<'_>,
    versions: &[Version],
    read_meta: bool,
    python_type: &Bound<'py, PyDict>,
    convert_bool: &Bound<'py, PyDict>,
    int_nodes: bool,
    classes: &Bound<'py, PyTuple>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let ns_of = |uri: &str| dom.namespaces.iter().position(|n| n == uri);
    let mut r = Reader {
        py,
        dom,
        ns: None,
        ns_viz: None,
        version_1_1: false,
        timeformat: None,
        python_type,
        convert_bool,
        int_nodes,
        directed: false,
        by_text: HashMap::new(),
        by_value: PyDict::new(py),
        values: Vec::new(),
        data: Vec::new(),
        pos: Vec::new(),
        order: Vec::new(),
        succ: Vec::new(),
        pred: Vec::new(),
        pairs: Vec::new(),
        pair_index: HashMap::default(),
        simple_graph: true,
    };
    // self.xml.find("{NS_GEXF}graph") for each version in turn.
    let mut found = None;
    for v in versions {
        r.ns = ns_of(&v.ns_gexf);
        if let Some(g) = r.find(0, r.ns, "graph") {
            r.ns_viz = ns_of(&v.ns_viz);
            r.version_1_1 = v.version == "1.1";
            found = Some(g);
            break;
        }
    }
    let Some(g) = found else { return Ok(None) };
    let meta = if read_meta {
        r.find(0, r.ns, "meta")
    } else {
        None
    };

    r.directed = r.get(g, "defaultedgetype") == Some("directed");
    let gattr = PyDict::new(py);
    if let Some(meta) = meta {
        for name in ["description", "keywords"] {
            if let Some(e) = r.find(meta, r.ns, name) {
                gattr.set_item(name, r.opt_str(r.el(e).text.as_deref()))?;
            }
        }
    }
    if let Some(name) = r.get(g, "name").filter(|n| !n.is_empty()) {
        gattr.set_item("name", name)?;
    }
    if let Some(start) = r.get(g, "start") {
        gattr.set_item("start", start)?;
    }
    if let Some(end) = r.get(g, "end") {
        gattr.set_item("end", end)?;
    }
    gattr.set_item(
        "mode",
        if r.get(g, "mode") == Some("dynamic") {
            "dynamic"
        } else {
            "static"
        },
    )?;
    r.timeformat = r
        .get(g, "timeformat")
        .map(|t| if t == "date" { "string" } else { t }.to_owned());

    let mut node_attr: Vec<(Option<&str>, AttrInfo)> = Vec::new();
    let mut edge_attr: Vec<(Option<&str>, AttrInfo)> = Vec::new();
    let node_default = PyDict::new(py);
    let edge_default = PyDict::new(py);
    let attribute_elements: Vec<usize> = r.findall(g, r.ns, "attributes").collect();
    for a in attribute_elements {
        match r.get(a, "class") {
            Some("node") => {
                r.attributes(a, &mut node_attr, &node_default)?;
                gattr.set_item("node_default", &node_default)?;
            }
            Some("edge") => {
                r.attributes(a, &mut edge_attr, &edge_default)?;
                gattr.set_item("edge_default", &edge_default)?;
            }
            _ => return Ok(None), // NetworkX raises
        }
    }
    // Gephi 0.7beta's weight attribute.
    set_attr(
        &mut edge_attr,
        Some("weight"),
        AttrInfo {
            title: Some("weight".into()),
            ty: Some("double".into()),
            dynamic: false,
        },
    );
    gattr.set_item("edge_default", &edge_default)?;

    if let Some(nodes) = r.find(g, r.ns, "nodes") {
        let list: Vec<usize> = r.findall(nodes, r.ns, "node").collect();
        for n in list {
            r.add_node(&node_attr, n, None)?;
        }
    }
    if let Some(edges) = r.find(g, r.ns, "edges") {
        let list: Vec<usize> = r.findall(edges, r.ns, "edge").collect();
        for e in list {
            r.add_edge(&edge_attr, e)?;
        }
    }

    let directed = r.directed;
    let multigraph = !r.simple_graph;
    let class_index = match (multigraph, directed) {
        (false, false) => 0,
        (false, true) => 1,
        (true, false) => 2,
        (true, true) => 3,
    };
    let graph = classes.get_item(class_index)?.call0()?;
    graph
        .getattr("graph")?
        .cast_into::<PyDict>()?
        .update(gattr.as_mapping())?;
    let n = r.values.len();
    let node_dict = PyDict::new(py);
    for &i in &r.order {
        node_dict.set_item(&r.values[i], &r.data[i])?;
    }
    let succ_rows: Vec<Bound<'py, PyDict>> = (0..n).map(|_| PyDict::new(py)).collect();
    let pred_rows: Vec<Bound<'py, PyDict>> = if directed {
        (0..n).map(|_| PyDict::new(py)).collect()
    } else {
        Vec::new()
    };
    if multigraph {
        let mut keydicts = Vec::with_capacity(r.pairs.len());
        for pair in &r.pairs {
            let keydict = PyDict::new(py);
            for (k, d) in &pair.entries {
                keydict.set_item(k, d)?;
            }
            keydicts.push(keydict);
        }
        for &u in &r.order {
            for &(v, p) in &r.succ[u] {
                succ_rows[u].set_item(&r.values[v], &keydicts[p])?;
            }
            if directed {
                for &(v, p) in &r.pred[u] {
                    pred_rows[u].set_item(&r.values[v], &keydicts[p])?;
                }
            }
        }
    } else {
        // nx.Graph(G) / nx.DiGraph(G): from_dict_of_dicts(G.adj,
        // multigraph_input=True), rows in node order; undirected, (u, v)
        // is skipped once v's row added it. Each keydict's data dicts are
        // merged into one.
        for &u in &r.order {
            let pu = r.pos[u];
            for &(v, p) in &r.succ[u] {
                if !directed && r.pos[v] < pu {
                    continue;
                }
                let entries = &r.pairs[p].entries;
                let data = if entries.len() == 1 {
                    entries[0].1.clone()
                } else {
                    let merged = PyDict::new(py);
                    for (_, d) in entries {
                        merged.update(d.as_mapping())?;
                    }
                    merged
                };
                succ_rows[u].set_item(&r.values[v], &data)?;
                if directed {
                    pred_rows[v].set_item(&r.values[u], &data)?;
                } else if u != v {
                    succ_rows[v].set_item(&r.values[u], &data)?;
                }
            }
        }
    }
    let adj = PyDict::new(py);
    for &i in &r.order {
        adj.set_item(&r.values[i], &succ_rows[i])?;
    }
    graph.setattr("_node", &node_dict)?;
    graph.setattr("_adj", &adj)?;
    if directed {
        let pred = PyDict::new(py);
        for &i in &r.order {
            pred.set_item(&r.values[i], &pred_rows[i])?;
        }
        graph.setattr("_succ", &adj)?;
        graph.setattr("_pred", &pred)?;
    }
    Ok(Some(graph))
}
