//! GraphML parsing for `read_graphml` / `parse_graphml`.
//!
//! Reads the document into a plain representation of what NetworkX's
//! `GraphMLReader` looks at (keys and defaults, then each graph's nodes,
//! edges and data, with element text as `xml.etree.ElementTree` gives it).
//! Anything NetworkX handles specially or that `ElementTree` would read
//! differently (yFiles data, ports, hyperedges, nested graphs, a DOCTYPE,
//! non-UTF-8 encodings, a missing GraphML namespace) makes the parse return
//! `None`, and NetworkX reads the file itself.

use quick_xml::events::{BytesRef, BytesStart, Event};
use quick_xml::name::ResolveResult;
use quick_xml::reader::NsReader;

const NS_GRAPHML: &[u8] = b"http://graphml.graphdrawing.org/xmlns";

/// A `<key>`: its `id`, `attr.name`, `attr.type` and `for`, and the text of
/// its `<default>` child (`None` if there is none).
pub struct Key {
    pub id: String,
    pub name: Option<String>,
    pub ty: Option<String>,
    pub for_: Option<String>,
    pub default: Option<Option<String>>,
    pub yfiles: bool,
}

/// A `<data>` element: its `key` attribute and text (`None` when empty).
pub type Data = (Option<String>, Option<String>);

pub struct Node {
    pub id: Option<String>,
    pub data: Vec<Data>,
}

pub struct Edge {
    pub id: Option<String>,
    pub source: Option<String>,
    pub target: Option<String>,
    pub directed: Option<String>,
    pub data: Vec<Data>,
}

pub struct Graph {
    pub edgedefault: Option<String>,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub data: Vec<Data>,
}

pub struct Document {
    pub keys: Vec<Key>,
    pub graphs: Vec<Graph>,
}

/// XML end-of-line handling, as expat applies it to literal text: `\r\n`
/// and lone `\r` become `\n`.
pub(crate) fn normalize_eol(s: &str) -> String {
    if !s.contains('\r') {
        return s.to_owned();
    }
    s.replace("\r\n", "\n").replace('\r', "\n")
}

/// A reference in text: the five predefined entities or a character
/// reference; `None` for anything else (an error for expat).
pub(crate) fn resolve_ref(r: &BytesRef<'_>) -> Option<String> {
    if r.is_char_ref() {
        return r.resolve_char_ref().ok().flatten().map(String::from);
    }
    let name = r.decode().ok()?;
    let c = match name.as_ref() {
        "lt" => '<',
        "gt" => '>',
        "amp" => '&',
        "apos" => '\'',
        "quot" => '"',
        _ => return None,
    };
    Some(c.to_string())
}

/// An element's attributes without a namespace prefix, read once:
/// values normalized as expat does (literal tab, newline and carriage
/// return become spaces, `\r\n` one space) and with references expanded.
struct Attrs<'a>(Vec<(&'a [u8], String)>);

impl<'a> Attrs<'a> {
    /// `None` for malformed input.
    fn read(e: &'a BytesStart<'a>) -> Option<Self> {
        let mut out: Vec<(&'a [u8], String)> = Vec::new();
        for a in e.attributes() {
            let a = a.ok()?;
            if a.key.prefix().is_some() {
                continue;
            }
            let key = a.key.into_inner();
            if out.iter().any(|(k, _)| *k == key) {
                return None; // duplicate attribute: expat rejects it
            }
            let raw = std::str::from_utf8(&a.value).ok()?;
            let value = if raw.bytes().any(|b| matches!(b, b'&' | b'\t' | b'\n' | b'\r')) {
                let raw = raw.replace("\r\n", " ").replace(['\t', '\n', '\r'], " ");
                quick_xml::escape::unescape(&raw).ok()?.into_owned()
            } else {
                raw.to_owned()
            };
            out.push((key, value));
        }
        Some(Attrs(out))
    }

    fn get(&self, name: &[u8]) -> Option<String> {
        self.0.iter().find(|(k, _)| *k == name).map(|(_, v)| v.clone())
    }

    fn take(&mut self, name: &[u8]) -> Option<String> {
        let i = self.0.iter().position(|(k, _)| *k == name)?;
        Some(std::mem::take(&mut self.0[i].1))
    }
}

/// What an open element is, for routing text and children.
#[derive(Clone, Copy, PartialEq)]
enum Ctx {
    Root,
    Key,
    Default,
    Graph,
    Node,
    Edge,
    Data,
    /// An element NetworkX doesn't look at (its subtree is ignored).
    Skip,
}

/// Parses a GraphML document; `None` to let NetworkX read it.
pub fn parse(bytes: &[u8]) -> Option<Document> {
    let mut reader = NsReader::from_reader(bytes);
    reader.config_mut().trim_text(false);
    reader.config_mut().check_end_names = true;
    let mut doc = Document {
        keys: Vec::new(),
        graphs: Vec::new(),
    };
    let mut stack: Vec<Ctx> = Vec::new();
    let mut seen_root = false;
    // Text of the current data/default element, and whether it has child
    // elements (yFiles data, which rustnx leaves to NetworkX).
    let mut text: Option<String> = None;
    // Which list the open <data> element was added to.
    let mut owner = Ctx::Graph;
    loop {
        let (ns, event) = reader.read_resolved_event().ok()?;
        let in_graphml = matches!(ns, ResolveResult::Bound(n) if n.as_ref() == NS_GRAPHML);
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
                let empty = matches!(event, Event::Empty(_));
                let local = e.local_name();
                let local = local.as_ref();
                let parent = stack.last().copied();
                let ctx = match parent {
                    None => {
                        if seen_root || !in_graphml || local != b"graphml" {
                            return None; // NetworkX retries or fails
                        }
                        seen_root = true;
                        Ctx::Root
                    }
                    Some(Ctx::Root) if in_graphml && local == b"key" => {
                        let a = Attrs::read(e)?;
                        doc.keys.push(Key {
                            id: a.get(b"id").unwrap_or_default(),
                            name: a.get(b"attr.name"),
                            ty: a.get(b"attr.type"),
                            for_: a.get(b"for"),
                            default: None,
                            yfiles: a.get(b"yfiles.type").is_some(),
                        });
                        Ctx::Key
                    }
                    Some(Ctx::Root) if in_graphml && local == b"graph" => {
                        if !doc.graphs.is_empty() {
                            return None; // NetworkX builds every graph
                        }
                        doc.graphs.push(Graph {
                            edgedefault: Attrs::read(e)?.get(b"edgedefault"),
                            nodes: Vec::new(),
                            edges: Vec::new(),
                            data: Vec::new(),
                        });
                        Ctx::Graph
                    }
                    Some(Ctx::Key) if in_graphml && local == b"default" => {
                        let key = doc.keys.last_mut()?;
                        if key.default.is_some() {
                            // NetworkX's find() takes the first <default>.
                            Ctx::Skip
                        } else {
                            key.default = Some(None);
                            text = None;
                            Ctx::Default
                        }
                    }
                    Some(Ctx::Graph) if in_graphml && local == b"node" => {
                        let mut a = Attrs::read(e)?;
                        if a.get(b"yfiles.foldertype").as_deref() == Some("group") {
                            return None; // a nested graph
                        }
                        doc.graphs.last_mut()?.nodes.push(Node {
                            id: a.take(b"id"),
                            data: Vec::new(),
                        });
                        Ctx::Node
                    }
                    Some(Ctx::Graph) if in_graphml && local == b"edge" => {
                        let mut a = Attrs::read(e)?;
                        doc.graphs.last_mut()?.edges.push(Edge {
                            id: a.take(b"id"),
                            source: a.take(b"source"),
                            target: a.take(b"target"),
                            directed: a.take(b"directed"),
                            data: Vec::new(),
                        });
                        Ctx::Edge
                    }
                    Some(Ctx::Graph) if in_graphml && local == b"hyperedge" => return None,
                    Some(Ctx::Node | Ctx::Edge) if in_graphml && local == b"port" => {
                        return None; // NetworkX warns
                    }
                    Some(c @ (Ctx::Graph | Ctx::Node | Ctx::Edge)) if in_graphml && local == b"data" => {
                        let key = Attrs::read(e)?.take(b"key");
                        let g = doc.graphs.last_mut()?;
                        let list = match c {
                            Ctx::Graph => &mut g.data,
                            Ctx::Node => &mut g.nodes.last_mut()?.data,
                            _ => &mut g.edges.last_mut()?.data,
                        };
                        list.push((key, None));
                        owner = c;
                        text = None;
                        Ctx::Data
                    }
                    // Child elements of data or default: yFiles extensions
                    // (or text NetworkX would cut at the first child).
                    Some(Ctx::Data | Ctx::Default) => return None,
                    _ => Ctx::Skip,
                };
                if empty {
                    finish(&mut doc, ctx, owner, &mut text)?;
                } else {
                    stack.push(ctx);
                }
            }
            Event::End(_) => {
                let ctx = stack.pop()?;
                finish(&mut doc, ctx, owner, &mut text)?;
            }
            Event::Text(t) => {
                if matches!(stack.last(), Some(Ctx::Data | Ctx::Default)) {
                    let s = t.decode().ok()?;
                    text.get_or_insert_with(String::new).push_str(&normalize_eol(&s));
                }
            }
            Event::CData(t) => {
                if matches!(stack.last(), Some(Ctx::Data | Ctx::Default)) {
                    let s = std::str::from_utf8(&t).ok()?;
                    text.get_or_insert_with(String::new).push_str(&normalize_eol(s));
                }
            }
            Event::GeneralRef(r) => {
                let s = resolve_ref(&r)?;
                if matches!(stack.last(), Some(Ctx::Data | Ctx::Default)) {
                    text.get_or_insert_with(String::new).push_str(&s);
                }
            }
            Event::Comment(_) | Event::PI(_) => {}
            Event::Eof => break,
        }
    }
    if !stack.is_empty() || !seen_root || doc.graphs.is_empty() {
        return None;
    }
    Some(doc)
}

/// Stores the text collected for a closed data element (added to the list
/// of `owner`) or default element. Empty text is `None`, as in ElementTree.
fn finish(doc: &mut Document, ctx: Ctx, owner: Ctx, text: &mut Option<String>) -> Option<()> {
    match ctx {
        Ctx::Data => {
            let g = doc.graphs.last_mut()?;
            let slot = match owner {
                Ctx::Node => g.nodes.last_mut()?.data.last_mut()?,
                Ctx::Edge => g.edges.last_mut()?.data.last_mut()?,
                _ => g.data.last_mut()?,
            };
            slot.1 = text.take().filter(|s| !s.is_empty());
        }
        Ctx::Default => {
            let key = doc.keys.last_mut()?;
            key.default = Some(text.take().filter(|s| !s.is_empty()));
        }
        _ => {}
    }
    Some(())
}

// --- Building NetworkX's graph -------------------------------------------------

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet, PyString, PyTuple};

/// A key's Python type, as `GraphML.python_type` maps `attr.type`.
#[derive(Clone, Copy)]
enum Ty {
    Int,
    Float,
    Bool,
    Str,
}

struct KeyInfo {
    name: String,
    ty: Ty,
    for_: Option<String>,
}

/// `GraphML.convert_bool[text.lower()]` for ASCII text; `None` (NetworkX
/// raises, or non-ASCII case folding) otherwise.
fn convert_bool(text: &str) -> Option<bool> {
    if !text.is_ascii() {
        return None;
    }
    match text.to_ascii_lowercase().as_str() {
        "true" | "1" => Some(true),
        "false" | "0" => Some(false),
        _ => None,
    }
}

/// `int(text)` without calling Python for plain ASCII decimals that fit
/// an `i64` (Python's `int` also takes whitespace, underscores and other
/// Unicode digits; those go through it).
pub(crate) fn plain_int(text: &str) -> Option<i64> {
    let b = text.as_bytes();
    let digits = match b.first()? {
        b'+' | b'-' => &b[1..],
        _ => b,
    };
    if digits.is_empty() || digits.len() > 18 || !digits.iter().all(u8::is_ascii_digit) {
        return None;
    }
    text.parse().ok()
}

/// `float(text)` without calling Python for plain ASCII decimals
/// (`[+-]digits[.digits][e[+-]digits]`): Rust and Python both round
/// these correctly, so the floats are identical.
pub(crate) fn plain_float(text: &str) -> Option<f64> {
    let b = text.as_bytes();
    let mut i = usize::from(matches!(b.first(), Some(b'+' | b'-')));
    let start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    let mut mantissa = i - start;
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let frac = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        mantissa += i - frac;
    }
    if mantissa == 0 {
        return None;
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        let exp = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp {
            return None;
        }
    }
    if i != b.len() {
        return None;
    }
    text.parse().ok()
}

/// `data_type(text)` as NetworkX converts it (Python's own `int`, `float`
/// and `str`).
fn convert<'py>(py: Python<'py>, ty: Ty, text: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    match ty {
        Ty::Int => {
            if let Some(i) = plain_int(text) {
                return Ok(Some(i.into_pyobject(py)?.into_any()));
            }
        }
        Ty::Float => {
            if let Some(f) = plain_float(text) {
                return Ok(Some(pyo3::types::PyFloat::new(py, f).into_any()));
            }
        }
        _ => {}
    }
    let s = PyString::new(py, text);
    Ok(Some(match ty {
        Ty::Str => s.into_any(),
        Ty::Int => py.get_type::<pyo3::types::PyInt>().call1((s,))?,
        Ty::Float => py.get_type::<pyo3::types::PyFloat>().call1((s,))?,
        Ty::Bool => match convert_bool(text) {
            Some(b) => pyo3::types::PyBool::new(py, b).to_owned().into_any(),
            None => return Ok(None),
        },
    }))
}

/// `decode_data_elements`: the data dict of one element; `None` where
/// NetworkX would raise.
fn decode<'py>(
    py: Python<'py>,
    keys: &HashMap<&str, KeyInfo>,
    data: &[Data],
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let out = PyDict::new(py);
    for (key, text) in data {
        let Some(info) = key.as_deref().and_then(|k| keys.get(k)) else {
            return Ok(None); // "Bad GraphML data: no key ..."
        };
        let value = match text {
            None => PyString::new(py, "").into_any(),
            Some(t) => match convert(py, info.ty, t)? {
                Some(v) => v,
                None => return Ok(None),
            },
        };
        out.set_item(&info.name, value)?;
    }
    Ok(Some(out))
}

/// NetworkX's graph from a parsed document (`GraphMLReader.make_graph`, then
/// the conversion to `Graph`/`DiGraph` and the `id` edge attributes), built
/// into `classes = (Graph, DiGraph, MultiGraph, MultiDiGraph)`; `None` lets
/// NetworkX read the file (where it raises, warns or does something rustnx
/// doesn't replay). Python errors while converting values also mean `None`.
pub fn build<'py>(
    py: Python<'py>,
    doc: &Document,
    node_type: &Bound<'py, PyAny>,
    edge_key_type: &Bound<'py, PyAny>,
    force_multigraph: bool,
    classes: &Bound<'py, PyTuple>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match build_inner(py, doc, node_type, edge_key_type, force_multigraph, classes) {
        Ok(g) => Ok(g),
        Err(_) => Ok(None),
    }
}

type Keys<'a, 'py> = (HashMap<&'a str, KeyInfo>, Vec<(&'a str, Bound<'py, PyAny>)>);

/// `find_graphml_keys`: the keys by id and the converted defaults, in
/// document order; `None` where NetworkX raises, warns or reads yFiles.
fn read_keys<'a, 'py>(py: Python<'py>, doc: &'a Document) -> PyResult<Option<Keys<'a, 'py>>> {
    let mut keys: HashMap<&str, KeyInfo> = HashMap::new();
    let mut defaults: Vec<(&str, Bound<'py, PyAny>)> = Vec::new();
    for k in &doc.keys {
        if k.yfiles || keys.contains_key(k.id.as_str()) {
            return Ok(None); // yFiles keys; duplicate ids (dict order quirks)
        }
        let (Some(name), Some(ty)) = (&k.name, &k.ty) else {
            return Ok(None); // NetworkX raises, or warns and uses "string"
        };
        let ty = match ty.as_str() {
            "integer" | "int" | "long" => Ty::Int,
            "float" | "double" => Ty::Float,
            "boolean" => Ty::Bool,
            "string" | "yfiles" => Ty::Str,
            _ => return Ok(None),
        };
        if let Some(default) = &k.default {
            let value = match (default, ty) {
                (Some(t), _) => match convert(py, ty, t)? {
                    Some(v) => v,
                    None => return Ok(None),
                },
                (None, Ty::Str) => PyString::new(py, "None").into_any(), // str(None)
                (None, _) => return Ok(None),
            };
            defaults.push((k.id.as_str(), value));
        }
        keys.insert(
            k.id.as_str(),
            KeyInfo {
                name: name.clone(),
                ty,
                for_: k.for_.clone(),
            },
        );
    }
    Ok(Some((keys, defaults)))
}

/// `G.graph` before the graph's own data: `node_default` and
/// `edge_default` from the keys' defaults.
fn graph_attrs<'py>(
    py: Python<'py>,
    keys: &HashMap<&str, KeyInfo>,
    defaults: &[(&str, Bound<'py, PyAny>)],
) -> PyResult<Bound<'py, PyDict>> {
    let gattr = PyDict::new(py);
    let node_default = PyDict::new(py);
    let edge_default = PyDict::new(py);
    for (id, value) in defaults {
        let info = &keys[id];
        // python_type(value) again: the identity for these types.
        match info.for_.as_deref() {
            Some("node") => node_default.set_item(&info.name, value)?,
            Some("edge") => edge_default.set_item(&info.name, value)?,
            _ => {}
        }
    }
    gattr.set_item("node_default", &node_default)?;
    gattr.set_item("edge_default", &edge_default)?;
    Ok(gattr)
}

fn build_inner<'py>(
    py: Python<'py>,
    doc: &Document,
    node_type: &Bound<'py, PyAny>,
    edge_key_type: &Bound<'py, PyAny>,
    force_multigraph: bool,
    classes: &Bound<'py, PyTuple>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let Some((keys, defaults)) = read_keys(py, doc)? else { return Ok(None) };
    let graph = &doc.graphs[0];
    let directed = graph.edgedefault.as_deref() == Some("directed");
    let is_str_type = node_type.is(py.get_type::<PyString>());
    let is_int_type = node_type.is(py.get_type::<pyo3::types::PyInt>());
    if is_str_type || is_int_type {
        return build_fast(py, graph, &keys, &defaults, is_int_type, edge_key_type, force_multigraph, classes);
    }

    // make_graph on a fresh MultiGraph / MultiDiGraph, as plain dicts.
    let node = PyDict::new(py);
    let succ = PyDict::new(py);
    let pred = if directed { Some(PyDict::new(py)) } else { None };
    let gattr = graph_attrs(py, &keys, &defaults)?;

    let make_node = |id: &Option<String>| -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(id) = id else { return Ok(None) };
        let s = PyString::new(py, id);
        Ok(Some(node_type.call1((s,))?))
    };
    let add_node = |n: &Bound<'py, PyAny>| -> PyResult<()> {
        if !node.contains(n)? {
            node.set_item(n, PyDict::new(py))?;
            succ.set_item(n, PyDict::new(py))?;
            if let Some(p) = &pred {
                p.set_item(n, PyDict::new(py))?;
            }
        }
        Ok(())
    };

    for nx_node in &graph.nodes {
        let Some(n) = make_node(&nx_node.id)? else { return Ok(None) };
        let Some(data) = decode(py, &keys, &nx_node.data)? else { return Ok(None) };
        add_node(&n)?;
        node.get_item(&n)?.unwrap().cast_into::<PyDict>()?.update(data.as_mapping())?;
    }

    let edge_ids = PyDict::new(py);
    let mut multigraph = force_multigraph;
    for e in &graph.edges {
        match (directed, e.directed.as_deref()) {
            (true, Some("false")) | (false, Some("true")) => return Ok(None),
            _ => {}
        }
        let Some(u) = make_node(&e.source)? else { return Ok(None) };
        let Some(v) = make_node(&e.target)? else { return Ok(None) };
        let Some(data) = decode(py, &keys, &e.data)? else { return Ok(None) };
        let key: Bound<'py, PyAny> = match e.id.as_deref().filter(|s| !s.is_empty()) {
            Some(id) => {
                let raw = PyString::new(py, id);
                edge_ids.set_item(PyTuple::new(py, [&u, &v])?, &raw)?;
                match edge_key_type.call1((&raw,)) {
                    Ok(k) => k,
                    Err(err) if err.is_instance_of::<pyo3::exceptions::PyValueError>(py) => {
                        raw.into_any()
                    }
                    Err(err) => return Err(err),
                }
            }
            None => match data.get_item("key")? {
                Some(k) => k,
                None => py.None().into_bound(py),
            },
        };
        // G.has_edge(source, target)
        let has = match succ.get_item(&u)? {
            Some(row) => row.cast_into::<PyDict>()?.contains(&v)?,
            None => false,
        };
        if has {
            multigraph = true;
        }
        // add_edges_from([(u, v, key, data)]): add_edge(u, v, key), then
        // update the edge's dict with data.
        add_node(&u)?;
        add_node(&v)?;
        let row_u = succ.get_item(&u)?.unwrap().cast_into::<PyDict>()?;
        let keydict = match row_u.get_item(&v)? {
            Some(kd) => kd.cast_into::<PyDict>()?,
            None => {
                let kd = PyDict::new(py);
                row_u.set_item(&v, &kd)?;
                let back = match &pred {
                    Some(p) => p.get_item(&v)?.unwrap().cast_into::<PyDict>()?,
                    None => succ.get_item(&v)?.unwrap().cast_into::<PyDict>()?,
                };
                back.set_item(&u, &kd)?;
                kd
            }
        };
        let key = if key.is_none() {
            // new_edge_key
            let mut k = keydict.len();
            while keydict.contains(k)? {
                k += 1;
            }
            k.into_pyobject(py)?.into_any()
        } else {
            key
        };
        let datadict = match keydict.get_item(&key)? {
            Some(d) => d.cast_into::<PyDict>()?,
            None => {
                let d = PyDict::new(py);
                keydict.set_item(&key, &d)?;
                d
            }
        };
        datadict.update(data.as_mapping())?;
    }
    let Some(gdata) = decode(py, &keys, &graph.data)? else { return Ok(None) };
    gattr.update(gdata.as_mapping())?;

    let class_index = match (multigraph, directed) {
        (false, false) => 0,
        (false, true) => 1,
        (true, false) => 2,
        (true, true) => 3,
    };
    let g = classes.get_item(class_index)?.call0()?;
    let set = |name: &str, value: &Bound<'py, PyDict>| g.setattr(name, value);
    if multigraph {
        g.getattr("graph")?.cast_into::<PyDict>()?.update(gattr.as_mapping())?;
        set("_node", &node)?;
        set("_adj", &succ)?;
        if let Some(p) = &pred {
            set("_succ", &succ)?;
            set("_pred", p)?;
        }
        return Ok(Some(g));
    }

    // nx.Graph(G) / nx.DiGraph(G): from_dict_of_dicts(G.adj,
    // multigraph_input=True), then the graph and node attributes.
    let hnode = PyDict::new(py);
    let hsucc = PyDict::new(py);
    let hpred = if directed { Some(PyDict::new(py)) } else { None };
    for n in node.keys() {
        hnode.set_item(&n, PyDict::new(py))?;
        hsucc.set_item(&n, PyDict::new(py))?;
        if let Some(p) = &hpred {
            p.set_item(&n, PyDict::new(py))?;
        }
    }
    let seen = PySet::empty(py)?;
    for (u, nbrs) in succ.iter() {
        let nbrs = nbrs.cast_into::<PyDict>()?;
        let row_u = hsucc.get_item(&u)?.unwrap().cast_into::<PyDict>()?;
        for (v, keydict) in nbrs.iter() {
            if !directed {
                let uv = PyTuple::new(py, [&u, &v])?;
                if seen.contains(&uv)? {
                    continue;
                }
            }
            for (_, data) in keydict.cast_into::<PyDict>()?.iter() {
                let datadict = match row_u.get_item(&v)? {
                    Some(d) => d.cast_into::<PyDict>()?,
                    None => PyDict::new(py),
                };
                datadict.update(data.cast_into::<PyDict>()?.as_mapping())?;
                row_u.set_item(&v, &datadict)?;
                let back = match &hpred {
                    Some(p) => p.get_item(&v)?.unwrap().cast_into::<PyDict>()?,
                    None => hsucc.get_item(&v)?.unwrap().cast_into::<PyDict>()?,
                };
                back.set_item(&u, &datadict)?;
            }
            if !directed {
                seen.add(PyTuple::new(py, [&v, &u])?)?;
            }
        }
    }
    g.getattr("graph")?.cast_into::<PyDict>()?.update(gattr.as_mapping())?;
    for (n, dd) in node.iter() {
        hnode.get_item(&n)?.unwrap().cast_into::<PyDict>()?.update(dd.cast_into::<PyDict>()?.as_mapping())?;
    }
    // nx.set_edge_attributes(G, values=edge_ids, name="id")
    for (uv, value) in edge_ids.iter() {
        let uv = uv.cast_into::<PyTuple>()?;
        let (u, v) = (uv.get_item(0)?, uv.get_item(1)?);
        if let Some(row) = hsucc.get_item(&u)? {
            if let Some(d) = row.cast_into::<PyDict>()?.get_item(&v)? {
                d.cast_into::<PyDict>()?.set_item("id", value)?;
            }
        }
    }
    set("_node", &hnode)?;
    set("_adj", &hsucc)?;
    if let Some(p) = &hpred {
        set("_succ", &hsucc)?;
        set("_pred", p)?;
    }
    Ok(Some(g))
}

/// An edge of the document, with its endpoints as node indices.
struct EdgeRec<'a, 'py> {
    u: usize,
    v: usize,
    data: Bound<'py, PyDict>,
    id: Option<&'a str>,
}

/// Node numbering for `build_fast`: nodes in `G._node` order, with their
/// attribute dicts.
struct Interner<'a, 'py> {
    py: Python<'py>,
    int_nodes: bool,
    nodes: Vec<Bound<'py, PyAny>>,
    data: Vec<Bound<'py, PyDict>>,
    by_text: HashMap<&'a str, usize>,
    /// int nodes: value -> index (different texts can give one int).
    by_value: Bound<'py, PyDict>,
}

impl<'a, 'py> Interner<'a, 'py> {
    /// The index of `node_type(id)`, added if new; `None` where NetworkX
    /// raises.
    fn intern(&mut self, id: &'a Option<String>) -> PyResult<Option<usize>> {
        let py = self.py;
        // node_type(None): str gives "None", int raises.
        let text = match id {
            Some(t) => t.as_str(),
            None if self.int_nodes => return Ok(None),
            None => "None",
        };
        if let Some(&i) = self.by_text.get(text) {
            return Ok(Some(i));
        }
        let i = if self.int_nodes {
            let value = match plain_int(text) {
                Some(i) => i.into_pyobject(py)?.into_any(),
                None => py.get_type::<pyo3::types::PyInt>().call1((text,))?,
            };
            match self.by_value.get_item(&value)? {
                Some(i) => i.extract::<usize>()?,
                None => {
                    self.by_value.set_item(&value, self.nodes.len())?;
                    self.push(value)
                }
            }
        } else {
            self.push(PyString::new(py, text).into_any())
        };
        self.by_text.insert(text, i);
        Ok(Some(i))
    }

    fn push(&mut self, node: Bound<'py, PyAny>) -> usize {
        self.nodes.push(node);
        self.data.push(PyDict::new(self.py));
        self.nodes.len() - 1
    }
}

/// `build_inner` for `node_type` `str` or `int`, whose equal nodes are
/// indistinguishable: nodes are numbered in the order NetworkX adds them,
/// and the graph is built from those numbers. A simple graph is built
/// directly in the order `nx.Graph(G)` / `nx.DiGraph(G)` would build it
/// from NetworkX's multigraph, without building the multigraph.
#[allow(clippy::too_many_arguments)]
fn build_fast<'a, 'py>(
    py: Python<'py>,
    graph: &'a Graph,
    keys: &HashMap<&str, KeyInfo>,
    defaults: &[(&str, Bound<'py, PyAny>)],
    int_nodes: bool,
    edge_key_type: &Bound<'py, PyAny>,
    force_multigraph: bool,
    classes: &Bound<'py, PyTuple>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let directed = graph.edgedefault.as_deref() == Some("directed");
    let gattr = graph_attrs(py, keys, defaults)?;

    let mut nodes = Interner {
        py,
        int_nodes,
        nodes: Vec::new(),
        data: Vec::new(),
        by_text: HashMap::new(),
        by_value: PyDict::new(py),
    };
    for n in &graph.nodes {
        let Some(i) = nodes.intern(&n.id)? else { return Ok(None) };
        let Some(data) = decode(py, keys, &n.data)? else { return Ok(None) };
        nodes.data[i].update(data.as_mapping())?;
    }

    // The edges, and whether NetworkX finds a parallel edge (has_edge
    // before adding each one).
    let key_is_int = edge_key_type.is(py.get_type::<pyo3::types::PyInt>());
    let key_is_str = edge_key_type.is(py.get_type::<PyString>());
    let mut edges: Vec<EdgeRec<'a, 'py>> = Vec::with_capacity(graph.edges.len());
    let mut present: HashSet<(usize, usize)> = HashSet::with_capacity(graph.edges.len());
    let mut multigraph = force_multigraph;
    for e in &graph.edges {
        match (directed, e.directed.as_deref()) {
            (true, Some("false")) | (false, Some("true")) => return Ok(None),
            _ => {}
        }
        // Evaluated as NetworkX does: source, target, data, then the key.
        let Some(u) = nodes.intern(&e.source)? else { return Ok(None) };
        let Some(v) = nodes.intern(&e.target)? else { return Ok(None) };
        let Some(data) = decode(py, keys, &e.data)? else { return Ok(None) };
        let id = e.id.as_deref().filter(|s| !s.is_empty());
        if let Some(id) = id {
            // A custom edge_key_type can raise or have side effects; int
            // raises only ValueError (caught) and str never raises.
            if !(key_is_int || key_is_str) {
                if let Err(err) = edge_key_type.call1((id,)) {
                    if !err.is_instance_of::<pyo3::exceptions::PyValueError>(py) {
                        return Err(err);
                    }
                }
            }
        }
        let pair = if directed || u <= v { (u, v) } else { (v, u) };
        if !present.insert(pair) {
            multigraph = true;
        }
        edges.push(EdgeRec { u, v, data, id });
    }
    let Some(gdata) = decode(py, keys, &graph.data)? else { return Ok(None) };
    gattr.update(gdata.as_mapping())?;

    let Interner { nodes, data: node_data, .. } = nodes;
    let n = nodes.len();
    let node_dict = PyDict::new(py);
    for (node, data) in nodes.iter().zip(&node_data) {
        node_dict.set_item(node, data)?;
    }
    let class_index = match (multigraph, directed) {
        (false, false) => 0,
        (false, true) => 1,
        (true, false) => 2,
        (true, true) => 3,
    };
    let g = classes.get_item(class_index)?.call0()?;
    g.getattr("graph")?.cast_into::<PyDict>()?.update(gattr.as_mapping())?;
    let succ_rows: Vec<Bound<'py, PyDict>> = (0..n).map(|_| PyDict::new(py)).collect();
    let pred_rows: Vec<Bound<'py, PyDict>> =
        if directed { (0..n).map(|_| PyDict::new(py)).collect() } else { Vec::new() };

    if multigraph {
        // make_graph's add_edges_from([(u, v, key, data)]) for each edge.
        for e in &edges {
            let key: Bound<'py, PyAny> = match e.id {
                Some(id) => {
                    let raw = PyString::new(py, id);
                    match edge_key_type.call1((&raw,)) {
                        Ok(k) => k,
                        Err(err) if err.is_instance_of::<pyo3::exceptions::PyValueError>(py) => {
                            raw.into_any()
                        }
                        Err(err) => return Err(err),
                    }
                }
                None => match e.data.get_item("key")? {
                    Some(k) => k,
                    None => py.None().into_bound(py),
                },
            };
            let row_u = &succ_rows[e.u];
            let v = &nodes[e.v];
            let keydict = match row_u.get_item(v)? {
                Some(kd) => kd.cast_into::<PyDict>()?,
                None => {
                    let kd = PyDict::new(py);
                    row_u.set_item(v, &kd)?;
                    let back = if directed { &pred_rows[e.v] } else { &succ_rows[e.v] };
                    back.set_item(&nodes[e.u], &kd)?;
                    kd
                }
            };
            let key = if key.is_none() {
                // new_edge_key
                let mut k = keydict.len();
                while keydict.contains(k)? {
                    k += 1;
                }
                k.into_pyobject(py)?.into_any()
            } else {
                key
            };
            match keydict.get_item(&key)? {
                Some(d) => d.cast_into::<PyDict>()?.update(e.data.as_mapping())?,
                None => keydict.set_item(&key, &e.data)?,
            }
        }
    } else {
        // The multigraph's adjacency order: each edge is new, so a row
        // lists edges in the order they touch the node.
        let mut madj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        for (k, e) in edges.iter().enumerate() {
            madj[e.u].push((e.v, k));
            if !directed && e.u != e.v {
                madj[e.v].push((e.u, k));
            }
        }
        // from_dict_of_dicts(G.adj, multigraph_input=True): rows in node
        // order; undirected, (u, v) is skipped once v's row added it.
        for (u, row) in madj.iter().enumerate() {
            for &(v, k) in row {
                if !directed && v < u {
                    continue;
                }
                let data = &edges[k].data;
                succ_rows[u].set_item(&nodes[v], data)?;
                if directed {
                    pred_rows[v].set_item(&nodes[u], data)?;
                } else if u != v {
                    succ_rows[v].set_item(&nodes[u], data)?;
                }
            }
        }
        // nx.set_edge_attributes(G, values=edge_ids, name="id"): each
        // (source, target) is one edge here.
        for e in &edges {
            if let Some(id) = e.id {
                e.data.set_item("id", id)?;
            }
        }
    }
    let adj = PyDict::new(py);
    for (node, row) in nodes.iter().zip(&succ_rows) {
        adj.set_item(node, row)?;
    }
    g.setattr("_node", &node_dict)?;
    g.setattr("_adj", &adj)?;
    if directed {
        let pred = PyDict::new(py);
        for (node, row) in nodes.iter().zip(&pred_rows) {
            pred.set_item(node, row)?;
        }
        g.setattr("_succ", &adj)?;
        g.setattr("_pred", &pred)?;
    }
    Ok(Some(g))
}
