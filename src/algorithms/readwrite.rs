//! Readers and parsers (batch 20): NetworkX's text formats parsed into a
//! list of graph operations (`Parsed`), which `lib.rs` replays onto a
//! NetworkX graph's dicts the way `add_node` / `add_edge` would.
//!
//! Every parser returns `None` ("bail") wherever NetworkX would raise, or
//! where Python's own conversions (`int`, `float`, `literal_eval`,
//! `shlex.split`) might do something this module doesn't model; the Python
//! side then reruns NetworkX's code on the same input. `apply` (which needs
//! the GIL) builds the result.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyString};
use std::collections::{HashMap, HashSet};

/// A node or attribute value.
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    /// GML's nested values: lists, dicts and `()`.
    List(Vec<Val>),
    Dict(Vec<(String, Val)>),
    EmptyTuple,
}

/// One `add_node(n, **attrs)` or `add_edge(u, v, **attrs)` call; attribute
/// keys index `Parsed::keys`.
#[derive(Debug)]
pub enum Op {
    Node(u32, Vec<(u32, Val)>),
    Edge(u32, u32, Vec<(u32, Val)>),
    /// `add_edge(u, v, key, **attrs)` on a multigraph, with a key not yet
    /// used between `u` and `v`.
    KeyedEdge(u32, u32, Val, Vec<(u32, Val)>),
}

#[derive(Debug, Default)]
pub struct Parsed {
    /// Distinct nodes, by index.
    pub nodes: Vec<Val>,
    /// Attribute names, by index.
    pub keys: Vec<String>,
    pub ops: Vec<Op>,
    /// For formats that pick the graph class: (multigraph, directed).
    pub class: (bool, bool),
    /// `G.graph["name"]`, if the format sets it.
    pub name: Option<String>,
    /// Other `G.graph` items, in order.
    pub graph_attrs: Vec<(u32, Val)>,
}

#[derive(Hash, PartialEq, Eq)]
enum NodeKey {
    Int(i64),
    Float(u64),
}

#[derive(Default)]
struct Builder {
    parsed: Parsed,
    str_index: HashMap<String, u32>,
    index: HashMap<NodeKey, u32>,
    key_index: HashMap<String, u32>,
}

impl Builder {
    fn node(&mut self, value: Val) -> u32 {
        let key = match &value {
            Val::Str(s) => return self.str_node(&s.clone()),
            Val::Int(i) => NodeKey::Int(*i),
            // 0.0 == -0.0 as dict keys (the first one seen is kept).
            Val::Float(f) => NodeKey::Float(if *f == 0.0 { 0 } else { f.to_bits() }),
            _ => unreachable!("nodes are str, int or float"),
        };
        let next = self.parsed.nodes.len() as u32;
        *self.index.entry(key).or_insert_with(|| {
            self.parsed.nodes.push(value);
            next
        })
    }

    fn str_node(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.str_index.get(s) {
            return i;
        }
        let i = self.parsed.nodes.len() as u32;
        self.parsed.nodes.push(Val::Str(s.to_string()));
        self.str_index.insert(s.to_string(), i);
        i
    }

    fn key(&mut self, k: &str) -> u32 {
        if let Some(&i) = self.key_index.get(k) {
            return i;
        }
        let i = self.parsed.keys.len() as u32;
        self.parsed.keys.push(k.to_string());
        self.key_index.insert(k.to_string(), i);
        i
    }

    fn finish(self) -> Parsed {
        self.parsed
    }
}

// --- Python string semantics ---------------------------------------------------

/// `str.isspace()` for one character (what `str.split()` and `str.strip()`
/// treat as whitespace).
pub fn py_isspace(c: char) -> bool {
    matches!(
        c,
        '\t'..='\r'
            | '\x1c'..='\x1f'
            | ' '
            | '\u{85}'
            | '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
    )
}

/// `s.split(delimiter)`.
pub fn py_split<'a>(s: &'a str, delimiter: Option<&str>) -> Vec<&'a str> {
    match delimiter {
        Some(d) => s.split(d).collect(),
        None => s.split(py_isspace).filter(|t| !t.is_empty()).collect(),
    }
}

/// `s.split(None, 1)`.
fn py_split_once(s: &str) -> Vec<&str> {
    let s = s.trim_start_matches(py_isspace);
    if s.is_empty() {
        return Vec::new();
    }
    match s.find(py_isspace) {
        None => vec![s],
        Some(p) => {
            let first = &s[..p];
            let rest = s[p..].trim_start_matches(py_isspace);
            if rest.is_empty() {
                vec![first]
            } else {
                vec![first, rest]
            }
        }
    }
}

fn py_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}

/// `int(s)` for a str: `Some(Ok(value))`, or `None` when unsure (non-ASCII
/// digits, underscores, values beyond i64) or when `int` would raise.
pub fn py_int(s: &str) -> Option<i64> {
    let t = py_strip(s);
    let digits = t.strip_prefix(['+', '-']).unwrap_or(t);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    t.parse::<i64>().ok()
}

/// What `float(s)` does for a str.
#[derive(Debug, PartialEq)]
pub enum FloatParse {
    Ok(f64),
    /// `float` raises ValueError.
    Invalid,
    /// Something this parser doesn't model (non-ASCII, underscores).
    Unsure,
}

pub fn py_float(s: &str) -> FloatParse {
    let t = py_strip(s);
    if !t.is_ascii() || t.contains('_') {
        return FloatParse::Unsure;
    }
    let (negative, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let lower = body.to_ascii_lowercase();
    if lower == "inf" || lower == "infinity" {
        return FloatParse::Ok(if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        });
    }
    if lower == "nan" {
        return FloatParse::Ok(if negative { -f64::NAN } else { f64::NAN });
    }
    if decimal_len(body.as_bytes()) != Some(body.len()) {
        return FloatParse::Invalid;
    }
    match t.parse::<f64>() {
        Ok(f) => FloatParse::Ok(f),
        Err(_) => FloatParse::Invalid,
    }
}

/// Length of the longest prefix of `b` of the form
/// `digits ['.' digits] [e [sign] digits]` with at least one mantissa digit,
/// or `None` if there's no mantissa digit.
fn decimal_len(b: &[u8]) -> Option<usize> {
    let mut i = 0;
    let mut mantissa = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
        mantissa += 1;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
            mantissa += 1;
        }
    }
    if mantissa == 0 {
        return None;
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let mut j = i + 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        let start = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > start {
            i = j;
        }
    }
    Some(i)
}

/// Whether `c` could continue a Python identifier or number token.
fn is_word_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c >= 0x80
}

// --- ast.literal_eval for flat dicts ---------------------------------------------

struct Lit<'a> {
    b: &'a [u8],
    s: &'a str,
    pos: usize,
}

impl Lit<'_> {
    fn skip_ws(&mut self) {
        while self.pos < self.b.len() && matches!(self.b[self.pos], b' ' | b'\t') {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }

    /// A plain one-line string literal without escapes; not followed by
    /// another string literal (which Python would concatenate).
    fn string(&mut self) -> Option<String> {
        let quote = self.peek()?;
        let start = self.pos + 1;
        let mut i = start;
        loop {
            let c = *self.b.get(i)?;
            if c == quote {
                break;
            }
            if matches!(c, b'\\' | b'\n' | b'\r' | 0) {
                return None;
            }
            i += 1;
        }
        let value = self.s[start..i].to_string();
        self.pos = i + 1;
        let after = self.pos;
        self.skip_ws();
        if matches!(self.peek(), Some(b'\'' | b'"')) {
            return None;
        }
        self.pos = after;
        Some(value)
    }

    fn word(&mut self, w: &str) -> bool {
        let end = self.pos + w.len();
        if self.b.len() >= end
            && &self.b[self.pos..end] == w.as_bytes()
            && !self.b.get(end).is_some_and(|&c| is_word_char(c))
        {
            self.pos = end;
            true
        } else {
            false
        }
    }

    fn number(&mut self) -> Option<Val> {
        let start = self.pos;
        let mut i = self.pos;
        if matches!(self.b[i], b'+' | b'-') {
            i += 1;
        }
        let body = &self.b[i..];
        let len = decimal_len(body)?;
        let text = &body[..len];
        let end = i + len;
        if self
            .b
            .get(end)
            .is_some_and(|&c| is_word_char(c) || c == b'.')
        {
            return None;
        }
        self.pos = end;
        let literal = &self.s[start..end];
        if text.iter().all(|c| c.is_ascii_digit()) {
            // Python rejects leading zeros in decimal ints ("00" is fine).
            if text.len() > 1 && text[0] == b'0' && text.iter().any(|&c| c != b'0') {
                return None;
            }
            literal.parse::<i64>().ok().map(Val::Int)
        } else {
            literal.parse::<f64>().ok().map(Val::Float)
        }
    }

    fn value(&mut self) -> Option<Val> {
        match self.peek()? {
            b'\'' | b'"' => self.string().map(Val::Str),
            b'0'..=b'9' | b'.' | b'+' | b'-' => self.number(),
            _ => {
                if self.word("True") {
                    Some(Val::Bool(true))
                } else if self.word("False") {
                    Some(Val::Bool(false))
                } else if self.word("None") {
                    Some(Val::None)
                } else {
                    None
                }
            }
        }
    }
}

/// `dict(ast.literal_eval(s))` for a flat dict display with str keys and
/// str/int/float/bool/None values (leading spaces and tabs allowed, as
/// `literal_eval` strips them). `None` for anything else.
pub fn literal_dict(s: &str) -> Option<Vec<(String, Val)>> {
    let mut p = Lit {
        b: s.as_bytes(),
        s,
        pos: 0,
    };
    p.skip_ws();
    if p.peek()? != b'{' {
        return None;
    }
    p.pos += 1;
    p.skip_ws();
    let mut out = Vec::new();
    if p.peek()? == b'}' {
        p.pos += 1;
    } else {
        loop {
            if !matches!(p.peek()?, b'\'' | b'"') {
                return None;
            }
            let key = p.string()?;
            p.skip_ws();
            if p.peek()? != b':' {
                return None;
            }
            p.pos += 1;
            p.skip_ws();
            let value = p.value()?;
            out.push((key, value));
            p.skip_ws();
            match p.peek()? {
                b',' => {
                    p.pos += 1;
                    p.skip_ws();
                    if p.peek()? == b'}' {
                        p.pos += 1;
                        break;
                    }
                }
                b'}' => {
                    p.pos += 1;
                    break;
                }
                _ => return None,
            }
        }
    }
    p.skip_ws();
    if p.pos != p.b.len() {
        return None;
    }
    Some(out)
}

/// Keyword names that `add_edge(u, v, **data)` would bind to a parameter
/// (or reject) instead of storing as an attribute.
fn reserved(key: &str) -> bool {
    matches!(
        key,
        "self" | "u_of_edge" | "v_of_edge" | "u_for_edge" | "v_for_edge" | "key"
    )
}

// --- edge lists and adjacency lists --------------------------------------------

/// How node strings are converted (`nodetype`).
#[derive(Clone, Copy, PartialEq)]
pub enum NodeType {
    Str,
    Int,
    Float,
}

impl NodeType {
    pub fn from_code(code: u8) -> Option<NodeType> {
        match code {
            0 => Some(NodeType::Str),
            1 => Some(NodeType::Int),
            2 => Some(NodeType::Float),
            _ => None,
        }
    }
}

/// `int`, `float` or `str` applied to a token.
fn convert(token: &str, kind: NodeType) -> Option<Val> {
    match kind {
        NodeType::Str => Some(Val::Str(token.to_string())),
        NodeType::Int => py_int(token).map(Val::Int),
        NodeType::Float => match py_float(token) {
            FloatParse::Ok(f) => Some(Val::Float(f)),
            _ => None,
        },
    }
}

fn node_of(b: &mut Builder, token: &str, kind: NodeType) -> Option<u32> {
    if kind == NodeType::Str {
        return Some(b.str_node(token));
    }
    match convert(token, kind)? {
        // NaN != NaN: every NaN would be its own node. -0.0 == 0.0, but
        // NetworkX keeps whichever object each row's dict saw first.
        Val::Float(f) if f.is_nan() || (f == 0.0 && f.is_sign_negative()) => None,
        v => Some(b.node(v)),
    }
}

/// Edge data handling in `parse_edgelist`.
pub enum EdgeData {
    /// `data=False`.
    Ignore,
    /// `data=True`: `literal_eval` of the remaining tokens joined by ","
    /// (if `comma`) or " ", then `strip()`ped if `strip`.
    Literal { comma: bool, strip: bool },
    /// `data=[(key, type), ...]`.
    Typed(Vec<(String, NodeType)>),
}

/// `parse_edgelist` (and bipartite `parse_edgelist` with `bipartite`,
/// which adds the `bipartite` node attribute and always strips comments).
pub fn edgelist(
    lines: &[&str],
    comments: Option<&str>,
    delimiter: Option<&str>,
    nodetype: NodeType,
    data: &EdgeData,
    bipartite: bool,
) -> Option<Parsed> {
    let mut b = Builder::default();
    let typed_keys: Vec<u32> = match data {
        EdgeData::Typed(spec) => spec.iter().map(|(k, _)| b.key(k)).collect(),
        _ => Vec::new(),
    };
    let side_key = if bipartite { b.key("bipartite") } else { 0 };
    for &raw in lines {
        let mut line = raw;
        if let Some(c) = comments {
            if let Some(p) = line.find(c) {
                line = &line[..p];
            }
            if line.is_empty() {
                continue;
            }
        }
        let s = py_split(line.trim_end_matches('\n'), delimiter);
        if s.len() < 2 {
            continue;
        }
        let u = node_of(&mut b, s[0], nodetype)?;
        let v = node_of(&mut b, s[1], nodetype)?;
        let d = &s[2..];
        let mut attrs = Vec::new();
        if !d.is_empty() {
            match data {
                EdgeData::Ignore => {}
                EdgeData::Literal { comma, strip } => {
                    let joined = d.join(if *comma { "," } else { " " });
                    let text = if *strip { py_strip(&joined) } else { &joined };
                    for (k, value) in literal_dict(text)? {
                        if reserved(&k) {
                            return None;
                        }
                        attrs.push((b.key(&k), value));
                    }
                }
                EdgeData::Typed(spec) => {
                    if d.len() != spec.len() {
                        return None;
                    }
                    for ((&key, (_, kind)), token) in typed_keys.iter().zip(spec).zip(d) {
                        attrs.push((key, convert(token, *kind)?));
                    }
                }
            }
        }
        if bipartite {
            b.parsed
                .ops
                .push(Op::Node(u, vec![(side_key, Val::Int(0))]));
            b.parsed
                .ops
                .push(Op::Node(v, vec![(side_key, Val::Int(1))]));
        }
        b.parsed.ops.push(Op::Edge(u, v, attrs));
    }
    Some(b.finish())
}

fn strip_comment<'a>(line: &'a str, comments: &str) -> &'a str {
    match line.find(comments) {
        Some(p) => &line[..p],
        None => line,
    }
}

/// `parse_adjlist`.
pub fn adjlist(
    lines: &[&str],
    comments: &str,
    delimiter: Option<&str>,
    nodetype: NodeType,
) -> Option<Parsed> {
    let mut b = Builder::default();
    for &raw in lines {
        let line = strip_comment(raw, comments);
        if line.is_empty() {
            continue;
        }
        let vlist = py_split(line.trim_end_matches('\n'), delimiter);
        let (&first, rest) = vlist.split_first()?;
        let u = node_of(&mut b, first, nodetype)?;
        b.parsed.ops.push(Op::Node(u, Vec::new()));
        let vs = rest
            .iter()
            .map(|t| node_of(&mut b, t, nodetype))
            .collect::<Option<Vec<u32>>>()?;
        for v in vs {
            b.parsed.ops.push(Op::Edge(u, v, Vec::new()));
        }
    }
    Some(b.finish())
}

/// `parse_multiline_adjlist`; `edgetype` is `None` for `literal_eval`.
pub fn multiline_adjlist(
    lines: &[&str],
    comments: &str,
    delimiter: Option<&str>,
    nodetype: NodeType,
    edgetype: Option<NodeType>,
) -> Option<Parsed> {
    let mut b = Builder::default();
    let weight = b.key("weight");
    let mut i = 0;
    while i < lines.len() {
        let line = strip_comment(lines[i], comments);
        i += 1;
        if line.is_empty() {
            continue;
        }
        let head = py_split(line.trim_end_matches('\n'), delimiter);
        if head.len() != 2 {
            return None;
        }
        let deg = py_int(head[1])?;
        let u = node_of(&mut b, head[0], nodetype)?;
        b.parsed.ops.push(Op::Node(u, Vec::new()));
        for _ in 0..deg.max(0) {
            let line = loop {
                let line = strip_comment(lines.get(i)?, comments);
                i += 1;
                if !line.is_empty() {
                    break line;
                }
            };
            let vlist = py_split(line.trim_end_matches('\n'), delimiter);
            let Some((&first, rest)) = vlist.split_first() else {
                continue; // isolated node
            };
            let data = rest.concat();
            let v = node_of(&mut b, first, nodetype)?;
            let attrs = match edgetype {
                Some(kind) => vec![(weight, convert(&data, kind)?)],
                // literal_eval("") raises, and NetworkX then uses {}.
                None if data.is_empty() => Vec::new(),
                None => {
                    let mut attrs = Vec::new();
                    for (k, value) in literal_dict(&data)? {
                        if reserved(&k) {
                            return None;
                        }
                        attrs.push((b.key(&k), value));
                    }
                    attrs
                }
            };
            b.parsed.ops.push(Op::Edge(u, v, attrs));
        }
    }
    Some(b.finish())
}

// --- LEDA -----------------------------------------------------------------------

/// `parse_leda`, from its lines (a str input already split at "\n").
pub fn leda(lines: &[&str]) -> Option<Parsed> {
    let kept: Vec<&str> = lines
        .iter()
        .filter(|l| !(l.starts_with('#') || l.starts_with('\n') || l.is_empty()))
        .map(|l| l.trim_end_matches('\n'))
        .collect();
    let mut it = kept.into_iter();
    for _ in 0..3 {
        it.next()?;
    }
    let du = py_int(it.next()?)?;
    let mut b = Builder::default();
    b.parsed.class = (false, du == -1);
    let n = py_int(it.next()?)?;
    let mut node = Vec::new();
    for i in 1..=n.max(0) {
        let raw = it.next()?;
        let symbol = raw
            .trim_end_matches(py_isspace)
            .trim_matches(['|', '{', '}', ' ']);
        let id = if symbol.is_empty() {
            b.str_node(&i.to_string())
        } else {
            b.str_node(symbol)
        };
        node.push(id);
    }
    for &id in &node {
        b.parsed.ops.push(Op::Node(id, Vec::new()));
    }
    let label_key = b.key("label");
    let m = py_int(it.next()?)?;
    for _ in 0..m.max(0) {
        let fields = py_split(it.next()?, None);
        if fields.len() != 4 {
            return None;
        }
        let lookup = |t: &str| -> Option<u32> {
            let i = py_int(t)?;
            if i >= 1 && i <= node.len() as i64 {
                Some(node[i as usize - 1])
            } else {
                None
            }
        };
        let s = lookup(fields[0])?;
        let t = lookup(fields[1])?;
        // label[2:-2], counted in characters.
        let chars: Vec<char> = fields[3].chars().collect();
        let label: String = if chars.len() > 4 {
            chars[2..chars.len() - 2].iter().collect()
        } else {
            String::new()
        };
        b.parsed
            .ops
            .push(Op::Edge(s, t, vec![(label_key, Val::Str(label))]));
    }
    Some(b.finish())
}

// --- Pajek ----------------------------------------------------------------------

/// Whether `s.lower().startswith(prefix)` for an ASCII lowercase `prefix`.
/// Only ASCII letters, U+0130 and the Kelvin sign lowercase to something
/// starting with an ASCII letter.
fn lower_starts_with(s: &str, prefix: &str) -> bool {
    let mut want = prefix.chars();
    let mut pending: Option<char> = None;
    let mut chars = s.chars();
    loop {
        let Some(w) = want.next() else {
            return true;
        };
        let c = match pending.take() {
            Some(c) => c,
            None => match chars.next() {
                None => return false,
                Some('\u{130}') => {
                    pending = Some('\u{307}');
                    'i'
                }
                Some('\u{212a}') => 'k',
                Some(c) => c.to_ascii_lowercase(),
            },
        };
        if c != w {
            return false;
        }
    }
}

/// `shlex.split(s)` (POSIX mode, no comments): tokens split at space, tab,
/// CR and LF; quotes group (and join adjacent text), a backslash escapes
/// the next character outside quotes and `"` or `\\` inside double quotes.
/// `None` where shlex raises (an unclosed quote, a trailing backslash).
fn shlex_split(s: &str) -> Option<Vec<String>> {
    let mut tokens = Vec::new();
    let mut token = String::new();
    let mut in_token = false;
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        match c {
            ' ' | '\t' | '\r' | '\n' => {
                if in_token {
                    tokens.push(std::mem::take(&mut token));
                    in_token = false;
                }
            }
            '\\' => {
                token.push(chars.next()?);
                in_token = true;
            }
            '\'' | '"' => {
                in_token = true;
                loop {
                    let q = chars.next()?;
                    if q == c {
                        break;
                    }
                    if c == '"' && q == '\\' {
                        let e = chars.next()?;
                        if e != '"' && e != '\\' {
                            token.push('\\');
                        }
                        token.push(e);
                    } else {
                        token.push(q);
                    }
                }
            }
            _ => {
                token.push(c);
                in_token = true;
            }
        }
    }
    if in_token {
        tokens.push(token);
    }
    Some(tokens)
}

/// `parse_pajek` for files with vertices and one `*edges` or `*arcs`
/// section (`*matrix` bails).
pub fn pajek(lines: &[&str]) -> Option<Parsed> {
    let lines: Vec<&str> = lines.iter().map(|l| l.trim_end_matches('\n')).collect();
    let mut b = Builder::default();
    b.parsed.class = (true, true); // MultiDiGraph
    let id_key = b.key("id");
    let (x_key, y_key, shape_key, weight_key) =
        (b.key("x"), b.key("y"), b.key("shape"), b.key("weight"));
    let mut nodelabels: Option<HashMap<String, u32>> = None;
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i];
        i += 1;
        if lower_starts_with(l, "*network") {
            let parts = py_split_once(l);
            if parts.len() == 2 {
                b.parsed.name = Some(parts[1].to_string());
            }
        } else if lower_starts_with(l, "*vertices") {
            let mut labels = HashMap::new();
            let head = py_split(l, None);
            if head.len() != 2 {
                return None;
            }
            for _ in 0..py_int(head[1])?.max(0) {
                let line = *lines.get(i)?;
                i += 1;
                let split = shlex_split(line)?;
                if split.len() < 2 {
                    return None;
                }
                let node = b.str_node(&split[1]);
                labels.insert(split[0].clone(), node);
                let mut attrs = vec![(id_key, Val::Str(split[0].clone()))];
                if split.len() >= 5 {
                    match py_float(&split[2]) {
                        FloatParse::Ok(x) => match py_float(&split[3]) {
                            FloatParse::Ok(y) => {
                                attrs.push((x_key, Val::Float(x)));
                                attrs.push((y_key, Val::Float(y)));
                                attrs.push((shape_key, Val::Str(split[4].clone())));
                            }
                            FloatParse::Invalid => {}
                            FloatParse::Unsure => return None,
                        },
                        FloatParse::Invalid => {}
                        FloatParse::Unsure => return None,
                    }
                }
                let extra = split.iter().skip(5).step_by(2);
                for (k, v) in extra.zip(split.iter().skip(6).step_by(2)) {
                    let k = b.key(k);
                    attrs.push((k, Val::Str(v.clone())));
                }
                b.parsed.ops.push(Op::Node(node, attrs));
            }
            nodelabels = Some(labels);
        } else if lower_starts_with(l, "*edges") || lower_starts_with(l, "*arcs") {
            if lower_starts_with(l, "*edge") {
                b.parsed.class = (true, false);
            }
            if lower_starts_with(l, "*arcs") {
                b.parsed.class = (true, true);
            }
            // NetworkX looks ids up in the last *vertices section.
            let labels = nodelabels.take()?;
            for &line in &lines[i..] {
                let split = shlex_split(line)?;
                if split.len() < 2 {
                    continue;
                }
                let mut ends = [0u32; 2];
                for (end, token) in ends.iter_mut().zip(&split[..2]) {
                    *end = match labels.get(token.as_str()) {
                        Some(&n) => n,
                        None => b.str_node(token),
                    };
                }
                let mut attrs = Vec::new();
                if split.len() >= 3 {
                    match py_float(&split[2]) {
                        FloatParse::Ok(w) => attrs.push((weight_key, Val::Float(w))),
                        FloatParse::Invalid => {}
                        FloatParse::Unsure => return None,
                    }
                }
                let extra = split.iter().skip(3).step_by(2);
                for (k, v) in extra.zip(split.iter().skip(4).step_by(2)) {
                    if reserved(k) {
                        return None;
                    }
                    let k = b.key(k);
                    attrs.push((k, Val::Str(v.clone())));
                }
                b.parsed.ops.push(Op::Edge(ends[0], ends[1], attrs));
            }
            break;
        } else if lower_starts_with(l, "*matrix") {
            return None;
        }
    }
    Some(b.finish())
}

// --- graph6 and sparse6 ----------------------------------------------------------

/// graph6's `data_to_n`: the node count and the rest of the data.
fn data_to_n(data: &[u8]) -> Option<(u64, &[u8])> {
    let d = |i: usize| -> Option<u64> { data.get(i).map(|&c| c as u64) };
    if d(0)? <= 62 {
        return Some((d(0)?, &data[1..]));
    }
    if d(1)? <= 62 {
        let n = (d(1)? << 12) + (d(2)? << 6) + d(3)?;
        return Some((n, &data[4..]));
    }
    let n = (d(2)? << 30) + (d(3)? << 24) + (d(4)? << 18) + (d(5)? << 12) + (d(6)? << 6) + d(7)?;
    Some((n, &data[8..]))
}

/// `bytes.strip()`: ASCII whitespace, including the vertical tab that
/// `<[u8]>::trim_ascii` keeps.
pub fn py_bytes_strip(b: &[u8]) -> &[u8] {
    let ws = |c: &u8| matches!(c, b' ' | b'\t' | b'\n' | b'\r' | b'\x0b' | b'\x0c');
    let start = b.iter().position(|c| !ws(c)).unwrap_or(b.len());
    let end = b.iter().rposition(|c| !ws(c)).map_or(start, |p| p + 1);
    &b[start..end]
}

/// Character values minus 63, if every character is in range(63, 127).
fn six_bit(bytes: &[u8]) -> Option<Vec<u8>> {
    bytes
        .iter()
        .map(|&c| {
            if (63..127).contains(&c) {
                Some(c - 63)
            } else {
                None
            }
        })
        .collect()
}

/// `from_graph6_bytes`; `strip_newline` for NetworkX 3.5+, which ignores
/// trailing newlines.
pub fn graph6(bytes: &[u8], strip_newline: bool) -> Option<Parsed> {
    let mut bytes = bytes.strip_prefix(b">>graph6<<").unwrap_or(bytes);
    if strip_newline {
        while let Some(rest) = bytes.strip_suffix(b"\n") {
            bytes = rest;
        }
    }
    let data = six_bit(bytes)?;
    let (n, data) = data_to_n(&data)?;
    let bits = n as u128 * (n as u128).saturating_sub(1) / 2;
    if data.len() as u128 != bits.div_ceil(6) {
        return None;
    }
    let mut b = Builder::default();
    for i in 0..n {
        b.parsed.nodes.push(Val::Int(i as i64));
        b.parsed.ops.push(Op::Node(i as u32, Vec::new()));
    }
    let mut bit = 0usize;
    for j in 1..n as u32 {
        for i in 0..j {
            if (data[bit / 6] >> (5 - bit % 6)) & 1 == 1 {
                b.parsed.ops.push(Op::Edge(i, j, Vec::new()));
            }
            bit += 1;
        }
    }
    Some(b.parsed)
}

/// `from_sparse6_bytes`: a MultiGraph if there are parallel edges, else a
/// Graph built the way `nx.Graph(multigraph)` copies it.
pub fn sparse6(bytes: &[u8]) -> Option<Parsed> {
    let bytes = bytes.strip_prefix(b">>sparse6<<").unwrap_or(bytes);
    let rest = bytes.strip_prefix(b":")?;
    let chars = six_bit(rest)?;
    let (n, data) = data_to_n(&chars)?;
    if n > u32::MAX as u64 / 2 {
        return None;
    }
    let mut k = 1u32;
    while (1u64 << k) < n {
        k += 1;
    }
    // parseData(): pairs (b, x).
    let mut pairs = Vec::new();
    let mut chunks = data.iter().map(|&c| c as u64);
    let mut d = 0u64;
    let mut d_len = 0u32;
    'outer: loop {
        if d_len < 1 {
            match chunks.next() {
                Some(c) => d = c,
                None => break,
            }
            d_len = 6;
        }
        d_len -= 1;
        let bit = (d >> d_len) & 1;
        let mut x = d & ((1u64 << d_len) - 1);
        let mut x_len = d_len;
        while x_len < k {
            match chunks.next() {
                Some(c) => d = c,
                None => break 'outer,
            }
            x = (x << 6) + d;
            x_len += 6;
        }
        x >>= x_len - k;
        d_len = x_len - k;
        pairs.push((bit, x));
    }
    let mut v = 0u64;
    let mut edges = Vec::new();
    let mut seen = HashSet::new();
    let mut multigraph = false;
    for (bit, x) in pairs {
        if bit == 1 {
            v += 1;
        }
        if x >= n || v >= n {
            break;
        } else if x > v {
            v = x;
        } else {
            let (a, c) = (x as u32, v as u32);
            if !seen.insert((a.min(c), a.max(c))) {
                multigraph = true;
            }
            edges.push((a, c));
        }
    }
    let mut b = Builder::default();
    for i in 0..n {
        b.parsed.nodes.push(Val::Int(i as i64));
        b.parsed.ops.push(Op::Node(i as u32, Vec::new()));
    }
    b.parsed.class = (multigraph, false);
    if multigraph {
        for (x, v) in edges {
            b.parsed.ops.push(Op::Edge(x, v, Vec::new()));
        }
        return Some(b.parsed);
    }
    // nx.Graph(G) walks G's adjacency (neighbors in first-edge order) and
    // adds each edge from the side it meets first.
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n as usize];
    for &(x, v) in &edges {
        adj[x as usize].push(v);
        if x != v {
            adj[v as usize].push(x);
        }
    }
    let mut added = HashSet::new();
    for (u, nbrs) in adj.iter().enumerate() {
        let u = u as u32;
        for &w in nbrs {
            if added.insert((u.min(w), u.max(w))) {
                b.parsed.ops.push(Op::Edge(u, w, Vec::new()));
            }
        }
    }
    Some(b.parsed)
}

// --- GML ------------------------------------------------------------------------

/// `str.splitlines()`.
fn py_splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        let boundary = matches!(
            c,
            '\n' | '\r'
                | '\x0b'
                | '\x0c'
                | '\x1c'
                | '\x1d'
                | '\x1e'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        );
        if boundary {
            out.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' {
                if let Some(&(j, '\n')) = it.peek() {
                    it.next();
                    end = j + 1;
                }
            }
            start = end;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Key(String),
    Real(f64),
    Int(i64),
    /// A string literal's text, without the quotes.
    Str(String),
    Open,
    Close,
    Eof,
}

/// The tokens of one (joined) line, as `parse_gml_lines`'s regular
/// expression finds them; `None` where it can't tokenize (NetworkX raises).
fn gml_tokens(line: &str, out: &mut Vec<Tok>) -> Option<()> {
    let b = line.as_bytes();
    let mut pos = 0;
    let digits = |mut i: usize| {
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        i
    };
    while pos < b.len() {
        let c = b[pos];
        if c.is_ascii_alphabetic() {
            // [A-Za-z][0-9A-Za-z_]*\b: a non-ASCII letter or digit next would
            // break the \b; such a line can't be tokenized anyway.
            let mut end = pos + 1;
            while end < b.len() && (b[end].is_ascii_alphanumeric() || b[end] == b'_') {
                end += 1;
            }
            if let Some(next) = line[end..].chars().next() {
                if !next.is_ascii() && !py_isspace(next) {
                    return None;
                }
            }
            out.push(Tok::Key(line[pos..end].to_string()));
            pos = end;
            continue;
        }
        if c == b'+' || c == b'-' || c == b'.' || c.is_ascii_digit() {
            // Reals: [+-]?(?:[0-9]*\.[0-9]+|[0-9]+\.[0-9]*|INF)(?:[Ee][+-]?[0-9]+)?
            let start = if c == b'+' || c == b'-' { pos + 1 } else { pos };
            let int_end = digits(start);
            let mut mantissa_end = None;
            if int_end < b.len() && b[int_end] == b'.' {
                let frac_end = digits(int_end + 1);
                if frac_end > int_end + 1 || int_end > start {
                    mantissa_end = Some(frac_end);
                }
            }
            let inf = b[start..].starts_with(b"INF");
            if mantissa_end.is_none() && inf {
                mantissa_end = Some(start + 3);
            }
            if let Some(m) = mantissa_end {
                let mut end = m;
                if end < b.len() && (b[end] == b'e' || b[end] == b'E') {
                    let mut j = end + 1;
                    if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                        j += 1;
                    }
                    let k = digits(j);
                    if k > j {
                        end = k;
                    }
                }
                let text = &line[pos..end];
                let value = if inf && mantissa_end == Some(start + 3) {
                    // float("+INFe5") raises.
                    if end != start + 3 {
                        return None;
                    }
                    if c == b'-' {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    }
                } else {
                    text.parse::<f64>().ok()?
                };
                out.push(Tok::Real(value));
                pos = end;
                continue;
            }
            // Ints: [+-]?[0-9]+
            if int_end > start {
                out.push(Tok::Int(line[pos..int_end].parse::<i64>().ok()?));
                pos = int_end;
                continue;
            }
            return None;
        }
        match c {
            b'"' => {
                let close = line[pos + 1..].find('"')? + pos + 1;
                out.push(Tok::Str(line[pos + 1..close].to_string()));
                pos = close + 1;
            }
            b'[' => {
                out.push(Tok::Open);
                pos += 1;
            }
            b']' => {
                out.push(Tok::Close);
                pos += 1;
            }
            b'#' => pos = b.len(),
            _ => {
                let mut end = pos;
                for ch in line[pos..].chars() {
                    if !py_isspace(ch) {
                        break;
                    }
                    end += ch.len_utf8();
                }
                if end == pos {
                    return None;
                }
                pos = end;
            }
        }
    }
    Some(())
}

/// `tokenize()` over all lines, including NetworkX's joining of string
/// values that span lines.
fn gml_tokenize(lines: &[&str]) -> Option<Vec<Tok>> {
    let mut out = Vec::new();
    let mut multilines: Vec<String> = Vec::new();
    for &raw in lines {
        let joined;
        let mut line = raw;
        if !multilines.is_empty() {
            multilines.push(py_strip(line).to_string());
            if !line.ends_with('"') {
                // line[-1] raises for an empty line.
                line.chars().last()?;
                continue;
            }
            joined = multilines.join(" ");
            multilines.clear();
            line = &joined;
        } else if line.matches('"').count() == 1 {
            let stripped = py_strip(line);
            if !stripped.starts_with('"') && !stripped.ends_with('"') {
                multilines.push(line.trim_end_matches(py_isspace).to_string());
                continue;
            }
        }
        gml_tokens(line, &mut out)?;
    }
    out.push(Tok::Eof);
    Some(out)
}

/// gml's `unescape`: numeric character references, and the named ones
/// `escape` and common writers produce. `None` for other named references
/// (whose handling needs the whole HTML entity table) and lone surrogates.
fn gml_unescape(text: &str) -> Option<String> {
    if !text.contains('&') {
        return Some(text.to_string());
    }
    let mut out = String::with_capacity(text.len());
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'&' {
            let ch = text[i..].chars().next()?;
            out.push(ch);
            i += ch.len_utf8();
            continue;
        }
        // &(?:[0-9A-Za-z]+|#(?:[0-9]+|x[0-9A-Fa-f]+));
        let rest = &b[i + 1..];
        let name_len = rest
            .iter()
            .take_while(|c| c.is_ascii_alphanumeric())
            .count();
        if name_len > 0 && rest.get(name_len) == Some(&b';') {
            let name = &text[i + 1..i + 1 + name_len];
            let ch = match name {
                "amp" => '&',
                "lt" => '<',
                "gt" => '>',
                "quot" => '"',
                _ => return None,
            };
            out.push(ch);
            i += name_len + 2;
            continue;
        }
        if rest.first() == Some(&b'#') {
            let (radix, start) = if rest.get(1) == Some(&b'x') {
                (16, 2)
            } else {
                (10, 1)
            };
            let n = rest[start..]
                .iter()
                .take_while(|c| {
                    if radix == 16 {
                        c.is_ascii_hexdigit()
                    } else {
                        c.is_ascii_digit()
                    }
                })
                .count();
            if n > 0 && rest.get(start + n) == Some(&b';') {
                let digits = &text[i + 1 + start..i + 1 + start + n];
                let end = i + 1 + start + n + 1;
                match u32::from_str_radix(digits, radix)
                    .ok()
                    .filter(|&c| c <= 0x10FFFF)
                {
                    // chr() of a surrogate gives a str Rust can't hold.
                    Some(code) => out.push(char::from_u32(code)?),
                    // chr() raises: the reference is left as it is.
                    None => out.push_str(&text[i..end]),
                }
                i = end;
                continue;
            }
        }
        out.push('&');
        i += 1;
    }
    Some(out)
}

/// A GML dict: keys in first-seen order, each with its values.
type GmlDict = Vec<(String, Val)>;

/// Nesting deeper than this falls back: NetworkX parses dicts recursively.
const GML_DEPTH_LIMIT: usize = 100;

struct GmlParser {
    toks: Vec<Tok>,
    pos: usize,
}

impl GmlParser {
    fn cur(&self) -> &Tok {
        &self.toks[self.pos]
    }

    fn advance(&mut self) -> Option<()> {
        // next(tokens) after the EOF token raises StopIteration.
        if self.pos + 1 >= self.toks.len() {
            return None;
        }
        self.pos += 1;
        Some(())
    }

    /// `parse_kv`, then `clean_dict_value` on each key's values.
    fn kv(&mut self, depth: usize) -> Option<GmlDict> {
        if depth > GML_DEPTH_LIMIT {
            return None;
        }
        let mut entries: Vec<(String, Vec<Val>)> = Vec::new();
        while let Tok::Key(key) = self.cur().clone() {
            self.advance()?;
            let value = match self.cur().clone() {
                Tok::Real(f) => {
                    self.advance()?;
                    Val::Float(f)
                }
                Tok::Int(i) => {
                    self.advance()?;
                    Val::Int(i)
                }
                Tok::Str(raw) => {
                    let text = gml_unescape(&raw)?;
                    self.advance()?;
                    match text.as_str() {
                        "()" => Val::EmptyTuple,
                        "[]" => Val::List(Vec::new()),
                        _ => Val::Str(text),
                    }
                }
                Tok::Open => {
                    self.advance()?;
                    let d = self.kv(depth + 1)?;
                    if *self.cur() != Tok::Close {
                        return None;
                    }
                    self.advance()?;
                    Val::Dict(d)
                }
                Tok::Key(word) => {
                    if matches!(key.as_str(), "id" | "label" | "source" | "target") {
                        self.advance()?;
                        Val::Str(word)
                    } else if word == "NAN" || word == "INF" {
                        self.advance()?;
                        Val::Float(if word == "NAN" {
                            f64::NAN
                        } else {
                            f64::INFINITY
                        })
                    } else {
                        return None;
                    }
                }
                Tok::Close | Tok::Eof => return None,
            };
            match entries.iter_mut().find(|(k, _)| *k == key) {
                Some((_, values)) => values.push(value),
                None => entries.push((key, vec![value])),
            }
        }
        Some(
            entries
                .into_iter()
                .map(|(k, mut values)| {
                    let v = if values.len() == 1 {
                        values.pop().expect("one value")
                    } else if values[0] == Val::Str("_networkx_list_start".into()) {
                        Val::List(values.split_off(1))
                    } else {
                        Val::List(values)
                    };
                    (k, v)
                })
                .collect(),
        )
    }
}

/// Python truthiness of a parsed value.
fn truthy(v: &Val) -> bool {
    match v {
        Val::None => false,
        Val::Bool(b) => *b,
        Val::Int(i) => *i != 0,
        Val::Float(f) => *f != 0.0,
        Val::Str(s) => !s.is_empty(),
        Val::List(l) => !l.is_empty(),
        Val::Dict(d) => !d.is_empty(),
        Val::EmptyTuple => false,
    }
}

/// `a == b` in Python, for hashable parsed values (None if either isn't).
fn py_eq(a: &Val, b: &Val) -> Option<bool> {
    Some(match (a, b) {
        (Val::Int(x), Val::Int(y)) => x == y,
        (Val::Float(x), Val::Float(y)) => x == y,
        (Val::Int(i), Val::Float(f)) | (Val::Float(f), Val::Int(i)) => {
            // Exact comparison, as Python does for int and float.
            f.fract() == 0.0 && *f >= -9.3e18 && *f <= 9.3e18 && (*f as i128) == (*i as i128)
        }
        (Val::Str(x), Val::Str(y)) => x == y,
        (Val::EmptyTuple, Val::EmptyTuple) => true,
        (Val::List(_) | Val::Dict(_), _) | (_, Val::List(_) | Val::Dict(_)) => return None,
        _ => false,
    })
}

/// Same value and type (and sign of zero): NetworkX would use this very
/// object, so the result can't tell them apart.
fn identical(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Float(x), Val::Float(y)) => x.to_bits() == y.to_bits(),
        _ => a == b,
    }
}

fn pop_key(d: &mut GmlDict, key: &str) -> Option<Val> {
    let i = d.iter().position(|(k, _)| k == key)?;
    Some(d.remove(i).1)
}

/// The nodes or edges of the graph dict: a list of dicts, or one dict.
fn gml_items(v: Option<Val>) -> Option<Vec<GmlDict>> {
    match v {
        None => Some(Vec::new()),
        Some(Val::Dict(d)) => Some(vec![d]),
        Some(Val::List(items)) => items
            .into_iter()
            .map(|item| match item {
                Val::Dict(d) => Some(d),
                _ => None,
            })
            .collect(),
        Some(_) => None,
    }
}

/// `parse_gml_lines(lines, label, None)`: the operations building the result
/// (after `relabel_nodes` when `label` names a node attribute).
pub fn gml(lines: &[&str], label: Option<&str>) -> Option<Parsed> {
    let mut p = GmlParser {
        toks: gml_tokenize(lines)?,
        pos: 0,
    };
    let mut top = p.kv(0)?;
    if *p.cur() != Tok::Eof {
        return None;
    }
    let Val::Dict(mut graph) = pop_key(&mut top, "graph")? else {
        return None;
    };
    let directed = pop_key(&mut graph, "directed").is_some_and(|v| truthy(&v));
    let multigraph = pop_key(&mut graph, "multigraph").is_some_and(|v| truthy(&v));
    let nodes = gml_items(pop_key(&mut graph, "node"))?;
    let edges = gml_items(pop_key(&mut graph, "edge"))?;
    let mut b = Builder::default();
    b.parsed.class = (multigraph, directed);
    for (k, v) in graph {
        let k = b.key(&k);
        b.parsed.graph_attrs.push((k, v));
    }
    let relabel = label.filter(|&l| l != "id");
    // Node ids (and labels): hashable, and no two equal.
    let mut ids: Vec<Val> = Vec::new();
    let mut labels: Vec<Val> = Vec::new();
    let mut node_attrs = Vec::new();
    for mut node in nodes {
        let id = pop_key(&mut node, "id")?;
        for other in &ids {
            if py_eq(&id, other)? {
                return None;
            }
        }
        if let Some(label) = relabel {
            let node_label = pop_key(&mut node, label)?;
            for other in &labels {
                if py_eq(&node_label, other)? {
                    return None;
                }
            }
            py_eq(&node_label, &node_label)?; // unhashable: NetworkX raises
            labels.push(node_label);
        }
        if py_eq(&id, &id).is_none() {
            return None;
        }
        let mut attrs = Vec::with_capacity(node.len());
        for (k, v) in node {
            if reserved_kwarg(&k, false, true) {
                return None;
            }
            attrs.push((b.key(&k), v));
        }
        ids.push(id);
        node_attrs.push(attrs);
    }
    let n = ids.len();
    let index_of = |v: &Val| -> Option<u32> {
        // `source in G`, then the edge's own object in G's rows: only the
        // node's own id (same type) is replayed exactly.
        let i = ids.iter().position(|id| py_eq(v, id) == Some(true))?;
        if identical(v, &ids[i]) {
            Some(i as u32)
        } else {
            None
        }
    };
    // G's rows: neighbor order, and each pair's (key, attrs) in key order.
    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut pair_edges: HashMap<(u32, u32), Vec<(Val, Vec<(u32, Val)>)>> = HashMap::new();
    let pair = |u: u32, v: u32| if directed || u <= v { (u, v) } else { (v, u) };
    for mut edge in edges {
        let s = index_of(&pop_key(&mut edge, "source")?)?;
        let t = index_of(&pop_key(&mut edge, "target")?)?;
        let key = if multigraph {
            pop_key(&mut edge, "key")
        } else {
            None
        };
        let mut attrs = Vec::with_capacity(edge.len());
        for (k, v) in edge {
            if reserved_kwarg(&k, multigraph, false) {
                return None;
            }
            attrs.push((b.key(&k), v));
        }
        let existing = pair_edges.entry(pair(s, t)).or_default();
        if existing.is_empty() {
            rows[s as usize].push(t);
            if s != t && !directed {
                rows[t as usize].push(s);
            }
        }
        let key = if !multigraph {
            if !existing.is_empty() {
                return None; // "is duplicated"
            }
            Val::None
        } else if let Some(key) = key {
            py_eq(&key, &key)?; // unhashable: NetworkX raises
            for (other, _) in existing.iter() {
                if py_eq(&key, other)? {
                    return None; // "is duplicated"
                }
            }
            key
        } else {
            // new_edge_key(): len(keydict), then the next unused.
            let mut next = existing.len() as i64;
            while existing
                .iter()
                .any(|(k, _)| py_eq(k, &Val::Int(next)) == Some(true))
            {
                next += 1;
            }
            Val::Int(next)
        };
        existing.push((key, attrs));
        if relabel.is_none() {
            let (k, attrs) = existing.last().expect("just pushed").clone();
            b.parsed.ops.push(if multigraph {
                Op::KeyedEdge(s, t, k, attrs)
            } else {
                Op::Edge(s, t, attrs)
            });
        }
    }
    b.parsed.nodes = match relabel {
        None => ids,
        Some(_) => labels,
    };
    let mut ops: Vec<Op> = node_attrs
        .into_iter()
        .enumerate()
        .map(|(i, attrs)| Op::Node(i as u32, attrs))
        .collect();
    if relabel.is_some() {
        // relabel_nodes(G, mapping) adds G.edges(keys=True, data=True) in
        // order: rows in node order, each undirected pair from the end seen
        // first.
        let mut seen = vec![false; n];
        for u in 0..n as u32 {
            for &v in &rows[u as usize] {
                if !directed && seen[v as usize] {
                    continue;
                }
                for (k, attrs) in &pair_edges[&pair(u, v)] {
                    ops.push(if multigraph {
                        Op::KeyedEdge(u, v, k.clone(), attrs.clone())
                    } else {
                        Op::Edge(u, v, attrs.clone())
                    });
                }
            }
            seen[u as usize] = true;
        }
    } else {
        ops.append(&mut b.parsed.ops);
    }
    b.parsed.ops = ops;
    Some(b.finish())
}

// --- applying the operations to a NetworkX graph (with the GIL) ------------------

fn to_py<'py>(py: Python<'py>, v: &Val) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Val::None => py.None().into_bound(py),
        Val::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Val::Int(i) => i.into_pyobject(py)?.into_any(),
        Val::Float(f) => PyFloat::new(py, *f).into_any(),
        Val::Str(s) => PyString::new(py, s).into_any(),
        Val::List(items) => {
            let items = items
                .iter()
                .map(|v| to_py(py, v))
                .collect::<PyResult<Vec<_>>>()?;
            pyo3::types::PyList::new(py, items)?.into_any()
        }
        Val::Dict(items) => {
            let d = PyDict::new(py);
            for (k, v) in items {
                d.set_item(k, to_py(py, v)?)?;
            }
            d.into_any()
        }
        Val::EmptyTuple => pyo3::types::PyTuple::empty(py).into_any(),
    })
}

/// A new, empty `networkx.(Multi)(Di)Graph`.
pub fn new_nx_graph(
    py: Python<'_>,
    multigraph: bool,
    directed: bool,
) -> PyResult<Bound<'_, PyAny>> {
    let name = match (multigraph, directed) {
        (false, false) => "Graph",
        (false, true) => "DiGraph",
        (true, false) => "MultiGraph",
        (true, true) => "MultiDiGraph",
    };
    py.import("networkx")?.getattr(name)?.call0()
}

struct Target<'py> {
    objs: Vec<Bound<'py, PyAny>>,
    node: Bound<'py, PyDict>,
    adj: Bound<'py, PyDict>,
    pred: Option<Bound<'py, PyDict>>,
    succ_of: Vec<Option<Bound<'py, PyDict>>>,
    pred_of: Vec<Option<Bound<'py, PyDict>>>,
    attr_of: Vec<Option<Bound<'py, PyDict>>>,
}

impl<'py> Target<'py> {
    /// The `if u not in self._node` part of `add_node` / `add_edge`.
    fn ensure(&mut self, py: Python<'py>, i: usize) -> PyResult<()> {
        if self.succ_of[i].is_some() {
            return Ok(());
        }
        let key = &self.objs[i];
        let succ = PyDict::new(py);
        self.adj.set_item(key, &succ)?;
        self.succ_of[i] = Some(succ);
        if let Some(pred) = &self.pred {
            let p = PyDict::new(py);
            pred.set_item(key, &p)?;
            self.pred_of[i] = Some(p);
        }
        let attrs = PyDict::new(py);
        self.node.set_item(key, &attrs)?;
        self.attr_of[i] = Some(attrs);
        Ok(())
    }

    /// Where `add_edge(u, v)` stores the reverse entry: `_pred[v]` or `_adj[v]`.
    fn back(&self, v: usize) -> &Bound<'py, PyDict> {
        match &self.pred {
            Some(_) => self.pred_of[v].as_ref(),
            None => self.succ_of[v].as_ref(),
        }
        .expect("node added")
    }
}

/// Replays `parsed.ops` onto the empty NetworkX graph `g` (a plain
/// Graph, DiGraph, MultiGraph or MultiDiGraph) as `add_node` and
/// `add_edge` would, filling its dicts directly.
pub fn apply<'py>(py: Python<'py>, parsed: &Parsed, g: &Bound<'py, PyAny>) -> PyResult<()> {
    let directed = g.call_method0("is_directed")?.is_truthy()?;
    let multigraph = g.call_method0("is_multigraph")?.is_truthy()?;
    let n = parsed.nodes.len();
    let mut t = Target {
        objs: parsed
            .nodes
            .iter()
            .map(|v| to_py(py, v))
            .collect::<PyResult<_>>()?,
        node: g.getattr("_node")?.cast_into::<PyDict>()?,
        adj: g.getattr("_adj")?.cast_into::<PyDict>()?,
        pred: if directed {
            Some(g.getattr("_pred")?.cast_into::<PyDict>()?)
        } else {
            None
        },
        succ_of: vec![None; n],
        pred_of: vec![None; n],
        attr_of: vec![None; n],
    };
    let keys: Vec<Bound<'py, PyString>> =
        parsed.keys.iter().map(|k| PyString::new(py, k)).collect();
    let fill = |d: &Bound<'py, PyDict>, attrs: &[(u32, Val)]| -> PyResult<()> {
        for (k, v) in attrs {
            d.set_item(&keys[*k as usize], to_py(py, v)?)?;
        }
        Ok(())
    };
    for op in &parsed.ops {
        match op {
            Op::Node(i, attrs) => {
                let i = *i as usize;
                t.ensure(py, i)?;
                fill(t.attr_of[i].as_ref().expect("node added"), attrs)?;
            }
            Op::Edge(u, v, attrs) | Op::KeyedEdge(u, v, _, attrs) => {
                let key = match op {
                    Op::KeyedEdge(_, _, key, _) => Some(to_py(py, key)?),
                    _ => None,
                };
                let (u, v) = (*u as usize, *v as usize);
                t.ensure(py, u)?;
                t.ensure(py, v)?;
                let succ = t.succ_of[u].as_ref().expect("node added");
                let existing = succ.get_item(&t.objs[v])?;
                if !multigraph {
                    match existing {
                        Some(d) => fill(d.cast::<PyDict>()?, attrs)?,
                        None => {
                            let d = PyDict::new(py);
                            fill(&d, attrs)?;
                            succ.set_item(&t.objs[v], &d)?;
                            t.back(v).set_item(&t.objs[u], &d)?;
                        }
                    }
                } else {
                    let d = PyDict::new(py);
                    fill(&d, attrs)?;
                    match existing {
                        Some(keydict) => {
                            let keydict = keydict.cast::<PyDict>()?;
                            match &key {
                                Some(key) => keydict.set_item(key, &d)?,
                                None => {
                                    // new_edge_key(): len(keydict), then the next unused.
                                    let mut key = keydict.len();
                                    while keydict.contains(key)? {
                                        key += 1;
                                    }
                                    keydict.set_item(key, &d)?;
                                }
                            }
                        }
                        None => {
                            let keydict = PyDict::new(py);
                            match &key {
                                Some(key) => keydict.set_item(key, &d)?,
                                None => keydict.set_item(0, &d)?,
                            }
                            succ.set_item(&t.objs[v], &keydict)?;
                            t.back(v).set_item(&t.objs[u], &keydict)?;
                        }
                    }
                }
            }
        }
    }
    if let Some(name) = &parsed.name {
        g.getattr("graph")?.set_item("name", name)?;
    }
    if !parsed.graph_attrs.is_empty() {
        let graph = g.getattr("graph")?;
        for (k, v) in &parsed.graph_attrs {
            graph.set_item(&keys[*k as usize], to_py(py, v)?)?;
        }
    }
    Ok(())
}

/// The `str` objects behind `lines`: each item of a list of exact `str`
/// (`mode` 0), or the one `str` given (`mode` 1: a decoded file, `mode` 2: a
/// `str` that NetworkX splits with `split("\n")`). `None` if an item isn't
/// an exact `str`.
pub fn string_items<'py>(lines: &Bound<'py, PyAny>, mode: u8) -> Option<Vec<Bound<'py, PyString>>> {
    if mode == 0 {
        let list = lines.cast::<pyo3::types::PyList>().ok()?;
        list.iter()
            .map(|item| item.cast_exact::<PyString>().ok().cloned())
            .collect()
    } else {
        Some(vec![lines.cast_exact::<PyString>().ok()?.clone()])
    }
}

/// `parse_gml`'s lines: list items lose one trailing "\n" (and may not
/// contain another), a file's text is split at "\n" (`mode` 1), a `str` at
/// every line boundary (`mode` 3).
pub fn gml_lines<'a>(items: &'a [Bound<'_, PyString>], mode: u8) -> Option<Vec<&'a str>> {
    use pyo3::types::PyStringMethods;
    let mut out = Vec::new();
    for item in items {
        let s = item.to_str().ok()?;
        match mode {
            0 => {
                let line = s.strip_suffix('\n').unwrap_or(s);
                if line.contains('\n') {
                    return None;
                }
                out.push(line);
            }
            1 => out.extend(
                s.split_inclusive('\n')
                    .map(|l| l.strip_suffix('\n').unwrap_or(l)),
            ),
            _ => out.extend(py_splitlines(s)),
        }
    }
    Some(out)
}

/// The lines of `string_items`: list items as they are, a file's text split
/// after each "\n" (as iterating over a binary file does), or a `str` split
/// at "\n". `None` if a string can't be encoded as UTF-8 (lone surrogates).
pub fn split_lines<'a>(items: &'a [Bound<'_, PyString>], mode: u8) -> Option<Vec<&'a str>> {
    use pyo3::types::PyStringMethods;
    let mut out = Vec::new();
    for item in items {
        let s = item.to_str().ok()?;
        match mode {
            0 => out.push(s),
            1 => out.extend(s.split_inclusive('\n')),
            _ => out.extend(s.split('\n')),
        }
    }
    Some(out)
}

// --- JSON graphs: NetworkX's own `add_node` / `add_edge` on Python objects --------
//
// `node_link_graph`, `adjacency_graph`, `cytoscape_graph` and `tree_graph`
// build graphs from already-parsed Python data, so these replay NetworkX's
// code on the same objects (each dict operation is the one NetworkX does).
// Any Python error, and anything NetworkX would raise for, gives `None`.

use pyo3::types::{PyList, PyTuple};

/// Whether `obj` is a str, int, float, bool or None, or a tuple of those:
/// node and key types whose hashing and equality can't run Python code.
fn simple(obj: &Bound<'_, PyAny>) -> bool {
    if obj.is_none()
        || obj.is_exact_instance_of::<PyString>()
        || obj.is_exact_instance_of::<pyo3::types::PyInt>()
        || obj.is_exact_instance_of::<PyFloat>()
        || obj.is_exact_instance_of::<PyBool>()
    {
        return true;
    }
    match obj.cast_exact::<PyTuple>() {
        Ok(t) => t.iter().all(|item| simple(&item)),
        Err(_) => false,
    }
}

fn exact_dict<'a, 'py>(obj: &'a Bound<'py, PyAny>) -> Option<&'a Bound<'py, PyDict>> {
    obj.cast_exact::<PyDict>().ok()
}

fn exact_list<'a, 'py>(obj: &'a Bound<'py, PyAny>) -> Option<&'a Bound<'py, PyList>> {
    obj.cast_exact::<PyList>().ok()
}

/// `d[key]`, `None` if missing.
fn item<'py>(d: &Bound<'py, PyDict>, key: &Bound<'py, PyAny>) -> Option<Bound<'py, PyAny>> {
    d.get_item(key).ok().flatten()
}

/// Keyword arguments `add_node` / `add_edge` would bind to a parameter.
fn reserved_kwarg(key: &str, multigraph: bool, node: bool) -> bool {
    match (node, multigraph) {
        (true, _) => matches!(key, "self" | "node_for_adding"),
        (false, false) => matches!(key, "self" | "u_of_edge" | "v_of_edge"),
        (false, true) => reserved(key),
    }
}

/// `{str(k): v for k, v in d.items() if k not in skip}` for `**` keyword
/// arguments: str keys only, none that the method takes itself.
fn kwargs<'py>(
    d: &Bound<'py, PyDict>,
    skip: &[&Bound<'py, PyAny>],
    multigraph: bool,
    node: bool,
) -> Option<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    let mut out = Vec::with_capacity(d.len());
    for (k, v) in d.iter() {
        let ks = k.cast_exact::<PyString>().ok()?;
        let mut skipped = false;
        for s in skip {
            if k.eq(s).ok()? {
                skipped = true;
                break;
            }
        }
        if skipped {
            continue;
        }
        if reserved_kwarg(ks.to_str().ok()?, multigraph, node) {
            return None;
        }
        out.push((k, v));
    }
    Some(out)
}

/// The dicts of a new NetworkX graph, with its `add_node` and `add_edge`.
pub struct PyBuilder<'py> {
    py: Python<'py>,
    node: Bound<'py, PyDict>,
    adj: Bound<'py, PyDict>,
    pred: Option<Bound<'py, PyDict>>,
    multigraph: bool,
}

type Pairs<'py> = [(Bound<'py, PyAny>, Bound<'py, PyAny>)];

impl<'py> PyBuilder<'py> {
    pub fn new(g: &Bound<'py, PyAny>) -> Option<PyBuilder<'py>> {
        let directed = g.call_method0("is_directed").ok()?.is_truthy().ok()?;
        Some(PyBuilder {
            py: g.py(),
            node: g.getattr("_node").ok()?.cast_into::<PyDict>().ok()?,
            adj: g.getattr("_adj").ok()?.cast_into::<PyDict>().ok()?,
            pred: if directed {
                Some(g.getattr("_pred").ok()?.cast_into::<PyDict>().ok()?)
            } else {
                None
            },
            multigraph: g.call_method0("is_multigraph").ok()?.is_truthy().ok()?,
        })
    }

    /// `if n not in self._node: ...` (None can't be a node).
    fn ensure(&self, n: &Bound<'py, PyAny>) -> Option<Bound<'py, PyDict>> {
        if let Some(attrs) = item(&self.node, n) {
            return attrs.cast_into::<PyDict>().ok();
        }
        if n.is_none() || !simple(n) {
            return None;
        }
        self.adj.set_item(n, PyDict::new(self.py)).ok()?;
        if let Some(pred) = &self.pred {
            pred.set_item(n, PyDict::new(self.py)).ok()?;
        }
        let attrs = PyDict::new(self.py);
        self.node.set_item(n, &attrs).ok()?;
        Some(attrs)
    }

    fn row(&self, d: &Bound<'py, PyDict>, n: &Bound<'py, PyAny>) -> Option<Bound<'py, PyDict>> {
        item(d, n)?.cast_into::<PyDict>().ok()
    }

    /// `add_node(n, **attrs)`; returns the node's attribute dict.
    pub fn add_node(
        &self,
        n: &Bound<'py, PyAny>,
        attrs: &Pairs<'py>,
    ) -> Option<Bound<'py, PyDict>> {
        let d = self.ensure(n)?;
        for (k, v) in attrs {
            d.set_item(k, v).ok()?;
        }
        Some(d)
    }

    /// `add_edge(u, v, [key,] **attrs)`; returns the edge's data dict.
    pub fn add_edge(
        &self,
        u: &Bound<'py, PyAny>,
        v: &Bound<'py, PyAny>,
        key: Option<&Bound<'py, PyAny>>,
        attrs: &Pairs<'py>,
    ) -> Option<Bound<'py, PyDict>> {
        self.ensure(u)?;
        self.ensure(v)?;
        let succ = self.row(&self.adj, u)?;
        let back = self.row(self.pred.as_ref().unwrap_or(&self.adj), v)?;
        if !self.multigraph {
            let datadict = match item(&succ, v) {
                Some(d) => d.cast_into::<PyDict>().ok()?,
                None => PyDict::new(self.py),
            };
            for (k, val) in attrs {
                datadict.set_item(k, val).ok()?;
            }
            succ.set_item(v, &datadict).ok()?;
            back.set_item(u, &datadict).ok()?;
            return Some(datadict);
        }
        let existing = item(&succ, v);
        let key = match key {
            Some(k) if !k.is_none() => {
                if !simple(k) {
                    return None;
                }
                k.clone()
            }
            // new_edge_key(): len(keydict), then the next unused.
            _ => {
                let mut next = match &existing {
                    Some(kd) => kd.cast::<PyDict>().ok()?.len(),
                    None => 0,
                };
                if let Some(kd) = &existing {
                    let kd = kd.cast::<PyDict>().ok()?;
                    while kd.contains(next).ok()? {
                        next += 1;
                    }
                }
                next.into_pyobject(self.py).ok()?.into_any()
            }
        };
        match existing {
            Some(kd) => {
                let kd = kd.cast_into::<PyDict>().ok()?;
                let datadict = match item(&kd, &key) {
                    Some(d) => d.cast_into::<PyDict>().ok()?,
                    None => PyDict::new(self.py),
                };
                for (k, val) in attrs {
                    datadict.set_item(k, val).ok()?;
                }
                kd.set_item(&key, &datadict).ok()?;
                Some(datadict)
            }
            None => {
                let datadict = PyDict::new(self.py);
                for (k, val) in attrs {
                    datadict.set_item(k, val).ok()?;
                }
                let kd = PyDict::new(self.py);
                kd.set_item(&key, &datadict).ok()?;
                succ.set_item(v, &kd).ok()?;
                back.set_item(u, &kd).ok()?;
                Some(datadict)
            }
        }
    }
}

/// `_to_tuple`: lists and tuples become tuples, recursively.
fn to_tuple<'py>(x: &Bound<'py, PyAny>) -> Option<Bound<'py, PyAny>> {
    let items: Vec<Bound<'py, PyAny>> = if let Ok(l) = x.cast_exact::<PyList>() {
        l.iter().collect()
    } else if let Ok(t) = x.cast_exact::<PyTuple>() {
        t.iter().collect()
    } else if x.is_instance_of::<PyList>() || x.is_instance_of::<PyTuple>() {
        return None;
    } else {
        return Some(x.clone());
    };
    let items = items.iter().map(to_tuple).collect::<Option<Vec<_>>>()?;
    PyTuple::new(x.py(), items).ok().map(|t| t.into_any())
}

/// `tuple(x) if isinstance(x, list) else x`.
fn list_to_tuple<'py>(x: Bound<'py, PyAny>) -> Option<Bound<'py, PyAny>> {
    if let Ok(l) = x.cast_exact::<PyList>() {
        return Some(l.to_tuple().into_any());
    }
    if x.is_instance_of::<PyList>() {
        return None;
    }
    Some(x)
}

pub struct NodeLinkNames<'py> {
    pub source: Bound<'py, PyAny>,
    pub target: Bound<'py, PyAny>,
    pub name: Bound<'py, PyAny>,
    pub key: Bound<'py, PyAny>,
}

/// The node and edge loops of `node_link_graph`.
pub fn node_link(
    b: &PyBuilder<'_>,
    nodes: &Bound<'_, PyAny>,
    edges: &Bound<'_, PyAny>,
    names: &NodeLinkNames<'_>,
) -> Option<()> {
    let py = b.py;
    for (i, d) in exact_list(nodes)?.iter().enumerate() {
        let d = exact_dict(&d)?;
        let node = match item(d, &names.name) {
            Some(n) => n,
            None => i.into_pyobject(py).ok()?.into_any(),
        };
        let node = to_tuple(&node)?;
        let attrs = kwargs(d, &[&names.name], false, true)?;
        b.add_node(&node, &attrs)?;
    }
    for d in exact_list(edges)?.iter() {
        let d = exact_dict(&d)?;
        let src = list_to_tuple(item(d, &names.source)?)?;
        let tgt = list_to_tuple(item(d, &names.target)?)?;
        if !b.multigraph {
            let attrs = kwargs(d, &[&names.source, &names.target], false, false)?;
            b.add_edge(&src, &tgt, None, &attrs)?;
        } else {
            let ky = item(d, &names.key);
            let skip = [&names.source, &names.target, &names.key];
            let attrs = kwargs(d, &skip, true, false)?;
            b.add_edge(&src, &tgt, ky.as_ref(), &attrs)?;
        }
    }
    Some(())
}

fn dict_copy<'py>(d: &Bound<'py, PyDict>) -> Option<Bound<'py, PyDict>> {
    d.copy().ok()
}

/// `d.pop(key)` (`default` None: KeyError) on a dict NetworkX copied.
fn pop<'py>(d: &Bound<'py, PyDict>, key: &Bound<'py, PyAny>) -> Option<Bound<'py, PyAny>> {
    let v = item(d, key)?;
    d.del_item(key).ok()?;
    Some(v)
}

fn update(d: &Bound<'_, PyDict>, from: &Bound<'_, PyDict>) -> Option<()> {
    d.update(from.as_mapping()).ok()
}

/// The node and adjacency loops of `adjacency_graph`; `key` is None for
/// graphs that aren't multigraphs.
pub fn adjacency(
    b: &PyBuilder<'_>,
    nodes: &Bound<'_, PyAny>,
    adjacency: &Bound<'_, PyAny>,
    id: &Bound<'_, PyAny>,
    key: &Bound<'_, PyAny>,
) -> Option<()> {
    let mut mapping = Vec::new();
    for d in exact_list(nodes)?.iter() {
        let node_data = dict_copy(exact_dict(&d)?)?;
        let node = pop(&node_data, id)?;
        let attrs = b.add_node(&node, &[])?;
        mapping.push(node);
        update(&attrs, &node_data)?;
    }
    for (i, d) in exact_list(adjacency)?.iter().enumerate() {
        let source = mapping.get(i)?;
        for tdata in exact_list(&d)?.iter() {
            let target_data = dict_copy(exact_dict(&tdata)?)?;
            let target = pop(&target_data, id)?;
            let datadict = if !b.multigraph {
                b.add_edge(source, &target, None, &[])?
            } else {
                let ky = item(&target_data, key);
                if ky.is_some() {
                    target_data.del_item(key).ok()?;
                }
                // add_edge(key=None) picks a new key, then graph[u][v][None]
                // raises KeyError.
                let ky = ky.filter(|k| !k.is_none())?;
                b.add_edge(source, &target, Some(&ky), &[])?
            };
            update(&datadict, &target_data)?;
        }
    }
    Some(())
}

/// The node and edge loops of `cytoscape_graph`.
pub fn cytoscape(
    b: &PyBuilder<'_>,
    nodes: &Bound<'_, PyAny>,
    edges: &Bound<'_, PyAny>,
    name: &Bound<'_, PyAny>,
    ident: &Bound<'_, PyAny>,
) -> Option<()> {
    let py = b.py;
    let data_key = PyString::new(py, "data").into_any();
    let value_key = PyString::new(py, "value").into_any();
    for d in exact_list(nodes)?.iter() {
        let data = item(exact_dict(&d)?, &data_key)?;
        let data = exact_dict(&data)?;
        let node_data = dict_copy(data)?;
        let node = item(data, &value_key)?;
        // `if d["data"].get(name): node_data[name] = ...` sets a key the copy
        // already has to the same value; only the truth tests are left.
        for k in [name, ident] {
            if let Some(v) = item(data, k) {
                v.is_truthy().ok()?;
            }
        }
        let attrs = b.add_node(&node, &[])?;
        update(&attrs, &node_data)?;
    }
    let source_key = PyString::new(py, "source").into_any();
    let target_key = PyString::new(py, "target").into_any();
    let key_key = PyString::new(py, "key").into_any();
    for d in exact_list(edges)?.iter() {
        let data = item(exact_dict(&d)?, &data_key)?;
        let data = exact_dict(&data)?;
        let edge_data = dict_copy(data)?;
        let sour = item(data, &source_key)?;
        let targ = item(data, &target_key)?;
        let datadict = if b.multigraph {
            let key = match item(data, &key_key) {
                Some(k) => k,
                None => 0i64.into_pyobject(py).ok()?.into_any(),
            };
            if key.is_none() {
                return None; // graph.edges[u, v, None] raises KeyError
            }
            b.add_edge(&sour, &targ, Some(&key), &[])?
        } else {
            b.add_edge(&sour, &targ, None, &[])?
        };
        update(&datadict, &edge_data)?;
    }
    Some(())
}

/// Deeper trees fall back: NetworkX recurses once per level and would hit
/// Python's recursion limit somewhere that depends on the caller's stack.
const TREE_DEPTH_LIMIT: usize = 200;

/// `tree_graph` into the empty DiGraph of `b`.
pub fn tree(
    b: &PyBuilder<'_>,
    data: &Bound<'_, PyAny>,
    ident: &Bound<'_, PyAny>,
    children: &Bound<'_, PyAny>,
) -> Option<()> {
    let data = exact_dict(data)?;
    let root = item(data, ident)?;
    let kids = item(data, children);
    let attrs = kwargs(data, &[ident, children], false, true)?;
    b.add_node(&root, &attrs)?;
    match kids {
        None => Some(()),
        Some(kids) => tree_children(b, &root, exact_list(&kids)?, ident, children, 1),
    }
}

fn tree_children<'py>(
    b: &PyBuilder<'py>,
    parent: &Bound<'py, PyAny>,
    kids: &Bound<'py, PyList>,
    ident: &Bound<'py, PyAny>,
    children: &Bound<'py, PyAny>,
    depth: usize,
) -> Option<()> {
    if depth > TREE_DEPTH_LIMIT {
        return None;
    }
    for data in kids.iter() {
        let data = exact_dict(&data)?;
        let child = item(data, ident)?;
        b.add_edge(parent, &child, None, &[])?;
        if let Some(grandchildren) = item(data, children) {
            let grandchildren = exact_list(&grandchildren)?;
            if !grandchildren.is_empty() {
                tree_children(b, &child, grandchildren, ident, children, depth + 1)?;
            }
        }
        let attrs = kwargs(data, &[ident, children], false, true)?;
        b.add_node(&child, &attrs)?;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_dicts() {
        assert_eq!(
            literal_dict("{'weight': 3, 'c': \"x y\", 'f': -1.5e3, 'b': True, 'n': None}"),
            Some(vec![
                ("weight".into(), Val::Int(3)),
                ("c".into(), Val::Str("x y".into())),
                ("f".into(), Val::Float(-1500.0)),
                ("b".into(), Val::Bool(true)),
                ("n".into(), Val::None),
            ])
        );
        assert_eq!(literal_dict("{}"), Some(vec![]));
        assert_eq!(
            literal_dict("{'a':1,}"),
            Some(vec![("a".into(), Val::Int(1))])
        );
        for bad in [
            "{'a': 07}",
            "{'a': 1_0}",
            "{'a': 'x' 'y'}",
            "{'a': 1j}",
            "{'a\\n': 1}",
            "{1: 2}",
            "[('a', 1)]",
            "{'a': [1]}",
            "{'a': Truex}",
            "{'a': --1}",
            "{'a': 1.2.3}",
        ] {
            assert_eq!(literal_dict(bad), None, "{bad}");
        }
    }

    #[test]
    fn python_numbers() {
        assert_eq!(py_int(" -12 "), Some(-12));
        assert_eq!(py_int("1_0"), None);
        assert_eq!(py_float("1e5"), FloatParse::Ok(1e5));
        assert_eq!(py_float("abc"), FloatParse::Invalid);
        assert_eq!(py_float("1_0"), FloatParse::Unsure);
        assert_eq!(py_float("-Infinity"), FloatParse::Ok(f64::NEG_INFINITY));
        assert!(lower_starts_with("*Vertices 3", "*vertices"));
        assert!(lower_starts_with("*networ\u{212a}", "*network"));
        assert!(!lower_starts_with("*net", "*network"));
    }
}
