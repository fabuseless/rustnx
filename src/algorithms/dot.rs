//! DOT parsing for `nx.nx_pydot.read_dot`: pydot 4's grammar
//! (`pydot.dot_parser`, written with pyparsing) reproduced as the same
//! parsing expression grammar: ordered choices, greedy repetition, comments
//! and whitespace skipped before every token except inside numbers, and
//! case-insensitive keywords that match as prefixes. The result holds what
//! `nx.nx_pydot.from_pydot` reads from the first graph: its type, strictness
//! and name, its attribute assignments, its nodes grouped by name and its
//! edges grouped by endpoints, in pydot's order. `None` lets NetworkX parse
//! (pydot fails, or raises, on the document).

/// `(name, value)` pairs in order; `None` for an attribute without a value.
pub type Attrs = Vec<(String, Option<String>)>;

/// An edge endpoint: a node id as written (quotes and port kept), or a
/// subgraph, given by the names of the nodes in it.
#[derive(Clone)]
pub enum Endpoint {
    Name(String),
    Subgraph(Vec<String>),
}

pub struct DotGraph {
    pub digraph: bool,
    pub strict: bool,
    pub name: String,
    pub attrs: Attrs,
    /// Node statements grouped by node name, in order of first appearance.
    pub nodes: Vec<(String, Vec<Attrs>)>,
    /// Edges grouped by their pair of endpoints (subgraph endpoints are
    /// never equal to another), in order of first appearance.
    pub edges: Vec<Vec<(Endpoint, Endpoint, Attrs)>>,
}

/// What a statement adds to the graph it is in.
enum Elem {
    Node(String, Attrs),
    Edge(Endpoint, Endpoint, Attrs),
    Assign(Attrs),
    /// A subgraph: kept by the graph, but `from_pydot` doesn't read it.
    Subgraph,
}

struct Parser<'a> {
    s: Vec<char>,
    ident: &'a [u8],
    /// A parse action would raise (an attribute named like a parameter, or a
    /// case-insensitive keyword compared with non-ASCII text).
    bail: bool,
}

type R<T> = Option<(usize, T)>;

impl<'a> Parser<'a> {
    fn at(&self, i: usize) -> Option<char> {
        self.s.get(i).copied()
    }

    fn starts(&self, i: usize, lit: &str) -> bool {
        lit.chars()
            .enumerate()
            .all(|(k, c)| self.at(i + k) == Some(c))
    }

    /// pyparsing's pre-parse: comments (`//` or `#` to the end of the line,
    /// `/* ... */`) and the whitespace " \t\n\r", in any mix.
    fn skip(&self, mut i: usize) -> usize {
        loop {
            while matches!(self.at(i), Some(' ' | '\t' | '\n' | '\r')) {
                i += 1;
            }
            if self.starts(i, "//") || self.at(i) == Some('#') {
                while !matches!(self.at(i), None | Some('\n')) {
                    i += 1;
                }
            } else if self.starts(i, "/*") {
                let mut j = i + 2;
                loop {
                    match self.at(j) {
                        None => return i,
                        Some('*') if self.at(j + 1) == Some('/') => break,
                        _ => j += 1,
                    }
                }
                i = j + 2;
            } else {
                return i;
            }
        }
    }

    fn lit(&self, i: usize, lit: &str) -> Option<usize> {
        let i = self.skip(i);
        self.starts(i, lit).then(|| i + lit.chars().count())
    }

    /// `CaselessLiteral(kw)`: the next characters, upper-cased, equal it.
    fn caseless(&mut self, i: usize, kw: &str) -> Option<usize> {
        let i = self.skip(i);
        for (off, k) in kw.chars().enumerate() {
            let c = self.at(i + off)?;
            if !c.is_ascii() {
                // Python's str.upper() is used there; leave it to pydot.
                self.bail = true;
                return None;
            }
            if !c.eq_ignore_ascii_case(&k) {
                return None;
            }
        }
        Some(i + kw.len())
    }

    fn is_ident(&self, c: char) -> bool {
        let u = c as u32;
        u < 0x10000 && (self.ident[(u >> 3) as usize] >> (u & 7)) & 1 == 1
    }

    fn identifier(&self, i: usize) -> R<String> {
        let i = self.skip(i);
        let mut j = i;
        while self.at(j).is_some_and(|c| self.is_ident(c)) {
            j += 1;
        }
        (j > i).then(|| (j, self.s[i..j].iter().collect()))
    }

    /// `QuotedString('"', multiline=True, unquote_results=False,
    /// esc_char="\\")`, then `push_dbl_quoted`.
    fn dbl_quoted(&self, i: usize) -> R<String> {
        let i = self.skip(i);
        if self.at(i) != Some('"') {
            return None;
        }
        let mut j = i + 1;
        loop {
            match self.at(j)? {
                '"' => break,
                '\\' => {
                    self.at(j + 1)?;
                    j += 2;
                }
                _ => j += 1,
            }
        }
        let raw: String = self.s[i..=j].iter().collect();
        let raw = if raw.contains('\\') {
            raw.replace("\\\r\n", "").replace("\\\n", "")
        } else {
            raw
        };
        Some((j + 1, raw))
    }

    /// The custom `HTML` token: `<` ... the matching `>`.
    fn html(&self, i: usize) -> R<String> {
        let i = self.skip(i);
        if self.at(i) != Some('<') {
            return None;
        }
        let mut open = 1;
        let mut j = i + 1;
        while let Some(c) = self.at(j) {
            if c == '<' {
                open += 1;
            } else if c == '>' {
                open -= 1;
            }
            j += 1;
            if open == 0 {
                return Some((j, self.s[i..j].iter().collect()));
            }
        }
        None
    }

    /// `ID`: concatenated quoted strings | a quoted string | an identifier
    /// | HTML, as `push_ID` gives it.
    fn id(&self, i: usize) -> R<String> {
        if let Some((mut j, first)) = self.dbl_quoted(i) {
            let mut parts = vec![first];
            while let Some(k) = self.lit(j, "+") {
                match self.dbl_quoted(k) {
                    Some((k, s)) => {
                        parts.push(s);
                        j = k;
                    }
                    None => break,
                }
            }
            if parts.len() >= 2 {
                let inner: String = parts
                    .iter()
                    .map(|s| {
                        let c: Vec<char> = s.chars().collect();
                        c[1..c.len() - 1].iter().collect::<String>()
                    })
                    .collect();
                return Some((j, format!("\"{inner}\"")));
            }
            let (j, s) = self.dbl_quoted(i)?;
            return Some((j, s));
        }
        if let Some(r) = self.identifier(i) {
            return Some(r);
        }
        self.html(i)
    }

    /// `float_number`: `-`? then digits and dots, with nothing between.
    fn float(&self, i: usize) -> R<String> {
        let i = self.skip(i);
        let mut j = i;
        if self.at(j) == Some('-') {
            j += 1;
        }
        let start = j;
        while matches!(self.at(j), Some('0'..='9' | '.')) {
            j += 1;
        }
        (j > start).then(|| (j, self.s[i..j].iter().collect()))
    }

    fn righthand(&self, i: usize) -> R<String> {
        self.float(i).or_else(|| self.id(i))
    }

    /// `node_id`: one to three IDs joined by `:`.
    fn node_id(&self, i: usize) -> R<String> {
        let (mut j, first) = self.id(i)?;
        let mut out = first;
        for _ in 0..2 {
            let Some(k) = self.lit(j, ":") else { break };
            let Some((k, part)) = self.id(k) else { break };
            out.push(':');
            out.push_str(&part);
            j = k;
        }
        Some((j, out))
    }

    /// `a_list`: `ID (= righthand)? ,?`, one or more.
    fn a_list(&self, i: usize) -> R<Attrs> {
        let mut out = Attrs::new();
        let mut j = i;
        while let Some((k, name)) = self.id(j) {
            let mut k = k;
            let mut value = None;
            if let Some(e) = self.lit(k, "=") {
                if let Some((e, v)) = self.righthand(e) {
                    k = e;
                    value = Some(v);
                }
            }
            if let Some(c) = self.lit(k, ",") {
                k = c;
            }
            out.push((name, value));
            j = k;
        }
        (!out.is_empty()).then_some((j, out))
    }

    /// `attr_list`: `[ a_list? ]`, one or more, merged.
    fn attr_list(&self, i: usize) -> R<Attrs> {
        let mut out = Attrs::new();
        let mut j = i;
        let mut any = false;
        while let Some(k) = self.lit(j, "[") {
            let (k, attrs) = match self.a_list(k) {
                Some((k, a)) => (k, a),
                None => (k, Attrs::new()),
            };
            let Some(k) = self.lit(k, "]") else { break };
            out.extend(attrs);
            j = k;
            any = true;
        }
        any.then_some((j, out))
    }

    /// Building `Node(name, **attrs)` / `Edge(src, dst, **attrs)` raises
    /// TypeError for an attribute named like one of their parameters.
    fn check_kwargs(&mut self, attrs: &Attrs, params: &[&str]) {
        if attrs.iter().any(|(k, _)| params.contains(&k.as_str())) {
            self.bail = true;
        }
    }

    /// `{ stmt_list? } ;?`: the statements' elements.
    fn graph_stmt(&mut self, i: usize) -> R<Vec<Elem>> {
        let j = self.lit(i, "{")?;
        let (j, elems) = match self.stmt_list(j) {
            Some(r) => r,
            None => (j, Vec::new()),
        };
        let mut j = self.lit(j, "}")?;
        if let Some(k) = self.lit(j, ";") {
            j = k;
        }
        Some((j, elems))
    }

    /// `subgraph ID? graph_stmt`.
    fn subgraph(&mut self, i: usize) -> R<Vec<Elem>> {
        let mut j = self.caseless(i, "subgraph")?;
        if let Some((k, _)) = self.id(j) {
            j = k;
        }
        self.graph_stmt(j)
    }

    /// The names of the nodes a subgraph holds (`obj_dict["nodes"]` keys).
    fn node_names(elems: &[Elem]) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        for e in elems {
            if let Elem::Node(n, _) = e {
                if !names.contains(n) {
                    names.push(n.clone());
                }
            }
        }
        names
    }

    fn edge_point(&mut self, i: usize) -> R<Endpoint> {
        if let Some((j, elems)) = self.subgraph(i) {
            return Some((j, Endpoint::Subgraph(Self::node_names(&elems))));
        }
        if let Some((j, elems)) = self.graph_stmt(i) {
            return Some((j, Endpoint::Subgraph(Self::node_names(&elems))));
        }
        self.node_id(i).map(|(j, s)| (j, Endpoint::Name(s)))
    }

    fn edgeop(&self, i: usize) -> Option<usize> {
        self.lit(i, "--").or_else(|| self.lit(i, "->"))
    }

    fn edge_stmt(&mut self, i: usize) -> R<Vec<Elem>> {
        let (mut j, first) = self.edge_point(i)?;
        let mut points = vec![first];
        while let Some(k) = self.edgeop(j) {
            let Some((k, p)) = self.edge_point(k) else {
                break;
            };
            points.push(p);
            j = k;
        }
        if points.len() < 2 {
            return None;
        }
        let attrs = match self.attr_list(j) {
            Some((k, a)) => {
                j = k;
                a
            }
            None => Attrs::new(),
        };
        self.check_kwargs(&attrs, &["src", "dst", "obj_dict"]);
        let edges = points
            .windows(2)
            .map(|w| Elem::Edge(w[0].clone(), w[1].clone(), attrs.clone()))
            .collect();
        Some((j, edges))
    }

    fn default_stmt(&mut self, i: usize) -> R<Vec<Elem>> {
        let (j, kind) = ["graph", "node", "edge"]
            .iter()
            .find_map(|k| self.caseless(i, k).map(|j| (j, *k)))?;
        let (j, attrs) = self.attr_list(j)?;
        self.check_kwargs(&attrs, &["name", "obj_dict"]);
        Some((j, vec![Elem::Node(kind.to_owned(), attrs)]))
    }

    fn assignment(&self, i: usize) -> R<Vec<Elem>> {
        let (j, name) = self.id(i)?;
        let j = self.lit(j, "=")?;
        let (j, value) = self.righthand(j)?;
        Some((j, vec![Elem::Assign(vec![(name, Some(value))])]))
    }

    fn node_stmt(&mut self, i: usize) -> R<Vec<Elem>> {
        let (mut j, name) = self.node_id(i)?;
        let attrs = match self.attr_list(j) {
            Some((k, a)) => {
                j = k;
                a
            }
            None => Attrs::new(),
        };
        if let Some(k) = self.lit(j, ";") {
            j = k;
        }
        self.check_kwargs(&attrs, &["name", "obj_dict"]);
        Some((j, vec![Elem::Node(strip_port(name), attrs)]))
    }

    fn stmt(&mut self, i: usize) -> R<Vec<Elem>> {
        if let Some(r) = self.assignment(i) {
            return Some(r);
        }
        if let Some(r) = self.edge_stmt(i) {
            return Some(r);
        }
        if let Some(r) = self.default_stmt(i) {
            return Some(r);
        }
        if let Some((j, _)) = self.subgraph(i) {
            return Some((j, vec![Elem::Subgraph]));
        }
        if let Some((j, _)) = self.graph_stmt(i) {
            return Some((j, vec![Elem::Subgraph]));
        }
        self.node_stmt(i)
    }

    fn stmt_list(&mut self, i: usize) -> R<Vec<Elem>> {
        let mut out = Vec::new();
        let mut j = i;
        let mut any = false;
        while let Some((k, elems)) = self.stmt(j) {
            out.extend(elems);
            j = self.lit(k, ";").unwrap_or(k);
            any = true;
            if self.bail {
                return None;
            }
        }
        any.then_some((j, out))
    }

    /// The first graph: `strict? (digraph|graph) ID? { stmt_list? } ;?`.
    fn first_graph(&mut self) -> Option<DotGraph> {
        let mut j = 0;
        let mut strict = false;
        if let Some(k) = self.caseless(j, "strict") {
            j = k;
            strict = true;
        }
        let (mut j, digraph) = match self.caseless(j, "digraph") {
            Some(k) => (k, true),
            None => (self.caseless(j, "graph")?, false),
        };
        let mut name = String::new();
        if let Some((k, id)) = self.id(j) {
            j = k;
            name = id;
        }
        let (_, elems) = self.graph_stmt(j)?;
        if self.bail {
            return None;
        }
        Some(build(digraph, strict, name, elems))
    }
}

/// `pydot.Node`'s compass point removal: `a:p` is node `a` (not for quoted
/// names).
fn strip_port(name: String) -> String {
    if name.starts_with('"') {
        return name;
    }
    match name.find(':') {
        Some(idx) if idx > 0 && idx + 1 < name.len() => name[..idx].to_owned(),
        _ => name,
    }
}

/// `add_elements` on a new `Dot`: nodes and edges grouped as pydot's
/// `obj_dict["nodes"]` / `obj_dict["edges"]` dicts group them.
fn build(digraph: bool, strict: bool, name: String, elems: Vec<Elem>) -> DotGraph {
    use std::collections::HashMap;
    let mut attrs = Attrs::new();
    let mut nodes: Vec<(String, Vec<Attrs>)> = Vec::new();
    let mut node_index: HashMap<String, usize> = HashMap::new();
    let mut edges: Vec<Vec<(Endpoint, Endpoint, Attrs)>> = Vec::new();
    let mut edge_index: HashMap<(String, String), usize> = HashMap::new();
    for e in elems {
        match e {
            Elem::Node(n, a) => match node_index.get(&n) {
                Some(&k) => nodes[k].1.push(a),
                None => {
                    node_index.insert(n.clone(), nodes.len());
                    nodes.push((n, vec![a]));
                }
            },
            Elem::Edge(u, v, a) => {
                let key = match (&u, &v) {
                    (Endpoint::Name(x), Endpoint::Name(y)) => Some((x.clone(), y.clone())),
                    _ => None,
                };
                match key.as_ref().and_then(|k| edge_index.get(k)) {
                    Some(&k) => edges[k].push((u, v, a)),
                    None => {
                        if let Some(k) = key {
                            edge_index.insert(k, edges.len());
                        }
                        edges.push(vec![(u, v, a)]);
                    }
                }
            }
            Elem::Assign(a) => attrs.extend(a),
            Elem::Subgraph => {}
        }
    }
    DotGraph {
        digraph,
        strict,
        name,
        attrs,
        nodes,
        edges,
    }
}

/// Parses `text` as pydot 4 does and returns its first graph; `ident` is a
/// bitmap of pyparsing's identifier characters in the Basic Multilingual
/// Plane (`unicode.BasicMultilingualPlane.alphanums + "_."`).
pub fn parse(text: &str, ident: &[u8]) -> Option<DotGraph> {
    if ident.len() != 0x10000 / 8 {
        return None;
    }
    let mut p = Parser {
        s: text.chars().collect(),
        ident,
        bail: false,
    };
    p.first_graph()
}
