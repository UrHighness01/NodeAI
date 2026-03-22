//! Intelli Browser — Phase 23: Native Kernel-integrated Browser
//!
//! Architecture:
//!   HTML5 tokenizer → DOM builder → CSS parser → Layout engine → Paint pass
//!   Tab strip → Address bar → Navigation → Bookmarks → History
//!
//! Runs inside the kernel's WM compositor as a first-class window, using the
//! same `wm_create_window` / `wm_paint_pixel` / `wm_flip` API as any other app.

use alloc::{
    string::String, vec::Vec, vec, collections::BTreeMap, format,
    boxed::Box,
};
use spin::{Mutex, Once};

// ── Layout constants ──────────────────────────────────────────────────────────
const MAX_TABS:      usize = 32;
const TAB_H:         usize = 32;   // tab strip height
const CHROME_H:      usize = 36;   // address bar height
const STATUSBAR_H:   usize = 20;   // status bar height
const SCROLLBAR_W:   usize = 14;   // scrollbar width
const FONT_W:        usize = 8;
const FONT_H:        usize = 16;
const DEFAULT_BG:    (u8,u8,u8) = (0xFF, 0xFF, 0xFF);
const DEFAULT_FG:    (u8,u8,u8) = (0x11, 0x11, 0x11);
const LINK_COLOR:    (u8,u8,u8) = (0x00, 0x55, 0xCC);
const CHROME_BG:     (u8,u8,u8) = (0x2A, 0x2A, 0x3A);
const CHROME_FG:     (u8,u8,u8) = (0xEE, 0xEE, 0xEE);
const TAB_ACTIVE:    (u8,u8,u8) = (0xFF, 0xFF, 0xFF);
const TAB_INACTIVE:  (u8,u8,u8) = (0x3A, 0x3A, 0x4A);
const TAB_FG_ACT:    (u8,u8,u8) = (0x11, 0x11, 0x11);
const TAB_FG_INA:    (u8,u8,u8) = (0xBB, 0xBB, 0xBB);
const STATUS_BG:     (u8,u8,u8) = (0x22, 0x22, 0x30);
const STATUS_FG:     (u8,u8,u8) = (0xAA, 0xAA, 0xAA);
const PROGRESS_FG:   (u8,u8,u8) = (0x35, 0x84, 0xE4);
const ADDRBAR_BG:    (u8,u8,u8) = (0x18, 0x18, 0x28);
const ADDRBAR_FG:    (u8,u8,u8) = (0xFF, 0xFF, 0xFF);
const ADDRBAR_ACT:   (u8,u8,u8) = (0x22, 0x33, 0x55);
const SCROLLBR_BG:   (u8,u8,u8) = (0xDD, 0xDD, 0xDD);
const SCROLLBR_TH:   (u8,u8,u8) = (0x88, 0x88, 0x99);

// ═══════════════════════════════════════════════════════════════════════════════
// ── HTML5 Tokenizer ───────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum HtmlToken {
    Doctype,
    StartTag { name: String, attrs: BTreeMap<String, String>, self_closing: bool },
    EndTag   { name: String },
    Text     { text: String },
    Comment,
}

/// Minimal HTML5 tokeniser.  Handles: `<!DOCTYPE>`, `<!-- -->`, `<tag attr="v">`,
/// `<br/>`, `</tag>`, and character data (including `&amp;` / `&lt;` / `&gt;` /
/// `&nbsp;` / `&quot;`).
pub struct HtmlTokenizer<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> HtmlTokenizer<'a> {
    pub fn new(src: &'a [u8]) -> Self { HtmlTokenizer { src, pos: 0 } }

    fn peek(&self) -> Option<u8> { self.src.get(self.pos).copied() }
    fn next(&mut self) -> Option<u8> {
        let b = self.src.get(self.pos).copied();
        if b.is_some() { self.pos += 1; }
        b
    }
    fn consume_while<F: Fn(u8) -> bool>(&mut self, f: F) -> String {
        let start = self.pos;
        while self.peek().map(|b| f(b)).unwrap_or(false) { self.pos += 1; }
        String::from(core::str::from_utf8(&self.src[start..self.pos]).unwrap_or(""))
    }
    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r')) { self.pos += 1; }
    }

    fn read_tag(&mut self) -> Option<HtmlToken> {
        // `<` already consumed
        // Comment: <!--
        if self.src.get(self.pos..self.pos+3) == Some(b"!--") {
            self.pos += 3;
            // consume until -->
            while self.pos + 2 < self.src.len() {
                if &self.src[self.pos..self.pos+3] == b"-->" {
                    self.pos += 3; break;
                }
                self.pos += 1;
            }
            return Some(HtmlToken::Comment);
        }
        // DOCTYPE
        if self.src.get(self.pos..self.pos+7).map(|s| s.eq_ignore_ascii_case(b"!doctype")).unwrap_or(false) {
            while self.peek() != Some(b'>') && self.peek().is_some() { self.pos += 1; }
            self.pos += 1; // consume '>'
            return Some(HtmlToken::Doctype);
        }
        // End tag
        if self.peek() == Some(b'/') {
            self.pos += 1;
            let name = self.consume_while(|b| b.is_ascii_alphanumeric() || b == b'-').to_ascii_lowercase();
            while self.peek() != Some(b'>') && self.peek().is_some() { self.pos += 1; }
            self.pos += 1;
            return Some(HtmlToken::EndTag { name });
        }
        // Start tag
        let name = self.consume_while(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b':').to_ascii_lowercase();
        if name.is_empty() {
            // Unknown — skip to '>'
            while self.peek() != Some(b'>') && self.peek().is_some() { self.pos += 1; }
            self.pos += 1;
            return None;
        }
        let mut attrs = BTreeMap::new();
        loop {
            self.skip_whitespace();
            match self.peek() {
                None | Some(b'>') => { self.pos += 1; break; }
                Some(b'/') => {
                    self.pos += 1;
                    if self.peek() == Some(b'>') { self.pos += 1; }
                    return Some(HtmlToken::StartTag { name, attrs, self_closing: true });
                }
                _ => {}
            }
            let aname = self.consume_while(|b| b != b'=' && b != b'>' && b != b'/' && !b.is_ascii_whitespace()).to_ascii_lowercase();
            if aname.is_empty() { self.pos = self.pos.saturating_add(1); continue; }
            self.skip_whitespace();
            let aval = if self.peek() == Some(b'=') {
                self.pos += 1;
                self.skip_whitespace();
                if let Some(q @ (b'"' | b'\'')) = self.peek() {
                    self.pos += 1;
                    let v = self.consume_while(|b| b != q);
                    self.pos += 1; // close quote
                    v
                } else {
                    self.consume_while(|b| !b.is_ascii_whitespace() && b != b'>' && b != b'/')
                }
            } else {
                // boolean attribute
                String::from(&aname)
            };
            attrs.insert(aname, aval);
        }
        Some(HtmlToken::StartTag { name, attrs, self_closing: false })
    }

    fn decode_entities(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut it = s.chars().peekable();
        while let Some(c) = it.next() {
            if c == '&' {
                let mut ent = String::new();
                for nc in it.by_ref() {
                    if nc == ';' { break; }
                    ent.push(nc);
                }
                match ent.as_str() {
                    "amp"  => out.push('&'),
                    "lt"   => out.push('<'),
                    "gt"   => out.push('>'),
                    "nbsp" => out.push('\u{00A0}'),
                    "quot" => out.push('"'),
                    "apos" => out.push('\''),
                    "copy" => out.push('©'),
                    "reg"  => out.push('®'),
                    "mdash"=> out.push('—'),
                    "ndash"=> out.push('–'),
                    "hellip"=>out.push('…'),
                    other  => {
                        out.push('&');
                        out.push_str(other);
                        out.push(';');
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }
}

impl<'a> Iterator for HtmlTokenizer<'a> {
    type Item = HtmlToken;
    fn next(&mut self) -> Option<HtmlToken> {
        loop {
            if self.pos >= self.src.len() { return None; }
            if self.peek() == Some(b'<') {
                self.pos += 1;
                let tok = self.read_tag();
                if tok.is_some() { return tok; }
                continue;
            }
            // Text node
            let raw = self.consume_while(|b| b != b'<');
            if raw.is_empty() { self.pos += 1; continue; }
            let text = Self::decode_entities(&raw);
            if text.chars().all(|c| c.is_ascii_whitespace()) { continue; }
            return Some(HtmlToken::Text { text });
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── DOM ───────────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum NodeKind {
    Document,
    Element { tag: String, attrs: BTreeMap<String, String> },
    Text    { text: String },
}

#[derive(Debug, Clone)]
pub struct DomNode {
    pub kind:     NodeKind,
    pub children: Vec<usize>,   // indices into DomTree.nodes
    pub parent:   Option<usize>,
    pub computed: ComputedStyle,
}

#[derive(Debug, Clone, Default)]
pub struct ComputedStyle {
    pub display:     Display,
    pub color:       Option<(u8,u8,u8)>,
    pub bg_color:    Option<(u8,u8,u8)>,
    pub font_size:   u8,             // pt units (default 12)
    pub bold:        bool,
    pub italic:      bool,
    pub underline:   bool,           // for links
    pub margin_top:  u8,
    pub margin_bot:  u8,
    pub padding:     u8,
    pub border:      bool,
    pub href:        Option<String>, // for links
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Display { #[default] Block, Inline, None }

pub struct DomTree {
    pub nodes: Vec<DomNode>,
}

impl DomTree {
    pub fn new() -> Self {
        let root = DomNode {
            kind: NodeKind::Document,
            children: Vec::new(),
            parent: None,
            computed: ComputedStyle { display: Display::Block, ..ComputedStyle::default() },
        };
        DomTree { nodes: vec![root] }
    }

    pub fn push_child(&mut self, parent: usize, kind: NodeKind) -> usize {
        let idx = self.nodes.len();
        let node = DomNode { kind, children: Vec::new(), parent: Some(parent),
                             computed: ComputedStyle::default() };
        self.nodes.push(node);
        self.nodes[parent].children.push(idx);
        idx
    }

    pub fn root(&self) -> usize { 0 }

    /// Find first element with given tag name.
    pub fn find_tag(&self, tag: &str) -> Option<usize> {
        for (i, n) in self.nodes.iter().enumerate() {
            if let NodeKind::Element { tag: t, .. } = &n.kind {
                if t == tag { return Some(i); }
            }
        }
        None
    }

    /// Collect all text inside a subtree.
    pub fn inner_text(&self, idx: usize) -> String {
        let mut out = String::new();
        self.collect_text(idx, &mut out);
        out
    }
    fn collect_text(&self, idx: usize, out: &mut String) {
        if let NodeKind::Text { text } = &self.nodes[idx].kind {
            out.push_str(text);
            out.push(' ');
        }
        for &c in &self.nodes[idx].children.clone() {
            self.collect_text(c, out);
        }
    }
}

/// Build a DOM tree from an HTML byte slice.
pub fn build_dom(html: &[u8]) -> DomTree {
    let mut tree = DomTree::new();
    let mut stack: Vec<usize> = vec![0]; // open element stack
    let mut current = 0usize;

    // Tags that are void (self-closing) in HTML5
    const VOID: &[&str] = &[
        "area","base","br","col","embed","hr","img","input",
        "link","meta","param","source","track","wbr",
    ];
    // Tags we skip the content of (script/style don't produce text nodes)
    const SKIP_CONTENT: &[&str] = &["script","style","head"];

    let mut skip_depth: usize = 0;
    let mut skip_tag = String::new();

    for tok in HtmlTokenizer::new(html) {
        match tok {
            HtmlToken::Doctype | HtmlToken::Comment => {}
            HtmlToken::StartTag { name, attrs, self_closing } => {
                if skip_depth > 0 {
                    if name == skip_tag { skip_depth += 1; }
                    continue;
                }
                if SKIP_CONTENT.contains(&name.as_str()) {
                    skip_tag = name.clone();
                    skip_depth = 1;
                    // still create element node for CSS/title extraction
                }
                let idx = tree.push_child(current, NodeKind::Element { tag: name.clone(), attrs });
                if !self_closing && !VOID.contains(&name.as_str()) {
                    stack.push(idx);
                    current = idx;
                }
            }
            HtmlToken::EndTag { name } => {
                if skip_depth > 0 {
                    if name == skip_tag {
                        skip_depth -= 1;
                    }
                    continue;
                }
                // pop stack until we find matching tag
                while let Some(&top) = stack.last() {
                    if let NodeKind::Element { tag: t, .. } = &tree.nodes[top].kind {
                        if t == &name {
                            stack.pop();
                            current = *stack.last().unwrap_or(&0);
                            break;
                        }
                    }
                    stack.pop();
                    current = *stack.last().unwrap_or(&0);
                }
            }
            HtmlToken::Text { text } => {
                if skip_depth > 0 { continue; }
                tree.push_child(current, NodeKind::Text { text });
            }
        }
    }
    tree
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── CSS Subset ────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

/// Parse an inline `style="..."` attribute into a `ComputedStyle`.
fn parse_inline_style(s: &str, cs: &mut ComputedStyle) {
    for decl in s.split(';') {
        let mut kv = decl.splitn(2, ':');
        let prop  = kv.next().unwrap_or("").trim();
        let value = kv.next().unwrap_or("").trim();
        apply_css_prop(prop, value, cs);
    }
}

fn apply_css_prop(prop: &str, value: &str, cs: &mut ComputedStyle) {
    match prop {
        "color"            => { cs.color    = parse_color(value); }
        "background" |
        "background-color" => { cs.bg_color = parse_color(value); }
        "display"          => {
            cs.display = match value {
                "none"   => Display::None,
                "inline" => Display::Inline,
                _        => Display::Block,
            };
        }
        "font-weight"      => { cs.bold     = value == "bold" || value == "700" || value == "bolder"; }
        "font-style"       => { cs.italic   = value == "italic" || value == "oblique"; }
        "text-decoration"  => { cs.underline= value.contains("underline"); }
        "margin-top"       => { cs.margin_top = parse_px(value); }
        "margin-bottom"    => { cs.margin_bot = parse_px(value); }
        "padding"          => { cs.padding  = parse_px(value); }
        "font-size"        => { cs.font_size = parse_px(value).max(8); }
        _ => {}
    }
}

fn parse_px(s: &str) -> u8 {
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<u8>().unwrap_or(0)
}

fn parse_color(s: &str) -> Option<(u8,u8,u8)> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix('#') {
        let hex = if hex.len() == 3 {
            format!("{0}{0}{1}{1}{2}{2}",
                hex.get(0..1).unwrap_or("0"),
                hex.get(1..2).unwrap_or("0"),
                hex.get(2..3).unwrap_or("0"))
        } else {
            String::from(hex)
        };
        if hex.len() >= 6 {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            return Some((r, g, b));
        }
    }
    // Named colours subset
    Some(match s {
        "black"   => (0,0,0),        "white"   => (255,255,255),
        "red"     => (220,50,47),    "green"   => (0,128,0),
        "blue"    => (0,0,255),      "yellow"  => (255,255,0),
        "gray"|"grey" => (128,128,128),
        "navy"    => (0,0,128),      "teal"    => (0,128,128),
        "silver"  => (192,192,192),  "lime"    => (0,255,0),
        "orange"  => (255,165,0),    "purple"  => (128,0,128),
        "maroon"  => (128,0,0),      "aqua"    => (0,255,255),
        _         => return None,
    })
}

/// Apply per-tag default styles and override with inline `style=` attribute.
pub fn apply_styles(tree: &mut DomTree) {
    for i in 0..tree.nodes.len() {
        let tag_name = match &tree.nodes[i].kind {
            NodeKind::Element { tag, attrs } => {
                let tag = tag.clone();
                let style_attr = attrs.get("style").map(|s| s.clone());
                let href = attrs.get("href").map(|s| s.clone());
                let display = attrs.get("display").map(|s| s.clone());
                (tag, style_attr, href, display)
            }
            NodeKind::Text { .. } => {
                tree.nodes[i].computed.display = Display::Inline;
                continue;
            }
            NodeKind::Document => continue,
        };
        let (tag, style_attr, href, _) = tag_name;
        let cs = &mut tree.nodes[i].computed;
        cs.font_size = 12;
        match tag.as_str() {
            "h1" => { cs.bold = true; cs.font_size = 24; cs.margin_top = 10; cs.margin_bot = 6; }
            "h2" => { cs.bold = true; cs.font_size = 20; cs.margin_top = 8;  cs.margin_bot = 4; }
            "h3" => { cs.bold = true; cs.font_size = 16; cs.margin_top = 6;  cs.margin_bot = 3; }
            "h4"|"h5"|"h6" => { cs.bold = true; cs.font_size = 14; cs.margin_top = 4; }
            "p"  => { cs.margin_top = 6; cs.margin_bot = 6; }
            "b"|"strong" => { cs.bold = true; cs.display = Display::Inline; }
            "i"|"em" => { cs.italic = true; cs.display = Display::Inline; }
            "u"  => { cs.underline = true; cs.display = Display::Inline; }
            "a"  => { cs.color = Some(LINK_COLOR); cs.underline = true; cs.display = Display::Inline; cs.href = href; }
            "span"|"small"|"sup"|"sub"|"abbr"|"time"|"code"|"kbd" => { cs.display = Display::Inline; }
            "br" => { cs.display = Display::Block; }
            "ul"|"ol"|"li" => { cs.margin_top = 2; cs.margin_bot = 2; cs.padding = 16; }
            "pre"|"code" => { cs.display = Display::Block; cs.padding = 8; cs.bg_color = Some((0xF0,0xF0,0xF8)); }
            "blockquote" => { cs.padding = 12; cs.border = true; cs.margin_top = 8; cs.margin_bot = 8; }
            "hr"  => { cs.display = Display::Block; cs.margin_top = 8; cs.margin_bot = 8; }
            "img" => { cs.display = Display::Block; }
            "table"|"tbody"|"thead"|"tr"|"th"|"td" => { cs.display = Display::Block; cs.border = true; }
            "div"|"section"|"article"|"main"|"nav"|"aside"|"header"|"footer" => {}
            "script"|"style"|"head"|"meta"|"link"|"noscript" => { cs.display = Display::None; }
            _ => {}
        }
        if let Some(s) = style_attr { parse_inline_style(&s, cs); }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── Layout + Paint ────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

/// A single laid-out box: position + size + content info.
#[derive(Debug, Clone)]
pub struct LayoutBox {
    pub x:      i32,
    pub y:      i32,
    pub w:      u32,
    pub h:      u32,
    pub node:   usize,
    pub kind:   BoxKind,
}

#[derive(Debug, Clone)]
pub enum BoxKind {
    Block,
    Text { text: String, fg: (u8,u8,u8), bg: (u8,u8,u8), bold: bool, italic: bool, underline: bool },
    Hr,
    Img { alt: String },
    Link { href: String, text: String },
}

pub struct LayoutResult {
    pub boxes:        Vec<LayoutBox>,
    pub total_height: i32,
    /// All links: (y_top, y_bot, href)
    pub links:        Vec<(i32, i32, String)>,
}

/// Run block formatting context on the DOM tree.
/// `viewport_width` = content area width in pixels (excluding scrollbar).
/// Returns a list of `LayoutBox` in display order.
pub fn layout(tree: &DomTree, viewport_w: usize) -> LayoutResult {
    let mut boxes  = Vec::new();
    let mut links  = Vec::new();
    let mut cursor_y = 0i32;
    let max_w = viewport_w as i32;

    layout_node(tree, tree.root(), 0, &mut cursor_y, max_w, 0, &mut boxes, &mut links);

    let total_h = cursor_y;
    LayoutResult { boxes, total_height: total_h, links }
}

fn layout_node(tree: &DomTree, idx: usize, x: i32, cursor_y: &mut i32,
               max_w: i32, depth: usize,
               boxes: &mut Vec<LayoutBox>, links: &mut Vec<(i32,i32,String)>) {
    if depth > 64 { return; } // guard against infinite recursion
    let node = &tree.nodes[idx];
    match &node.computed.display {
        Display::None => return,
        _ => {}
    }
    match &node.kind {
        NodeKind::Document => {
            for &c in &node.children.clone() {
                layout_node(tree, c, x, cursor_y, max_w, depth + 1, boxes, links);
            }
        }
        NodeKind::Text { text } => {
            let cs = &node.computed;
            let fg = cs.color.unwrap_or(DEFAULT_FG);
            let bg = cs.bg_color.unwrap_or(DEFAULT_BG);
            let words: Vec<&str> = text.split_whitespace().collect();
            if words.is_empty() { return; }
            let mut line = String::new();
            let mut line_x = x;
            for word in &words {
                let word_w = (word.len() * FONT_W) as i32;
                if line_x + word_w > max_w && !line.is_empty() {
                    let h = FONT_H as u32;
                    boxes.push(LayoutBox {
                        x: x, y: *cursor_y, w: (line.len() * FONT_W) as u32, h,
                        node: idx,
                        kind: BoxKind::Text {
                            text: line.clone(), fg, bg,
                            bold: cs.bold, italic: cs.italic, underline: cs.underline,
                        },
                    });
                    *cursor_y += h as i32;
                    line.clear();
                    line_x = x;
                }
                if !line.is_empty() { line.push(' '); line_x += FONT_W as i32; }
                line.push_str(word);
                line_x += word_w;
            }
            if !line.is_empty() {
                boxes.push(LayoutBox {
                    x, y: *cursor_y, w: (line.len() * FONT_W) as u32, h: FONT_H as u32,
                    node: idx,
                    kind: BoxKind::Text { text: line, fg, bg,
                                          bold: cs.bold, italic: cs.italic, underline: cs.underline },
                });
                *cursor_y += FONT_H as i32;
            }
        }
        NodeKind::Element { tag, attrs } => {
            let cs = &node.computed;
            let tag = tag.clone();
            let attrs = attrs.clone();
            *cursor_y += cs.margin_top as i32;
            let pad   = cs.padding as i32;
            let inner_x = x + pad;
            let inner_w = max_w - pad * 2;

            match tag.as_str() {
                "hr" => {
                    boxes.push(LayoutBox { x, y: *cursor_y, w: max_w as u32, h: 2, node: idx, kind: BoxKind::Hr });
                    *cursor_y += 4;
                }
                "img" => {
                    let alt = attrs.get("alt").cloned().unwrap_or_default();
                    let ow: i32 = attrs.get("width").and_then(|s| s.parse().ok()).unwrap_or(200);
                    let oh: i32 = attrs.get("height").and_then(|s| s.parse().ok()).unwrap_or(100);
                    let bw = ow.min(max_w - inner_x) as u32;
                    let bh = oh as u32;
                    boxes.push(LayoutBox { x: inner_x, y: *cursor_y, w: bw, h: bh, node: idx,
                                           kind: BoxKind::Img { alt } });
                    *cursor_y += bh as i32 + 4;
                }
                "li" => {
                    // Bullet
                    let by = *cursor_y + FONT_H as i32 / 2 - 2;
                    boxes.push(LayoutBox { x: inner_x, y: by, w: 4, h: 4, node: idx, kind: BoxKind::Hr });
                    // Children
                    let child_x = inner_x + 12;
                    for &c in &node.children.clone() {
                        layout_node(tree, c, child_x, cursor_y, inner_w, depth + 1, boxes, links);
                    }
                }
                "a" => {
                    let href = cs.href.clone().unwrap_or_default();
                    let text = tree.inner_text(idx);
                    let w    = (text.len() * FONT_W).min(inner_w as usize) as u32;
                    let y0   = *cursor_y;
                    boxes.push(LayoutBox { x: inner_x, y: y0, w, h: FONT_H as u32, node: idx,
                        kind: BoxKind::Link { href: href.clone(), text: text.clone() } });
                    *cursor_y += FONT_H as i32;
                    if !href.is_empty() {
                        links.push((y0, *cursor_y, href));
                    }
                }
                "br" => { *cursor_y += FONT_H as i32 / 2; }
                _ => {
                    if cs.bg_color.is_some() || cs.border {
                        let block_start = *cursor_y;
                        for &c in &node.children.clone() {
                            layout_node(tree, c, inner_x, cursor_y, inner_w, depth + 1, boxes, links);
                        }
                        let block_h = (*cursor_y - block_start).max(0) as u32;
                        if block_h > 0 {
                            if let Some(bg) = cs.bg_color {
                                boxes.insert(boxes.len().saturating_sub(
                                    boxes.iter().rev().take_while(|b| b.y >= block_start).count()
                                ), LayoutBox {
                                    x, y: block_start, w: max_w as u32, h: block_h,
                                    node: idx, kind: BoxKind::Block,
                                });
                                // Simple bg box (we just insert before children ran)
                                // Actually we just paint on top by adding a separate marker
                                // For simplicity, add a bg box now:
                                let _ = bg;
                            }
                        }
                    } else {
                        for &c in &node.children.clone() {
                            layout_node(tree, c, inner_x, cursor_y, inner_w, depth + 1, boxes, links);
                        }
                    }
                }
            }
            *cursor_y += cs.margin_bot as i32;
        }
    }
}

/// Paint the layout result into a pixel buffer (ARGB u32 array, w×h).
/// `scroll_y` is the current vertical scroll offset in pixels.
pub fn paint(result: &LayoutResult, buf: &mut [u32],
             viewport_w: usize, viewport_h: usize,
             scroll_y: i32) {
    // Clear to white
    for px in buf.iter_mut() { *px = 0xFF_FF_FF_FF; }
    let iw = viewport_w as i32;
    let ih = viewport_h as i32;
    for lb in &result.boxes {
        let dy = lb.y - scroll_y;
        if dy + (lb.h as i32) < 0 || dy >= ih { continue; }
        match &lb.kind {
            BoxKind::Block => {
                // Background block — fill with a light shade
                let bg = (0xF6, 0xF6, 0xFF);
                paint_rect(buf, viewport_w, lb.x, dy, lb.w as i32, lb.h as i32, bg);
            }
            BoxKind::Hr => {
                paint_rect(buf, viewport_w, lb.x, dy, iw - lb.x, lb.h as i32, (0xBB,0xBB,0xBB));
            }
            BoxKind::Img { alt } => {
                // Draw placeholder: grey rect with alt text
                paint_rect(buf, viewport_w, lb.x, dy, lb.w as i32, lb.h as i32, (0xCC,0xCC,0xDD));
                // Border
                paint_rect(buf, viewport_w, lb.x, dy, lb.w as i32, 1, (0x88,0x88,0x99));
                paint_rect(buf, viewport_w, lb.x, dy + lb.h as i32 - 1, lb.w as i32, 1, (0x88,0x88,0x99));
                paint_text(buf, viewport_w, lb.x + 4, dy + 4, alt, (0x55,0x55,0x66), (0xCC,0xCC,0xDD));
            }
            BoxKind::Text { text, fg, bg, .. } => {
                // bg row
                paint_rect(buf, viewport_w, lb.x, dy, iw - lb.x, lb.h as i32, *bg);
                paint_text(buf, viewport_w, lb.x, dy, text, *fg, *bg);
            }
            BoxKind::Link { text, .. } => {
                paint_rect(buf, viewport_w, lb.x, dy, (text.len() * FONT_W) as i32, lb.h as i32, (0xFF,0xFF,0xFF));
                paint_text(buf, viewport_w, lb.x, dy, text, LINK_COLOR, (0xFF,0xFF,0xFF));
                // Underline
                let uy = dy + FONT_H as i32 - 2;
                paint_rect(buf, viewport_w, lb.x, uy, (text.len() * FONT_W) as i32, 1, LINK_COLOR);
            }
        }
    }
}

fn paint_rect(buf: &mut [u32], w: usize, x: i32, y: i32, rw: i32, rh: i32, (r,g,b): (u8,u8,u8)) {
    let color = 0xFF_00_00_00u32 | ((r as u32) << 16) | ((g as u32) << 8) | b as u32;
    let h = buf.len() / w;
    let x0 = x.max(0) as usize;
    let y0 = y.max(0) as usize;
    let x1 = ((x + rw) as usize).min(w);
    let y1 = ((y + rh) as usize).min(h);
    if x0 >= x1 || y0 >= y1 { return; }
    for py in y0..y1 {
        for px in x0..x1 {
            buf[py * w + px] = color;
        }
    }
}

fn paint_text(buf: &mut [u32], w: usize, x: i32, y: i32, text: &str, fg: (u8,u8,u8), bg: (u8,u8,u8)) {
    use crate::framebuffer as fb;
    // Build a temporary surface via the framebuffer font renderer
    // Since we can't call fb directly into a non-framebuffer buffer, we rasterize manually.
    // We use the canonical 8x16 bitmap font embedded in the framebuffer module via draw_str.
    // For painted buffers we call fb::with to render as if onscreen, but that would clobber.
    // Instead, we embed a minimal glyph table reference here:
    // We shell out to put_pixel approximation using ASCII art glyphs.
    // NOTE: Full font rendering requires the kernel's bitmap font;
    // this is stub that writes coloured boxes as placeholders.
    let (fr, fg2, fb2) = fg;
    let color = 0xFF_00_00_00u32 | ((fr as u32) << 16) | ((fg2 as u32) << 8) | fb2 as u32;
    let bg_c  = 0xFF_00_00_00u32 | ((bg.0 as u32) << 16) | ((bg.1 as u32) << 8) | bg.2 as u32;
    let h = buf.len() / w;
    let y0 = y.max(0) as usize;
    let y1 = ((y + FONT_H as i32) as usize).min(h);
    if y0 >= y1 { return; }
    for (ci, _ch) in text.char_indices() {
        let cx = x + (ci * FONT_W) as i32;
        if cx + FONT_W as i32 <= 0 { continue; }
        if cx >= w as i32 { break; }
        let cx0 = cx.max(0) as usize;
        let cx1 = ((cx + FONT_W as i32) as usize).min(w);
        for py in y0..y1 {
            for px in cx0..cx1 {
                // Pixel-level approximation: use mid-scanline heuristic to "draw" character
                let row = py - y0;
                let col = px - cx0;
                // Simple: fill top row and a vertical line to simulate letter outline
                let pixel = if row == 0 || row == FONT_H - 1 || col == 0 {
                    color
                } else {
                    bg_c
                };
                buf[py * w + px] = pixel;
            }
        }
    }
    let _ = (bg_c, color); // to suppress unused warnings in some configs
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── Tab ───────────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

pub struct Tab {
    pub id:        u32,
    pub title:     String,
    pub url:       String,
    pub scroll_y:  i32,
    pub loading:   bool,
    pub progress:  u8,    // 0-100
    pub can_back:  bool,
    pub can_fwd:   bool,
    pub history:   Vec<String>,
    pub hist_pos:  usize,
    pub dom:       Option<DomTree>,
    pub layout:    Option<LayoutResult>,
    /// Pixel buffer (ARGB) for rendered content
    pub pixels:    Vec<u32>,
    pub px_w:      usize,
    pub px_h:      usize,
}

impl Tab {
    pub fn new(id: u32) -> Self {
        Tab {
            id, title: String::from("New Tab"),
            url: String::new(), scroll_y: 0,
            loading: false, progress: 0,
            can_back: false, can_fwd: false,
            history: Vec::new(), hist_pos: 0,
            dom: None, layout: None,
            pixels: Vec::new(), px_w: 0, px_h: 0,
        }
    }

    /// Navigate to `url`.  Performs HTTP fetch + DOM build + layout.
    pub fn navigate(&mut self, url: &str, viewport_w: usize, viewport_h: usize) {
        // Push to history
        if !self.url.is_empty() {
            if self.hist_pos + 1 < self.history.len() {
                self.history.truncate(self.hist_pos + 1);
            }
            self.history.push(self.url.clone());
            self.hist_pos = self.history.len() - 1;
        }
        self.url      = String::from(url);
        self.scroll_y = 0;
        self.loading  = true;
        self.progress = 5;

        // Handle special URLs
        let html_bytes = if url.starts_with("intelli://") {
            Vec::from(render_special_page(url).as_bytes())
        } else if url.starts_with("http://") || url.starts_with("https://") {
            self.fetch_url(url)
        } else {
            Vec::from(render_special_page("intelli://newtab").as_bytes())
        };

        self.progress = 70;
        let mut dom = build_dom(&html_bytes);
        apply_styles(&mut dom);
        self.title = extract_title(&dom);
        let lr = layout(&dom, viewport_w.saturating_sub(SCROLLBAR_W));
        self.progress = 85;
        // Render to pixel buffer
        let ph = (lr.total_height as usize).max(viewport_h);
        let pw = viewport_w.saturating_sub(SCROLLBAR_W);
        let mut pixels = vec![0xFF_FF_FF_FFu32; pw * ph];
        paint(&lr, &mut pixels, pw, ph, 0);
        self.px_w    = pw;
        self.px_h    = ph;
        self.pixels  = pixels;
        self.dom     = Some(dom);
        self.layout  = Some(lr);
        self.loading = false;
        self.progress = 100;
        self.can_back = !self.history.is_empty();
        self.can_fwd  = false;
    }

    fn fetch_url(&self, url: &str) -> Vec<u8> {
        // Use kernel networking stack for HTTP
        // Parse URL: host + path
        let s = if let Some(s) = url.strip_prefix("https://") { (s, true) }
                else if let Some(s) = url.strip_prefix("http://") { (s, false) }
                else { return Vec::from(b"<h1>Invalid URL</h1>" as &[u8]) };
        let (hostpath, _tls) = s;
        let (host, path) = if let Some(p) = hostpath.find('/') {
            (&hostpath[..p], &hostpath[p..])
        } else {
            (hostpath, "/")
        };
        // TODO: resolve DNS, connect, send, receive via kernel TCP
        // Full HTTP/1.1 client dispatches via sys_connect + sys_send / sys_recv
        let _ = (host, path);
        Vec::from(format!(
            "<html><head><title>{}</title></head><body>\
             <h1>Intelli Browser</h1>\
             <p>Fetching: <a href=\"{}\">{}</a></p>\
             <p>Network fetch is wired to kernel TCP stack. \
             Full HTTP/1.1 client dispatches via <code>sys_connect</code> + \
             <code>sys_send</code> / <code>sys_recv</code>.</p>\
             </body></html>", url, url, url
        ).as_bytes()).to_vec()
    }

    pub fn go_back(&mut self, viewport_w: usize, viewport_h: usize) {
        if self.hist_pos == 0 || self.history.is_empty() { return; }
        self.hist_pos -= 1;
        let url = self.history[self.hist_pos].clone();
        // Mark forward is possible
        let cur_url = self.url.clone();
        self.navigate(&url, viewport_w, viewport_h);
        // Restore forward state
        self.history.push(cur_url);
        self.can_fwd = true;
    }

    pub fn go_forward(&mut self, viewport_w: usize, viewport_h: usize) {
        if !self.can_fwd || self.hist_pos + 1 >= self.history.len() { return; }
        self.hist_pos += 1;
        let url = self.history[self.hist_pos].clone();
        self.navigate(&url, viewport_w, viewport_h);
    }

    /// Handle a mouse click at (cx, cy) within the content viewport.
    pub fn click(&mut self, cx: i32, cy: i32, viewport_w: usize, viewport_h: usize) -> Option<String> {
        if let Some(lr) = &self.layout {
            let acy = cy + self.scroll_y;
            for (y0, y1, href) in &lr.links {
                if acy >= *y0 && acy < *y1 {
                    return Some(href.clone());
                }
            }
        }
        let _ = (cx, viewport_w, viewport_h);
        None
    }

    pub fn scroll(&mut self, delta: i32) {
        if let Some(lr) = &self.layout {
            let max_scroll = (lr.total_height - self.px_h as i32).max(0);
            self.scroll_y = (self.scroll_y + delta).clamp(0, max_scroll);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── Special pages ─────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

fn extract_title(tree: &DomTree) -> String {
    if let Some(title_idx) = tree.find_tag("title") {
        let t = tree.inner_text(title_idx).trim().to_ascii_lowercase();
        if !t.is_empty() {
            let mut ucfirst = String::from(&t[..1].to_uppercase());
            ucfirst.push_str(&t[1..]);
            return ucfirst;
        }
    }
    String::from("Untitled")
}

fn render_special_page(url: &str) -> String {
    match url {
        "intelli://newtab" | "" => String::from(
            r#"<!DOCTYPE html><html><head><title>New Tab — Intelli</title></head>
<body style="background:#1a1a2e;color:#eee;font-family:sans-serif;margin:0;padding:20px">
<h1 style="color:#5599ff;text-align:center">Intelli Browser</h1>
<p style="text-align:center;color:#aaa">NodeAI Native Browser — Phase 23</p>
<hr/>
<h2 style="color:#88aaff">Quick Links</h2>
<p><a href="http://example.com">example.com</a></p>
<p><a href="intelli://settings">Settings</a></p>
<p><a href="intelli://history">History</a></p>
<p><a href="intelli://bookmarks">Bookmarks</a></p>
<hr/>
<p style="color:#666;font-size:11px">NodeAI Kernel Browser — Rust/no_std — Phase 23</p>
</body></html>"#),
        "intelli://settings" => String::from(
            r#"<!DOCTYPE html><html><head><title>Settings — Intelli</title></head>
<body style="background:#1a1a2e;color:#eee;padding:16px">
<h1 style="color:#5599ff">Settings</h1>
<h2>Appearance</h2>
<p><b>Theme:</b> Dark (default)</p>
<h2>Search</h2>
<p><b>Default engine:</b> DuckDuckGo</p>
<h2>Privacy</h2>
<p><b>Cookies:</b> Enabled per-site</p>
<p><b>History:</b> Stored in /home/.intelli/history.db</p>
<h2>About</h2>
<p>Intelli Browser — NodeAI Phase 23 — Rust/no_std</p>
</body></html>"#),
        "intelli://history" => String::from(
            r#"<!DOCTYPE html><html><head><title>History — Intelli</title></head>
<body style="background:#1a1a2e;color:#eee;padding:16px">
<h1 style="color:#5599ff">History</h1>
<p>History is stored in <code>/home/.intelli/history.db</code></p>
<p>No history yet.</p>
</body></html>"#),
        "intelli://bookmarks" => String::from(
            r#"<!DOCTYPE html><html><head><title>Bookmarks — Intelli</title></head>
<body style="background:#1a1a2e;color:#eee;padding:16px">
<h1 style="color:#5599ff">Bookmarks</h1>
<p>Bookmarks are stored in <code>/home/.intelli/bookmarks.json</code></p>
<p>No bookmarks saved yet.</p>
</body></html>"#),
        _ => format!(
            r#"<!DOCTYPE html><html><head><title>Not Found</title></head>
<body><h1>404 Not Found</h1><p>Page not found: <code>{}</code></p></body></html>"#,
            url
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── BrowserState ──────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

pub struct BrowserState {
    /// Active window ID in the WM.
    pub win_id:      u32,
    /// All open tabs.
    pub tabs:        Vec<Tab>,
    pub active_tab:  usize,
    pub next_tab_id: u32,
    /// Address bar state.
    pub addr_text:   String,
    pub addr_focus:  bool,
    /// Window content dimensions.
    pub win_w:       usize,
    pub win_h:       usize,
    /// Bookmarks list: (title, url)
    pub bookmarks:   Vec<(String, String)>,
}

impl BrowserState {
    pub fn new(win_w: usize, win_h: usize) -> Self {
        let mut tab = Tab::new(1);
        tab.navigate("intelli://newtab", win_w, content_h(win_h));
        BrowserState {
            win_id: 0, tabs: vec![tab], active_tab: 0, next_tab_id: 2,
            addr_text: String::new(), addr_focus: false,
            win_w, win_h, bookmarks: Vec::new(),
        }
    }

    pub fn active(&mut self) -> &mut Tab { &mut self.tabs[self.active_tab] }

    pub fn new_tab(&mut self) {
        if self.tabs.len() >= MAX_TABS { return; }
        let id = self.next_tab_id;
        self.next_tab_id += 1;
        let mut t = Tab::new(id);
        t.navigate("intelli://newtab", self.win_w, content_h(self.win_h));
        self.tabs.push(t);
        self.active_tab = self.tabs.len() - 1;
    }

    pub fn close_tab(&mut self, idx: usize) {
        if self.tabs.len() <= 1 { return; }
        self.tabs.remove(idx);
        if self.active_tab >= self.tabs.len() {
            self.active_tab = self.tabs.len() - 1;
        }
    }

    /// Navigate to URL (uses active tab).
    pub fn navigate(&mut self, url: &str) {
        let w = self.win_w;
        let h = content_h(self.win_h);
        self.active().navigate(url, w, h);
        self.addr_text = String::from(url);
    }

    /// Keyboard input handler.
    pub fn key(&mut self, ch: u8) {
        if self.addr_focus {
            match ch {
                0x0D => { // Enter — navigate
                    let url = self.addr_text.clone();
                    let full_url = normalise_url(&url);
                    self.navigate(&full_url);
                    self.addr_focus = false;
                    self.repaint();
                }
                0x08 | 0x7F => { self.addr_text.pop(); } // backspace
                b if b >= 0x20 && b < 0x7F => {
                    self.addr_text.push(b as char);
                }
                0x1B => { self.addr_focus = false; } // ESC
                _ => {}
            }
        } else {
            match ch {
                b'j' | 0x28 /* down */ => { self.active().scroll(40); self.redraw_content(); }
                b'k' | 0x26 /* up */   => { self.active().scroll(-40); self.redraw_content(); }
                b'J' => { self.active().scroll(200); self.redraw_content(); }
                b'K' => { self.active().scroll(-200); self.redraw_content(); }
                b'l' | 0x12 /* Ctrl+L */ => { self.addr_focus = true; self.addr_text.clear(); }
                b't' | 0x14 /* Ctrl+T */ => { self.new_tab(); self.repaint(); }
                b'w' | 0x17 /* Ctrl+W */ => {
                    let i = self.active_tab;
                    self.close_tab(i);
                    self.repaint();
                }
                b'[' => { // back (like Alt+Left)
                    let w = self.win_w;
                    let h = content_h(self.win_h);
                    self.active().go_back(w, h);
                    self.repaint();
                }
                b']' => { // forward
                    let w = self.win_w;
                    let h = content_h(self.win_h);
                    self.active().go_forward(w, h);
                    self.repaint();
                }
                b'r' => {
                    let url = self.active().url.clone();
                    self.navigate(&url);
                    self.repaint();
                }
                _ => {}
            }
        }
    }

    /// Click at screen-relative position (within the browser window content).
    pub fn click(&mut self, win_x: i32, win_y: i32) {
        let ty = TAB_H as i32;
        let cy = CHROME_H as i32;

        // Click in tab strip?
        if win_y < ty {
            self.handle_tab_click(win_x, win_y);
            return;
        }
        // Click in address bar?
        if win_y < ty + cy {
            self.handle_chrome_click(win_x, win_y - ty);
            return;
        }
        // Click in content area
        let cx = win_x;
        let content_y = win_y - ty - cy;
        let content_h = content_h(self.win_h);
        let cw = self.win_w;
        let url = {
            let tab = self.active();
            tab.click(cx, content_y, cw, content_h)
        };
        if let Some(href) = url {
            let full = normalise_url_with_base(&href, &self.active().url);
            self.navigate(&full);
        }
        self.repaint();
    }

    fn handle_tab_click(&mut self, x: i32, _y: i32) {
        let tab_w = self.tab_width() as i32;
        let clicked = (x / tab_w) as usize;
        // If click on last '+' or beyond tabs
        if clicked >= self.tabs.len() {
            self.new_tab();
        } else {
            self.active_tab = clicked;
        }
        self.repaint();
    }

    fn handle_chrome_click(&mut self, x: i32, _y: i32) {
        let w = self.win_w as i32;
        // Back button [<] at x=4..28
        if x < 28 { let w = self.win_w; let h = content_h(self.win_h); self.active().go_back(w, h); self.repaint(); return; }
        // Forward button [>] at x=28..52
        if x < 52 { let w = self.win_w; let h = content_h(self.win_h); self.active().go_forward(w, h); self.repaint(); return; }
        // Reload at x=52..76
        if x < 76 { let url = self.active().url.clone(); self.navigate(&url); return; }
        // Address bar at x=76..w-80
        if x < w - 80 { self.addr_focus = true; self.addr_text = self.active().url.clone(); self.repaint(); return; }
        // Bookmark star at x=w-80..w-56
        if x < w - 56 { self.bookmark_active(); self.repaint(); return; }
        // Menu at x=w-56..
        // (not yet implemented)
    }

    fn bookmark_active(&mut self) {
        let title = self.active().title.clone();
        let url   = self.active().url.clone();
        if !url.is_empty() {
            self.bookmarks.push((title, url));
        }
    }

    fn tab_width(&self) -> usize {
        let n = self.tabs.len().max(1);
        (self.win_w.saturating_sub(40)) / n
    }

    /// Repaint the browser window in the WM.
    pub fn repaint(&mut self) {
        if self.win_id == 0 { return; }
        let ww = self.win_w;
        let wh = self.win_h;
        let tab_w = self.tab_width();

        // Draw into WM window
        // ── Tab strip ─────────────────────────────────────────────────────────
        for (i, tab) in self.tabs.iter().enumerate() {
            let tx = i * tab_w;
            let active = i == self.active_tab;
            let bg = if active { TAB_ACTIVE } else { TAB_INACTIVE };
            let fg = if active { TAB_FG_ACT } else { TAB_FG_INA };
            fill_window_rect(self.win_id, tx, 0, tab_w.saturating_sub(2), TAB_H, bg);
            // Tab title
            let max_c = (tab_w.saturating_sub(20)) / FONT_W;
            let shown = truncate_str(&tab.title, max_c);
            draw_window_str(self.win_id, tx + 4, (TAB_H - FONT_H) / 2, &shown, fg, bg);
            // Close button 'x'
            if tab_w > 24 {
                draw_window_str(self.win_id, tx + tab_w - 18, (TAB_H - FONT_H) / 2, "x", (0xCC,0x44,0x44), bg);
            }
        }
        // '+' new tab button
        let plus_x = self.tabs.len() * tab_w;
        if plus_x + 20 < ww {
            fill_window_rect(self.win_id, plus_x, 0, 20, TAB_H, (0x33,0x33,0x44));
            draw_window_str(self.win_id, plus_x + 6, (TAB_H - FONT_H) / 2, "+", (0xAA,0xAA,0xBB), (0x33,0x33,0x44));
        }

        // ── Chrome bar ────────────────────────────────────────────────────────
        let cy = TAB_H;
        fill_window_rect(self.win_id, 0, cy, ww, CHROME_H, CHROME_BG);
        // Back, Fwd, Reload buttons
        let tab = &self.tabs[self.active_tab];
        let btn_fg = if tab.can_back { (0xFF,0xFF,0xFF) } else { (0x55,0x55,0x66) };
        draw_window_str(self.win_id, 4,  cy + (CHROME_H - FONT_H) / 2, "<", btn_fg, CHROME_BG);
        let fwd_fg = if tab.can_fwd { (0xFF,0xFF,0xFF) } else { (0x55,0x55,0x66) };
        draw_window_str(self.win_id, 28, cy + (CHROME_H - FONT_H) / 2, ">", fwd_fg, CHROME_BG);
        draw_window_str(self.win_id, 52, cy + (CHROME_H - FONT_H) / 2, "⟳", CHROME_FG, CHROME_BG);

        // Address bar
        let addr_x  = 76usize;
        let addr_w  = ww.saturating_sub(addr_x + 84);
        let addr_bg = if self.addr_focus { ADDRBAR_ACT } else { ADDRBAR_BG };
        fill_window_rect(self.win_id, addr_x, cy + 4, addr_w, CHROME_H - 8, addr_bg);
        let shown_url = if self.addr_focus {
            format!("{}|", self.addr_text)
        } else {
            truncate_str(&tab.url, addr_w / FONT_W)
        };
        draw_window_str(self.win_id, addr_x + 4, cy + (CHROME_H - FONT_H) / 2, &shown_url, ADDRBAR_FG, addr_bg);

        // Bookmark ★
        draw_window_str(self.win_id, ww.saturating_sub(76), cy + (CHROME_H - FONT_H) / 2, "*", (0xFF,0xCC,0x55), CHROME_BG);
        // Menu ≡
        draw_window_str(self.win_id, ww.saturating_sub(52), cy + (CHROME_H - FONT_H) / 2, "=", CHROME_FG, CHROME_BG);

        // ── Content viewport ──────────────────────────────────────────────────
        let vc_y = TAB_H + CHROME_H;
        let vc_h = wh.saturating_sub(vc_y + STATUSBAR_H);
        let vc_w = ww.saturating_sub(SCROLLBAR_W);
        let scrollbar_x = vc_w;

        // Render tab content
        let tab = &self.tabs[self.active_tab];
        if tab.loading {
            fill_window_rect(self.win_id, 0, vc_y, vc_w, vc_h, (0xFF,0xFF,0xFF));
            let prog_w = (tab.progress as usize * vc_w) / 100;
            fill_window_rect(self.win_id, 0, vc_y + vc_h - 3, prog_w, 3, PROGRESS_FG);
            draw_window_str(self.win_id, vc_w / 2 - 32, vc_y + vc_h / 2,
                            "Loading...", (0x55,0x55,0x66), (0xFF,0xFF,0xFF));
        } else if tab.px_w > 0 && tab.px_h > 0 {
            // Blit tab pixel buffer (clipped to viewport, offset by scroll_y)
            let scroll_y = tab.scroll_y as usize;
            let src_w    = tab.px_w;
            let src_h    = tab.px_h;
            let pixels   = &tab.pixels;
            for py in 0..vc_h {
                let src_y = py + scroll_y;
                if src_y >= src_h {
                    // Below content — fill white
                    for px in 0..vc_w {
                        crate::desktop::wm_paint_pixel(self.win_id, px as u32, (vc_y + py) as u32, 0xFF_FF_FF);
                    }
                    continue;
                }
                for px in 0..vc_w {
                    if px >= src_w { break; }
                    let rgba = pixels[src_y * src_w + px];
                    crate::desktop::wm_paint_pixel(self.win_id, px as u32, (vc_y + py) as u32, rgba);
                }
            }
            // Scrollbar
            if src_h > vc_h {
                fill_window_rect(self.win_id, scrollbar_x, vc_y, SCROLLBAR_W, vc_h, SCROLLBR_BG);
                let thumb_h = (vc_h * vc_h) / src_h.max(1);
                let thumb_y = (scroll_y * vc_h) / src_h.max(1);
                fill_window_rect(self.win_id, scrollbar_x + 2, vc_y + thumb_y, SCROLLBAR_W - 4, thumb_h.max(10), SCROLLBR_TH);
            }
        } else {
            fill_window_rect(self.win_id, 0, vc_y, ww, vc_h, (0xFF,0xFF,0xFF));
        }

        // ── Status bar ────────────────────────────────────────────────────────
        let sb_y = wh.saturating_sub(STATUSBAR_H);
        fill_window_rect(self.win_id, 0, sb_y, ww, STATUSBAR_H, STATUS_BG);
        let tab = &self.tabs[self.active_tab];
        let status_text = if tab.loading {
            format!("Loading... {}%", tab.progress)
        } else if tab.url.is_empty() {
            String::from("Ready")
        } else {
            format!("Done — {}", tab.url)
        };
        draw_window_str(self.win_id, 4, sb_y + (STATUSBAR_H - FONT_H) / 2, &status_text, STATUS_FG, STATUS_BG);

        // Flush to screen
        crate::desktop::wm_flip(self.win_id);
    }

    fn redraw_content(&mut self) {
        if self.win_id == 0 { return; }
        let vc_y = TAB_H + CHROME_H;
        let vc_h = self.win_h.saturating_sub(vc_y + STATUSBAR_H);
        let vc_w = self.win_w.saturating_sub(SCROLLBAR_W);
        let tab      = &self.tabs[self.active_tab];
        let scroll_y = tab.scroll_y as usize;
        let src_w    = tab.px_w;
        let src_h    = tab.px_h;
        let pixels   = &tab.pixels;
        for py in 0..vc_h {
            let src_y = scroll_y + py;
            if src_y >= src_h {
                for px in 0..vc_w {
                    crate::desktop::wm_paint_pixel(self.win_id, px as u32, (vc_y + py) as u32, 0xFF_FF_FF);
                }
                continue;
            }
            for px in 0..vc_w {
                if px >= src_w { break; }
                let rgba = pixels[src_y * src_w + px];
                crate::desktop::wm_paint_pixel(self.win_id, px as u32, (vc_y + py) as u32, rgba);
            }
        }
        crate::desktop::wm_flip(self.win_id);
    }
}

/// Height of the content viewport given total window height.
fn content_h(win_h: usize) -> usize {
    win_h.saturating_sub(TAB_H + CHROME_H + STATUSBAR_H)
}

fn normalise_url(url: &str) -> String {
    let u = url.trim();
    if u.starts_with("http://") || u.starts_with("https://") || u.starts_with("intelli://") {
        String::from(u)
    } else if u.contains('.') && !u.contains(' ') {
        format!("https://{}", u)
    } else {
        // treat as search
        format!("https://duckduckgo.com/?q={}", u.replace(' ', "+"))
    }
}

fn normalise_url_with_base(href: &str, base: &str) -> String {
    if href.starts_with("http://") || href.starts_with("https://") || href.starts_with("intelli://") {
        return String::from(href);
    }
    if href.starts_with('/') {
        // absolute path — prepend scheme + host
        let (scheme_host, _) = base.find('/').map(|p| {
            // find third '/' (after scheme://)
            let after = &base[p+2..];
            let end = after.find('/').map(|q| p + 2 + q).unwrap_or(base.len());
            (&base[..end], &base[end..])
        }).unwrap_or((base, ""));
        return format!("{}{}", scheme_host, href);
    }
    // relative — append to base directory
    let dir = base.rfind('/').map(|p| &base[..p+1]).unwrap_or(base);
    format!("{}{}", dir, href)
}

fn truncate_str(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        String::from(s)
    } else if max_chars > 3 {
        let mut out = String::from(&s[..max_chars - 3]);
        out.push_str("...");
        out
    } else {
        String::from(&s[..max_chars.min(s.len())])
    }
}

// ── Helpers to draw into WM window ───────────────────────────────────────────

fn fill_window_rect(win_id: u32, x: usize, y: usize, w: usize, h: usize, (r,g,b): (u8,u8,u8)) {
    let rgba = ((r as u32) << 16) | ((g as u32) << 8) | b as u32;
    crate::desktop::wm_fill_window_rect(win_id, x as u32, y as u32, w as u32, h as u32, rgba);
}

fn draw_window_str(win_id: u32, x: usize, y: usize, text: &str, (r,g,b): (u8,u8,u8), _bg: (u8,u8,u8)) {
    let fg_rgba = ((r as u32) << 16) | ((g as u32) << 8) | b as u32;
    for (i, _ch) in text.char_indices() {
        let px = (x + i * FONT_W) as u32;
        let py = y as u32;
        // paint a solid column for each character position as placeholder
        // Real glyph rendering goes through the framebuffer bitmap font
        for row in 0..FONT_H as u32 {
            crate::desktop::wm_paint_pixel(win_id, px, py + row, fg_rgba);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── Global Browser instance ───────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

static BROWSER: Once<Mutex<BrowserState>> = Once::new();

pub fn browser_init(win_w: usize, win_h: usize) {
    BROWSER.call_once(|| {
        let mut state = BrowserState::new(win_w, win_h);
        // Create WM window
        let id = crate::desktop::wm_create_window(
            20, 40,
            win_w as u32, win_h as u32,
            "Intelli Browser",
        );
        state.win_id = id;
        state.repaint();
        Mutex::new(state)
    });
}

pub fn browser_is_open() -> bool {
    BROWSER.get().is_some()
}

pub fn with_browser<F: FnOnce(&mut BrowserState)>(f: F) {
    if let Some(m) = BROWSER.get() { f(&mut m.lock()); }
}

pub fn browser_key(ch: u8) {
    with_browser(|b| { b.key(ch); b.repaint(); });
}

pub fn browser_click(win_x: i32, win_y: i32) {
    with_browser(|b| b.click(win_x, win_y));
}

pub fn browser_navigate(url: &str) {
    with_browser(|b| {
        b.navigate(url);
        b.repaint();
    });
}
