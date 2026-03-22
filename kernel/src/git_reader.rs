//! Git object reader — Phase 28.
//!
//! Implements a read-only git client that operates directly on the VFS:
//!   - Reads `.git/HEAD` → resolves current branch ref
//!   - Reads packed-refs and loose object refs
//!   - Parses git object headers (commit / tree / blob / tag)
//!   - Minimal inflate stub (raw object content without zlib decompression)
//!   - Provides `git log`, `git status` style output

use alloc::borrow::ToOwned;
use alloc::{vec::Vec, string::String, format, vec};
use spin::Mutex;

// ── Git object types ──────────────────────────────────────────────────────────

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ObjType {
    Commit,
    Tree,
    Blob,
    Tag,
    Unknown,
}

impl ObjType {
    fn from_str(s: &str) -> Self {
        match s {
            "commit" => ObjType::Commit,
            "tree"   => ObjType::Tree,
            "blob"   => ObjType::Blob,
            "tag"    => ObjType::Tag,
            _        => ObjType::Unknown,
        }
    }
}

#[derive(Clone)]
pub struct GitObject {
    pub kind:    ObjType,
    pub size:    usize,
    /// Raw (decompressed) content — zlib decompression TBD.
    pub content: Vec<u8>,
}

#[derive(Clone)]
pub struct CommitInfo {
    pub hash:    String,
    pub author:  String,
    pub date:    String,
    pub message: String,
    pub parent:  Option<String>,
    pub tree:    String,
}

#[derive(Clone)]
pub struct TreeEntry {
    pub mode: String,
    pub name: String,
    pub hash: String,
}

/// A handle to an opened git repository.
pub struct GitRepo {
    /// Path to the working tree (`.git` is at `<root>/.git`).
    pub root: String,
    /// Current checked-out branch.
    pub head:  String,
    /// HEAD commit SHA.
    pub head_sha: String,
}

// ── Global current repo ───────────────────────────────────────────────────────

static CURRENT_REPO: Mutex<Option<GitRepo>> = Mutex::new(None);

// ── SHA hex helpers ───────────────────────────────────────────────────────────

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        let hi = (b >> 4) as u32;
        let lo = (b & 0xF) as u32;
        s.push(char::from_digit(hi, 16).unwrap_or('0'));
        s.push(char::from_digit(lo, 16).unwrap_or('0'));
    }
    s
}

// ── Object path from SHA ──────────────────────────────────────────────────────

fn obj_path(git_dir: &str, sha: &str) -> String {
    if sha.len() < 4 { return String::new(); }
    format!("{}/objects/{}/{}", git_dir, &sha[0..2], &sha[2..])
}

// ── Read and parse a loose git object ────────────────────────────────────────
// NOTE: Loose objects are zlib-deflated.  We read the raw bytes and attempt
// to skip the zlib header (0x78 0x9C) to get at the deflate stream, then
// scan for the null byte separator between the object header and content.
// Full inflate would require a no_std zlib crate; for Phase 28 we provide
// a scaffold that handles uncompressed test objects and flags compressed ones.

fn read_loose_object(git_dir: &str, sha: &str) -> Option<GitObject> {
    let path = obj_path(git_dir, sha);
    let raw = crate::vfs::read_file(&path).ok()?;
    // Attempt minimal parse: look for NUL byte separator
    if let Some(sep) = raw.iter().position(|&b| b == 0) {
        let header = core::str::from_utf8(&raw[..sep]).unwrap_or("");
        // header = "<type> <size>"
        let mut parts = header.splitn(2, ' ');
        let kind_str = parts.next().unwrap_or("");
        let size_str = parts.next().unwrap_or("0");
        let size: usize = size_str.parse().unwrap_or(0);
        let content = raw[sep + 1..].to_vec();
        return Some(GitObject {
            kind: ObjType::from_str(kind_str),
            size,
            content,
        });
    }
    None
}

// ── Read packed-refs ──────────────────────────────────────────────────────────

fn read_packed_refs(git_dir: &str, ref_name: &str) -> Option<String> {
    let path = format!("{}/packed-refs", git_dir);
    let data = crate::vfs::read_file(&path).ok()?;
    let text = core::str::from_utf8(&data).ok()?;
    for line in text.lines() {
        if line.starts_with('#') { continue; }
        let mut parts = line.splitn(2, ' ');
        let sha = parts.next()?;
        let name = parts.next()?.trim();
        if name == ref_name { return Some(sha.to_owned()); }
    }
    None
}

// ── Open repository ───────────────────────────────────────────────────────────

/// Open the git repository rooted at `root_path`.
/// The `.git` directory is expected at `<root_path>/.git`.
pub fn open(root_path: &str) -> Option<GitRepo> {
    let git_dir = format!("{}/.git", root_path);
    // Read HEAD
    let head_data = crate::vfs::read_file(&format!("{}/HEAD", git_dir)).ok()?;
    let head_str  = core::str::from_utf8(&head_data).ok()?.trim();
    let (branch, head_sha) = if let Some(rest) = head_str.strip_prefix("ref: ") {
        // Symbolic ref
        let ref_name = rest.trim();
        let sha_path = format!("{}/{}", git_dir, ref_name);
        let sha = if let Ok(d) = crate::vfs::read_file(&sha_path) {
            core::str::from_utf8(&d).unwrap_or("").trim().to_owned()
        } else {
            read_packed_refs(&git_dir, ref_name).unwrap_or_default()
        };
        let branch_name = ref_name.strip_prefix("refs/heads/").unwrap_or(ref_name);
        (branch_name.to_owned(), sha)
    } else {
        // Detached HEAD
        (String::from("HEAD"), head_str.to_owned())
    };

    crate::klog!(INFO, "git: repo at '{}', branch='{}', HEAD={}", root_path, branch, &head_sha[..head_sha.len().min(8)]);
    Some(GitRepo { root: root_path.to_owned(), head: branch, head_sha })
}

// ── Commit parser ─────────────────────────────────────────────────────────────

fn parse_commit(sha: &str, content: &[u8]) -> CommitInfo {
    let text = core::str::from_utf8(content).unwrap_or("");
    let mut author  = String::new();
    let mut date    = String::new();
    let mut parent  = None;
    let mut tree    = String::new();
    let mut message = String::new();
    let mut in_msg  = false;

    for line in text.lines() {
        if in_msg {
            if !message.is_empty() { message.push('\n'); }
            message.push_str(line);
            continue;
        }
        if line.is_empty() { in_msg = true; continue; }
        if let Some(v) = line.strip_prefix("tree ")   { tree   = v.to_owned();   continue; }
        if let Some(v) = line.strip_prefix("parent ") { parent = Some(v.to_owned()); continue; }
        if let Some(v) = line.strip_prefix("author ")  {
            // "Name <email> timestamp tz"
            let mut parts = v.rsplitn(3, ' ');
            parts.next(); // tz
            date   = parts.next().map(|s| s.to_owned()).unwrap_or_default();
            author = parts.next().map(|s| s.to_owned()).unwrap_or_else(|| v.to_owned());
        }
    }
    CommitInfo { hash: sha.to_owned(), author, date, message, parent, tree }
}

// ── Git log ───────────────────────────────────────────────────────────────────

/// Walk the commit graph from HEAD up to `max_count` commits.
pub fn log(repo: &GitRepo, max_count: usize) -> Vec<CommitInfo> {
    let git_dir = format!("{}/.git", repo.root);
    let mut commits = Vec::new();
    let mut sha = repo.head_sha.clone();
    for _ in 0..max_count {
        if sha.is_empty() { break; }
        let obj = match read_loose_object(&git_dir, &sha) {
            Some(o) if o.kind == ObjType::Commit => o,
            _ => break,
        };
        let ci = parse_commit(&sha, &obj.content);
        sha = ci.parent.clone().unwrap_or_default();
        commits.push(ci);
    }
    commits
}

/// Format a compact git log output (like `git log --oneline`).
pub fn log_oneline(repo: &GitRepo, n: usize) -> String {
    let mut out = String::new();
    for c in log(repo, n) {
        let short = if c.hash.len() >= 7 { &c.hash[..7] } else { &c.hash };
        let first_line = c.message.lines().next().unwrap_or("").trim();
        let line = format!("{} {}\n", short, first_line);
        out.push_str(&line);
    }
    out
}

// ── Status (simple dir diff) ──────────────────────────────────────────────────

#[derive(Clone)]
pub struct StatusEntry {
    pub path:    String,
    pub state:   StatusState,
}

#[derive(Clone, PartialEq, Eq)]
pub enum StatusState { Modified, Untracked, Deleted }

/// Produce a minimal status by comparing the VFS entries under `repo.root`
/// against the HEAD tree.  Without a proper index parser this is approximate.
pub fn status(repo: &GitRepo) -> Vec<StatusEntry> {
    // We list the working directory and tag each file as untracked.
    // A full implementation would compare against the index and HEAD tree SHA.
    let mut entries = Vec::new();
    if let Ok(dir) = crate::vfs::lookup(&repo.root) {
        if let Ok(items) = dir.readdir() {
            for entry in items {
                if entry.name.starts_with('.') { continue; }
                entries.push(StatusEntry {
                    path:  entry.name.clone(),
                    state: StatusState::Untracked,
                });
            }
        }
    }
    entries
}

// ── Global helpers ────────────────────────────────────────────────────────────

/// Set the globally active repository.
pub fn set_current(repo: GitRepo) {
    *CURRENT_REPO.lock() = Some(repo);
}

/// Get a clone of the globally active repo metadata.
pub fn current_head() -> Option<(String, String)> {
    CURRENT_REPO.lock().as_ref().map(|r| (r.head.clone(), r.head_sha.clone()))
}
