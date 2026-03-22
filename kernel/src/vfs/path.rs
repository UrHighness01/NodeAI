//! Path resolution for the VFS — Phase 7.
//!
//! Walks the mount table and descends through directory entries.

use alloc::{sync::Arc, string::String};
use super::{VfsNode, VfsResult, VfsError, MOUNTS};

/// Resolve an absolute path to a `VfsNode`.
pub fn resolve(path: &str) -> VfsResult<Arc<dyn VfsNode>> {
    // Find the deepest mount point that is a prefix of `path`.
    let mounts = MOUNTS.read();
    let mut best_len = 0usize;
    let mut best_idx: Option<usize> = None;

    for (i, m) in mounts.iter().enumerate() {
        if path.starts_with(m.path.as_str()) && m.path.len() >= best_len {
            best_len = m.path.len();
            best_idx = Some(i);
        }
    }

    let (root, rest) = match best_idx {
        Some(i) => {
            let m = &mounts[i];
            let rest = &path[m.path.len()..];
            (m.root.clone(), rest)
        }
        None => return Err(VfsError::NotFound),
    };
    drop(mounts);

    walk(root, rest)
}

/// Descend into `node` through the `/`-separated components of `path`.
fn walk(mut node: Arc<dyn VfsNode>, path: &str) -> VfsResult<Arc<dyn VfsNode>> {
    for component in path.split('/').filter(|s| !s.is_empty()) {
        match component {
            "."  => {}
            ".." => {
                // Simplified: ".." at root stays at root.
                // A real implementation would track parent pointers.
            }
            name => {
                node = node.lookup(name)?;
            }
        }
    }
    Ok(node)
}

/// Return the parent path and the final component of a path.
pub fn split_parent(path: &str) -> (&str, &str) {
    let path = path.trim_end_matches('/');
    if let Some(pos) = path.rfind('/') {
        let parent = if pos == 0 { "/" } else { &path[..pos] };
        let name = &path[pos + 1..];
        (parent, name)
    } else {
        ("/", path)
    }
}
