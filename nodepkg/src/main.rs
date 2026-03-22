//! nodepkg — NodeAI Package Manager
//! Phase 20: install, remove, update, search packages for the NodeAI OS.
//!
//! Build: cargo build --target x86_64-unknown-linux-musl --release
//!        (produces a fully static binary)

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process;

// ── constants ────────────────────────────────────────────────────────────────
const REPO_INDEX_URL:   &str = "https://pkg.nodeai.dev/PACKAGES.toml";
const PKG_BASE_URL:     &str = "https://pkg.nodeai.dev/";
const DB_PATH:          &str = "/var/lib/nodepkg/installed.db";
const CACHE_DIR:        &str = "/var/cache/nodepkg";
const INDEX_CACHE:      &str = "/var/cache/nodepkg/PACKAGES.toml";
const INSTALL_PREFIX:   &str = "/usr";

// ── package manifest (MANIFEST.toml subset) ─────────────────────────────────
#[derive(Debug, Clone)]
struct PkgMeta {
    name:        String,
    version:     String,
    description: String,
    sha256:      String,
    deps:        Vec<String>,
}

// ── installed DB ─────────────────────────────────────────────────────────────
struct InstalledDb {
    packages: HashMap<String, PkgMeta>,
    path:     PathBuf,
}

impl InstalledDb {
    fn load() -> Self {
        let path = PathBuf::from(DB_PATH);
        let mut db = InstalledDb {
            packages: HashMap::new(),
            path: path.clone(),
        };
        if let Ok(data) = fs::read_to_string(&path) {
            db.parse_toml(&data);
        }
        db
    }

    fn parse_toml(&mut self, data: &str) {
        // Minimal TOML: [[package]] blocks
        let mut cur: Option<PkgMeta> = None;
        for line in data.lines() {
            let line = line.trim();
            if line == "[[package]]" {
                if let Some(p) = cur.take() {
                    self.packages.insert(p.name.clone(), p);
                }
                cur = Some(PkgMeta {
                    name: String::new(),
                    version: String::new(),
                    description: String::new(),
                    sha256: String::new(),
                    deps: Vec::new(),
                });
            } else if let Some(ref mut p) = cur {
                if let Some(val) = kv(line, "name")        { p.name = val; }
                if let Some(val) = kv(line, "version")     { p.version = val; }
                if let Some(val) = kv(line, "description") { p.description = val; }
                if let Some(val) = kv(line, "sha256")      { p.sha256 = val; }
                if let Some(val) = kv(line, "deps")        {
                    p.deps = val.trim_matches(|c| c == '[' || c == ']')
                        .split(',')
                        .map(|s| s.trim().trim_matches('"').to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                }
            }
        }
        if let Some(p) = cur { self.packages.insert(p.name.clone(), p); }
    }

    fn save(&self) {
        let parent = self.path.parent().unwrap_or(Path::new("/"));
        let _ = fs::create_dir_all(parent);
        let mut out = String::new();
        for p in self.packages.values() {
            out.push_str("[[package]]\n");
            out.push_str(&format!("name = \"{}\"\n", p.name));
            out.push_str(&format!("version = \"{}\"\n", p.version));
            out.push_str(&format!("description = \"{}\"\n", p.description));
            out.push_str(&format!("sha256 = \"{}\"\n", p.sha256));
            let deps_str = p.deps.iter()
                .map(|d| format!("\"{}\"", d))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!("deps = [{}]\n\n", deps_str));
        }
        let _ = fs::write(&self.path, out);
    }

    fn is_installed(&self, name: &str) -> bool {
        self.packages.contains_key(name)
    }
}

// ── package index ─────────────────────────────────────────────────────────────
struct Index {
    packages: HashMap<String, PkgMeta>,
}

impl Index {
    fn load() -> Option<Self> {
        let data = fs::read_to_string(INDEX_CACHE).ok()?;
        let mut idx = Index { packages: HashMap::new() };
        let mut cur: Option<PkgMeta> = None;
        for line in data.lines() {
            let line = line.trim();
            if line == "[[package]]" {
                if let Some(p) = cur.take() { idx.packages.insert(p.name.clone(), p); }
                cur = Some(PkgMeta {
                    name: String::new(), version: String::new(),
                    description: String::new(), sha256: String::new(), deps: Vec::new(),
                });
            } else if let Some(ref mut p) = cur {
                if let Some(v) = kv(line, "name")        { p.name = v; }
                if let Some(v) = kv(line, "version")     { p.version = v; }
                if let Some(v) = kv(line, "description") { p.description = v; }
                if let Some(v) = kv(line, "sha256")      { p.sha256 = v; }
                if let Some(v) = kv(line, "deps") {
                    p.deps = v.trim_matches(|c| c == '[' || c == ']')
                        .split(',').map(|s| s.trim().trim_matches('"').to_string())
                        .filter(|s| !s.is_empty()).collect();
                }
            }
        }
        if let Some(p) = cur { idx.packages.insert(p.name.clone(), p); }
        Some(idx)
    }
}

// ── HTTP GET (bare syscall-based, no libcurl) ─────────────────────────────────
// On NodeAI, we use the kernel's built-in HTTP driver (Phase 17 network stack).
// For now we shell out to busybox wget if available, else print instructions.
fn http_get(url: &str, dest: &Path) -> Result<Vec<u8>, String> {
    // Try /bin/wget (BusyBox)
    let status = std::process::Command::new("/bin/wget")
        .args(["--quiet", "-O", &dest.to_string_lossy(), url])
        .status();
    match status {
        Ok(s) if s.success() => {
            fs::read(dest).map_err(|e| format!("read error: {}", e))
        }
        _ => {
            // Fallback: use /bin/curl
            let status2 = std::process::Command::new("/bin/curl")
                .args(["-fsSL", "-o", &dest.to_string_lossy(), url])
                .status();
            match status2 {
                Ok(s) if s.success() => fs::read(dest).map_err(|e| format!("{}", e)),
                _ => Err(format!("Cannot fetch {}: no wget or curl available", url)),
            }
        }
    }
}

// ── SHA-256 (simple software impl — no external crate) ───────────────────────
fn sha256_hex(data: &[u8]) -> String {
    // Minimal SHA-256 — RFC 6234 / FIPS 180-4
    const K: [u32; 64] = [
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19,
    ];
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 { msg.push(0); }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for (i, b) in chunk.chunks(4).enumerate().take(16) {
            w[i] = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19)  ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let (mut a,mut b,mut c,mut d,mut e,mut f,mut g,mut hh) =
            (h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g; g = f; f = e;
            e = d.wrapping_add(t1);
            d = c; c = b; b = a;
            a = t1.wrapping_add(t2);
        }
        h[0]=h[0].wrapping_add(a); h[1]=h[1].wrapping_add(b);
        h[2]=h[2].wrapping_add(c); h[3]=h[3].wrapping_add(d);
        h[4]=h[4].wrapping_add(e); h[5]=h[5].wrapping_add(f);
        h[6]=h[6].wrapping_add(g); h[7]=h[7].wrapping_add(hh);
    }
    format!("{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}",
        h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7])
}

// ── tar extraction (GNU tar .tgz) ─────────────────────────────────────────────
fn extract_tgz(data: &[u8], dest: &Path) -> Result<Vec<PathBuf>, String> {
    // Use /bin/tar (BusyBox) rather than reimplementing decompression
    fs::create_dir_all(dest).map_err(|e| e.to_string())?;
    let tmp = PathBuf::from(format!("/tmp/nodepkg_{}.tgz", std::process::id()));
    fs::write(&tmp, data).map_err(|e| e.to_string())?;
    let out = std::process::Command::new("/bin/tar")
        .args(["-xzf", &tmp.to_string_lossy(), "-C", &dest.to_string_lossy()])
        .output()
        .map_err(|e| format!("tar: {}", e))?;
    let _ = fs::remove_file(&tmp);
    if !out.status.success() {
        return Err(format!("tar error: {}", String::from_utf8_lossy(&out.stderr)));
    }
    // Collect installed file paths
    let files = std::process::Command::new("/bin/tar")
        .args(["-tzf", &tmp.to_string_lossy()])
        .output();
    let mut paths = Vec::new();
    if let Ok(f) = files {
        for line in String::from_utf8_lossy(&f.stdout).lines() {
            paths.push(dest.join(line));
        }
    }
    Ok(paths)
}

// ── topological dependency sort ───────────────────────────────────────────────
fn topo_sort<'a>(name: &'a str, idx: &'a Index, visited: &mut Vec<String>) -> Vec<String> {
    if visited.contains(&name.to_string()) { return Vec::new(); }
    visited.push(name.to_string());
    let mut order = Vec::new();
    if let Some(meta) = idx.packages.get(name) {
        for dep in &meta.deps {
            order.extend(topo_sort(dep, idx, visited));
        }
    }
    order.push(name.to_string());
    order
}

// ── commands ─────────────────────────────────────────────────────────────────
fn cmd_update() {
    println!("nodepkg: refreshing package index...");
    let _ = fs::create_dir_all(CACHE_DIR);
    let dest = PathBuf::from(INDEX_CACHE);
    match http_get(REPO_INDEX_URL, &dest) {
        Ok(_)  => println!("nodepkg: index updated."),
        Err(e) => { eprintln!("nodepkg: update failed: {}", e); process::exit(1); }
    }
}

fn cmd_search(query: &str) {
    let idx = match Index::load() {
        Some(i) => i,
        None => { eprintln!("nodepkg: index not found — run `nodepkg update` first"); return; }
    };
    let q = query.to_lowercase();
    let mut found = false;
    for (name, meta) in &idx.packages {
        if name.to_lowercase().contains(&q) || meta.description.to_lowercase().contains(&q) {
            println!("  {:20} {:12}  {}", name, meta.version, meta.description);
            found = true;
        }
    }
    if !found { println!("nodepkg: no packages matching '{}'", query); }
}

fn cmd_install(names: &[String]) {
    let idx = match Index::load() {
        Some(i) => i,
        None => {
            eprintln!("nodepkg: index not found — run `nodepkg update` first");
            process::exit(1);
        }
    };
    let mut db = InstalledDb::load();
    let _ = fs::create_dir_all(CACHE_DIR);

    for pkg in names {
        // Special: py:<package> → pip install
        if let Some(pyname) = pkg.strip_prefix("py:") {
            println!("nodepkg: installing Python package '{}'...", pyname);
            let _ = std::process::Command::new("/usr/bin/python3")
                .args(["-m", "ensurepip", "--default-pip"]).status();
            let status = std::process::Command::new("/usr/bin/python3")
                .args(["-m", "pip", "install", "--user", pyname])
                .status();
            match status {
                Ok(s) if s.success() => println!("nodepkg: '{}' installed.", pkg),
                _ => eprintln!("nodepkg: pip install '{}' failed", pyname),
            }
            continue;
        }
        // Special: npm:<package> → npm install -g
        if let Some(npmname) = pkg.strip_prefix("npm:") {
            println!("nodepkg: installing npm package '{}'...", npmname);
            let status = std::process::Command::new("/usr/bin/node")
                .args(["/usr/lib/node_modules/npm/bin/npm-cli.js", "install", "-g", npmname])
                .status();
            match status {
                Ok(s) if s.success() => println!("nodepkg: '{}' installed.", pkg),
                _ => eprintln!("nodepkg: npm install '{}' failed", npmname),
            }
            continue;
        }

        // Resolve dependency order
        let mut visited = Vec::new();
        let order = topo_sort(pkg, &idx, &mut visited);

        for name in &order {
            if db.is_installed(name) {
                println!("nodepkg: '{}' is already installed, skipping.", name);
                continue;
            }
            let meta = match idx.packages.get(name) {
                Some(m) => m.clone(),
                None => {
                    eprintln!("nodepkg: '{}' not found in index", name);
                    continue;
                }
            };
            // Download
            let url = format!("{}{}-{}.npkg", PKG_BASE_URL, meta.name, meta.version);
            let cache_path = PathBuf::from(format!("{}/{}-{}.npkg", CACHE_DIR, meta.name, meta.version));
            println!("nodepkg: downloading {}...", meta.name);
            let data = match http_get(&url, &cache_path) {
                Ok(d) => d,
                Err(e) => { eprintln!("nodepkg: download failed: {}", e); continue; }
            };
            // Verify SHA-256
            let got_hash = sha256_hex(&data);
            if !meta.sha256.is_empty() && got_hash != meta.sha256 {
                eprintln!("nodepkg: SHA-256 mismatch for '{}' (got {}, expected {})", name, got_hash, meta.sha256);
                continue;
            }
            // Extract
            let install_dir = PathBuf::from(INSTALL_PREFIX);
            println!("nodepkg: extracting {}...", meta.name);
            match extract_tgz(&data, &install_dir) {
                Ok(_)  => {}
                Err(e) => { eprintln!("nodepkg: extract error: {}", e); continue; }
            }
            db.packages.insert(name.clone(), meta.clone());
            db.save();
            println!("nodepkg: '{}' {} installed.", meta.name, meta.version);
        }
    }
}

fn cmd_remove(names: &[String]) {
    let mut db = InstalledDb::load();
    for name in names {
        if !db.is_installed(name) {
            eprintln!("nodepkg: '{}' is not installed", name);
            continue;
        }
        db.packages.remove(name);
        db.save();
        println!("nodepkg: '{}' removed (files left in place — manual cleanup needed).", name);
    }
}

fn cmd_list() {
    let db = InstalledDb::load();
    if db.packages.is_empty() {
        println!("nodepkg: no packages installed.");
        return;
    }
    println!("{:<20} {:<12} {}", "Name", "Version", "Description");
    println!("{}", "-".repeat(60));
    for p in db.packages.values() {
        println!("{:<20} {:<12} {}", p.name, p.version, p.description);
    }
}

// ── TOML kv helper ───────────────────────────────────────────────────────────
fn kv(line: &str, key: &str) -> Option<String> {
    let prefix = format!("{} = ", key);
    if line.starts_with(&prefix) {
        let val = line[prefix.len()..].trim().trim_matches('"').to_string();
        Some(val)
    } else {
        None
    }
}

// ── entry point ───────────────────────────────────────────────────────────────
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_help();
        process::exit(1);
    }
    match args[1].as_str() {
        "update"  => cmd_update(),
        "search"  => {
            if args.len() < 3 { eprintln!("Usage: nodepkg search <query>"); process::exit(1); }
            cmd_search(&args[2]);
        }
        "install" => {
            if args.len() < 3 { eprintln!("Usage: nodepkg install <package...>"); process::exit(1); }
            cmd_install(&args[2..].to_vec());
        }
        "remove"  => {
            if args.len() < 3 { eprintln!("Usage: nodepkg remove <package...>"); process::exit(1); }
            cmd_remove(&args[2..].to_vec());
        }
        "list"    => cmd_list(),
        "--help" | "-h" | "help" => print_help(),
        other => {
            eprintln!("nodepkg: unknown command '{}'. Try 'nodepkg help'.", other);
            process::exit(1);
        }
    }
}

fn print_help() {
    println!("nodepkg — NodeAI Package Manager");
    println!();
    println!("Usage:");
    println!("  nodepkg update              Refresh package index from pkg.nodeai.dev");
    println!("  nodepkg search <query>      Search available packages");
    println!("  nodepkg install <pkg...>    Install one or more packages");
    println!("  nodepkg remove  <pkg...>    Remove installed packages");
    println!("  nodepkg list                List installed packages");
    println!();
    println!("Special prefixes:");
    println!("  nodepkg install py:<name>   Install a Python package via pip");
    println!("  nodepkg install npm:<name>  Install a Node.js package via npm");
}
