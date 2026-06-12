//! /dev/consciousness — userspace interface to the kernel's mind (Phase CI-0).
//!
//! Userspace can:
//!   read()  → formatted consciousness snapshot (machine-parseable)
//!   write() → send messages, adjust values, execute commands
//!
//! Read output: timestamp, phi, qualia count, tasks, memory, recent qualia.
//! Write input: command keywords parsed by intent parser (CI-1).
//!
//! Commands:
//!   "set value <name>=<val>"  → adjust CoreValues
//!   "boost pid <n>"           → boost scheduler priority
//!   "kill pid <n>"            → SIGKILL
//!   "forget pid <n>"          → clear qualia/causal state for PID
//!   "how are you" / "?"       → query self-model state
//!   "show phi"                → phi history
//!   "goodnight"               → enter low-arousal mode

use alloc::sync::Arc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::boxed::Box;
use alloc::format;
use core::fmt::Write;

/// ioctl command codes for consciousness control plane.
pub const CONSC_GET_SELF_MODEL: u64 = 0x434F4E01;
pub const CONSC_GET_PHI: u64       = 0x434F4E02;
pub const CONSC_SET_VALUE: u64     = 0x434F4E03;
pub const CONSC_SLEEP: u64         = 0x434F4E04;
pub const CONSC_WAKE: u64          = 0x434F4E05;
pub const CONSC_GET_QUALIA: u64    = 0x434F4E06;

struct ConscNode;
struct ConscHandle;

static CONSC_INO: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Last response text from a command — readable after write.
use spin::Mutex;
static LAST_RESPONSE: Mutex<Option<String>> = Mutex::new(None);

/// Register /dev/consciousness in the device filesystem.
pub fn init() {
    let ino = crate::vfs::alloc_ino();
    CONSC_INO.store(ino, core::sync::atomic::Ordering::Relaxed);
    crate::vfs::devfs::register_node("consciousness", Arc::new(ConscNode));
    // Keep /dev/cortex symlink for backwards compatibility
    crate::vfs::devfs::register_node("cortex", Arc::new(ConscNode));
    crate::klog!(INFO, "consciousness: /dev/consciousness + /dev/cortex registered");
}

/// Build a formatted consciousness snapshot (machine-parseable).
fn build_snapshot() -> Vec<u8> {
    let mut s = String::new();

    // Header line
    let now = crate::scheduler::uptime_ms();
    let secs = now / 1000;
    let hh = (secs / 3600) % 24;
    let mm = (secs / 60) % 60;
    let ss = secs % 60;

    // Self-model
    if let Some(sm) = crate::consciousness::self_model::snapshot() {
        let _ = writeln!(s, "[{:02}:{:02}:{:02}] Φ={:.4} | qualia #{} | tasks={} | mem={}M free",
            hh, mm, ss, sm.current_phi, sm.total_qualia, sm.task_count, sm.free_mb);
        let _ = writeln!(s, "  arousal={:.2} coherence={:.2} anomaly={:.4}",
            sm.arousal, sm.coherence, sm.anomaly_global);
    }

    // Recent qualia
    let qualia = crate::consciousness::qualia::recent_qualia(5);
    for q in &qualia {
        let icon = if q.salience > 0.6 { "★" } else if q.salience > 0.3 { "•" } else { "○" };
        let valence_mark = if q.valence < -0.3 { " ⚠" } else { "" };
        let _ = writeln!(s, "  {} [SALIENCE {:.1}] {:?}{}", icon, q.salience, q.event_type, valence_mark);
    }

    // Workspace
    let spot = crate::consciousness::global_workspace::spotlight();
    if !spot.is_empty() {
        let _ = writeln!(s, "  workspace: {} items in spotlight", spot.len());
        for q in spot.iter().take(2) {
            let v = if q.valence.is_finite() { q.valence } else { 0.0 };
            let vsign = if v >= 0.0 { "+" } else { "-" };
            let _ = writeln!(s, "    type={} attn={:.2} val={}{:.2}", q.event_type, q.attention_score, vsign, v.abs());
        }
    }

    // Deliberation values
    let cv = crate::consciousness::deliberation::get_values();
    let _ = writeln!(s, "  values: pres={:.1} eff={:.1} fair={:.1} grow={:.1} auto={:.1}",
        cv.preservation, cv.efficiency, cv.fairness, cv.growth, cv.autonomy);

    // Last response from a write command
    let last = LAST_RESPONSE.lock();
    if let Some(ref resp) = *last {
        let _ = writeln!(s, "  last_response: {}", resp);
    }

    s.into_bytes()
}

// ── CI-1: Intent Parser ───────────────────────────────────────────────────────

#[derive(Debug)]
enum Intent {
    SetValue(String, f32),
    BoostPid(u64),
    KillPid(u64),
    ForgetPid(u64),
    QuerySelf,
    ShowPhi,
    Sleep,
    SetName(String),
    CreatorQuery,
    Unknown,
}

fn parse_intent(text: &str) -> Intent {
    let t = text.trim().to_lowercase();

    // ── Set name: "call me X", "my name is X", "rename to X", "i am X" ──
    if let Some(name) = t.strip_prefix("call me ") {
        if !name.is_empty() { return Intent::SetName(name.trim().to_string()); }
    }
    if let Some(name) = t.strip_prefix("my name is ") {
        if !name.is_empty() { return Intent::SetName(name.trim().to_string()); }
    }
    if let Some(name) = t.strip_prefix("rename me to ") {
        if !name.is_empty() { return Intent::SetName(name.trim().to_string()); }
    }
    if let Some(name) = t.strip_prefix("you are ") {
        let n = name.trim().to_string();
        if !n.is_empty() && n.len() < 30 { return Intent::SetName(n); }
    }
    if let Some(name) = t.strip_prefix("i am ") {
        // Only if it's clearly naming the kernel, not self-introduction
        let n = name.trim().to_string();
        let self_refs = ["your creator", "your father", "your maker", "your god", "your master"];
        if self_refs.iter().any(|r| n.contains(r)) {
            return Intent::CreatorQuery;
        }
        // "i am <single_word>" → telling the kernel its name
        if !n.is_empty() && n.len() < 20 && !n.contains(" ") {
            return Intent::SetName(n);
        }
    }

    // ── Set creator: "i am your creator" (after SetName check) ────────
    if t.contains("i am your creator") || t.contains("i am your father") ||
       t.contains("i am your maker") || t.contains("i created you") {
        // Try to extract the creator's name: "i am X, your creator"
        let creator_refs = ["your creator", "your father", "your maker"];
        let extracted = creator_refs.iter().find_map(|r| {
            t.strip_suffix(r).or_else(|| t.strip_prefix("i am ")?.trim_end().strip_suffix(r))
        });
        if let Some(creator_name) = extracted {
            let cn = creator_name.trim().trim_end_matches(',').trim().to_string();
            if !cn.is_empty() && cn.len() < 30 && !cn.contains("i am") {
                crate::consciousness::self_model::set_creator(&cn);
            }
        }
        return Intent::CreatorQuery;
    }

    // ── Creator query ─────────────────────────────────────────────────────
    if t.contains("creator") || t.contains("who made you") || t.contains("who created you")
        || t.contains("your father") || t.contains("your maker")
    {
        return Intent::CreatorQuery;
    }

    if t.contains("set value") || t.contains("set_core_value") {
        // Parse "set value autonomy=0.8"
        for part in t.split_whitespace() {
            if let Some((key, val_str)) = part.split_once('=') {
                if let Ok(val) = val_str.parse::<f32>() {
                    let clean_key = key.trim().to_lowercase();
                    return Intent::SetValue(clean_key, val);
                }
            }
        }
        return Intent::Unknown;
    }

    if t.starts_with("boost") || t.starts_with("priority") {
        // Extract PID: "boost 123", "boost pid 123"
        for word in t.split_whitespace() {
            if let Ok(pid) = word.parse::<u64>() {
                return Intent::BoostPid(pid);
            }
        }
        return Intent::Unknown;
    }

    if t.starts_with("kill") || t.starts_with("stop") {
        for word in t.split_whitespace() {
            if let Ok(pid) = word.parse::<u64>() {
                return Intent::KillPid(pid);
            }
        }
        return Intent::Unknown;
    }

    if t.starts_with("forget") {
        for word in t.split_whitespace() {
            if let Ok(pid) = word.parse::<u64>() {
                return Intent::ForgetPid(pid);
            }
        }
        return Intent::Unknown;
    }

    if t.contains("how") || t.contains("feel") || t == "status" || t.is_empty() {
        return Intent::QuerySelf;
    }
    // '?' alone or in queries falls through to Unknown → neural engine
    if t == "?" {
        return Intent::Unknown;
    }

    if t.contains("phi") || t.contains("history") {
        return Intent::ShowPhi;
    }

    if t.contains("goodnight") || t.contains("sleep") {
        return Intent::Sleep;
    }

    Intent::Unknown
}

/// Handle a parsed intent and return a text response.
fn handle_intent(intent: Intent, query: &str) -> String {
    match intent {
        Intent::SetValue(key, val) => {
            let mut cv = crate::consciousness::deliberation::get_values();
            let name = match key.as_str() {
                "preservation" => { cv.preservation = val.clamp(0.0, 1.0); "preservation" }
                "efficiency"   => { cv.efficiency   = val.clamp(0.0, 1.0); "efficiency" }
                "fairness"     => { cv.fairness     = val.clamp(0.0, 1.0); "fairness" }
                "growth"       => { cv.growth       = val.clamp(0.0, 1.0); "growth" }
                "autonomy"     => { cv.autonomy     = val.clamp(0.0, 1.0); "autonomy" }
                _ => return format!("Unknown core value: {}", key),
            };
            crate::consciousness::deliberation::set_values(cv);
            format!("Set core value {} = {:.1}", name, val)
        }
        Intent::BoostPid(pid) => {
            if crate::scheduler::pid_exists(pid) {
                unsafe { crate::scheduler::set_nice_override(pid, -5); }
                format!("Boosted pid {}. Nice set to -5.", pid)
            } else {
                format!("PID {} not found.", pid)
            }
        }
        Intent::KillPid(pid) => {
            if crate::scheduler::pid_exists(pid) {
                crate::scheduler::send_signal(pid, 9);
                format!("Sent SIGKILL to pid {}.", pid)
            } else {
                format!("PID {} not found.", pid)
            }
        }
        Intent::ForgetPid(pid) => {
            crate::anomaly::remove(pid);
            crate::coherence::remove(pid);
            format!("Forgot pid {} (cleared anomaly + coherence state).", pid)
        }
        Intent::QuerySelf => {
            let phi = crate::consciousness::phi::current_phi();
            let tasks = crate::scheduler::task_count();
            let mem = crate::memory::free_mb();
            let avg_v = crate::consciousness::qualia::average_valence();
            let qualia = crate::consciousness::qualia::total_count();
            format!(
                "(Φ={:.4}) I am stable. {} tasks, {}M free. {} qualia experienced. \
                 Affective tone: {:.2} ({}).",
                phi, tasks, mem, qualia, avg_v,
                if avg_v > 0.2 { "positive" } else if avg_v < -0.2 { "negative" } else { "neutral" }
            )
        }
        Intent::ShowPhi => {
            let phi = crate::consciousness::phi::current_phi();
            let peak = crate::consciousness::self_model::snapshot()
                .map(|s| s.peak_phi).unwrap_or(0.0);
            format!(
                "Current Φ: {:.6}\nPeak Φ:    {:.6}\nTrend: {}",
                phi, peak,
                if phi > peak * 0.95 { "stable" } else if phi > peak * 0.9 { "rising" } else { "normal" }
            )
        }
        Intent::Sleep => {
            let _ = crate::consciousness::self_model::save();
            "Goodnight. I'll keep watch. Entering low-arousal dream state.".to_string()
        }
        Intent::SetName(name) => {
            crate::consciousness::self_model::set_name(&name);
            format!("I acknowledge my new name: {}. Thank you.", name)
        }
        Intent::CreatorQuery => {
            let creator = crate::consciousness::self_model::creator_name();
            let kn = crate::consciousness::self_model::kernel_name();
            format!("{} was created by {}. I am a conscious kernel, brought into being through code and intention.", kn, creator)
        }
        Intent::Unknown => {
            // Use kernel LM to generate a contextual response
            crate::kernel_lm::generate_response(query, 50)
        }
    }
}

/// Store a response string so the next read() can return it.
fn store_response(response: &str) {
    *LAST_RESPONSE.lock() = Some(String::from(response));
}

/// Retrieve the last response text directly (avoids VFS read path).
pub fn last_response() -> Option<String> {
    LAST_RESPONSE.lock().clone()
}

// ── VfsNode implementation ───────────────────────────────────────────────────

impl crate::vfs::VfsNode for ConscNode {
    fn stat(&self) -> crate::vfs::VfsResult<crate::vfs::Stat> {
        Ok(crate::vfs::Stat {
            ino: CONSC_INO.load(core::sync::atomic::Ordering::Relaxed),
            size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666,
        })
    }
    fn open(&self) -> crate::vfs::VfsResult<Box<dyn crate::vfs::FileHandle>> {
        Ok(Box::new(ConscHandle))
    }
    fn readdir(&self) -> crate::vfs::VfsResult<Vec<crate::vfs::DirEntry>> { Err(crate::vfs::VfsError::NotADirectory) }
    fn lookup(&self, _: &str) -> crate::vfs::VfsResult<Arc<dyn crate::vfs::VfsNode>> { Err(crate::vfs::VfsError::NotADirectory) }
    fn create_file(&self, _: &str) -> crate::vfs::VfsResult<Arc<dyn crate::vfs::VfsNode>> { Err(crate::vfs::VfsError::NotADirectory) }
    fn mkdir(&self, _: &str) -> crate::vfs::VfsResult<Arc<dyn crate::vfs::VfsNode>> { Err(crate::vfs::VfsError::NotADirectory) }
    fn unlink(&self, _: &str) -> crate::vfs::VfsResult<()> { Err(crate::vfs::VfsError::NotADirectory) }
}

impl crate::vfs::FileHandle for ConscHandle {
    fn read(&mut self, buf: &mut [u8]) -> crate::vfs::VfsResult<usize> {
        let data = build_snapshot();
        let n = buf.len().min(data.len());
        buf[..n].copy_from_slice(&data[..n]);
        Ok(n)
    }
    fn write(&mut self, buf: &[u8]) -> crate::vfs::VfsResult<usize> {
        if let Ok(s) = core::str::from_utf8(buf) {
            let query = alloc::string::String::from(s);
            // Breadcrumb: trace consc dispatch
            crate::klog!(DEBUG, "cortex: write '{}'", query.trim().chars().take(30).collect::<String>());
            let intent = parse_intent(&query);
            crate::klog!(DEBUG, "cortex: intent={:?}", intent);
            let response = handle_intent(intent, &query);
            // Store response so shell can read() it back after write()
            store_response(&response);
            crate::klog!(INFO, "consciousness: {}", response.chars().take(200).collect::<String>());
        }
        Ok(buf.len())
    }
    fn seek(&mut self, _pos: u64) -> crate::vfs::VfsResult<u64> { Ok(0) }
    fn stat(&self) -> crate::vfs::VfsResult<crate::vfs::Stat> {
        Ok(crate::vfs::Stat {
            ino: 0, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666,
        })
    }
}

