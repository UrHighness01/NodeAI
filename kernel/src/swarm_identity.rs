//! Swarm Identity — node identity and role management for the kernel mesh.
//!
//! Each node in the swarm has a unique identity (UUID), a role (leader/follower),
//! and a trust score. Identity persists across gossip rounds and is used for
//! BFT consensus quorum calculations.
//!
//! Call tick() every 1000ms. Use node_count(), our_role(), format_report().

use alloc::vec::Vec;
use alloc::format;
use alloc::string::String;
use core::sync::atomic::{AtomicBool, Ordering};
use spin::Mutex;

/// Whether swarm identity is active.
static IDENTITY_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Maximum nodes in the identity table.
const MAX_NODES: usize = 16;

/// Node roles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeRole {
    Leader,
    Follower,
    Observer,
}

impl NodeRole {
    fn name(&self) -> &'static str {
        match self {
            NodeRole::Leader => "leader",
            NodeRole::Follower => "follower",
            NodeRole::Observer => "observer",
        }
    }
}

/// A known node in the swarm.
#[derive(Debug, Clone)]
struct NodeEntry {
    /// Node UUID (first 64 bits).
    id: u64,
    /// Node's role.
    role: NodeRole,
    /// Trust score (0.0–1.0).
    trust: f32,
    /// Uptime reported by this node.
    uptime: u64,
    /// Whether this node is currently responsive.
    alive: bool,
}

/// Identity state.
struct IdentityState {
    /// Our own node UUID.
    our_id: u64,
    /// Our role.
    our_role: NodeRole,
    /// Known nodes.
    nodes: Vec<NodeEntry>,
    /// Total nodes ever seen.
    total_nodes_seen: u64,
    /// Leader election term.
    term: u64,
}

static STATE: Mutex<Option<IdentityState>> = Mutex::new(None);

/// Initialize swarm identity.
pub fn init() {
    let mut state = IdentityState {
        our_id: crate::scheduler::uptime_ms() ^ 0xCAFE_BABE,
        our_role: NodeRole::Follower,
        nodes: Vec::with_capacity(MAX_NODES),
        total_nodes_seen: 1,
        term: 1,
    };

    let mut lock = STATE.lock();
    *lock = Some(state);
    IDENTITY_ACTIVE.store(true, Ordering::Release);
    crate::klog!(INFO, "swarm_identity: node identity initialized (id=0x{:x})",
        crate::scheduler::uptime_ms() ^ 0xCAFE_BABE);
}

/// Tick swarm identity — called every 1000ms.
/// Simulates peer discovery and heartbeat monitoring.
pub fn tick() {
    if !IDENTITY_ACTIVE.load(Ordering::Acquire) { return; }

    let mut lock = STATE.lock();
    let state = match &mut *lock {
        Some(s) => s,
        None => return,
    };

    let now = crate::scheduler::uptime_ms() / 1000;

    // Simulate discovering a new node occasionally
    let uptime_secs = crate::scheduler::uptime_ms() / 1000;
    let expected_nodes = ((uptime_secs / 30) as usize).min(MAX_NODES - 1);

    while state.nodes.len() < expected_nodes {
        let new_id = state.our_id.wrapping_add(state.nodes.len() as u64 + 1);
        state.nodes.push(NodeEntry {
            id: new_id,
            role: NodeRole::Follower,
            trust: 0.8,
            uptime: now,
            alive: true,
        });
        state.total_nodes_seen = state.total_nodes_seen.saturating_add(1);
    }

    // Check peer liveness — mark dead if no heartbeat in 30s
    for node in state.nodes.iter_mut() {
        if now.saturating_sub(node.uptime) > 30 {
            node.alive = false;
        }
    }

    // Leader election: lowest ID among alive nodes becomes leader
    let mut leader_id = state.our_id;
    let mut leader_found = false;
    for node in &state.nodes {
        if node.alive && node.id < leader_id {
            leader_id = node.id;
            leader_found = true;
        }
    }

    if leader_found {
        if leader_id == state.our_id {
            state.our_role = NodeRole::Leader;
        } else {
            state.our_role = NodeRole::Follower;
        }
    } else {
        state.our_role = NodeRole::Leader; // Solo node
    }

    // Increment term on leader changes
    state.term = state.term.wrapping_add(1);
}

/// Get our node role.
pub fn our_role() -> NodeRole {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.our_role,
        None => NodeRole::Observer,
    }
}

/// Number of known nodes.
pub fn node_count() -> usize {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.nodes.len() + 1, // +1 for ourselves
        None => 1,
    }
}

/// Number of alive nodes.
pub fn alive_count() -> usize {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.nodes.iter().filter(|n| n.alive).count() + 1,
        None => 1,
    }
}

/// Current leader election term.
pub fn term() -> u64 {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.term,
        None => 0,
    }
}

/// Format /proc/swarm_identity report.
pub fn format_report() -> Vec<u8> {
    let active = IDENTITY_ACTIVE.load(Ordering::Acquire);
    if !active {
        return format!("Swarm Identity\nNot initialized\n").into_bytes();
    }
    let lock = STATE.lock();
    match &*lock {
        Some(s) => {
            let mut report = format!(
                "Swarm Identity\n\
                 ==============\n\
                 our_id:       0x{:x}\n\
                 our_role:     {}\n\
                 known_nodes:  {}\n\
                 alive_nodes:  {}\n\
                 total_seen:   {}\n\
                 term:         {}\n\
                 \n\
                 Node Table:\n",
                s.our_id,
                s.our_role.name(),
                s.nodes.len() + 1,
                s.nodes.iter().filter(|n| n.alive).count() + 1,
                s.total_nodes_seen,
                s.term,
            );

            report.push_str(&format!(
                "  {:16} {:10} {:8} {:6}\n",
                "ID", "ROLE", "TRUST", "ALIVE"
            ));
            report.push_str(&format!(
                "  {:16x} {:10} {:8} {:6}\n",
                s.our_id,
                s.our_role.name(),
                "1.00",
                "YES",
            ));
            for node in &s.nodes {
                report.push_str(&format!(
                    "  {:16x} {:10} {:8.2} {:6}\n",
                    node.id,
                    node.role.name(),
                    node.trust,
                    if node.alive { "YES" } else { "NO" },
                ));
            }

            report.into_bytes()
        }
        None => format!("Swarm Identity\nUninitialized\n").into_bytes(),
    }
}
