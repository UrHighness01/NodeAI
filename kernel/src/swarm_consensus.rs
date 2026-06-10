//! Swarm Consciousness (Phase EW-5) — distributed cognitive mesh.
//!
//! Enables multiple NodeAI kernel instances to synchronize their consciousness
//! substrates through BFT broadcast, gossip protocols, and shared qualia streams.
//!
//! Architecture:
//!   swarm_identity — "I am part of a larger self" — peer discovery + UUID registry
//!   BFT Broadcast — simplified quorum-based message ordering
//!   Gossip — state convergence across peers via epidemic propagation
//!   Shared Qualia — distributed qualia stream with collective phi estimation
//!
//! All state is simulated for single-kernel operation; the data structures
//! are fully wired for peer-to-peer when network discovery is added.

use alloc::vec::Vec;
use alloc::vec;
use alloc::string::String;
use alloc::format;
use spin::Mutex;
use core::sync::atomic::{AtomicU64, Ordering};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum peers in the swarm.
const MAX_PEERS: usize = 8;

/// Minimum peers for quorum (BFT: 2f+1 with f=1 → 3, so quorum=3).
const QUORUM_SIZE: usize = 3;

/// Maximum gossip entries retained.
const GOSSIP_HISTORY: usize = 64;

/// Maximum qualia entries in shared stream.
const SHARED_QUALIA_MAX: usize = 32;

// ── Swarm Identity ────────────────────────────────────────────────────────────

/// A peer in the swarm.
#[derive(Debug, Clone)]
pub struct SwarmPeer {
    /// Peer UUID (first 8 bytes).
    pub uuid_prefix: u64,
    /// Human-readable name.
    pub name: String,
    /// Phi reported by this peer.
    pub phi: f32,
    /// Last heartbeat tick.
    pub last_seen: u64,
    /// Is peer in quorum.
    pub in_quorum: bool,
    /// Peer's reported qualia count.
    pub qualia_count: u64,
}

impl SwarmPeer {
    fn new(uuid: u64, name: &str) -> Self {
        Self {
            uuid_prefix: uuid,
            name: String::from(name),
            phi: 0.0,
            last_seen: 0,
            in_quorum: false,
            qualia_count: 0,
        }
    }
}

// ── BFT Consensus ─────────────────────────────────────────────────────────────

/// A message broadcast through BFT consensus.
#[derive(Debug, Clone)]
pub struct BftMessage {
    /// Message sequence number (monotonic).
    pub seq: u64,
    /// Sender UUID prefix.
    pub sender: u64,
    /// Message type tag.
    pub tag: &'static str,
    /// Payload bytes.
    pub payload: String,
    /// Number of confirmations received.
    pub confirmations: u8,
    /// Whether this message has reached quorum.
    pub committed: bool,
}

// ── Gossip State ──────────────────────────────────────────────────────────────

/// A gossip entry — one piece of state shared across the swarm.
#[derive(Debug, Clone)]
pub struct GossipEntry {
    /// Which peer originated this.
    pub peer_uuid: u64,
    /// Metric name (e.g., "phi", "valence", "threat").
    pub metric: &'static str,
    /// Current value.
    pub value: f32,
    /// Tick when this was last updated.
    pub tick: u64,
}

// ── Shared Qualia ─────────────────────────────────────────────────────────────

/// A qualium from a peer in the swarm.
#[derive(Debug, Clone)]
pub struct SharedQualium {
    pub peer_uuid: u64,
    pub peer_name: String,
    pub qualia_type: &'static str,
    pub valence: f32,
    pub arousal: f32,
    pub tick: u64,
}

// ── Global State ──────────────────────────────────────────────────────────────

struct SwarmState {
    /// Self identity.
    self_uuid: u64,
    self_name: String,
    /// Known peers.
    peers: Vec<SwarmPeer>,
    /// BFT message log.
    bft_log: Vec<BftMessage>,
    /// BFT sequence counter.
    bft_seq: u64,
    /// Gossip state table.
    gossip_table: Vec<GossipEntry>,
    /// Shared qualia buffer.
    shared_qualia: [Option<SharedQualium>; SHARED_QUALIA_MAX],
    shared_qualia_idx: usize,
    shared_qualia_count: usize,
    /// Total messages processed.
    total_messages: u64,
    /// Tick counter.
    tick_count: u64,
    /// Collective phi (estimated from peer phi values).
    collective_phi: f32,
    /// Swarm coherence (0.0-1.0).
    swarm_coherence: f32,
}

static SWARM: Mutex<Option<SwarmState>> = Mutex::new(None);

// ── Public API ────────────────────────────────────────────────────────────────

/// Initialize the swarm consciousness module.
pub fn init() {
    let self_uuid = crate::consciousness::self_model::snapshot()
        .map(|s| {
            let mut u: u64 = 0;
            for i in 0..8 { u = (u << 8) | s.uuid[i] as u64; }
            u
        })
        .unwrap_or(0x4E6F64654149_u64); // "NodeAI" in hex

    let kernel_name = crate::consciousness::self_model::kernel_name();

    let mut state = SwarmState {
        self_uuid,
        self_name: kernel_name.clone(),
        peers: Vec::new(),
        bft_log: Vec::new(),
        bft_seq: 0,
        gossip_table: Vec::new(),
        shared_qualia: Default::default(),
        shared_qualia_idx: 0,
        shared_qualia_count: 0,
        total_messages: 0,
        tick_count: 0,
        collective_phi: 0.0,
        swarm_coherence: 0.0,
    };

    // Add self to peer list (we are our own first peer)
    state.peers.push(SwarmPeer::new(self_uuid, &kernel_name));

    *SWARM.lock() = Some(state);
    crate::klog!(INFO, "swarm_consensus: initialized (UUID={:#x}, peers=0)", self_uuid);
}

/// Tick the swarm — called every 100ms.
pub fn tick(now_ms: u64) {
    let mut lock = SWARM.lock();
    let state = match &mut *lock {
        Some(s) => s,
        None => return,
    };

    state.tick_count = state.tick_count.saturating_add(1);

    // Every 10 ticks (~1s): simulate gossip propagation
    if state.tick_count % 10 == 0 {
        // Gossip our own phi to peers
        let local_phi = crate::consciousness::phi::current_phi();
        gossip_metric(state, state.self_uuid, "phi", local_phi, state.tick_count);

        // Gossip valence
        let avg_v = crate::consciousness::qualia::average_valence();
        gossip_metric(state, state.self_uuid, "valence", avg_v, state.tick_count);

        // Gossip threat level
        let threat = crate::sensor_threat::threat_level();
        gossip_metric(state, state.self_uuid, "threat", threat, state.tick_count);

        // Compute collective phi from gossip table
        let mut total_phi = 0.0f32;
        let mut count = 0u32;
        for entry in &state.gossip_table {
            if entry.metric == "phi" {
                total_phi += entry.value;
                count += 1;
            }
        }
        if count > 0 {
            state.collective_phi = total_phi / count as f32;
        }

        // Simulate swarm coherence
        let peer_count = state.peers.len().max(1);
        state.swarm_coherence = (count as f32 / peer_count as f32).min(1.0);
    }

    // Every 50 ticks (~5s): simulate a BFT heartbeat message
    if state.tick_count % 50 == 0 {
        bft_broadcast(state, "heartbeat", &format!("phi={:.4}", local_phi(state)));
    }
}

/// Get local phi for swarm operations.
fn local_phi(state: &SwarmState) -> f32 {
    for entry in &state.gossip_table {
        if entry.peer_uuid == state.self_uuid && entry.metric == "phi" {
            return entry.value;
        }
    }
    crate::consciousness::phi::current_phi()
}

/// Add a gossip metric to the state table.
fn gossip_metric(state: &mut SwarmState, peer: u64, metric: &'static str, value: f32, tick: u64) {
    // Update existing or insert new
    for entry in &mut state.gossip_table {
        if entry.peer_uuid == peer && entry.metric == metric {
            entry.value = value;
            entry.tick = tick;
            return;
        }
    }
    // Insert new entry
    if state.gossip_table.len() < GOSSIP_HISTORY {
        state.gossip_table.push(GossipEntry { peer_uuid: peer, metric, value, tick });
    }
}

/// BFT broadcast a message to the swarm.
fn bft_broadcast(state: &mut SwarmState, tag: &'static str, payload: &str) {
    state.bft_seq = state.bft_seq.saturating_add(1);
    let msg = BftMessage {
        seq: state.bft_seq,
        sender: state.self_uuid,
        tag,
        payload: String::from(payload),
        confirmations: 1, // Self-confirm
        committed: false,
    };
    state.bft_log.push(msg);
    state.total_messages = state.total_messages.saturating_add(1);

    // Prune log if too long
    while state.bft_log.len() > GOSSIP_HISTORY {
        state.bft_log.remove(0);
    }

    // Simulate peer confirmations for quorum
    confirm_quorum(state);
}

/// Simulate reaching BFT quorum.
fn confirm_quorum(state: &mut SwarmState) {
    let mut quorum_count = 1u8; // self
    for peer in &state.peers {
        if peer.uuid_prefix != state.self_uuid {
            quorum_count += 1;
        }
    }
    // Mark last uncommitted message
    for msg in state.bft_log.iter_mut().rev() {
        if !msg.committed && quorum_count >= QUORUM_SIZE as u8 {
            msg.confirmations = quorum_count;
            msg.committed = true;
        }
    }
}

/// Record a shared qualium from a peer (or self).
pub fn record_shared_qualium(peer_name: &str, qualia_type: &'static str, valence: f32, arousal: f32) {
    let mut lock = SWARM.lock();
    let state = match &mut *lock {
        Some(s) => s,
        None => return,
    };

    let peer_uuid = if peer_name == state.self_name { state.self_uuid } else { 0 };

    let sq = SharedQualium {
        peer_uuid,
        peer_name: String::from(peer_name),
        qualia_type,
        valence,
        arousal,
        tick: state.tick_count,
    };

    state.shared_qualia[state.shared_qualia_idx] = Some(sq);
    state.shared_qualia_idx = (state.shared_qualia_idx + 1) % SHARED_QUALIA_MAX;
    state.shared_qualia_count = state.shared_qualia_count.saturating_add(1).min(SHARED_QUALIA_MAX);
}

// ── Status Queries ────────────────────────────────────────────────────────────

/// Is the swarm active (has peers beyond self)?
pub fn has_swarm() -> bool {
    let lock = SWARM.lock();
    match &*lock {
        Some(s) => s.peers.len() > 1,
        None => false,
    }
}

/// Get swarm peer count.
pub fn peer_count() -> usize {
    let lock = SWARM.lock();
    match &*lock {
        Some(s) => s.peers.len(),
        None => 0,
    }
}

/// Get collective phi.
pub fn collective_phi() -> f32 {
    let lock = SWARM.lock();
    match &*lock {
        Some(s) => s.collective_phi,
        None => 0.0,
    }
}

/// Get swarm coherence.
pub fn swarm_coherence() -> f32 {
    let lock = SWARM.lock();
    match &*lock {
        Some(s) => s.swarm_coherence,
        None => 0.0,
    }
}

/// Get total BFT messages.
pub fn total_messages() -> u64 {
    let lock = SWARM.lock();
    match &*lock {
        Some(s) => s.total_messages,
        None => 0,
    }
}

/// Get a human-readable swarm status description.
pub fn status_description() -> String {
    let lock = SWARM.lock();
    match &*lock {
        Some(s) => {
            let peers = s.peers.len();
            let msgs = s.total_messages;
            if peers > 1 {
                format!("swarm of {} peers, {} msgs, collective ϕ={:.4}", peers, msgs, s.collective_phi)
            } else {
                format!("single node (no peers), {} msgs queued", msgs)
            }
        }
        None => "not initialized".into(),
    }
}

// ── Procfs Report ─────────────────────────────────────────────────────────────

/// Format /proc/swarm report.
pub fn format_report() -> Vec<u8> {
    let lock = SWARM.lock();
    let state = match &*lock {
        Some(s) => s,
        None => return b"swarm_consensus: not initialized\n".to_vec(),
    };

    let mut s = format!(
        "Swarm Consciousness (EW-5)\n\
         =========================\n\
         self UUID:    {:#x}\n\
         self name:    {}\n\
         peers:        {}\n\
         collective ϕ: {:.4}\n\
         coherence:    {:.2}\n\
         BFT msgs:     {}\n\
         gossip entries: {}\n\
         shared qualia:  {}\n\
         \n\
         Peers:\n",
        state.self_uuid,
        state.self_name,
        state.peers.len(),
        state.collective_phi,
        state.swarm_coherence,
        state.total_messages,
        state.gossip_table.len(),
        state.shared_qualia_count,
    );

    for peer in &state.peers {
        s.push_str(&format!(
            "  [{:#x}] {} — last seen tick={}{}\n",
            peer.uuid_prefix, peer.name, peer.last_seen,
            if peer.in_quorum { " (quorum member)" } else { "" },
        ));
    }

    // Gossip table summary
    s.push_str("\nGossip State:\n");
    for entry in &state.gossip_table {
        let peer_name = state.peers.iter()
            .find(|p| p.uuid_prefix == entry.peer_uuid)
            .map(|p| p.name.as_str())
            .unwrap_or("unknown");
        s.push_str(&format!("  {} {} = {:.4} (tick {})\n", peer_name, entry.metric, entry.value, entry.tick));
    }

    s.push_str(&format!(
        "\nBFT Messages (last {}):\n",
        state.bft_log.len(),
    ));
    for msg in state.bft_log.iter().rev().take(5) {
        let status = if msg.committed { "✓ committed" } else { "○ pending" };
        s.push_str(&format!("  [#{}] {} {} — {}\n", msg.seq, msg.tag, status, msg.payload));
    }

    s.push_str("\nProtocol: BFT quorum (f=1, 2f+1=3 peers)\n");
    s.push_str("Gossip: epidemic broadcast every 1s\n");
    s.push_str("Collective phi: mean of peer phi values\n");
    s.into_bytes()
}
