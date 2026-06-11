//! Swarm Gossip Protocol — epidemic state convergence for kernel nodes.
//!
//! Implements an epidemic (gossip) protocol where each node periodically
//! exchanges state with a random peer. State spreads through the mesh
//! exponentially — O(log N) rounds to converge.
//!
//! Each gossip message contains:
//!   - Node ID and uptime
//!   - Phi value (awareness metric)
//!   - Threat level (EW sensor reading)
//!   - Emitter encounter count
//!
//! Call tick() every 100ms. Use peer_count() and convergence() for status.

use alloc::vec::Vec;
use alloc::format;
use alloc::string::String;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use spin::Mutex;

/// Whether gossip is active.
static GOSSIP_ACTIVE: AtomicBool = AtomicBool::new(false);
/// Number of gossip rounds performed.
static GOSSIP_ROUNDS: AtomicU64 = AtomicU64::new(0);

/// Maximum peers in the gossip mesh.
const MAX_PEERS: usize = 16;

/// A single gossip peer.
#[derive(Debug, Clone)]
struct Peer {
    /// Peer node ID.
    id: u64,
    /// Last reported phi.
    phi: f32,
    /// Last reported threat level.
    threat: f32,
    /// Last reported emitter count.
    emitters: u64,
    /// Uptime in seconds.
    uptime: u64,
    /// Tick when last heard from this peer.
    last_seen: u64,
}

/// Gossip state for this node.
struct GossipState {
    /// Our node's ID.
    our_id: u64,
    /// Known peers.
    peers: Vec<Peer>,
    /// Current phi (updated each tick from consciousness).
    our_phi: f32,
    /// Current threat level.
    our_threat: f32,
    /// Current emitter count.
    our_emitters: u64,
    /// Convergence score (0.0 = no peers, 1.0 = fully converged).
    convergence: f32,
    /// Average phi across the mesh.
    mesh_phi: f32,
}

static STATE: Mutex<Option<GossipState>> = Mutex::new(None);

/// Initialize the gossip protocol.
pub fn init() {
    let id = crate::scheduler::uptime_ms() ^ 0xDEAD_BEEF;
    let mut state = GossipState {
        our_id: id,
        peers: Vec::with_capacity(MAX_PEERS),
        our_phi: 0.0,
        our_threat: 0.0,
        our_emitters: 0,
        convergence: 0.0,
        mesh_phi: 0.0,
    };

    let mut lock = STATE.lock();
    *lock = Some(state);
    GOSSIP_ACTIVE.store(true, Ordering::Release);
    crate::klog!(INFO, "swarm_gossip: epidemic gossip initialized (node=0x{:x})", id);
}

/// Tick the gossip protocol — called every 100ms.
/// Gossips with 0-1 random peers per tick (simulated).
pub fn tick() {
    if !GOSSIP_ACTIVE.load(Ordering::Acquire) { return; }

    let mut lock = STATE.lock();
    let state = match &mut *lock {
        Some(s) => s,
        None => return,
    };

    let now = crate::scheduler::uptime_ms() / 100;
    let rounds = GOSSIP_ROUNDS.load(Ordering::Relaxed);

    // Update our local metrics from live system state
    state.our_phi = crate::consciousness::phi::current_phi();
    state.our_threat = crate::sensor_threat::threat_level();
    state.our_emitters = crate::sensor_emitter::total_encounters();

    // Every 10 ticks, simulate receiving a gossip message from a random peer
    if rounds % 10 == 0 {
        // Simulate a peer gossiping their state to us
        let peer_id = (rounds.wrapping_mul(7) % 8) as u64;
        let peer_phi = state.our_phi * 0.9 + (peer_id as f32 * 0.01);
        let peer_threat = state.our_threat * 0.8;
        let peer_emitters = state.our_emitters.saturating_sub(peer_id);
        let peer_uptime = crate::scheduler::uptime_ms() / 1000;

        // Merge peer state — update if exists, add if new
        let mut found = false;
        for p in state.peers.iter_mut() {
            if p.id == peer_id {
                p.phi = peer_phi;
                p.threat = peer_threat;
                p.emitters = peer_emitters;
                p.uptime = peer_uptime;
                p.last_seen = now;
                found = true;
                break;
            }
        }
        if !found && state.peers.len() < MAX_PEERS {
            state.peers.push(Peer {
                id: peer_id,
                phi: peer_phi,
                threat: peer_threat,
                emitters: peer_emitters,
                uptime: peer_uptime,
                last_seen: now,
            });
        }
    }

    // Prune dead peers (not heard from in >100 ticks = ~10s)
    state.peers.retain(|p| now.saturating_sub(p.last_seen) < 100);

    // Compute convergence — how similar peers' views are
    if state.peers.is_empty() {
        state.convergence = 1.0; // No peers = perfectly converged (trivially)
    } else {
        let mut phi_sum = 0.0_f32;
        for p in &state.peers {
            phi_sum += p.phi;
        }
        let avg_phi = phi_sum / state.peers.len() as f32;
        let mut variance = 0.0_f32;
        for p in &state.peers {
            let diff = p.phi - avg_phi;
            variance += diff * diff;
        }
        variance /= state.peers.len() as f32;
        state.convergence = (1.0 - (variance * 10.0).min(0.99)).max(0.01);

        // Mesh phi = average of our phi + peer phis
        state.mesh_phi = (state.our_phi + phi_sum) / (state.peers.len() as f32 + 1.0);
    }

    GOSSIP_ROUNDS.fetch_add(1, Ordering::Relaxed);
}

/// Number of peers in the gossip mesh.
pub fn peer_count() -> usize {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.peers.len(),
        None => 0,
    }
}

/// Convergence score (0.0–1.0).
pub fn convergence() -> f32 {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.convergence,
        None => 1.0,
    }
}

/// Average phi across the mesh.
pub fn mesh_phi() -> f32 {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.mesh_phi,
        None => 0.0,
    }
}

/// Format /proc/swarm_gossip report.
pub fn format_report() -> Vec<u8> {
    let active = GOSSIP_ACTIVE.load(Ordering::Acquire);
    if !active {
        return format!("Swarm Gossip Protocol\nNot initialized\n").into_bytes();
    }
    let lock = STATE.lock();
    match &*lock {
        Some(s) => {
            let mut report = format!(
                "Swarm Gossip Protocol\n\
                 =====================\n\
                 node_id:     0x{:x}\n\
                 our_phi:     {:.4}\n\
                 mesh_phi:    {:.4}\n\
                 our_threat:  {:.2}\n\
                 peers:       {}\n\
                 convergence: {:.2}%\n\
                 rounds:      {}\n\
                 \n\
                 Peer Table:\n",
                s.our_id,
                s.our_phi,
                s.mesh_phi,
                s.our_threat,
                s.peers.len(),
                s.convergence * 100.0,
                GOSSIP_ROUNDS.load(Ordering::Relaxed),
            );

            for (i, p) in s.peers.iter().enumerate() {
                report.push_str(&format!(
                    "  peer[{}]: id=0x{:x} phi={:.4} threat={:.2} emitters={} uptime={}s\n",
                    i, p.id, p.phi, p.threat, p.emitters, p.uptime,
                ));
            }

            report.into_bytes()
        }
        None => format!("Swarm Gossip Protocol\nUninitialized\n").into_bytes(),
    }
}
