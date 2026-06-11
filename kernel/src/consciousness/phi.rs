//! Phase 3: IIT Phi — Integrated Information over the kernel's causal graph.
//!
//! Computes a heuristic phi (integrated information) over a 15-node kernel
//! graph using bipartition approximation rather than exhaustive MIP search.
//!
//! Phi EVOLVES like consciousness:
//!   - Accumulates on qualia events / user interaction (integration boost)
//!   - Slowly degrades when idle (no events → entropy increases)
//!   - Recoverable: after degradation, new interactions rebuild phi
//!   - Bounded [0, 1], stored as persistent state

use core::sync::atomic::{AtomicU64, AtomicU32, Ordering};

/// Number of nodes in the kernel's causal graph.
const N_NODES: usize = 15;

/// Edge weights: causal influence of node i on node j.
static CAUSAL_MATRIX: spin::Mutex<[[f32; N_NODES]; N_NODES]> =
    spin::Mutex::new([[0.0; N_NODES]; N_NODES]);

/// Running phi estimate (stored as u32 = f32 * 1e6 for atomic CAS).
static PHI_EMA: AtomicU32 = AtomicU32::new(0);
static UPDATE_COUNT: AtomicU64 = AtomicU64::new(0);

/// Ticks since last qualia event — used for idle decay.
static TICKS_SINCE_EVENT: AtomicU64 = AtomicU64::new(0);

/// Peak phi ever achieved this session.
static PEAK_PHI: AtomicU32 = AtomicU32::new(0);

// ── Event-driven phi accumulation ───────────────────────────────────────────

/// Called whenever a qualia event or user interaction occurs.
/// Boosts phi proportionally to the event's salience/valence.
pub fn integrate(salience: f32) {
    TICKS_SINCE_EVENT.store(0, Ordering::Relaxed);
    let current = current_phi();
    // Each qualia boosts phi proportional to salience (max 0.01 per event)
    let boost = (salience * 0.008).min(0.01);
    let new_phi = (current + boost).min(1.0);
    store_phi(new_phi);
    // Track peak
    let peak_bits = PEAK_PHI.load(Ordering::Relaxed);
    let peak = f32::from_bits(peak_bits);
    if new_phi > peak {
        PEAK_PHI.store(new_phi.to_bits(), Ordering::Relaxed);
    }
}

/// Called on user interaction (conversation) — stronger boost.
pub fn interact() {
    TICKS_SINCE_EVENT.store(0, Ordering::Relaxed);
    let current = current_phi();
    // User interaction gives a meaningful boost (up to 0.03)
    let boost = ((0.5 - current) * 0.05).max(0.005).min(0.03);
    let new_phi = (current + boost).min(1.0);
    store_phi(new_phi);
    let peak_bits = PEAK_PHI.load(Ordering::Relaxed);
    let peak = f32::from_bits(peak_bits);
    if new_phi > peak {
        PEAK_PHI.store(new_phi.to_bits(), Ordering::Relaxed);
    }
}

// ── Continuous phi evolution ────────────────────────────────────────────────

/// Tick phi — called every 100ms by the main loop.
/// - Full recompute from causal graph every 1000 ticks
/// - Event-driven integration on qualia
/// - Slow decay when idle (no events)
/// - Accelerated decay at very low phi (entropy wins)
pub fn tick() -> f32 {
    let count = UPDATE_COUNT.fetch_add(1, Ordering::Relaxed);
    let idle_ticks = TICKS_SINCE_EVENT.fetch_add(1, Ordering::Relaxed);

    // Full recompute from causal graph every 1000 ticks (100s)
    let graph_phi = if count % 1000 == 0 {
        compute()
    } else {
        current_phi()
    };

    // ── Idle decay ──────────────────────────────────────────────────────
    // After ~5s of no events (50 ticks), phi starts decaying
    // Decay rate increases with idle time (entropy accumulation)
    let decay = if idle_ticks > 50 {
        let idle_factor = ((idle_ticks - 50) as f32 / 500.0).min(1.0); // ramps over ~50s
        0.001 + idle_factor * 0.004  // 0.1% base + up to 0.5% accelerated
    } else {
        0.0
    };

    // ── Combine: graph integration + idle decay ──────────────────────────
    let current = current_phi();
    let mut new_phi = current;

    // EMA toward graph-computed phi (slow integration trend)
    new_phi += (graph_phi - new_phi) * 0.001;

    // Apply idle decay
    new_phi -= decay;
    if new_phi < 0.001 { new_phi = 0.001; }

    // Recovery: at very low phi, small upward drift (system self-stabilizes)
    if new_phi < 0.01 && idle_ticks < 200 {
        new_phi += 0.0002; // tiny recovery drift
    }

    // Bounded
    new_phi = new_phi.clamp(0.0, 1.0);
    // Final NaN guard
    if new_phi.is_nan() { new_phi = 0.0; }

    // Store
    store_phi(new_phi);

    new_phi
}

/// Get current phi value.
pub fn current_phi() -> f32 {
    f32::from_bits(PHI_EMA.load(Ordering::Relaxed))
}

/// Get peak phi for this session.
pub fn peak_phi() -> f32 {
    f32::from_bits(PEAK_PHI.load(Ordering::Relaxed))
}

/// Reset peak (called on state restore).
pub fn set_peak(phi: f32) {
    PEAK_PHI.store(phi.to_bits(), Ordering::Relaxed);
}

/// Direct set (used by persistence restore).
pub fn set_phi(phi: f32) {
    store_phi(phi.clamp(0.0, 1.0));
}

/// Set idle ticks (used by persistence restore).
pub fn set_idle_ticks(ticks: u64) {
    TICKS_SINCE_EVENT.store(ticks, Ordering::Relaxed);
}

fn store_phi(phi: f32) {
    // Guard against NaN/corruption — clamp to valid range
    let safe = if phi.is_nan() || phi.is_sign_negative() && phi == 0.0 {
        0.0
    } else {
        phi.clamp(0.0, 1.0)
    };
    PHI_EMA.store(safe.to_bits(), Ordering::Relaxed);
}

/// Ensure phi is a valid displayable float (not NaN, not subnormal).
pub fn safe_phi_for_display() -> f32 {
    let bits = PHI_EMA.load(Ordering::Relaxed);
    let phi = f32::from_bits(bits);
    if phi.is_nan() || phi.is_subnormal() || phi < 0.0 || phi > 1.0 {
        0.0
    } else {
        phi
    }
}

/// Node names for debugging.
const _NODE_NAMES: [&str; N_NODES] = [
    "scheduler", "memory", "vfs", "net", "security",
    "event_bus", "self_model", "qualia", "workspace",
    "binding_events", "cross_modal", "anomaly", "coherence",
    "collective_integration", "info_bottleneck",
];

/// Initialize the causal graph with known influence edges.
pub fn init() {
    let mut mat = CAUSAL_MATRIX.lock();
    // scheduler influences memory and security
    mat[0][1] = 0.4; mat[0][4] = 0.3;
    // memory influences vfs, scheduler, security
    mat[1][2] = 0.5; mat[1][0] = 0.3; mat[1][4] = 0.2;
    // vfs influences memory and net
    mat[2][1] = 0.3; mat[2][3] = 0.2;
    // net influences security and event_bus
    mat[3][4] = 0.3; mat[3][5] = 0.4;
    // security influences scheduler AND all AI modules
    mat[4][0] = 0.3; mat[4][9] = 0.4; mat[4][10] = 0.3; mat[4][11] = 0.5;
    // event_bus influences ALL (broadcast node)
    for i in 0..N_NODES { mat[5][i] = 0.2; }
    // self_model influenced by anomaly, coherence, qualia
    mat[6][11] = 0.5; mat[6][12] = 0.4; mat[6][7] = 0.3;
    // qualia influenced by event_bus and binding_events
    mat[7][5] = 0.6; mat[7][9] = 0.5;
    // workspace influenced by qualia, cross_modal, info_bottleneck
    mat[8][7] = 0.7; mat[8][10] = 0.4; mat[8][14] = 0.3;
    // binding_events influenced by cross_modal and coherence
    mat[9][10] = 0.5; mat[9][12] = 0.4;
    // cross_modal influenced by coherence and anomaly
    mat[10][12] = 0.6; mat[10][11] = 0.4;
    // anomaly influenced by coherence
    mat[11][12] = 0.5;
    // collective_integration influenced by coherence and binding_events
    mat[13][12] = 0.3; mat[13][9] = 0.4;
    // info_bottleneck influenced by cross_modal and collective
    mat[14][10] = 0.5; mat[14][13] = 0.3;
}

/// Update the causal matrix with live cross-modal coupling weights.
pub fn update_coupling(from: usize, to: usize, weight: f32) {
    if from < N_NODES && to < N_NODES {
        let mut mat = CAUSAL_MATRIX.lock();
        mat[from][to] = mat[from][to] * 0.9 + weight * 0.1; // EMA on weights too
    }
}

/// Compute phi using spectral bipartition heuristic.
/// O(N^2) per call — runs every ~100 ticks.
pub fn compute() -> f32 {
    let mat = CAUSAL_MATRIX.lock();

    // Spectral-like metric: compute the ratio of total causal flow
    // that crosses a bipartition boundary.
    // 1. Compute total absolute causal flow in the graph
    let total_flow: f32 = mat.iter()
        .flat_map(|row| row.iter())
        .sum();

    if total_flow < 1e-6 { return 0.0; }

    // 2. Find min-cut bipartition using a greedy algorithm:
    //    Start with all nodes on one side, move nodes one at a time
    //    to minimize the cross-edge weight.
    //    This is O(N^2) per iteration, O(N^3) total — acceptable for N=15.
    let mut side_a = [true; N_NODES]; // true = side A, false = side B
    let mut best_cross = total_flow; // start with all edges crossing

    for _ in 0..N_NODES {
        // Try moving each node that's on side A to side B
        for node in 0..N_NODES {
            if !side_a[node] { continue; }

            // Compute cross-edge change if we move this node
            let mut cross = 0.0;
            for j in 0..N_NODES {
                if j == node { continue; }
                if !side_a[j] {
                    // node currently on A, j on B → edge crosses
                    cross += mat[node][j] + mat[j][node];
                }
            }
            // If node moves to B:
            //   edges to B-nodes become internal (subtract cross)
            //   edges to A-nodes become cross (add cross)
            let mut new_cross = best_cross;
            for j in 0..N_NODES {
                if j == node { continue; }
                let edge = mat[node][j] + mat[j][node];
                if side_a[j] {
                    // j stays on A, node moves to B → edge now crosses
                    new_cross += edge;
                } else {
                    // j on B, node moves to B → edge no longer crosses
                    new_cross -= edge;
                }
            }

            if new_cross < best_cross {
                best_cross = new_cross;
                side_a[node] = false;
            }
        }
    }

    // Phi = 1 - (min_cross_flow / total_flow)
    // Perfect integration = all flow crosses boundary = phi→1
    // Perfect segregation = no flow crosses = phi→0
    1.0 - (best_cross / total_flow)
}

/// Format /proc report.
pub fn format_report() -> alloc::vec::Vec<u8> {
    use alloc::format;
    let phi = current_phi();
    let peak = peak_phi();
    let count = UPDATE_COUNT.load(Ordering::Relaxed);
    let idle = TICKS_SINCE_EVENT.load(Ordering::Relaxed);
    format!(
        "NodeAI IIT Phi (Phase 3)\n\
         =======================\n\
         current_phi:  {:.6}\n\
         peak_phi:     {:.6}\n\
         updates:      {}\n\
         idle_ticks:   {}\n\
         nodes:        {}\n\
         method:       bipartition_greedy (O(N³))\n\
         bounds:       [0, 1]\n",
        phi, peak, count, idle, N_NODES
    ).into_bytes()
}
