//! Cross-Modal Predictive Coupling — cross-domain coherence tracking (Project-C port).
//!
//! Measures whether one kernel subsystem's state predicts another's beyond chance.
//! For example: does scheduler coherence predict memory pressure 5 seconds later?
//! If yes, the subsystems form a coupled system — if no, they operate independently.
//!
//! Uses a simplified lagged cross-correlation: for each ordered pair of signals
//! (source → target), we track whether source's direction change at time t
//! predicts target's direction change at time t+lag.
//!
//! This is the first OS kernel to measure cross-subsystem predictive coupling.

use alloc::vec::Vec;
use spin::Mutex;

/// Number of signal domains tracked.
const N_DOMAINS: usize = 5;

/// Sliding window of observations per domain.
const WINDOW_SIZE: usize = 64;

/// The five kernel subsystem signals we track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Domain {
    Scheduler = 0,  // coherence horizon mean
    Memory    = 1,  // free MB delta
    Anomaly   = 2,  // anomaly rate
    Syscall   = 3,  // syscall rate
    Spectrum  = 4,  // RF energy density (EW sensory cortex)
}

impl Domain {
    pub fn name(self) -> &'static str {
        match self {
            Domain::Scheduler => "scheduler",
            Domain::Memory    => "memory",
            Domain::Anomaly   => "anomaly",
            Domain::Syscall   => "syscall",
            Domain::Spectrum  => "spectrum",
        }
    }

    pub fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Domain::Scheduler),
            1 => Some(Domain::Memory),
            2 => Some(Domain::Anomaly),
            3 => Some(Domain::Syscall),
            4 => Some(Domain::Spectrum),
            _ => None,
        }
    }
}

/// Ring buffer of normalized directional changes per domain.
struct DomainBuffer {
    /// +1 = up, -1 = down, 0 = unchanged for each tick.
    directions: Vec<i8>,
    /// Last raw value for computing direction.
    last_value: f32,
}

impl DomainBuffer {
    fn new() -> Self {
        Self {
            directions: Vec::with_capacity(WINDOW_SIZE),
            last_value: 0.0,
        }
    }

    fn observe(&mut self, value: f32) {
        let dir = if value > self.last_value * 1.01 { 1 }
                  else if value < self.last_value * 0.99 { -1 }
                  else { 0 };
        self.last_value = value;
        self.directions.push(dir);
        if self.directions.len() > WINDOW_SIZE {
            self.directions.remove(0);
        }
    }

    fn as_slice(&self) -> &[i8] {
        &self.directions
    }
}

struct CrossModalState {
    buffers: [DomainBuffer; N_DOMAINS],
    /// Number of times each domain has been updated.
    update_count: [u64; N_DOMAINS],
}

impl CrossModalState {
    const fn new() -> Self {
        const EMPTY: DomainBuffer = DomainBuffer {
            directions: Vec::new(),
            last_value: 0.0,
        };
        Self {
            buffers: [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
            update_count: [0; N_DOMAINS],
        }
    }

    fn observe(&mut self, domain: Domain, value: f32) {
        let idx = domain as usize;
        self.buffers[idx].observe(value);
        self.update_count[idx] = self.update_count[idx].wrapping_add(1);
    }

    /// Compute lagged cross-correlation: does src[t] direction predict tgt[t+lag]?
    /// Returns [-1.0, 1.0] where 1.0 = always same direction, -1.0 = always opposite.
    fn lagged_cross_corr(&self, src: Domain, tgt: Domain, lag: usize) -> f32 {
        let s = self.buffers[src as usize].as_slice();
        let t = self.buffers[tgt as usize].as_slice();

        if s.len() < lag + 4 || t.len() < lag + 4 {
            return 0.0; // insufficient data
        }

        let mut same = 0usize;
        let mut total = 0usize;

        // Align: s[i] predicts t[i + lag]
        for i in 0..(s.len().min(t.len()).saturating_sub(lag)) {
            let sd = s[i];
            let td = t[i + lag];
            // Only count non-zero directions
            if sd == 0 || td == 0 { continue; }
            total += 1;
            if sd == td { same += 1; }
        }

        if total == 0 { 0.0 }
        else { (same as f32 / total as f32) * 2.0 - 1.0 }
    }
}

static CROSS_MODAL: Mutex<CrossModalState> = Mutex::new(CrossModalState::new());

/// Record an observation for a domain. Called from telemetry::tick or equivalent.
pub fn observe(domain: Domain, value: f32) {
    CROSS_MODAL.lock().observe(domain, value);
}

/// Return the lagged cross-correlation between two domains.
pub fn coupling(src: Domain, tgt: Domain, lag: usize) -> f32 {
    CROSS_MODAL.lock().lagged_cross_corr(src, tgt, lag)
}

/// Format a /proc report.
pub fn format_report() -> Vec<u8> {
    use alloc::format;
    use alloc::string::String;

    let state = CROSS_MODAL.lock();
    let mut out = String::from("NodeAI Cross-Modal Coupling (Project-C)\n");
    out.push_str("=========================================\n");

    for si in 0..N_DOMAINS {
        for ti in 0..N_DOMAINS {
            if si == ti { continue; }
            let src = Domain::from_index(si).unwrap();
            let tgt = Domain::from_index(ti).unwrap();
            // Compute coupling at lags 1, 2, 3
            for lag in 1..=3 {
                let c = state.lagged_cross_corr(src, tgt, lag);
                if c.abs() > 0.1 { // only show meaningful couplings
                    out.push_str(&format!(
                        "  {} → {} (lag={}): {:.3}\n",
                        src.name(), tgt.name(), lag, c
                    ));
                }
            }
        }
    }

    out.into_bytes()
}
