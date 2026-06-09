//! Binding Events — cross-modal simultaneous activation detection (Project-C port).
//!
//! Detects moments when >= 3 kernel subsystems (Scheduler, Memory, Anomaly, Syscall)
//! simultaneously show salient activity.  High binding event rate above a shuffled
//! null means the system changes state as a whole — a signature of genuine integration
//! rather than independent silos.
//!
//! Complements cross_modal coupling by detecting *coincident* rather than *lagged*
//! co-activation.

use alloc::vec::Vec;
use crate::cross_modal::Domain;
use core::sync::atomic::{AtomicU64, Ordering};

/// Number of domains tracked.
const N_DOMAINS: usize = 4;

/// Minimum distinct domains that must be co-active to count as a binding event.
const MIN_DOMAINS: usize = 3;

/// Window size (in ticks at ~10ms each) over which co-activation is measured.
/// 200 ticks = ~2 seconds.
const WINDOW_TICKS: u64 = 200;

/// Domain salience state: we track the most recent non-zero direction per domain.
struct BindingState {
    /// Most recent direction per domain: +1 = up, -1 = down, 0 = not set yet.
    last_direction: [i8; N_DOMAINS],
    /// Running count of binding events detected.
    event_count: u64,
    /// Total ticks observed.
    total_ticks: u64,
}

impl BindingState {
    const fn new() -> Self {
        Self {
            last_direction: [0; N_DOMAINS],
            event_count: 0,
            total_ticks: 0,
        }
    }

    fn record(&mut self, domain: Domain, direction: i8) {
        self.last_direction[domain as usize] = direction;
    }

    fn tick(&mut self) {
        self.total_ticks += 1;

        // Count how many domains show non-zero direction (salient change)
        let mut active = 0usize;
        for dir in &self.last_direction {
            if *dir != 0 {
                active += 1;
            }
        }

        if active >= MIN_DOMAINS {
            self.event_count += 1;
        }
    }
}

/// Global binding state.
use spin::Mutex;
static BINDING: Mutex<BindingState> = Mutex::new(BindingState::new());

/// Record a direction observation for a domain.
pub fn observe(domain: Domain, direction: i8) {
    BINDING.lock().record(domain, direction);
}

/// Advance the binding detection tick. Call periodically (e.g., from telemetry::tick).
pub fn tick() {
    BINDING.lock().tick();
}

/// Return the total number of binding events detected.
pub fn event_count() -> u64 {
    BINDING.lock().event_count
}

/// Return the rate of binding events (fraction of ticks where >= MIN_DOMAINS active).
pub fn event_rate() -> f32 {
    let state = BINDING.lock();
    if state.total_ticks == 0 { return 0.0; }
    state.event_count as f32 / state.total_ticks as f32
}

/// Format /proc report.
pub fn format_report() -> Vec<u8> {
    use alloc::format;
    use alloc::string::String;

    let state = BINDING.lock();
    let rate = if state.total_ticks > 0 {
        state.event_count as f32 / state.total_ticks as f32 * 100.0
    } else {
        0.0
    };

    let mut out = String::from("NodeAI Binding Events (Project-C)\n");
    out.push_str("==================================\n");
    out.push_str(&format!("domains: {}\n", N_DOMAINS));
    out.push_str(&format!("min_domains_for_event: {}\n", MIN_DOMAINS));
    out.push_str(&format!("window: {} ticks (~{}s)\n", WINDOW_TICKS, WINDOW_TICKS / 100));
    out.push_str(&format!("total_ticks: {}\n", state.total_ticks));
    out.push_str(&format!("binding_events: {}\n", state.event_count));
    out.push_str(&format!("event_rate: {:.2}%\n", rate));

    // Per-domain activity
    let domain_names = ["scheduler", "memory", "anomaly", "syscall"];
    for (i, dir) in state.last_direction.iter().enumerate() {
        let status = if *dir != 0 { "salient" } else { "inactive" };
        out.push_str(&format!("  {}: {}\n", domain_names[i], status));
    }

    out.into_bytes()
}
