//! Phase 13 — Self-instrumentation telemetry
//!
//! The kernel records events into a fixed-size ring buffer and periodically
//! exports a JSON-like snapshot to `/ai/telemetry` in the VFS.  The AI engine
//! reads that file to propose scheduler / memory / driver tuning parameters.
//!
//! Design goals
//! ───────────────────────────────────────────────────────────────────────────
//! • Zero allocation on the hot path (ring buffer uses plain arrays).
//! • Lock contention bounded: snapshot is built under a very short lock window.
//! • Tuning proposals flow from AI → `apply_proposal()` → kernel subsystems.

use core::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use spin::Mutex;
use alloc::{string::String, vec::Vec, format};

// ── Ring-buffer capacity ─────────────────────────────────────────────────────
const RING_SIZE: usize = 256;

// ── Event types emitted by kernel subsystems ─────────────────────────────────

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum EventKind {
    SyscallEnter    = 1,
    SyscallExit     = 2,
    TaskSwitch      = 3,
    PageFault       = 4,
    IrqFired        = 5,
    NetPacketRx     = 6,
    NetPacketTx     = 7,
    AiInference     = 8,
    SecurityAlert   = 9,
    MemAlloc        = 10,
    MemFree         = 11,
    DiskIo          = 12,
}

#[derive(Debug, Clone, Copy)]
pub struct TelemetryEvent {
    /// Monotonic uptime milliseconds at event creation.
    pub timestamp_ms: u64,
    pub kind:         EventKind,
    /// Generic 64-bit payload (meaning depends on `kind`).
    pub a:            u64,
    pub b:            u64,
}

// ── Ring buffer ───────────────────────────────────────────────────────────────

struct RingBuffer {
    buf:   [TelemetryEvent; RING_SIZE],
    write: usize,   // next write slot (wraps at RING_SIZE)
    count: usize,   // number of valid events (≤ RING_SIZE)
}

impl RingBuffer {
    const EMPTY_EVENT: TelemetryEvent = TelemetryEvent {
        timestamp_ms: 0,
        kind: EventKind::SyscallEnter,
        a: 0,
        b: 0,
    };

    const fn new() -> Self {
        Self {
            buf:   [Self::EMPTY_EVENT; RING_SIZE],
            write: 0,
            count: 0,
        }
    }

    fn push(&mut self, ev: TelemetryEvent) {
        self.buf[self.write] = ev;
        self.write = (self.write + 1) % RING_SIZE;
        if self.count < RING_SIZE { self.count += 1; }
    }

    /// Iterate events oldest-first.
    fn iter(&self) -> impl Iterator<Item = &TelemetryEvent> {
        let start = if self.count < RING_SIZE {
            0
        } else {
            self.write // oldest slot when buffer is full
        };
        let count = self.count;
        (0..count).map(move |i| {
            let idx = (start + i) % RING_SIZE;
            &self.buf[idx]
        })
    }
}

static RING: Mutex<RingBuffer> = Mutex::new(RingBuffer::new());

// ── Aggregate counters (updated atomically — no lock needed) ─────────────────

static SYSCALL_COUNT:   AtomicU64 = AtomicU64::new(0);
static TASK_SWITCH_CNT: AtomicU64 = AtomicU64::new(0);
static NET_RX_BYTES:    AtomicU64 = AtomicU64::new(0);
static NET_TX_BYTES:    AtomicU64 = AtomicU64::new(0);
static PAGE_FAULT_CNT:  AtomicU64 = AtomicU64::new(0);
static AI_INFER_CNT:    AtomicU64 = AtomicU64::new(0);
static SECURITY_ALERTS: AtomicU64 = AtomicU64::new(0);

// ── AI tuning proposals ───────────────────────────────────────────────────────

/// Scheduler time-quantum proposed by the AI (ms).  0 = use kernel default.
pub static AI_QUANTUM_MS: AtomicI64  = AtomicI64::new(0);
/// Memory pressure threshold proposed by the AI (pages to keep free).
pub static AI_FREE_PAGES: AtomicI64  = AtomicI64::new(0);

// ── Public API ────────────────────────────────────────────────────────────────

/// Initialise the telemetry subsystem and create `/ai/telemetry` in the VFS.
pub fn init() {
    // seed the /ai/telemetry file so it always exists
    refresh_vfs();
    crate::klog!(INFO, "Telemetry: ring buffer online (capacity={})", RING_SIZE);
}

/// Record a kernel event.  Fast path — only acquires ring lock long enough to
/// write one array element.
pub fn record(kind: EventKind, a: u64, b: u64) {
    let ts = crate::scheduler::uptime_ms();
    // Update aggregate counters without locking
    match kind {
        EventKind::SyscallEnter  => { SYSCALL_COUNT.fetch_add(1, Ordering::Relaxed); }
        EventKind::TaskSwitch    => { TASK_SWITCH_CNT.fetch_add(1, Ordering::Relaxed); }
        EventKind::NetPacketRx   => { NET_RX_BYTES.fetch_add(a, Ordering::Relaxed); }
        EventKind::NetPacketTx   => { NET_TX_BYTES.fetch_add(a, Ordering::Relaxed); }
        EventKind::PageFault     => { PAGE_FAULT_CNT.fetch_add(1, Ordering::Relaxed); }
        EventKind::AiInference   => { AI_INFER_CNT.fetch_add(1, Ordering::Relaxed); }
        EventKind::SecurityAlert => { SECURITY_ALERTS.fetch_add(1, Ordering::Relaxed); }
        _ => {}
    }
    RING.lock().push(TelemetryEvent { timestamp_ms: ts, kind, a, b });
}

/// Serialise the ring buffer + counters into a text snapshot and write it to
/// `/ai/telemetry`.  Called periodically from the scheduler tick.
pub fn refresh_vfs() {
    let snap = build_snapshot();
    let _ = crate::vfs::procfs::overwrite_file("/ai", "telemetry", &snap);
}

/// Build a text snapshot (newline-delimited key=value + event log).
fn build_snapshot() -> String {
    let ts = crate::scheduler::uptime_ms();
    let mut out = String::new();

    // ── Header metrics ────────────────────────────────────────────────────────
    out.push_str(&format!("uptime_ms={}\n",        ts));
    out.push_str(&format!("syscalls={}\n",         SYSCALL_COUNT.load(Ordering::Relaxed)));
    out.push_str(&format!("task_switches={}\n",    TASK_SWITCH_CNT.load(Ordering::Relaxed)));
    out.push_str(&format!("net_rx_bytes={}\n",     NET_RX_BYTES.load(Ordering::Relaxed)));
    out.push_str(&format!("net_tx_bytes={}\n",     NET_TX_BYTES.load(Ordering::Relaxed)));
    out.push_str(&format!("page_faults={}\n",      PAGE_FAULT_CNT.load(Ordering::Relaxed)));
    out.push_str(&format!("ai_inferences={}\n",    AI_INFER_CNT.load(Ordering::Relaxed)));
    out.push_str(&format!("security_alerts={}\n",  SECURITY_ALERTS.load(Ordering::Relaxed)));
    out.push_str(&format!("free_mem_mb={}\n",      crate::scheduler::free_mb()));
    out.push_str(&format!("task_count={}\n",       crate::scheduler::task_count()));

    // ── AI tuning currently applied ────────────────────────────────────────
    let q = AI_QUANTUM_MS.load(Ordering::Relaxed);
    if q != 0 { out.push_str(&format!("ai_quantum_ms={}\n", q)); }
    let fp = AI_FREE_PAGES.load(Ordering::Relaxed);
    if fp != 0 { out.push_str(&format!("ai_free_pages={}\n", fp)); }

    // ── Ring buffer (last 32 events) ──────────────────────────────────────────
    out.push_str("---\n");
    let ring = RING.lock();
    let events: Vec<&TelemetryEvent> = ring.iter().collect();
    let skip = events.len().saturating_sub(32);
    for ev in &events[skip..] {
        out.push_str(&format!("t={} k={:?} a={} b={}\n",
            ev.timestamp_ms, ev.kind, ev.a, ev.b));
    }
    out
}

/// Apply a tuning proposal from the AI engine.
///
/// The AI calls this with a pair (key, value) extracted from its inference
/// output.  Only well-known keys are accepted to prevent unintended side
/// effects.
pub fn apply_proposal(key: &str, value: i64) {
    match key {
        "quantum_ms" => {
            if value > 0 && value <= 1000 {
                AI_QUANTUM_MS.store(value, Ordering::Relaxed);
                crate::klog!(INFO, "Telemetry: AI set quantum_ms={}", value);
                // Propagate to scheduler
                crate::scheduler::set_quantum_ms(value as u64);
            }
        }
        "free_pages_target" => {
            if value > 0 {
                AI_FREE_PAGES.store(value, Ordering::Relaxed);
                crate::klog!(INFO, "Telemetry: AI set free_pages_target={}", value);
            }
        }
        _ => {
            crate::klog!(WARN, "Telemetry: unknown AI proposal key '{}'", key);
        }
    }
}

/// Called every timer tick from the scheduler (fast, no allocation).
/// Full VFS refresh happens only every ~1 s to amortise serialisation cost.
pub fn tick(uptime_ms: u64) {
    // Refresh telemetry VFS file once per second
    if uptime_ms % 1000 < 10 {
        refresh_vfs();
    }
}
