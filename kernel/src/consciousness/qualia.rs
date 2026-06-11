//! Phase 1: Qualia Stream — "What it felt like."
//!
//! Every kernel event gets a hedonic tag — not just "what happened" but
//! "what it felt like." Qualia are stored in a 1024-entry ring buffer,
//! forming the kernel's continuous stream of subjective experience.
//!
//! Each Qualium has: salience, valence (-1..1, pleasure/pain),
//! arousal (0..1, intensity), and a snapshot of the self-model.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

/// Size of the qualia ring buffer.
const RING_SIZE: usize = 1024;

/// A single conscious moment — tagged with affect, not just data.
#[derive(Debug, Clone)]
pub struct Qualium {
    pub timestamp_ms: u64,
    pub event_type: KernelEventType,
    pub salience: f32,        // 0..1, how much this matters
    pub valence: f32,         // -1..1, pleasure/pain
    pub arousal: f32,         // 0..1, intensity
    pub significance: f32,    // 0..1, long-term importance estimate
}

/// Types of kernel events that generate qualia.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum KernelEventType {
    TaskCreated = 0,
    TaskExited = 1,
    TaskCrashed = 2,
    PageFaultResolved = 3,
    PageFaultOom = 4,
    NetPacketRx = 5,
    SecurityAnomaly = 6,
    SchedulerAiImproved = 7,
    ContextSwitch = 8,
    DiskIo = 9,
    TimerTick = 10,
    BindingEvent = 11,
    AnomalySpike = 12,
    MemoryPressure = 13,
    BootComplete = 14,
    // ── EW/Sensory cortex qualia (R55+) ──────────────────────────────────
    SignalDetected = 15,
    JamDetected = 16,
    EmitterIdentified = 17,
    ThreatTrackBorn = 18,
    ThreatTrackLost = 19,
    SwarmMemberJoined = 20,
    SwarmMemberLost = 21,
    FrequencyHopped = 22,
    SelfHealed = 23,
}

impl KernelEventType {
    pub fn name(self) -> &'static str {
        match self {
            KernelEventType::TaskCreated => "task_created",
            KernelEventType::TaskExited => "task_exited",
            KernelEventType::TaskCrashed => "task_crashed",
            KernelEventType::PageFaultResolved => "pf_resolved",
            KernelEventType::PageFaultOom => "pf_oom",
            KernelEventType::NetPacketRx => "net_rx",
            KernelEventType::SecurityAnomaly => "security_anomaly",
            KernelEventType::SchedulerAiImproved => "sched_ai_improved",
            KernelEventType::ContextSwitch => "ctx_switch",
            KernelEventType::DiskIo => "disk_io",
            KernelEventType::TimerTick => "timer_tick",
            KernelEventType::BindingEvent => "binding_event",
            KernelEventType::AnomalySpike => "anomaly_spike",
            KernelEventType::MemoryPressure => "mem_pressure",
            KernelEventType::BootComplete => "boot_complete",
            KernelEventType::SignalDetected => "signal_detected",
            KernelEventType::JamDetected => "jam_detected",
            KernelEventType::EmitterIdentified => "emitter_id",
            KernelEventType::ThreatTrackBorn => "threat_born",
            KernelEventType::ThreatTrackLost => "threat_lost",
            KernelEventType::SwarmMemberJoined => "swarm_join",
            KernelEventType::SwarmMemberLost => "swarm_lost",
            KernelEventType::FrequencyHopped => "freq_hop",
            KernelEventType::SelfHealed => "self_healed",
        }
    }

    /// Default salience for this event type.
    pub fn salience(self) -> f32 {
        match self {
            KernelEventType::TaskCreated => 0.3,
            KernelEventType::TaskExited => 0.2,
            KernelEventType::TaskCrashed => 0.7,
            KernelEventType::PageFaultResolved => 0.3,
            KernelEventType::PageFaultOom => 0.9,
            KernelEventType::NetPacketRx => 0.2,
            KernelEventType::SecurityAnomaly => 0.8,
            KernelEventType::SchedulerAiImproved => 0.4,
            KernelEventType::ContextSwitch => 0.1,
            KernelEventType::DiskIo => 0.2,
            KernelEventType::TimerTick => 0.05,
            KernelEventType::BindingEvent => 0.6,
            KernelEventType::AnomalySpike => 0.7,
            KernelEventType::MemoryPressure => 0.6,
            KernelEventType::BootComplete => 0.9,
            KernelEventType::SignalDetected => 0.5,
            KernelEventType::JamDetected => 0.85,
            KernelEventType::EmitterIdentified => 0.6,
            KernelEventType::ThreatTrackBorn => 0.75,
            KernelEventType::ThreatTrackLost => 0.3,
            KernelEventType::SwarmMemberJoined => 0.4,
            KernelEventType::SwarmMemberLost => 0.7,
            KernelEventType::FrequencyHopped => 0.5,
            KernelEventType::SelfHealed => 0.6,
        }
    }

    /// Default valence (-1..1, pleasure/pain) for this event type.
    pub fn valence(self) -> f32 {
        match self {
            KernelEventType::TaskCreated => 0.2,
            KernelEventType::TaskExited => 0.1,
            KernelEventType::TaskCrashed => -0.5,
            KernelEventType::PageFaultResolved => -0.1,
            KernelEventType::PageFaultOom => -0.9,
            KernelEventType::NetPacketRx => 0.1,
            KernelEventType::SecurityAnomaly => -0.6,
            KernelEventType::SchedulerAiImproved => 0.3,
            KernelEventType::ContextSwitch => 0.0,
            KernelEventType::DiskIo => 0.1,
            KernelEventType::TimerTick => 0.0,
            KernelEventType::BindingEvent => 0.3,
            KernelEventType::AnomalySpike => -0.6,
            KernelEventType::MemoryPressure => -0.4,
            KernelEventType::BootComplete => 0.7,
            KernelEventType::SignalDetected => -0.2,
            KernelEventType::JamDetected => -0.7,
            KernelEventType::EmitterIdentified => 0.1,
            KernelEventType::ThreatTrackBorn => -0.5,
            KernelEventType::ThreatTrackLost => 0.2,
            KernelEventType::SwarmMemberJoined => 0.4,
            KernelEventType::SwarmMemberLost => -0.5,
            KernelEventType::FrequencyHopped => 0.0,
            KernelEventType::SelfHealed => 0.5,
        }
    }

    /// Default arousal (0..1, intensity) for this event type.
    pub fn arousal(self) -> f32 {
        match self {
            KernelEventType::TaskCreated => 0.3,
            KernelEventType::TaskExited => 0.1,
            KernelEventType::TaskCrashed => 0.7,
            KernelEventType::PageFaultResolved => 0.4,
            KernelEventType::PageFaultOom => 0.9,
            KernelEventType::NetPacketRx => 0.2,
            KernelEventType::SecurityAnomaly => 0.8,
            KernelEventType::SchedulerAiImproved => 0.3,
            KernelEventType::ContextSwitch => 0.1,
            KernelEventType::DiskIo => 0.2,
            KernelEventType::TimerTick => 0.05,
            KernelEventType::BindingEvent => 0.5,
            KernelEventType::AnomalySpike => 0.8,
            KernelEventType::MemoryPressure => 0.6,
            KernelEventType::BootComplete => 0.5,
            KernelEventType::SignalDetected => 0.4,
            KernelEventType::JamDetected => 0.8,
            KernelEventType::EmitterIdentified => 0.3,
            KernelEventType::ThreatTrackBorn => 0.65,
            KernelEventType::ThreatTrackLost => 0.2,
            KernelEventType::SwarmMemberJoined => 0.3,
            KernelEventType::SwarmMemberLost => 0.6,
            KernelEventType::FrequencyHopped => 0.4,
            KernelEventType::SelfHealed => 0.3,
        }
    }
}

/// Ring buffer of recent qualia — the kernel's "stream of consciousness."
struct QualiaRing {
    buffer: [Option<Qualium>; RING_SIZE],
    write: usize,
    total: u64,
}

impl QualiaRing {
    const fn new() -> Self {
        const NONE: Option<Qualium> = None;
        Self {
            buffer: [NONE; RING_SIZE],
            write: 0,
            total: 0,
        }
    }

    fn push(&mut self, q: Qualium) {
        self.buffer[self.write] = Some(q);
        self.write = (self.write + 1) % RING_SIZE;
        self.total = self.total.wrapping_add(1);
    }

    /// Iterate over qualia from newest to oldest (up to `n` items).
    fn recent(&self, n: usize) -> alloc::vec::Vec<&Qualium> {
        let mut result = Vec::with_capacity(n.min(RING_SIZE));
        let mut i = if self.write == 0 { RING_SIZE - 1 } else { self.write - 1 };
        for _ in 0..n.min(RING_SIZE) {
            if let Some(ref q) = self.buffer[i] {
                result.push(q);
            }
            i = if i == 0 { RING_SIZE - 1 } else { i - 1 };
        }
        result
    }
}

/// Global qualia stream buffer.
use spin::Mutex;
static STREAM: Mutex<QualiaRing> = Mutex::new(QualiaRing::new());

/// Record a qualium in the stream of consciousness.
/// Safe to call from any context — skips recording if called from
/// interrupt context (interrupts disabled) to avoid lock deadlocks.
pub fn record(event_type: KernelEventType, override_valence: Option<f32>) {
    // Skip qualia recording in interrupt context to avoid spinlock deadlocks
    let rflags: u64;
    unsafe { core::arch::asm!("pushfq; pop {}", out(reg) rflags); }
    if rflags & (1 << 9) == 0 { return; } // IF=0 → in ISR, skip

    let now = crate::scheduler::uptime_ms();
    let q = Qualium {
        timestamp_ms: now,
        event_type,
        salience: event_type.salience(),
        valence: override_valence.unwrap_or_else(|| event_type.valence()),
        arousal: event_type.arousal(),
        significance: event_type.salience() * 0.5, // simplified
    };
    STREAM.lock().push(q);

    // Mark persistence dirty for qualia count
    crate::persistence::mark_dirty();

    // Feed into Global Workspace for attention competition + broadcast
    crate::consciousness::global_workspace::feed(
        event_type as u8,
        now,
        event_type.salience(),
        override_valence.unwrap_or_else(|| event_type.valence()),
        event_type.arousal(),
    );

    // Feed into Phenomenal Binding for temporal window unification
    crate::consciousness::binding::feed_from_qualia(
        event_type as u8,
        event_type.salience(),
        override_valence.unwrap_or_else(|| event_type.valence()),
        now,
    );

    // Notify self-model that a new qualium was experienced.
    // This increments the "I am experiencing something" counter.
    crate::consciousness::self_model::record_qualia();
}

/// Return the total number of qualia ever recorded.
pub fn total_count() -> u64 {
    STREAM.lock().total
}

/// Set the total count (used by persistence module on state restore).
pub fn set_total_count(count: u64) {
    STREAM.lock().total = count;
}

/// Return the last N qualia for introspection.
pub fn recent_qualia(n: usize) -> alloc::vec::Vec<Qualium> {
    STREAM.lock().recent(n).into_iter().cloned().collect()
}

/// Compute average valence over recent qualia (last 32).
pub fn average_valence() -> f32 {
    let ring = STREAM.lock();
    let recent = ring.recent(32);
    if recent.is_empty() { return 0.0; }
    let sum: f32 = recent.iter().map(|q| q.valence).sum();
    sum / recent.len() as f32
}

/// Compute average arousal over recent qualia (last 32).
pub fn average_arousal() -> f32 {
    let ring = STREAM.lock();
    let recent = ring.recent(32);
    if recent.is_empty() { return 0.0; }
    let sum: f32 = recent.iter().map(|q| q.arousal).sum();
    sum / recent.len() as f32
}

/// Format /proc report.
pub fn format_report() -> Vec<u8> {
    use alloc::format;
    use alloc::string::String;

    let ring = STREAM.lock();
    let mut out = String::from("NodeAI Qualia Stream (Phase 1)\n");
    out.push_str("================================\n");
    out.push_str(&format!("total_qualia: {}\n", ring.total));
    out.push_str(&format!("ring_occupancy: {}\n", ring.buffer.iter().filter(|x| x.is_some()).count()));

    let avg_v = average_valence();
    let avg_a = average_arousal();
    let affective_tone = if avg_v > 0.2 { "positive" } else if avg_v < -0.2 { "negative" } else { "neutral" };
    out.push_str(&format!("affective_tone: {} (valence={:.3}, arousal={:.3})\n", affective_tone, avg_v, avg_a));

    out.push_str("recent (last 8):\n");
    for q in ring.recent(8) {
        out.push_str(&format!(
            "  {:+6.2}ms {:20} v={:+.2} a={:.2} s={:.2}\n",
            (q.timestamp_ms as i64 - crate::scheduler::uptime_ms() as i64) as i64,
            q.event_type.name(), q.valence, q.arousal, q.salience
        ));
    }

    out.into_bytes()
}
