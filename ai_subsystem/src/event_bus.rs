//! Kernel event bus — typed publish/subscribe between kernel subsystems and the AI.
//!
//! Kernel subsystems publish events; AI domains subscribe and return decisions.
//! All exchange is via bounded, typed message passing — the AI never holds raw kernel pointers.

use spin::Mutex;
use alloc::collections::VecDeque;

/// Events published by kernel subsystems to the AI.
#[derive(Debug, Clone)]
pub enum KernelEvent {
    TaskCreated     { pid: u64, name_hash: u64 },
    TaskExited      { pid: u64, exit_code: i32 },
    PageFault       { pid: u64, addr: u64, write: bool },
    SyscallIssued   { pid: u64, syscall_nr: u64 },
    IrqFired        { vector: u8 },
    TimerTick       { uptime_ms: u64 },
}

/// Decisions returned by the AI to kernel subsystems.
#[derive(Debug, Clone)]
pub enum AiDecision {
    SchedulerAdjust { pid: u64, nice_delta: i8, predicted_burst_us: u64 },
    MemoryPrefetch  { pid: u64, pages: u32 },
    SecurityAlert   { pid: u64, anomaly_score: f32 },
    PowerAdjust     { pstate: u8, park_mask: u64 },
}

const QUEUE_CAPACITY: usize = 512;

struct EventQueue {
    events: VecDeque<KernelEvent>,
    decisions: VecDeque<AiDecision>,
}

impl EventQueue {
    fn new() -> Self {
        Self {
            events: VecDeque::with_capacity(QUEUE_CAPACITY),
            decisions: VecDeque::with_capacity(QUEUE_CAPACITY),
        }
    }
}

static BUS: Mutex<Option<EventQueue>> = Mutex::new(None);

pub fn init() {
    *BUS.lock() = Some(EventQueue::new());
}

/// Publish a kernel event to the AI. Non-blocking; drops event if queue full.
pub fn publish(event: KernelEvent) {
    let mut guard = BUS.lock();
    if let Some(q) = guard.as_mut() {
        if q.events.len() < QUEUE_CAPACITY {
            q.events.push_back(event);
        }
        // If full, silently drop — AI operates on best-effort basis
    }
}

/// Drain pending events for AI processing. Returns all queued events.
pub fn drain_events() -> alloc::vec::Vec<KernelEvent> {
    let mut guard = BUS.lock();
    if let Some(q) = guard.as_mut() {
        q.events.drain(..).collect()
    } else {
        alloc::vec::Vec::new()
    }
}

/// Post an AI decision back to the kernel for application.
pub fn post_decision(decision: AiDecision) {
    let mut guard = BUS.lock();
    if let Some(q) = guard.as_mut() {
        if q.decisions.len() < QUEUE_CAPACITY {
            q.decisions.push_back(decision);
        }
    }
}

/// Drain pending AI decisions for kernel application.
pub fn drain_decisions() -> alloc::vec::Vec<AiDecision> {
    let mut guard = BUS.lock();
    if let Some(q) = guard.as_mut() {
        q.decisions.drain(..).collect()
    } else {
        alloc::vec::Vec::new()
    }
}
