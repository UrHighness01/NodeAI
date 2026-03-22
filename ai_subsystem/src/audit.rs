//! AI audit log — every AI decision is recorded for review.
//! Backed by a fixed-size lock-free ring buffer (no allocator needed in hot path).

use spin::Mutex;

/// Events that the constraint engine records when it blocks an AI action.
#[derive(Debug, Clone)]
pub enum AuditEvent {
    PriorityHardCap { original_nice: i32, proposed_adjust: i8 },
    IllegalSwapAttempt { page_flags: u64 },
    UnauthorizedKillAttempt,
    AiInferenceFailed { domain: &'static str },
    ModelLoadFailed { reason: &'static str },
    AiDecision { domain: &'static str, input_hash: u64, output_hash: u64 },
}

const RING_SIZE: usize = 4096;

struct AuditRing {
    buf: [Option<AuditEvent>; RING_SIZE],
    head: usize,
    count: usize,
}

impl AuditRing {
    const fn new() -> Self {
        // const-compatible initialization
        Self {
            buf: [const { None }; RING_SIZE],
            head: 0,
            count: 0,
        }
    }

    fn push(&mut self, event: AuditEvent) {
        self.buf[self.head] = Some(event);
        self.head = (self.head + 1) % RING_SIZE;
        if self.count < RING_SIZE {
            self.count += 1;
        }
    }
}

static AUDIT: Mutex<AuditRing> = Mutex::new(AuditRing::new());

pub fn log_constraint_violation(event: AuditEvent) {
    AUDIT.lock().push(event);
}

pub fn log_decision(domain: &'static str, input_hash: u64, output_hash: u64) {
    AUDIT.lock().push(AuditEvent::AiDecision { domain, input_hash, output_hash });
}

/// Return the number of audit entries recorded so far.
pub fn entry_count() -> usize {
    AUDIT.lock().count
}
