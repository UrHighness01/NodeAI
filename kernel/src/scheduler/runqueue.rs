//! Run queue — round-robin with AI priority hints.
//!
//! Stores tasks directly in a heap-allocated Vec.  Phase 5 will split this
//! into per-CPU queues once SMP is established.

use alloc::collections::VecDeque;
use alloc::boxed::Box;
use spin::Mutex;
use super::task::{Pid, Task, TaskState};

// ── Storage ──────────────────────────────────────────────────────────────────
// We store all tasks here; the runqueue holds PIDs.

static RUNQUEUE: Mutex<RunQueue> = Mutex::new(RunQueue::new());

struct RunQueue {
    queue:   VecDeque<Pid>,
    // Simple time-quota per task in ticks before forced preemption.
    timeslice: u32,
    ticks_left: u32,
    current_pid: Option<Pid>,
}

impl RunQueue {
    const DEFAULT_TIMESLICE: u32 = 10; // 10 ms ticks before preemption

    const fn new() -> Self {
        RunQueue {
            queue:       VecDeque::new(),
            timeslice:   Self::DEFAULT_TIMESLICE,
            ticks_left:  Self::DEFAULT_TIMESLICE,
            current_pid: None,
        }
    }

    fn enqueue(&mut self, pid: Pid) {
        self.queue.push_back(pid);
    }

    fn current(&self) -> Option<Pid> {
        self.current_pid
    }

    /// Called on each timer tick. Returns the next PID to run if a switch is
    /// needed, `None` if the current task keeps running.
    fn tick(&mut self) -> Option<Pid> {
        self.ticks_left = self.ticks_left.saturating_sub(1);
        if self.ticks_left == 0 {
            return self.schedule_next();
        }
        None
    }

    /// Select the next task from the queue (round-robin).
    fn schedule_next(&mut self) -> Option<Pid> {
        // Put current task back at the tail if it exists
        if let Some(cur) = self.current_pid.take() {
            self.queue.push_back(cur);
        }
        self.current_pid = self.queue.pop_front();
        self.ticks_left = self.timeslice;
        self.current_pid
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

pub fn init() {
    // Queue starts empty; the idle task is a special case handled by main.
}

pub fn enqueue(pid: Pid) {
    RUNQUEUE.lock().enqueue(pid);
}

pub fn dequeue_next() -> Option<Pid> {
    RUNQUEUE.lock().schedule_next()
}

pub fn current_pid() -> Option<Pid> {
    RUNQUEUE.lock().current()
}

/// Remove a PID from the queue (e.g. when putting a task to sleep).
pub fn remove(pid: Pid) {
    let mut rq = RUNQUEUE.lock();
    rq.queue.retain(|&p| p != pid);
    if rq.current_pid == Some(pid) {
        rq.current_pid = None;
    }
}

/// Called from the APIC timer interrupt.
/// Returns the next `Pid` to switch to if preemption is needed.
pub fn tick() -> Option<Pid> {
    RUNQUEUE.lock().tick()
}

