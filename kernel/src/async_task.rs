//! Async Task Queue — non-blocking background processing for MHS and future engines.
//!
//! Enables the kernel to run long computations (neural inference, complex analysis)
//! without freezing the shell. Tasks are enqueued, processed one step per tick
//! (cooperative, not preemptive — single-threaded kernel), and results are
//! retrieved on demand.
//!
//! This is a general-purpose async task system — usable for anything from MHS
//! inference to sensor data analysis to swarm communication.

use alloc::vec::Vec;
use alloc::string::String;
use alloc::format;
use spin::Mutex;
use core::sync::atomic::{AtomicU64, Ordering};

/// Maximum number of async tasks in the queue.
const MAX_TASKS: usize = 16;

/// Maximum result length stored.
const MAX_RESULT_LEN: usize = 512;

/// State of an async task.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskState {
    /// Queued and waiting to start.
    Pending,
    /// Currently being processed incrementally.
    Running,
    /// Completed successfully. Result available.
    Completed,
    /// Failed or produced empty result.
    Failed,
}

impl TaskState {
    pub fn describe(&self) -> &'static str {
        match self {
            TaskState::Pending => "pending",
            TaskState::Running => "running",
            TaskState::Completed => "completed",
            TaskState::Failed => "failed",
        }
    }
}

/// A single async task in the queue.
#[derive(Debug, Clone)]
pub struct AsyncTask {
    /// Monotonically increasing task ID.
    pub id: u64,
    /// Original user query.
    pub query: String,
    /// Current state.
    pub state: TaskState,
    /// Completed result (when state == Completed).
    pub result: String,
    /// Whether the user has been notified about this result.
    pub notified: bool,
    /// Task type tag.
    pub tag: &'static str,
}

/// Internal task queue state.
struct TaskQueueState {
    /// Ring buffer of tasks.
    tasks: Vec<AsyncTask>,
    /// Next task ID.
    next_id: u64,
    /// Total tasks ever enqueued.
    total_enqueued: u64,
    /// Total tasks completed.
    total_completed: u64,
    /// Whether MHS is currently running a step.
    mhs_running: bool,
}

static QUEUE: Mutex<Option<TaskQueueState>> = Mutex::new(None);

/// Initialize the async task system.
pub fn init() {
    *QUEUE.lock() = Some(TaskQueueState {
        tasks: Vec::with_capacity(MAX_TASKS),
        next_id: 1,
        total_enqueued: 0,
        total_completed: 0,
        mhs_running: false,
    });
    crate::klog!(INFO, "async_task: background task queue initialized (max={})", MAX_TASKS);
}

/// Tick the async task system — called every 100ms from idle_loop.
/// If there's a pending task, advance MHS generation by one step.
/// NOTE: MHS inference is DISABLED for now — the 6-layer GLA forward pass
/// causes heap fragmentation and #PF crashes in the constrained kernel heap.
/// The queue remains functional for future use with Project-K nano model.
pub fn tick() {
    let mut lock = QUEUE.lock();
    let q = match &mut *lock {
        Some(q) => q,
        None => return,
    };

    // MHS inference DISABLED (heap fragmentation cause #PF crashes).
    // All pending tasks are immediately marked as completed with a note.
    for i in 0..q.tasks.len() {
        if q.tasks[i].state == TaskState::Pending {
            q.tasks[i].state = TaskState::Completed;
            q.total_completed = q.total_completed.saturating_add(1);
            q.tasks[i].result = String::from(
                "(MHS disabled - use templates for instant response. Will be enabled when nano model is ready.)"
            );
        }
    }
    q.mhs_running = false;
}

/// Enqueue a new async task.
/// Returns the task ID, or None if queue is full.
pub fn enqueue(query: &str, tag: &'static str) -> Option<u64> {
    let mut lock = QUEUE.lock();
    let q = match &mut *lock {
        Some(q) => q,
        None => return None,
    };

    if q.tasks.len() >= MAX_TASKS {
        // Remove oldest completed/failed task to make room
        q.tasks.retain(|t| t.state == TaskState::Pending || t.state == TaskState::Running);
        if q.tasks.len() >= MAX_TASKS {
            return None; // Still full
        }
    }

    let id = q.next_id;
    q.next_id = q.next_id.saturating_add(1);
    q.total_enqueued = q.total_enqueued.saturating_add(1);

    q.tasks.push(AsyncTask {
        id,
        query: String::from(query),
        state: TaskState::Pending,
        result: String::new(),
        notified: false,
        tag,
    });

    Some(id)
}

/// Get all completed tasks that haven't been notified yet.
pub fn get_new_results() -> Vec<(u64, String, String)> {
    let mut lock = QUEUE.lock();
    let q = match &mut *lock {
        Some(q) => q,
        None => return Vec::new(),
    };

    let mut results = Vec::new();
    for task in &mut q.tasks {
        if task.state == TaskState::Completed && !task.notified {
            task.notified = true;
            results.push((task.id, task.query.clone(), task.result.clone()));
        }
    }
    results
}

/// Get all tasks in the queue (for display).
pub fn get_all_tasks() -> Vec<(u64, String, TaskState, bool)> {
    let lock = QUEUE.lock();
    let q = match &*lock {
        Some(q) => q,
        None => return Vec::new(),
    };
    q.tasks.iter().map(|t| (t.id, t.query.clone(), t.state, t.notified)).collect()
}

/// Get queue statistics.
pub fn stats() -> (u64, u64, usize, bool) {
    let lock = QUEUE.lock();
    let q = match &*lock {
        Some(q) => q,
        None => return (0, 0, 0, false),
    };
    (q.total_enqueued, q.total_completed, q.tasks.len(), q.mhs_running)
}

/// Format /proc/async_tasks report.
pub fn format_report() -> Vec<u8> {
    let (enqueued, completed, queued_len, running) = stats();
    let tasks = get_all_tasks();
    let mut s = format!(
        "Async Task Queue\n\
         ===============\n\
         total enqueued:  {}\n\
         total completed: {}\n\
         in queue:        {}\n\
         running:         {}\n\
         \n\
         Tasks:\n",
        enqueued, completed, queued_len, if running { "yes (MHS active)" } else { "no" },
    );

    if tasks.is_empty() {
        s.push_str("  (none)\n");
    } else {
        for (id, query, state, notified) in &tasks {
            let q_trunc: String = query.chars().take(30).collect();
            s.push_str(&format!(
                "  [#{}] {} — {}{}\n",
                id, q_trunc, state.describe(),
                if *notified { " (notified)" } else { "" },
            ));
        }
    }
    s.into_bytes()
}
