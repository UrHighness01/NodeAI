//! Scheduler — preemptive round-robin with AI-augmented priority (Phase 4).

pub mod task;
mod runqueue;
mod context_switch;

pub use task::{Task, TaskState, Pid};

use alloc::collections::BTreeMap;
use spin::Mutex;

/// Global task table: PID → Task.
static TASKS: Mutex<BTreeMap<Pid, Task>> = Mutex::new(BTreeMap::new());
static NEXT_PID: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(1);

/// Monotonic uptime counter — incremented once per `tick()` call (~1 ms).
static UPTIME_MS: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(0);

/// AI-requested scheduler time-quantum (ms). 0 means use the default 10 ms.
static QUANTUM_MS: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(0);

pub fn init() {
    runqueue::init();
    crate::klog!(INFO, "Scheduler: round-robin initialized");
}

/// Allocate a new PID.
pub fn alloc_pid() -> Pid {
    NEXT_PID.fetch_add(1, core::sync::atomic::Ordering::Relaxed)
}

/// Spawn a new kernel thread.
pub fn spawn_kernel_thread(name: &str, entry: fn() -> !) {
    // Allocate a 16 KiB kernel stack from the buddy allocator.
    const STACK_PAGES: usize = 4;
    let stack_phys = crate::memory::alloc_frames(2) // 2^2 = 4 pages
        .expect("out of memory for kernel stack");

    let stack_top = crate::memory::phys_offset()
        + stack_phys
        + (STACK_PAGES as u64 * crate::memory::PAGE_SIZE);

    let pid = alloc_pid();
    let task = Task::new_kernel_thread(pid, name, entry as u64, stack_top);

    crate::klog!(INFO, "Scheduler: spawning '{}' pid={} entry={:#x}", name, pid, entry as u64);

    TASKS.lock().insert(pid, task);
    runqueue::enqueue(pid);
}

/// Called from the APIC timer interrupt handler every ~10 ms (100 Hz).
/// Increments uptime by 10 so `uptime_ms()` always returns true milliseconds.
pub fn tick() {
    let uptime_ms = UPTIME_MS.fetch_add(10, core::sync::atomic::Ordering::Relaxed);

    // Drive the AI inference pipeline on every tick
    crate::ai_engine::process_tick(uptime_ms);

    // Refresh the graphical desktop if the framebuffer is active
    crate::desktop::tick(uptime_ms);

    // Update telemetry ring and (periodically) flush to VFS
    crate::telemetry::tick(uptime_ms);

    // Pump the audio DMA ring so ongoing PCM playback stays filled
    crate::audio::tick();

    let quantum = QUANTUM_MS.load(core::sync::atomic::Ordering::Relaxed);
    let _ = quantum; // used by runqueue when it is extended
    if let Some(_next_pid) = runqueue::tick() {
        crate::klog!(TRACE, "Scheduler: preemption tick → pid={}", _next_pid);
    }
}

/// Return the current kernel uptime in milliseconds.
pub fn uptime_ms() -> u64 {
    UPTIME_MS.load(core::sync::atomic::Ordering::Relaxed)
}

/// Return the number of free megabytes of RAM.
pub fn free_mb() -> u64 {
    crate::memory::free_mb()
}

/// Apply an AI-proposed scheduler quantum.  0 = reset to default.
pub fn set_quantum_ms(ms: u64) {
    QUANTUM_MS.store(ms, core::sync::atomic::Ordering::Relaxed);
    crate::klog!(INFO, "Scheduler: quantum set to {} ms", ms);
}

/// Voluntarily yield the CPU to the next runnable task.
pub fn yield_cpu() {
    x86_64::instructions::interrupts::disable();
    let _ = runqueue::dequeue_next();
    x86_64::instructions::interrupts::enable();
}

/// Return the PID of the currently running task, or 1 (init) as default.
pub fn current_pid() -> Pid {
    runqueue::current_pid().unwrap_or(1)
}

/// Return the number of living tasks in the task table.
pub fn task_count() -> usize {
    TASKS.lock().len()
}

/// Mark the current task as a zombie and yield the CPU.
/// This is called from `sys_exit`; control does not return here.
pub fn exit_current(code: i32) -> ! {
    crate::klog!(INFO, "Scheduler: exit_current(code={}) — halting task", code);
    // Mark task zombie in the task table if we can identify it.
    if let Some(pid) = runqueue::current_pid() {
        let mut tasks = TASKS.lock();
        if let Some(task) = tasks.get_mut(&pid) {
            task.state     = crate::scheduler::task::TaskState::Zombie;
            task.exit_code = Some(code);
        }
    }
    loop {
        x86_64::instructions::hlt();
    }
}

/// Fork: create a shallow clone of `parent_pid` and return the new child PID.
/// Full CoW page-table cloning is deferred to Phase 21; for now both tasks
/// share the kernel CR3 (sufficient for in-kernel fork/exec patterns).
pub fn fork_task(parent_pid: Pid) -> Option<Pid> {
    let child_pid = alloc_pid();
    let mut tasks = TASKS.lock();
    let parent = tasks.get(&parent_pid)?.clone_shallow(child_pid)?;
    tasks.insert(child_pid, parent);
    runqueue::enqueue(child_pid);
    crate::klog!(INFO, "Scheduler: fork parent={} → child={}", parent_pid, child_pid);
    Some(child_pid)
}

/// Wait for any zombie child of `parent_pid`.  Returns (child_pid, exit_code).
/// Spins (without yielding CPU) until a zombie child is found or timeout.
pub fn wait_for_child(parent_pid: Pid) -> Option<(Pid, i32)> {
    // Up to 10 seconds of spinning at 100 Hz ticks (1 000 000 iterations).
    for _ in 0..1_000_000u32 {
        let result = {
            let mut tasks = TASKS.lock();
            let child_entry = tasks
                .iter()
                .find(|(_, t)| t.parent_pid == parent_pid
                    && t.state == crate::scheduler::task::TaskState::Zombie)
                .map(|(&pid, t)| (pid, t.exit_code.unwrap_or(0)));
            if let Some((cpid, code)) = child_entry {
                tasks.remove(&cpid);
                Some((cpid, code))
            } else {
                None
            }
        };
        if result.is_some() {
            return result;
        }
        // Light pause to avoid hammering the lock
        for _ in 0..1000u32 {
            core::hint::spin_loop();
        }
    }
    None
}

/// Set the user-space program break for `pid`.
pub fn set_user_brk(pid: Pid, brk: u64) {
    let mut tasks = TASKS.lock();
    if let Some(t) = tasks.get_mut(&pid) {
        t.user_brk = brk;
    }
}

/// Get the user-space program break for `pid`.
pub fn get_user_brk(pid: Pid) -> u64 {
    TASKS.lock().get(&pid).map(|t| t.user_brk).unwrap_or(0)
}

/// Get (uid, gid, euid, egid) for a task.
pub fn get_credentials(pid: Pid) -> (u32, u32, u32, u32) {
    TASKS.lock()
        .get(&pid)
        .map(|t| (t.uid, t.gid, t.euid, t.egid))
        .unwrap_or((0, 0, 0, 0))
}

/// Set the FS base (TLS pointer) for a task.
pub fn set_fs_base(pid: Pid, base: u64) {
    let mut tasks = TASKS.lock();
    if let Some(t) = tasks.get_mut(&pid) {
        t.fs_base = base;
    }
}

/// Get the FS base (TLS pointer) for a task.
pub fn get_fs_base(pid: Pid) -> u64 {
    TASKS.lock().get(&pid).map(|t| t.fs_base).unwrap_or(0)
}

/// Get the parent PID of a task (0 = no parent).
pub fn get_parent_pid(pid: Pid) -> Pid {
    TASKS.lock().get(&pid).map(|t| t.parent_pid).unwrap_or(0)
}

/// Force-kill a task (mark zombie with given code).
pub fn kill_task(pid: Pid, code: i32) {
    let mut tasks = TASKS.lock();
    if let Some(t) = tasks.get_mut(&pid) {
        t.state     = crate::scheduler::task::TaskState::Zombie;
        t.exit_code = Some(code);
    }
}

/// Spawn a user-space thread (POSIX thread / CLONE_THREAD).
/// Creates a new task with shared address space (CR3) but independent stack.
/// The child task starts with the same RIP as the parent (SYSCALL return point)
/// but with RSP = `new_stack` and RAX = 0 (child return value from clone).
pub fn spawn_user_thread(parent_pid: Pid, new_stack: u64, tls: u64, settls: bool) -> Option<Pid> {
    let child_pid = alloc_pid();
    let mut tasks = TASKS.lock();
    let mut child = tasks.get(&parent_pid)?.clone_shallow(child_pid)?;
    // Override stack and return value for thread semantics
    child.context.rsp = new_stack;
    child.context.rax = 0; // child sees 0 as return from clone
    if settls {
        child.fs_base = tls;
    }
    // Thread shares parent tgid (use parent_pid as thread-group leader)
    child.parent_pid = parent_pid;
    tasks.insert(child_pid, child);
    runqueue::enqueue(child_pid);
    crate::klog!(INFO, "Scheduler: thread spawn parent={} → tid={}", parent_pid, child_pid);
    Some(child_pid)
}

/// Set a signal handler for the given signal number.
pub fn set_signal_handler(pid: Pid, signum: usize, handler: u64) {
    let mut tasks = TASKS.lock();
    if let Some(t) = tasks.get_mut(&pid) {
        if signum < 64 {
            t.signal_handlers[signum] = handler;
        }
    }
}

/// Return total RAM in 4 KiB pages.
pub fn total_ram_pages() -> u64 {
    crate::memory::total_ram_pages()
}

// ── Phase 29 additions ────────────────────────────────────────────────────────

/// Approximate CPU utilisation as a percentage (0-100).
/// Calculated from the ratio of idle ticks to total ticks in the last window.
pub fn cpu_usage_pct() -> u8 {
    let tasks = TASKS.lock();
    let running = tasks.values()
        .filter(|t| t.state == crate::scheduler::task::TaskState::Running ||
                    t.state == crate::scheduler::task::TaskState::Runnable)
        .count();
    // Heuristic: >4 ready tasks → high load.
    ((running * 25).min(100)) as u8
}

/// Number of user-space (non-kernel) processes.
pub fn user_process_count() -> usize {
    // Kernel threads have names starting with '['.
    TASKS.lock().values()
        .filter(|t| !t.name.starts_with('['))
        .count()
}

/// Return a snapshot of all active PIDs.
pub fn all_pids() -> alloc::vec::Vec<Pid> {
    TASKS.lock().keys().copied().collect()
}

/// Return the name of a task, or None if it does not exist.
pub fn task_name(pid: Pid) -> Option<alloc::string::String> {
    TASKS.lock().get(&pid).map(|t| t.name.clone())
}

/// Return an estimate of memory used by a task in bytes (user_brk as a proxy).
pub fn task_mem_bytes(pid: Pid) -> u64 {
    TASKS.lock().get(&pid).map(|t| t.user_brk).unwrap_or(0)
}
