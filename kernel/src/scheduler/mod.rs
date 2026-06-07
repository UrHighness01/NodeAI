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

/// Called from the naked APIC timer handler with interrupts disabled.
///
/// 1. Saves `old_rsp` (bottom of the saved interrupt frame on the current task's
///    kernel stack) into the current task.
/// 2. Runs per-tick subsystem work.
/// 3. Advances the run queue.
/// 4. If a switch is needed: loads new task's CR3, updates TSS.RSP0, returns
///    new task's `saved_kernel_rsp` so the naked handler can switch stacks.
///    If no switch: returns `old_rsp` unchanged.
///
/// # Safety
/// Must be called with interrupts disabled and `old_rsp` pointing at a valid
/// 160-byte saved interrupt frame on the current task's kernel stack.
#[no_mangle]
pub unsafe extern "C" fn schedule_from_interrupt(old_rsp: u64) -> u64 {
    // EOI first — acknowledges the timer before we do any work.
    crate::interrupts::apic::eoi();

    let uptime_ms = UPTIME_MS.fetch_add(10, core::sync::atomic::Ordering::Relaxed);

    // Step 1: save old_rsp into the CURRENT (outgoing) task.
    let old_pid = runqueue::current_pid();
    if let Some(pid) = old_pid {
        let mut tasks = TASKS.lock();
        if let Some(task) = tasks.get_mut(&pid) {
            task.saved_kernel_rsp = old_rsp;
            if task.state == task::TaskState::Running {
                task.state = task::TaskState::Runnable;
            }
        }
    }

    // Step 2: per-tick subsystem work.
    crate::ai_engine::process_tick(uptime_ms);
    crate::desktop::tick(uptime_ms);
    crate::telemetry::tick(uptime_ms);
    crate::audio::tick();

    // Step 3: advance the run queue.
    let next_pid = match runqueue::tick() {
        Some(pid) => pid,
        None      => return old_rsp, // same task, no switch
    };

    // Step 4: set up the incoming task and return its kernel RSP.
    crate::klog!(TRACE, "Scheduler: {:?} → {}", old_pid, next_pid);

    // Log BEFORE loading new CR3 — after the load, old stack may be unmapped
    // (matters once per-process page tables are active).
    let next_rsp = {
        let mut tasks = TASKS.lock();
        match tasks.get_mut(&next_pid) {
            Some(t) => {
                t.state = task::TaskState::Running;
                crate::gdt::update_rsp0(t.kernel_stack_top);
                let cr3    = t.cr3;
                let fs     = t.fs_base;
                let rsp    = t.saved_kernel_rsp;
                // Load new page table last — after all stack accesses.
                core::arch::asm!("mov cr3, {}", in(reg) cr3, options(nostack, nomem));
                if fs != 0 {
                    x86_64::registers::model_specific::FsBase::write(
                        x86_64::VirtAddr::new(fs));
                }
                rsp
            }
            None => old_rsp,
        }
    };
    next_rsp
}

/// Legacy tick() — kept for call sites not yet migrated.
/// Real work is in schedule_from_interrupt (called by naked timer handler).
pub fn tick() {}

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

/// Put the current task to sleep (remove from run queue).
/// The task will not be scheduled again until `wake_pid` is called.
/// Caller must yield immediately after to hand off the CPU.
pub fn sleep_current() {
    if let Some(pid) = runqueue::current_pid() {
        let mut tasks = TASKS.lock();
        if let Some(t) = tasks.get_mut(&pid) {
            t.state = crate::scheduler::task::TaskState::Sleeping;
        }
        runqueue::remove(pid);
    }
}

/// Wake a sleeping task — mark Runnable and re-add to run queue.
/// No-op if the task is not sleeping.
pub fn wake_pid(pid: Pid) {
    let mut tasks = TASKS.lock();
    if let Some(t) = tasks.get_mut(&pid) {
        if t.state == crate::scheduler::task::TaskState::Sleeping {
            t.state = crate::scheduler::task::TaskState::Runnable;
            drop(tasks);
            runqueue::enqueue(pid);
        }
    }
}

/// Return the PID of the currently running task, or 1 (init) as default.
pub fn current_pid() -> Pid {
    runqueue::current_pid().unwrap_or(1)
}

/// Return the number of living tasks in the task table.
pub fn task_count() -> usize {
    TASKS.lock().len()
}

/// Terminate task `pid` with `code` — marks zombie, wakes parent, removes from queue.
/// If `pid` is the current task, halts; otherwise returns normally.
pub fn exit_current_direct(pid: Pid, code: i32) -> ! {
    crate::klog!(INFO, "Scheduler: exit pid={} code={}", pid, code);
    let parent_pid = {
        let mut tasks = TASKS.lock();
        let ppid = tasks.get(&pid).map(|t| t.parent_pid).unwrap_or(0);
        if let Some(task) = tasks.get_mut(&pid) {
            task.state     = task::TaskState::Zombie;
            task.exit_code = Some(code);
        }
        drop(tasks);
        runqueue::remove(pid);
        ppid
    };
    if parent_pid != 0 {
        wake_pid(parent_pid);
        send_signal(parent_pid, 17); // SIGCHLD
    }
    loop { x86_64::instructions::hlt(); }
}

/// Mark the current task as a zombie, wake a waiting parent, and halt.
/// Called from `sys_exit`; never returns.
pub fn exit_current(code: i32) -> ! {
    crate::klog!(INFO, "Scheduler: exit_current(code={})", code);
    let parent_pid = if let Some(pid) = runqueue::current_pid() {
        let mut tasks = TASKS.lock();
        let ppid = tasks.get(&pid).map(|t| t.parent_pid).unwrap_or(0);
        if let Some(task) = tasks.get_mut(&pid) {
            task.state     = crate::scheduler::task::TaskState::Zombie;
            task.exit_code = Some(code);
        }
        drop(tasks);
        runqueue::remove(pid);
        ppid
    } else {
        0
    };

    // Wake the parent and send SIGCHLD.
    if parent_pid != 0 {
        wake_pid(parent_pid);
        send_signal(parent_pid, 17); // SIGCHLD
    }

    loop { x86_64::instructions::hlt(); }
}

/// Find and remove one zombie child of `parent_pid`.
/// Returns `(child_pid, exit_code)` if found, `None` otherwise.
pub fn reap_zombie_child(parent_pid: Pid) -> Option<(Pid, i32)> {
    let mut tasks = TASKS.lock();
    let entry = tasks
        .iter()
        .find(|(_, t)| t.parent_pid == parent_pid
            && t.state == crate::scheduler::task::TaskState::Zombie)
        .map(|(&pid, t)| (pid, t.exit_code.unwrap_or(0)));
    if let Some((cpid, _)) = entry {
        tasks.remove(&cpid);
    }
    entry
}

/// Fork: create a clone of `parent_pid` with its own page table (kernel half shared,
/// user half has a fresh L4 — full CoW copy deferred to Phase 21).
pub fn fork_task(parent_pid: Pid) -> Option<Pid> {
    let child_pid = alloc_pid();

    // Allocate a new L4 for the child (kernel half copied from parent).
    let child_cr3 = crate::memory::alloc_user_cr3().unwrap_or_else(|| {
        // Fall back to sharing kernel CR3 if OOM (safer than crash).
        let v: u64;
        unsafe { core::arch::asm!("mov {}, cr3", out(reg) v, options(nomem, nostack)); }
        v & !0xFFF
    });

    let mut tasks = TASKS.lock();
    let mut child = tasks.get(&parent_pid)?.clone_shallow(child_pid)?;
    child.cr3 = child_cr3;
    tasks.insert(child_pid, child);
    runqueue::enqueue(child_pid);
    crate::klog!(INFO, "Scheduler: fork parent={} → child={} cr3={:#x}",
        parent_pid, child_pid, child_cr3);
    Some(child_pid)
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

/// Update the CR3 (page table) for a task — called by execve when it creates a new address space.
pub fn set_task_cr3(pid: Pid, cr3: u64) {
    let mut tasks = TASKS.lock();
    if let Some(t) = tasks.get_mut(&pid) { t.cr3 = cr3; }
}

/// Apply an AI-suggested priority adjustment to a task (clamped to ±20).
pub fn adjust_priority(pid: Pid, delta: i8) {
    let mut tasks = TASKS.lock();
    if let Some(t) = tasks.get_mut(&pid) {
        t.priority = (t.priority + delta as i32).clamp(-20, 20);
    }
}

/// Force-kill a task by sending SIGKILL (default action = terminate).
pub fn kill_task(pid: Pid, _code: i32) {
    send_signal(pid, 9); // SIGKILL
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

/// Send a signal to a task: set the pending bit, wake if sleeping.
pub fn send_signal(pid: Pid, signum: u8) {
    if signum as usize >= 64 { return; }
    let should_wake = {
        let mut tasks = TASKS.lock();
        if let Some(t) = tasks.get_mut(&pid) {
            t.pending_signals |= 1u64 << signum;
            t.state == task::TaskState::Sleeping
        } else {
            false
        }
    };
    if should_wake { wake_pid(pid); }
}

/// Take the highest-priority pending unmasked signal for `pid`.
/// Returns (signum, handler_va) where handler_va=0 means default action.
pub fn take_pending_signal(pid: Pid) -> Option<(u8, u64)> {
    let mut tasks = TASKS.lock();
    let t = tasks.get_mut(&pid)?;
    let deliverable = t.pending_signals & !t.signal_mask;
    if deliverable == 0 { return None; }
    let signum = deliverable.trailing_zeros() as u8;
    t.pending_signals &= !(1u64 << signum);
    let handler = t.signal_handlers[signum as usize];
    Some((signum, handler))
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
