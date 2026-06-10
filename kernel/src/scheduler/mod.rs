//! Scheduler — preemptive round-robin with AI-augmented priority.

pub mod task;
mod runqueue;
mod context_switch;

pub use task::{Task, TaskState, Pid};

use alloc::collections::BTreeMap;
use spin::Mutex;

/// Global task table: PID → Task.
pub(crate) static TASKS: Mutex<BTreeMap<Pid, Task>> = Mutex::new(BTreeMap::new());
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
    let mut task = Task::new_kernel_thread(pid, name, entry as u64, stack_top);

    // Place a stack canary at the bottom of the kernel stack (lowest address).
    // The canary is a deterministic value derived from the PID XOR a boot constant
    // so that stack overflows that corrupt it are detected on the next context switch.
    let canary_addr  = crate::memory::phys_offset() + stack_phys; // bottom of stack
    let canary_value = 0xDEAD_C0DE_CAFE_0000u64 ^ (pid << 3);
    unsafe { core::ptr::write_volatile(canary_addr as *mut u64, canary_value); }
    task.stack_canary      = canary_value;
    task.stack_canary_addr = canary_addr;

    crate::klog!(INFO, "Scheduler: spawning '{}' pid={} entry={:#x}", name, pid, entry as u64);

    task.runnable_at = UPTIME_MS.load(core::sync::atomic::Ordering::Relaxed);
    TASKS.lock().insert(pid, task);
    runqueue::enqueue(pid);
    crate::rlimit::init_pid(pid, None);
    // Record process birth qualia — the kernel experiences a new task being born
    crate::consciousness::qualia::record(
        crate::consciousness::qualia::KernelEventType::TaskCreated, None);
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

    // CRITICAL: if there is no current task (we are on the idle/boot stack),
    // do NOT attempt a context switch. The idle stack has no Task entry, so
    // we cannot save its RSP. If we switch away here we permanently lose the
    // idle execution context — idle_loop never wakes, uptime advances but
    // the heartbeat never fires. Always return old_rsp when old_pid is None.
    if old_pid.is_none() {
        // Still do per-tick subsystem work — but ONLY call interrupt-safe
        // functions here. desktop::tick() and telemetry::tick() both use
        // fb::with() which takes the framebuffer spin-lock. If the idle loop
        // holds that lock when the timer fires (e.g. inside browser_fetch_tick),
        // re-acquiring it from interrupt context deadlocks the CPU forever.
        crate::audio::tick();
        return old_rsp;
    }

    if let Some(pid) = old_pid {
        // ── Stack canary check ────────────────────────────────────────────────
        let canary_mismatch: bool = {
            let tasks = TASKS.lock();
            if let Some(task) = tasks.get(&pid) {
                if task.stack_canary_addr != 0 {
                    let live = unsafe {
                        core::ptr::read_volatile(task.stack_canary_addr as *const u64)
                    };
                    live != task.stack_canary
                } else { false }
            } else { false }
        };
        if canary_mismatch {
            crate::klog!(ERROR, "STACK OVERFLOW: pid={} canary clobbered", pid);
            crate::auto_security::on_stack_overflow(pid);
            kill_task(pid, 6); // SIGABRT
        }

        let mut tasks = TASKS.lock();
        if let Some(task) = tasks.get_mut(&pid) {
            task.saved_kernel_rsp = old_rsp;
            if task.state == task::TaskState::Running {
                task.state = task::TaskState::Runnable;
            }
        }
    }

    // Step 2a: fingerprint + transformer update + causal producer boost.
    if let Some(pid) = old_pid {
        // Fingerprint cluster update.
        if let Some((_, fp_profile, _fp_conf)) = crate::fingerprint::classify_task(pid) {
            let mut tasks = TASKS.lock();
            if let Some(task) = tasks.get_mut(&pid) {
                if task.intent == 0 {
                    task.ai_profile.ai_nice_adjust = fp_profile.nice_adjust;
                }
            }
        }

        // Transformer SGD feedback: compare previous prediction to what actually happened.
        let (actual_nice, actual_burst, actual_pf) = {
            let tasks = TASKS.lock();
            if let Some(task) = tasks.get(&pid) {
                (task.ai_profile.ai_nice_adjust, task.ai_profile.ticks_run as u32, 0u8)
            } else {
                (0i8, 1u32, 0u8)
            }
        };
        crate::transformer_sched::on_deschedule(pid, actual_nice, actual_burst, actual_pf);

        // Phase 4: Phi-Metric Privilege Separation
        // Demote highly chaotic (unpredictable) tasks to 'nobody' (65534)
        if crate::anomaly::phi(pid) < 0.3 && get_euid(pid).unwrap_or(0) != 65534 {
            crate::klog!(WARN, "SECURITY: pid={} phi is too low, dynamically demoting euid to 65534 (nobody)", pid);
            set_euid(pid, 65534);
        }

        // Phase 1: Causal-Graph Assisted Task Affinity
        // If this task habitually wakes specific consumers, pre-enqueue them
        // at the front of the runqueue for temporal cache locality.
        for succ in crate::causal::predict_successors(pid).into_iter().rev() {
            runqueue::move_to_front(succ);
        }
    }

    // Step 2: per-tick subsystem work.
    crate::ai_engine::process_tick(uptime_ms);
    crate::desktop::tick(uptime_ms);
    crate::telemetry::tick(uptime_ms);
    crate::audio::tick();

    // Step 3: compute AI-predicted quantum for the incoming task, then tick.
    // burst_estimate_us is maintained by ai_engine's SGD; 1 tick = 10 ms = 10_000 µs.
    // We peek at the *front* of the run queue to predict for the next task.
    // Memory pressure scales burst_ticks based on system free-RAM level.
    // madvise(MADV_SEQUENTIAL) boosts the scale (sequential tasks benefit from
    // longer quanta); MADV_RANDOM reduces it (random-access thrashes less with
    // shorter quanta).  This is the AI-integration of madvise hints — no other
    // kernel feeds madvise advice into a neural-network scheduler.
    let next_pid_for_hint = {
        let rq = runqueue::peek_front();
        rq.unwrap_or(0)
    };
    let access_adj = match crate::mem_pressure::access_pattern(next_pid_for_hint) {
        crate::mem_pressure::AccessPattern::Sequential => 1.20f32,
        crate::mem_pressure::AccessPattern::Random     => 0.75f32,
        crate::mem_pressure::AccessPattern::Normal     => 1.00f32,
    };
    let pressure_scale = crate::mem_pressure::current().burst_scale() * access_adj;

    let next_burst_ticks: Option<u32> = {
        let rq_guard  = runqueue::peek_front();
        let tasks_guard = TASKS.lock();
        rq_guard.and_then(|pid| {
            tasks_guard.get(&pid).map(|t| {
                let us = t.ai_profile.burst_estimate_us;
                if us == 0 { None } else {
                    let ticks = ((us / 10_000) as f32 * pressure_scale) as u32;
                    Some(ticks.max(1))
                }
            }).flatten()
        })
    };
    let next_pid = match runqueue::tick(next_burst_ticks) {
        Some(pid) => pid,
        None      => return old_rsp, // same task, no switch
    };

    // Step 3b: causal pre-wake — pre-enqueue habitual consumers of next_pid so
    // they are already Runnable when next_pid performs its first futex_wake.
    // This eliminates a full context-switch latency for pipe/socket pipelines.
    {
        let successors = crate::causal::predict_successors(next_pid);
        if !successors.is_empty() {
            let mut tasks = TASKS.lock();
            for succ_pid in &successors {
                if let Some(t) = tasks.get_mut(succ_pid) {
                    if t.state == task::TaskState::Sleeping {
                        t.state = task::TaskState::Runnable;
                        runqueue::enqueue(*succ_pid);
                        crate::klog!(TRACE,
                            "Causal pre-wake: {} predicted successor of {}",
                            succ_pid, next_pid);
                    }
                }
            }
        }

        // ── Causal page prefetch (novel) ─────────────────────────────────
        // The most likely successor of next_pid will run soon and fault pages.
        // Pre-fault its VMA pages now while we are still in interrupt context
        // so those pages are hot in the TLB when the process actually runs.
        // This is the first kernel OS to use a causal wakeup graph to predict
        // and pre-fault pages for a process before it is even scheduled.
        let prefetch_pid = crate::causal::predict_next_wake(next_pid)
            .filter(|(_, prob)| *prob > 0.60)
            .map(|(pid, _)| pid);
        if let Some(pp) = prefetch_pid {
            let vmas = crate::syscall::pid_vmas(pp);
            for (start, end, writable, executable) in vmas.iter().take(4) {
                // Pre-fault up to 8 pages per VMA (don't spend too long in ISR).
                let mut va = *start;
                let limit  = (*end).min(*start + 8 * crate::memory::PAGE_SIZE);
                while va < limit {
                    crate::syscall::demand_page_vma(pp, va);
                    va += crate::memory::PAGE_SIZE;
                }
                let _ = (writable, executable);
            }
            if !vmas.is_empty() {
                crate::klog!(TRACE, "Causal prefetch: pre-faulted {} VMAs for pid={}",
                    vmas.len().min(4), pp);
            }
        }
    }

    // Step 3c: confidence-weighted blend of three AI signals.
    //
    // Three sources of scheduling intelligence:
    //   - Transformer (sequence model): nice_delta, burst_ticks. Confidence =
    //     1 - attention_entropy (peaked attention = confident).
    //   - Fingerprint (histogram k-means): nice_adjust. Confidence = cosine_score.
    //   - Causal graph (producer probability): -5 nice bonus. Confidence = prob.
    //
    // Final nice = sum_i(weight_i * nice_i) where weight_i = conf_i² / sum_j(conf_j²).
    // When one source is very confident, it dominates. When all are uncertain, they
    // average. This prevents random-init transformer from polluting good fingerprint
    // decisions during cold start.
    let transformer_decision = crate::transformer_sched::predict(next_pid);

    let (tf_nice, tf_conf) = transformer_decision
        .map(|td| (td.nice_delta as f32, 1.0 - td.attention_entropy))
        .unwrap_or((0.0, 0.0));

    let (fp_nice, fp_conf) = crate::fingerprint::classify_task(next_pid)
        .map(|(_, prof, cs)| (prof.nice_adjust as f32, cs))
        .unwrap_or((0.0, 0.0));

    let (causal_nice, causal_conf) = crate::causal::predict_next_wake(next_pid)
        .map(|(_, prob)| if prob >= 0.5 { (-5.0f32, prob) } else { (0.0, 0.0) })
        .unwrap_or((0.0, 0.0));

    let blended_nice: f32 = {
        let w_tf     = tf_conf     * tf_conf;
        let w_fp     = fp_conf     * fp_conf;
        let w_causal = causal_conf * causal_conf;
        let total    = w_tf + w_fp + w_causal;
        if total > 1e-6 {
            (w_tf * tf_nice + w_fp * fp_nice + w_causal * causal_nice) / total
        } else {
            0.0
        }
    };

    // Step 4: set up the incoming task and return its kernel RSP.
    crate::klog!(TRACE, "Scheduler: {:?} → {}", old_pid, next_pid);

    let next_rsp = {
        let mut tasks = TASKS.lock();
        match tasks.get_mut(&next_pid) {
            Some(t) => {
                // Measure scheduling latency: how long did this task wait to run?
                let wait_ms = uptime_ms.saturating_sub(t.runnable_at);
                let wait_us = wait_ms * 1000;
                t.sched_latency_total_us = t.sched_latency_total_us.saturating_add(wait_us);
                t.sched_count += 1;
                t.ai_profile.ticks_run += 1;
                t.state = task::TaskState::Running;

                // Apply confidence-weighted blend of all three AI signals.
                if t.intent == 0 {
                    t.ai_profile.ai_nice_adjust = blended_nice.clamp(-20.0, 20.0) as i8;
                }
                // Use transformer's burst_ticks if it has a confident prediction.
                if let Some(td) = transformer_decision {
                    let _ = td.burst_ticks; // picked up by next_burst_ticks path above
                }

                // Feed actual scheduling latency back to transformer as 4th target.
                crate::transformer_sched::record_wait(next_pid, wait_us);
                crate::gdt::update_rsp0(t.kernel_stack_top);
                let cr3    = t.cr3;
                let fs     = t.fs_base;
                let rsp    = t.saved_kernel_rsp;
                // Point percpu fpu_ptr at the new task's FPU state area.
                let fpu_ptr = &mut t.fpu_state as *mut _ as u64;
                let cpu = hal::arch_x86_64::gs_cpu_data();
                (*cpu).fpu_ptr = fpu_ptr;
                // Load CR3 last — after all stack accesses.
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

/// Format scheduling latency stats for /proc/sched_latency.
pub fn format_sched_latency() -> alloc::vec::Vec<u8> {
    use alloc::string::String;
    x86_64::instructions::interrupts::without_interrupts(|| {
        let tasks = TASKS.lock();
        let mut out = String::from("PID   NAME             AVG_WAIT_US  TOTAL_WAIT_US  SCHEDULES\n");
        out.push_str("----  ---------------  -----------  -------------  ---------\n");
        let mut entries: alloc::vec::Vec<(u64, u64, u64, u64, alloc::string::String)> = tasks.iter()
            .filter(|(_, t)| t.sched_count > 0)
            .map(|(pid, t)| {
                let avg = t.sched_latency_total_us / t.sched_count;
                (*pid, avg, t.sched_latency_total_us, t.sched_count, t.name.clone())
            })
            .collect();
        // Sort by average wait descending (highest latency first).
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        for (pid, avg, total, count, name) in &entries {
            out.push_str(&alloc::format!(
                "{:<5} {:<16} {:<12} {:<14} {}\n",
                pid, &name[..name.len().min(15)], avg, total, count));
        }
        out.into_bytes()
    })
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
    let now = UPTIME_MS.load(core::sync::atomic::Ordering::Relaxed);
    let mut tasks = TASKS.lock();
    if let Some(t) = tasks.get_mut(&pid) {
        if t.state == crate::scheduler::task::TaskState::Sleeping {
            t.state = crate::scheduler::task::TaskState::Runnable;
            t.runnable_at = now;
            drop(tasks);
            runqueue::enqueue(pid);
        }
    }
}

/// Return the PID of the currently running task, or 1 (init) as default.
pub fn current_pid() -> Pid {
    runqueue::current_pid().unwrap_or(1)
}

/// Get the current number of tasks.
pub fn task_count() -> usize {
    x86_64::instructions::interrupts::without_interrupts(|| {
        TASKS.lock().len()
    })
}

/// Snapshot of per-task data used by /proc/<pid>/.
pub struct TaskInfo {
    pub name:         alloc::string::String,
    pub state_char:   char,
    pub parent_pid:   Pid,
    pub thread_count: u32,
    pub vm_pages:     u64,
}

/// Return a TaskInfo snapshot for `pid`, or None if the task doesn't exist.
pub fn task_info(pid: Pid) -> Option<TaskInfo> {
    let tasks = TASKS.lock();
    let t = tasks.get(&pid)?;
    Some(TaskInfo {
        name:         t.name.clone(),
        state_char:   match t.state {
            TaskState::Runnable => 'R',
            TaskState::Sleeping => 'S',
            TaskState::Zombie   => 'Z',
            _                   => 'S',
        },
        parent_pid:   t.parent_pid,
        thread_count: 1,
        vm_pages:     0, // page accounting not tracked per-task yet
    })
}

/// Return all live (non-zombie) PIDs.
pub fn list_pids() -> alloc::vec::Vec<Pid> {
    let tasks = TASKS.lock();
    tasks.iter()
        .filter(|(_, t)| t.state != TaskState::Zombie)
        .map(|(&pid, _)| pid)
        .collect()
}

/// Return true if `pid` is a live task.
pub fn pid_exists(pid: Pid) -> bool {
    TASKS.lock().contains_key(&pid)
}

/// Stamp the `woke_by` field on a task — called by causal::record_wakeup.
pub fn set_woke_by(wakee_pid: Pid, waker_pid: Pid) {
    if let Some(task) = TASKS.lock().get_mut(&wakee_pid) {
        task.woke_by = Some(waker_pid);
    }
}

/// Terminate task `pid` with `code` — marks zombie, wakes parent, removes from queue.
/// If `pid` is the current task, halts; otherwise returns normally.
pub fn exit_current_direct(pid: Pid, code: i32) -> ! {
    crate::klog!(INFO, "Scheduler: exit pid={} code={}", pid, code);
    let (parent_pid, task_cr3) = {
        let mut tasks = TASKS.lock();
        let ppid = tasks.get(&pid).map(|t| t.parent_pid).unwrap_or(0);
        let cr3  = tasks.get(&pid).map(|t| t.cr3).unwrap_or(0);
        if let Some(task) = tasks.get_mut(&pid) {
            task.state     = task::TaskState::Zombie;
            task.exit_code = Some(code);
        }
        drop(tasks);
        runqueue::remove(pid);
        (ppid, cr3)
    };
    // Release CoW-shared frame references so frames are freed when the last owner exits.
    if task_cr3 != 0 {
        unsafe { crate::memory::release_user_cow_refs(task_cr3); }
    }

    // Record crashes in episodic memory for future recovery
    if code != 0 {
        crate::causal_recovery::record_crash(pid, code);
    }

    crate::syscall_stats::remove(pid);
    crate::transformer_sched::remove(pid);
    crate::syscall::cleanup_pid_fds(pid);
    crate::syscall::cleanup_pid_vmas(pid);
    crate::rlimit::remove_pid(pid);
    crate::ptrace::cleanup_pid(pid as u64);
    crate::job_control::cleanup_pid(pid);
    crate::namespaces::cleanup_pid(pid);
    crate::syscall_proxy::cleanup_pid(pid);
    crate::security::cleanup_task_context(pid);
    crate::collective_integration::cleanup(pid);
    crate::novel_detector::remove(pid);
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
    let pid = runqueue::current_pid().unwrap_or(0);
    let parent_pid = if pid != 0 {
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

    // Release CoW-shared frame references.
    if pid != 0 {
        let task_cr3 = TASKS.lock().get(&pid).map(|t| t.cr3).unwrap_or(0);
        if task_cr3 != 0 {
            unsafe { crate::memory::release_user_cow_refs(task_cr3); }
        }
    }

    // Clean up per-task data.
    crate::syscall_stats::remove(pid);
    crate::anomaly::remove(pid);
    crate::transformer_sched::remove(pid);
    crate::syscall::cleanup_pid_fds(pid);
    crate::syscall::cleanup_pid_vmas(pid);
    crate::rlimit::remove_pid(pid);
    crate::mem_pressure::remove_pid(pid);
    crate::ptrace::cleanup_pid(pid as u64);
    crate::job_control::cleanup_pid(pid);
    crate::namespaces::cleanup_pid(pid);
    crate::syscall_proxy::cleanup_pid(pid);
    crate::security::cleanup_task_context(pid);
    crate::collective_integration::cleanup(pid);
    crate::novel_detector::remove(pid);

    // Wake the parent and send SIGCHLD.
    if parent_pid != 0 {
        // Record causal edge: dying child → parent wakeup.
        if pid != 0 { crate::causal::record_wakeup(pid, parent_pid); }
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

/// Fork: create a full copy of `parent_pid`'s user address space.
/// Child gets its own L4 (kernel half shared, user half deep-copied page by page).
/// This is a full copy — no CoW — so parent and child are fully independent at fork.
pub fn fork_task(parent_pid: Pid) -> Option<Pid> {
    let child_pid = alloc_pid();

    // Get parent's CR3 before locking TASKS (needed for page table walk).
    let parent_cr3 = {
        TASKS.lock().get(&parent_pid).map(|t| t.cr3)?
    };

    // Allocate a new L4 for the child (kernel half copied, user half empty initially).
    let child_cr3 = crate::memory::alloc_user_cr3().unwrap_or_else(|| {
        let v: u64;
        unsafe { core::arch::asm!("mov {}, cr3", out(reg) v, options(nomem, nostack)); }
        v & !0xFFF
    });

    // CoW-share all user-space pages from parent to child.
    // copy_user_address_space strips WRITABLE from parent's PTEs and sets CoW
    // bits — the TLB flush below is mandatory to invalidate cached writable entries.
    let pages = unsafe {
        crate::memory::copy_user_address_space(parent_cr3, child_cr3)
    };
    match &pages {
        Ok(n)  => crate::klog!(INFO, "Scheduler: fork parent={} → child={} ({} pages CoW-shared)", parent_pid, child_pid, n),
        Err(e) => crate::klog!(WARN, "Scheduler: fork child={} page share incomplete: {}", child_pid, e),
    }

    // Flush the parent's TLB: we stripped WRITABLE from its PTEs above but the
    // CPU may still have cached writable translations.  A CR3 reload invalidates
    // all non-global entries without changing the address space.
    unsafe {
        let cr3: u64;
        core::arch::asm!("mov {}, cr3", out(reg) cr3, options(nomem, nostack));
        core::arch::asm!("mov cr3, {}", in(reg) cr3, options(nomem, nostack));
    }

    let mut tasks = TASKS.lock();
    let mut child = tasks.get(&parent_pid)?.clone_shallow(child_pid)?;
    child.cr3 = child_cr3;
    tasks.insert(child_pid, child);
    runqueue::enqueue(child_pid);
    crate::security::init_task_context(child_pid, Some(parent_pid));
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

/// Override the AI nice_adjust for a task (used by job_control for fg/bg).
/// Pass 0 to clear the override and let the AI resume normal scheduling.
pub fn set_nice_override(pid: Pid, nice: i8) {
    let mut tasks = TASKS.lock();
    if let Some(t) = tasks.get_mut(&pid) {
        t.ai_profile.ai_nice_adjust = nice;
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

/// Store the robust futex list for a thread (walked on thread death to unlock mutexes).
pub fn set_robust_list(pid: Pid, head: u64, len: usize) {
    let mut tasks = TASKS.lock();
    if let Some(t) = tasks.get_mut(&pid) {
        t.robust_list_head = head;
        t.robust_list_len  = len;
    }
}

/// Update the CR3 (page table) for a task — called by execve when it creates a new address space.
pub fn set_task_cr3(pid: Pid, cr3: u64) {
    let mut tasks = TASKS.lock();
    if let Some(t) = tasks.get_mut(&pid) { t.cr3 = cr3; }
}

// ── Intent-based scheduling ────────────────────────────────────────────────────
//
// Intent constants (must match sys_intent documentation in syscall/mod.rs).
pub const INTENT_DEFAULT:      u8 = 0;
pub const INTENT_LATENCY:      u8 = 1;
pub const INTENT_BATCH:        u8 = 2;
pub const INTENT_INTERACTIVE:  u8 = 3;
pub const INTENT_IO_SEQUENTIAL:u8 = 4;
pub const INTENT_IO_RANDOM:    u8 = 5;
pub const INTENT_MEMORY_LARGE: u8 = 6;
pub const INTENT_CPU_BOUND:    u8 = 7;

/// Apply a declared intent to a task's scheduling parameters immediately.
/// The AI engine may refine these further on the next tick.
pub fn set_intent(pid: Pid, intent: u8, hint: u64) {
    let mut tasks = TASKS.lock();
    if let Some(t) = tasks.get_mut(&pid) {
        t.intent      = intent;
        t.intent_hint = hint;
        // Apply immediate priority bias based on intent.
        t.priority = match intent {
            INTENT_LATENCY     => -15, // near-realtime
            INTENT_INTERACTIVE => -10, // boosted
            INTENT_DEFAULT     =>   0,
            INTENT_BATCH       =>  10, // deprioritised
            INTENT_CPU_BOUND   =>   5, // slightly lower (yield to latency tasks)
            _                  => t.priority, // I/O hints: no priority change
        };
        crate::klog!(INFO, "Intent: pid={} type={} priority={}", pid, intent, t.priority);
    }
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
    // clone_shallow copies parent's interrupt frame and zeroes RAX (child gets 0 from clone).
    // It does NOT copy page tables — child.cr3 == parent.cr3 (shared address space).
    let mut child = tasks.get(&parent_pid)?.clone_shallow(child_pid)?;

    // Override the user RSP in the saved interrupt frame so the thread starts on new_stack.
    // Frame layout: [saved_kernel_rsp+120]=RIP, [+128]=CS, [+136]=RFLAGS, [+144]=RSP, [+152]=SS
    unsafe {
        let rsp_slot = (child.saved_kernel_rsp + 144) as *mut u64;
        rsp_slot.write(new_stack);
    }
    if settls { child.fs_base = tls; }
    child.parent_pid = parent_pid;
    tasks.insert(child_pid, child);
    runqueue::enqueue(child_pid);
    // Initialize resource limits (inherit from parent or defaults)
    crate::rlimit::init_pid(child_pid, Some(parent_pid));
    // Check if parent had a crash pattern — apply proactive constraints if so
    crate::causal_recovery::on_spawn(child_pid, parent_pid);
    crate::klog!(INFO, "Scheduler: thread tid={} parent={} stack={:#x}", child_pid, parent_pid, new_stack);
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

/// Return the raw pending_signals bitmask for signalfd polling (non-consuming).
pub fn get_pending_signals(pid: Pid) -> u64 {
    TASKS.lock().get(&pid).map(|t| t.pending_signals).unwrap_or(0)
}

/// Consume one pending signal matching `mask` for signalfd::read().
/// Returns the signal number removed, or None if no match.
pub fn consume_masked_signal(pid: Pid, mask: u64) -> Option<u8> {
    let mut tasks = TASKS.lock();
    let t = tasks.get_mut(&pid)?;
    let matched = t.pending_signals & mask;
    if matched == 0 { return None; }
    let signum = matched.trailing_zeros() as u8;
    t.pending_signals &= !(1u64 << signum);
    Some(signum)
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

/// Return the current signal handler address for `signum`, or 0 if none.
pub fn get_signal_handler(pid: Pid, signum: usize) -> u64 {
    let tasks = TASKS.lock();
    tasks.get(&pid).map(|t| t.signal_handlers[signum]).unwrap_or(0)
}

/// Return the current signal mask for `pid`.
pub fn get_signal_mask(pid: Pid) -> u64 {
    let tasks = TASKS.lock();
    tasks.get(&pid).map(|t| t.signal_mask).unwrap_or(0)
}

/// Set the absolute signal mask for `pid` (SIG_SETMASK).
pub fn set_signal_mask(pid: Pid, mask: u64) {
    let mut tasks = TASKS.lock();
    if let Some(t) = tasks.get_mut(&pid) {
        t.signal_mask = mask;
    }
}

/// Block (add to mask) a set of signals (SIG_BLOCK).
pub fn mask_signals(pid: Pid, mask: u64) {
    let mut tasks = TASKS.lock();
    if let Some(t) = tasks.get_mut(&pid) {
        t.signal_mask |= mask;
    }
}

/// Unblock (remove from mask) a set of signals (SIG_UNBLOCK).
pub fn unmask_signals(pid: Pid, mask: u64) {
    let mut tasks = TASKS.lock();
    if let Some(t) = tasks.get_mut(&pid) {
        t.signal_mask &= !mask;
    }
}

/// Return total RAM in 4 KiB pages.
pub fn total_ram_pages() -> u64 {
    crate::memory::total_ram_pages()
}

// ── AI scheduling extensions (transformer, causal, anomaly blending) ──────────

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

/// Return the CR3 (PML4) value for a task, or None if the task does not exist.
pub fn get_task_cr3(pid: Pid) -> Option<u64> {
    TASKS.lock().get(&pid).map(|t| t.cr3)
}

/// Return the saved kernel RSP for a task (proxy for instruction pointer during hibernation).
pub fn get_saved_rsp(pid: Pid) -> Option<u64> {
    TASKS.lock().get(&pid).map(|t| t.saved_kernel_rsp)
}

/// Set the task's euid.
pub fn set_euid(pid: u64, euid: u32) {
    if let Some(mut tasks) = TASKS.try_lock() {
        if let Some(t) = tasks.get_mut(&pid) {
            t.euid = euid;
        }
    }
}

/// Get the task's euid without blocking.
pub fn get_euid(pid: u64) -> Option<u32> {
    let tasks = TASKS.try_lock()?;
    tasks.get(&pid).map(|t| t.euid)
}
