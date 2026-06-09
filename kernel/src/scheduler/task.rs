//! Task — the fundamental unit of execution in NodeAI.

use alloc::string::String;
use alloc::vec::Vec;

/// 512-byte FXSAVE area — saves x87 FPU + SSE state.
/// Must be 16-byte aligned; repr(C,align(16)) ensures this.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct FpuState(pub [u8; 512]);

impl FpuState {
    /// A valid reset state: FCW=0x037F (all exceptions masked), MXCSR=0x1F80.
    /// Captured once at boot via capture_init_fpu_state() and copied to every new task.
    pub const fn default_state() -> Self {
        let mut s = [0u8; 512];
        // x87 FCW: offset 0, value 0x037F (little-endian)
        s[0] = 0x7F; s[1] = 0x03;
        // MXCSR: offset 24, value 0x00001F80
        s[24] = 0x80; s[25] = 0x1F;
        // MXCSR_MASK: offset 28, value 0xFFFF
        s[28] = 0xFF; s[29] = 0xFF;
        Self(s)
    }
}

// ── Synthetic interrupt-frame layout constants ────────────────────────────────
//
// The naked timer handler pushes GPRs in this order (first push is highest addr):
//   push rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11, rbx, rbp, r12, r13, r14, r15
// then the CPU has already pushed the IRET frame (RIP, CS, RFLAGS, RSP, SS).
//
// After all pushes, RSP points to r15 (lowest addr = bottom of save area).
// Layout relative to saved_kernel_rsp (which = address of r15 slot):
//   +0:   r15   +8:  r14   +16: r13  +24: r12  +32: rbp  +40: rbx
//   +48:  r11   +56: r10   +64: r9   +72: r8   +80: rdi  +88: rsi
//   +96:  rdx  +104: rcx  +112: rax
//   +120: RIP  +128: CS   +136: RFLAGS  +144: RSP  +152: SS
//
// Total frame size = 15 GPRs × 8 + 5 IRET words × 8 = 160 bytes.

/// Byte offset of the RAX slot within a saved interrupt frame (from frame base = r15 slot).
pub const FRAME_RAX_OFFSET: usize = 112;
/// Total size of one saved interrupt frame in bytes.
pub const FRAME_SIZE: usize = 160;
/// Offset of the RIP slot (IRET[0]) within a saved interrupt frame.
pub const FRAME_RIP_OFFSET: usize = 120;

/// Unique process/thread identifier.
pub type Pid = u64;

/// Task execution state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    /// Eligible to run; waiting for CPU time.
    Runnable,
    /// Currently executing on a CPU.
    Running,
    /// Sleeping — waiting for an event (I/O, signal, timer).
    Sleeping,
    /// Terminated — waiting to be reaped.
    Zombie,
}

/// Per-task CPU register snapshot (x86_64).
/// Layout matches the push order in `context_switch.S` / inline asm.
/// MUST be kept in sync with `switch_context` in `context_switch.rs`.
#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct CpuContext {
    // Callee-saved registers (saved/restored in context switch)
    pub r15: u64, pub r14: u64, pub r13: u64, pub r12: u64,
    pub rbp: u64, pub rbx: u64,
    // Caller-saved + argument registers (also saved for preemption)
    pub r11: u64, pub r10: u64, pub r9:  u64, pub r8:  u64,
    pub rax: u64, pub rcx: u64, pub rdx: u64, pub rsi: u64, pub rdi: u64,
    // Return address / stack state
    pub rip:    u64,
    pub cs:     u64,
    pub rflags: u64,
    pub rsp:    u64,
    pub ss:     u64,
}

/// AI-maintained behavioral fingerprint — updated every scheduler tick.
#[derive(Debug, Default, Clone)]
pub struct AiProfile {
    /// Estimated remaining CPU burst in microseconds (AI prediction).
    pub burst_estimate_us: u64,
    /// Ratio of I/O waits to CPU time (0.0 = CPU-bound, 1.0 = I/O-bound).
    pub io_cpu_ratio: f32,
    /// Cache miss rate from hardware performance counters.
    pub cache_miss_rate: f32,
    /// AI-assigned priority adjustment in range [-20, 20] (like nice values).
    pub ai_nice_adjust: i8,
    /// Accumulated scheduler ticks run.
    pub ticks_run: u64,
}

/// File descriptor table entry.
#[derive(Debug, Clone)]
pub struct FdEntry {
    pub fd: u32,
    // TODO: link to VfsNode for per-task fd table (currently managed in syscall FD_TABLE)
}

/// The Task Control Block.
pub struct Task {
    pub pid:      Pid,
    pub name:     String,
    pub state:    TaskState,
    pub priority: i32,
    /// Top of the kernel stack for this task.
    pub kernel_stack_top: u64,
    /// Saved kernel RSP — points at the bottom of the saved interrupt frame on
    /// this task's kernel stack.  Set by the timer handler on preemption and
    /// restored by `schedule_from_interrupt`.
    pub saved_kernel_rsp: u64,
    /// CR3 value (page table physical address) for this task.
    pub cr3: u64,
    pub context: CpuContext,
    pub ai_profile: AiProfile,
    /// File descriptors (placeholder — real fd table lives in syscall::FD_TABLE).
    pub fds: Vec<FdEntry>,
    /// Signal mask (64 standard POSIX signals).
    pub signal_mask: u64,
    /// Signal handlers: index = signal number, value = user-space handler VA (0 = default).
    pub signal_handlers: [u64; 64],
    /// Per-process credentials.
    pub uid:  u32,
    pub euid: u32,
    pub gid:  u32,
    pub egid: u32,
    /// Parent PID (0 = no parent / init).
    pub parent_pid: Pid,
    /// Exit code set by sys_exit; None while alive.
    pub exit_code: Option<i32>,
    /// POSIX robust futex list head (set by set_robust_list, walked on thread death).
    pub robust_list_head: u64,
    pub robust_list_len:  usize,
    /// Declared scheduling intent via sys_intent (0 = default/unset).
    pub intent: u8,
    /// Hint value accompanying the intent (e.g. I/O stride, memory estimate).
    pub intent_hint: u64,
    /// Pending signal bitmap (bit N = signal N is pending delivery).
    pub pending_signals: u64,
    /// Per-task FPU/SSE state — saved/restored by the timer handler via fxsave64/fxrstor64.
    pub fpu_state: FpuState,
    /// User-space program break (top of heap) for sys_brk.
    pub user_brk: u64,
    /// Thread-local storage FS base (ARCH_SET_FS).
    pub fs_base: u64,
    /// Causal wakeup: which PID last woke this task (via futex, pipe, socket, or exit).
    /// None = self-woken (initial run, or woke from timer with no blocked wait).
    pub woke_by: Option<Pid>,
    /// Uptime (ms) when this task last became Runnable (woken or enqueued for first time).
    /// Used to compute scheduling latency: time between becoming runnable and actually running.
    pub runnable_at: u64,
    /// Cumulative scheduling latency (µs) — sum of all wait times for this task.
    pub sched_latency_total_us: u64,
    /// Number of times this task was scheduled — for computing average latency.
    pub sched_count: u64,
    /// Kernel stack canary value — written to the bottom of the kernel stack at
    /// task creation.  Checked on every context switch; corruption = stack overflow.
    pub stack_canary: u64,
    /// Physical-offset virtual address where the canary lives (bottom of stack).
    pub stack_canary_addr: u64,
    /// Snapshot of the MHS FastState (Phase 3)
    pub mhs_snapshot: Option<[f32; 16]>,
}

impl Task {
    /// Create a kernel thread with its own stack (stack_top is the high end).
    ///
    /// Lays a synthetic interrupt frame at the top of the stack so that
    /// `schedule_from_interrupt` can restore this task exactly like any other
    /// preempted task — no special "first-run" path needed.
    pub fn new_kernel_thread(pid: Pid, name: &str, entry: u64, stack_top: u64) -> Self {
        let cr3: u64;
        unsafe {
            core::arch::asm!("mov {}, cr3", out(reg) cr3, options(nomem, nostack));
        }

        // Build the synthetic frame in memory (stack grows down):
        //   [stack_top - 8]  : SS
        //   [stack_top - 16] : RSP (thread's initial stack pointer = stack_top)
        //   [stack_top - 24] : RFLAGS
        //   [stack_top - 32] : CS
        //   [stack_top - 40] : RIP (entry point)
        //   [stack_top - 160]: 15 × u64 GPR save area, all zero
        // saved_kernel_rsp = stack_top - FRAME_SIZE (= stack_top - 160)
        let frame_base = stack_top - FRAME_SIZE as u64;
        unsafe {
            let p = frame_base as *mut u64;
            // Zero the entire frame (GPRs and IRET slots).
            core::ptr::write_bytes(p as *mut u8, 0, FRAME_SIZE);
            // Write IRET frame at offsets 120..160
            p.add(FRAME_RIP_OFFSET / 8).write(entry);          // RIP
            p.add(FRAME_RIP_OFFSET / 8 + 1).write(0x08);       // CS  = kernel code
            p.add(FRAME_RIP_OFFSET / 8 + 2).write(0x202);      // RFLAGS = IF
            p.add(FRAME_RIP_OFFSET / 8 + 3).write(stack_top);  // RSP (thread uses full stack)
            p.add(FRAME_RIP_OFFSET / 8 + 4).write(0x10);       // SS  = kernel data
        }

        let mut ctx = CpuContext::default();
        ctx.rip    = entry;
        ctx.rsp    = stack_top;
        ctx.cs     = 0x08;
        ctx.ss     = 0x10;
        ctx.rflags = 0x202;

        Task {
            pid,
            name:             String::from(name),
            state:            TaskState::Runnable,
            priority:         0,
            kernel_stack_top: stack_top,
            saved_kernel_rsp: frame_base,
            cr3,
            context:          ctx,
            ai_profile:       AiProfile::default(),
            fds:              Vec::new(),
            signal_mask:      0,
            signal_handlers:  [0u64; 64],
            uid:  0,
            euid: 0,
            gid:  0,
            egid: 0,
            parent_pid:          0,
            exit_code:           None,
            robust_list_head:    0,
            robust_list_len:     0,
            intent:              0,
            intent_hint:         0,
            pending_signals: 0,
            fpu_state:       FpuState::default_state(),
            user_brk:              0,
            fs_base:               0,
            woke_by:               None,
            runnable_at:           0,
            sched_latency_total_us: 0,
            sched_count:           0,
            stack_canary:      0,
            stack_canary_addr: 0,
            mhs_snapshot:      None,
        }
    }

    /// Shallow clone for fork(): copy everything except give the child a new PID
    /// and mark parent_pid.  Both share the same CR3 (CoW deferred).
    ///
    /// The child's saved interrupt frame is a copy of the parent's, with one
    /// change: RAX = 0 so the child sees 0 as its return value from fork().
    pub fn clone_shallow(&self, child_pid: Pid) -> Option<Task> {
        const STACK_PAGES: usize = 4;
        let stack_phys = crate::memory::alloc_frames(2)?;
        // Map the physical frame via the PHYS_OFFSET window (same as map_user_range).
        let stack_top = crate::memory::phys_offset() + stack_phys
            + (STACK_PAGES as u64 * crate::memory::PAGE_SIZE);

        // Copy parent's saved interrupt frame onto the child's new kernel stack.
        let child_frame_base = stack_top - FRAME_SIZE as u64;
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.saved_kernel_rsp as *const u8,
                child_frame_base as *mut u8,
                FRAME_SIZE,
            );
            // Child must return 0 from fork() — zero the RAX slot.
            let rax_ptr = (child_frame_base + FRAME_RAX_OFFSET as u64) as *mut u64;
            rax_ptr.write(0);
        }

        Some(Task {
            pid:              child_pid,
            name:             self.name.clone(),
            state:            TaskState::Runnable,
            priority:         self.priority,
            kernel_stack_top: stack_top,
            saved_kernel_rsp: child_frame_base,
            cr3:              self.cr3,
            context:          self.context,
            ai_profile:       AiProfile::default(),
            fds:              self.fds.clone(),
            signal_mask:      self.signal_mask,
            signal_handlers:  self.signal_handlers,
            uid:              self.uid,
            euid:             self.euid,
            gid:              self.gid,
            egid:             self.egid,
            parent_pid:           self.pid,
            exit_code:            None,
            robust_list_head:     0,
            robust_list_len:      0,
            intent:               self.intent,
            intent_hint:          self.intent_hint,
            pending_signals:  0,
            fpu_state:        self.fpu_state, // copy parent's FPU state to child
            user_brk:               self.user_brk,
            fs_base:                self.fs_base,
            woke_by:                None,
            runnable_at:            0,
            sched_latency_total_us: 0,
            sched_count:            0,
            stack_canary:           0,
            stack_canary_addr:      0,
            mhs_snapshot:           self.mhs_snapshot.clone(),
        })
    }
}
