//! Task — the fundamental unit of execution in NodeAI.

use alloc::string::String;
use alloc::vec::Vec;

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
    // Phase 7: replace with Arc<dyn VfsNode>
}

/// The Task Control Block.
pub struct Task {
    pub pid:      Pid,
    pub name:     String,
    pub state:    TaskState,
    pub priority: i32,
    /// Top of the kernel stack for this task (used during context switch).
    pub kernel_stack_top: u64,
    /// CR3 value (page table physical address) for this task.
    pub cr3: u64,
    pub context: CpuContext,
    pub ai_profile: AiProfile,
    /// File descriptors (Phase 7: link to VFS).
    pub fds: Vec<FdEntry>,
    /// Signal mask (64 standard POSIX signals).
    pub signal_mask: u64,
    /// Signal handlers: index = signal number, value = user-space handler VA (0 = default).
    pub signal_handlers: [u64; 64],
    /// Per-process credentials (Phase 14).
    pub uid:  u32,
    pub euid: u32,
    pub gid:  u32,
    pub egid: u32,
    /// Parent PID (0 = no parent / init).
    pub parent_pid: Pid,
    /// Exit code set by sys_exit; None while alive.
    pub exit_code: Option<i32>,
    /// User-space program break (top of heap) for sys_brk.
    pub user_brk: u64,
    /// Thread-local storage FS base (ARCH_SET_FS).
    pub fs_base: u64,
}

impl Task {
    /// Create a kernel thread with its own stack (stack_top is stack end).
    pub fn new_kernel_thread(pid: Pid, name: &str, entry: u64, stack_top: u64) -> Self {
        let mut ctx = CpuContext::default();
        ctx.rip    = entry;
        ctx.rsp    = stack_top;
        ctx.cs     = 0x08;      // kernel code segment
        ctx.ss     = 0x10;      // kernel data segment
        ctx.rflags = 0x202;     // IF=1 (interrupts enabled)

        // Read current CR3 — kernel threads share the kernel page table.
        let cr3: u64;
        unsafe {
            core::arch::asm!("mov {}, cr3", out(reg) cr3, options(nomem, nostack));
        }

        Task {
            pid,
            name:             String::from(name),
            state:            TaskState::Runnable,
            priority:         0,
            kernel_stack_top: stack_top,
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
            parent_pid:  0,
            exit_code:   None,
            user_brk:    0,
            fs_base:     0,
        }
    }

    /// Shallow clone for fork(): copy everything except give the child a new PID
    /// and mark parent_pid.  Both share the same CR3 (Phase 21 adds CoW).
    pub fn clone_shallow(&self, child_pid: Pid) -> Option<Task> {
        // Allocate a new kernel stack for the child.
        const STACK_PAGES: usize = 4;
        let stack_phys = crate::memory::alloc_frames(2)?;
        let stack_top  = stack_phys + (STACK_PAGES as u64 * crate::memory::PAGE_SIZE);
        Some(Task {
            pid:              child_pid,
            name:             self.name.clone(),
            state:            TaskState::Runnable,
            priority:         self.priority,
            kernel_stack_top: stack_top,
            cr3:              self.cr3,          // shared for now
            context:          self.context,
            ai_profile:       AiProfile::default(),
            fds:              self.fds.clone(),
            signal_mask:      self.signal_mask,
            signal_handlers:  self.signal_handlers,
            uid:              self.uid,
            euid:             self.euid,
            gid:              self.gid,
            egid:             self.egid,
            parent_pid:       self.pid,
            exit_code:        None,
            user_brk:         self.user_brk,
            fs_base:          self.fs_base,
        })
    }
}

