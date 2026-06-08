//! System call fast-path (SYSCALL/SYSRET, x86_64).
//!
//! Sets up the `SYSCALL`/`SYSRET` mechanism (AMD-64 / Intel 64) by writing:
//!   - `MSR_IA32_STAR`  — segment selectors used on SYSCALL/SYSRET
//!   - `MSR_IA32_LSTAR` — 64-bit SYSCALL target RIP
//!   - `MSR_IA32_FMASK` — RFLAGS bits to clear on SYSCALL (we clear IF)
//!   - `MSR_IA32_EFER`  — enables SCE (SYSCALL Enable) bit
//!
//! The assembly stub (`_syscall_entry`) saves the small set of volatile
//! registers needed for `SYSRETQ` (RCX/R11 = user RIP/RFLAGS), switches to
//! the per-CPU kernel stack, shuffles syscall args into the C ABI register
//! order, and calls `syscall_dispatch_extern`.

use core::sync::atomic::{AtomicU64, Ordering};
use alloc::collections::BTreeMap;
use spin::Mutex;

// ── Syscall telemetry ─────────────────────────────────────────────────────────
static SYSCALL_COUNT: AtomicU64 = AtomicU64::new(0);

/// Total syscall invocations since boot (for desktop telemetry).
pub fn syscall_count() -> u64 {
    SYSCALL_COUNT.load(Ordering::Relaxed)
}
use hal::arch_x86_64::{
    wrmsr, rdmsr,
    MSR_IA32_STAR, MSR_IA32_LSTAR, MSR_IA32_FMASK, MSR_IA32_EFER,
    PercpuData,
};

// ── Error numbers (POSIX subset) ──────────────────────────────────────────────
pub const EPERM:    i64 = -1;
pub const ENOENT:   i64 = -2;
pub const EBADF:    i64 = -9;
pub const EFAULT:   i64 = -14;
pub const EINVAL:   i64 = -22;
pub const EAGAIN:   i64 = -11;
pub const ENOTSOCK: i64 = -88;
pub const ENOSYS:   i64 = -38;

// ── Global file-descriptor table ─────────────────────────────────────────────
//
// Key: (pid, fd_number).  Fd 0/1/2 are stdin/stdout/stderr (handled inline).
// All other fds come from open().

type FdKey = (u64, u64);
static FD_TABLE: Mutex<BTreeMap<FdKey, alloc::boxed::Box<dyn crate::vfs::FileHandle>>>
    = Mutex::new(BTreeMap::new());
static NEXT_FD:  Mutex<BTreeMap<u64, u64>> = Mutex::new(BTreeMap::new());
/// Maps (pid, fd) → file path for AI readahead — populated by sys_open.
static FD_PATH_TABLE: Mutex<BTreeMap<FdKey, alloc::string::String>> = Mutex::new(BTreeMap::new());
/// Bound port for socket fds — separate from FD_TABLE to avoid trait-object downcast.
static SOCKET_PORTS: Mutex<BTreeMap<FdKey, u16>> = Mutex::new(BTreeMap::new());

// ── epoll interest-list tables ────────────────────────────────────────────────
/// Per-interest-list entry: the events mask and the opaque user data word.
#[derive(Clone, Copy)]
struct EpollInterest { events: u32, data: u64 }

/// One epoll instance: maps watched-fd → interest.
struct EpollInstance { interests: alloc::collections::BTreeMap<i32, EpollInterest> }

impl EpollInstance {
    fn new() -> Self { Self { interests: alloc::collections::BTreeMap::new() } }
}

/// (pid, epfd) → EpollInstance
static EPOLL_TABLE: Mutex<alloc::collections::BTreeMap<FdKey, EpollInstance>>
    = Mutex::new(alloc::collections::BTreeMap::new());

const EPOLL_CTL_ADD: i32 = 1;
const EPOLL_CTL_DEL: i32 = 2;
const EPOLL_CTL_MOD: i32 = 3;
const EPOLLIN:  u32 = 0x0001;
const EPOLLOUT: u32 = 0x0004;
const EPOLLERR: u32 = 0x0008;
const EPOLLHUP: u32 = 0x0010;
// Edge-triggered flag — we implement level-triggered; ET flag is accepted but ignored.
const EPOLLET: u32 = 1 << 31;

/// VfsNode for directory fds — used by getdents64 to call readdir().
static DIR_NODES: spin::Mutex<BTreeMap<FdKey, alloc::sync::Arc<dyn crate::vfs::VfsNode>>>
    = spin::Mutex::new(BTreeMap::new());

/// Called by exit_current to free all per-pid fd state.
pub fn cleanup_pid_fds(pid: u64) {
    FD_TABLE.lock().retain(|&(p, _), _| p != pid);
    FD_PATH_TABLE.lock().retain(|&(p, _), _| p != pid);
    EPOLL_TABLE.lock().retain(|&(p, _), _| p != pid);
    NEXT_FD.lock().remove(&pid);
    // DIR_NODES and SOCKET_PORTS share the same key layout.
    DIR_NODES.lock().retain(|&(p, _), _| p != pid);
    SOCKET_PORTS.lock().retain(|&(p, _), _| p != pid);
}

/// Return the list of open fd numbers for a given PID (used by /proc/<pid>/fd/).
pub fn list_pid_fds(pid: u64) -> alloc::vec::Vec<u64> {
    FD_TABLE.lock().keys().filter_map(|&(p, fd)| if p == pid { Some(fd) } else { None }).collect()
}

/// Return the filesystem path for (pid, fd), if known.
pub fn fd_path(pid: u64, fd: u64) -> Option<alloc::string::String> {
    FD_PATH_TABLE.lock().get(&(pid, fd)).cloned()
}

/// Allocate the next available fd for a given pid (first-fit above 2).
fn alloc_fd(pid: u64) -> u64 {
    let mut map = NEXT_FD.lock();
    let fd = *map.get(&pid).unwrap_or(&3u64);
    map.insert(pid, fd + 1);
    fd
}

// ── NodeAI syscall numbers (Linux x86_64 ABI compatible) ─────────────────────
pub mod nr {
    pub const READ:            u64 = 0;
    pub const WRITE:           u64 = 1;
    pub const OPEN:            u64 = 2;
    pub const CLOSE:           u64 = 3;
    pub const STAT:            u64 = 4;
    pub const FSTAT:           u64 = 5;
    pub const LSTAT:           u64 = 6;
    pub const POLL:            u64 = 7;
    pub const LSEEK:           u64 = 8;
    pub const MMAP:            u64 = 9;
    pub const MPROTECT:        u64 = 10;
    pub const MUNMAP:          u64 = 11;
    pub const BRK:             u64 = 12;
    pub const RT_SIGACTION:    u64 = 13;
    pub const RT_SIGPROCMASK:  u64 = 14;
    pub const RT_SIGRETURN:    u64 = 15;
    pub const SIGALTSTACK:     u64 = 131;
    pub const IOCTL:           u64 = 16;
    pub const PREAD64:         u64 = 17;
    pub const PWRITE64:        u64 = 18;
    pub const READV:           u64 = 19;
    pub const WRITEV:          u64 = 20;
    pub const SELECT:          u64 = 23;
    pub const NANOSLEEP:       u64 = 35;
    pub const SENDFILE:        u64 = 40;
    pub const SOCKET:          u64 = 41;
    pub const CONNECT:         u64 = 42;
    pub const ACCEPT:          u64 = 43;
    pub const SENDTO:          u64 = 44;
    pub const RECVFROM:        u64 = 45;
    pub const SHUTDOWN:        u64 = 48;
    pub const BIND:            u64 = 49;
    pub const LISTEN:          u64 = 50;
    pub const GETSOCKNAME:     u64 = 51;
    pub const GETPEERNAME:     u64 = 52;
    pub const SETSOCKOPT:      u64 = 54;
    pub const GETSOCKOPT:      u64 = 55;
    pub const CLONE:           u64 = 56;
    pub const FORK:            u64 = 57;
    pub const EXECVE:          u64 = 59;
    pub const EXIT:            u64 = 60;
    pub const WAIT4:           u64 = 61;
    pub const KILL:            u64 = 62;
    pub const UNAME:           u64 = 63;
    pub const FCNTL:           u64 = 72;
    pub const GETDENTS64:      u64 = 217;
    pub const TRUNCATE:        u64 = 76;
    pub const FTRUNCATE:       u64 = 77;
    pub const GETCWD:          u64 = 79;
    pub const CHDIR:           u64 = 80;
    pub const MKDIR:           u64 = 83;
    pub const RMDIR:           u64 = 84;
    pub const CREAT:           u64 = 85;
    pub const LINK:            u64 = 86;
    pub const UNLINK:          u64 = 87;
    pub const SYMLINK:         u64 = 88;
    pub const RENAME:          u64 = 82;
    pub const CHMOD:           u64 = 90;
    pub const CHOWN:           u64 = 92;
    pub const GETPID:          u64 = 39;
    pub const PIPE:            u64 = 22;
    pub const DUP:             u64 = 32;
    pub const DUP2:            u64 = 33;
    pub const GETPPID:         u64 = 110;
    pub const GETUID:          u64 = 102;
    pub const GETGID:          u64 = 104;
    pub const SETUID:          u64 = 105;
    pub const SETGID:          u64 = 106;
    pub const GETEUID:         u64 = 107;
    pub const GETEGID:         u64 = 108;
    pub const SETPGID:         u64 = 109;
    pub const SETSID:          u64 = 112;
    pub const GETPGID:         u64 = 121;
    pub const ARCH_PRCTL:      u64 = 158;
    pub const PRCTL:           u64 = 157;
    pub const GETTID:          u64 = 186;
    pub const FUTEX:           u64 = 202;
    pub const SET_TID_ADDRESS: u64 = 218;
    pub const CLOCK_GETTIME:   u64 = 228;
    pub const EXIT_GROUP:      u64 = 231;
    pub const EPOLL_CTL:       u64 = 233;
    pub const EPOLL_WAIT:      u64 = 232;
    pub const SET_ROBUST_LIST: u64 = 273;
    pub const GET_ROBUST_LIST: u64 = 274;
    pub const PIPE2:           u64 = 293;
    pub const DUP3:            u64 = 292;
    pub const EPOLL_CREATE1:   u64 = 291;
    pub const EVENTFD2:        u64 = 290;
    pub const ACCEPT4:         u64 = 288;
    pub const PRLIMIT64:       u64 = 302;
    pub const GETRANDOM:       u64 = 318;
    pub const MEMFD_CREATE:    u64 = 319;
    pub const STATX:           u64 = 332;
    pub const OPENAT:          u64 = 257;
    pub const FSTATAT:         u64 = 262;
    pub const GETTIMEOFDAY:    u64 = 96;
    pub const SYSINFO:         u64 = 99;
    pub const TIMES:           u64 = 100;
    pub const GETRLIMIT:       u64 = 97;
    pub const SETRLIMIT:       u64 = 160;
    pub const UMASK:           u64 = 95;
    pub const MADVISE:         u64 = 28;
    pub const MINCORE:         u64 = 27;
    pub const MSYNC:           u64 = 26;
    pub const MLOCK:           u64 = 149;
    pub const MUNLOCK:         u64 = 150;
    pub const STATFS:          u64 = 137;
    pub const FSTATFS:         u64 = 138;
    pub const AI_QUERY:        u64 = 200;
    pub const AI_LOG:          u64 = 201;
    pub const SYS_INTENT:      u64 = 202; // NodeAI-specific: declare scheduling intent
    pub const TKILL:           u64 = 200;   // re-use slot — routed same as AI_QUERY when called correctly
    pub const TGKILL:          u64 = 234;
    // Phase 24 additions (new constants not already defined above)
    pub const READLINK:        u64 = 89;
    pub const INOTIFY_INIT1:   u64 = 294;
    pub const INOTIFY_ADD_WATCH: u64 = 254;
    pub const INOTIFY_RM_WATCH: u64 = 255;
    pub const FACCESSAT:       u64 = 269;
    pub const MKDIRAT:         u64 = 258;
    pub const UNLINKAT:        u64 = 263;
    pub const RENAMEAT:        u64 = 264;
    pub const RENAMEAT2:       u64 = 316;
    pub const SYMLINKAT:       u64 = 266;
    pub const READLINKAT:      u64 = 267;
    pub const TIMERFD_CREATE:  u64 = 283;
    pub const TIMERFD_SETTIME: u64 = 286;
    pub const TIMERFD_GETTIME: u64 = 287;
    pub const SIGNALFD4:       u64 = 289;
    pub const CLONE3:          u64 = 435;
    pub const SETTIMEOFDAY:    u64 = 164;
    pub const ACCESS:          u64 = 21;
    pub const LCHOWN:          u64 = 94;
    pub const FCHMOD:          u64 = 91;
    pub const FCHOWN:          u64 = 93;
    pub const MKNOD:           u64 = 133;
    pub const GETGROUPS:       u64 = 115;
    pub const SETGROUPS:       u64 = 116;
    pub const ALARM:           u64 = 37;
    pub const PAUSE:           u64 = 34;
}

// ── Per-CPU kernel stack for syscall handling ─────────────────────────────────

const SYSCALL_STACK_PAGES: usize = 4;   // 16 KiB
const SYSCALL_STACK_SIZE:  usize = SYSCALL_STACK_PAGES * 4096;

#[repr(C, align(16))]
struct AlignedStack([u8; SYSCALL_STACK_SIZE]);

static mut BOOT_SYSCALL_STACK: AlignedStack = AlignedStack([0u8; SYSCALL_STACK_SIZE]);

// Per-CPU data area for the boot CPU.
static mut BOOT_PERCPU: PercpuData = PercpuData {
    self_ptr:          0,
    cpu_id:            0,
    _pad:              0,
    kernel_rsp:        0,
    user_rsp:          0,
    ticks_per_ms:      0,
    _pad2:             0,
    signal_new_rip:    0,
    signal_new_rsp:    0,
    signal_new_rflags: 0,
    signal_signum:     0,
    fpu_ptr:           0, // set to boot task's FPU area during init_lstar
};

// ── Assembly syscall entry stub ───────────────────────────────────────────────
//
// On SYSCALL entry (x86_64 ABI):
//   RAX = syscall number
//   RDI = arg0, RSI = arg1, RDX = arg2, R10 = arg3, R8 = arg4, R9 = arg5
//   RCX = saved user RIP  (by CPU)
//   R11 = saved user RFLAGS (by CPU)
//
// C calling convention (used by syscall_dispatch_extern):
//   RDI = arg0, RSI = arg1, RDX = arg2, RCX = arg3, R8 = arg4, R9 = arg5
//
// We shuffle: nr→RDI, arg0→RSI, arg1→RDX, arg2→RCX, arg3(R10)→R8, arg4(R8)→R9

#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(r#"
.global _syscall_entry
.align 16
_syscall_entry:
    swapgs
    mov     qword ptr gs:24, rsp        # save user RSP → PercpuData.user_rsp
    mov     rsp, qword ptr gs:16        # load kernel RSP

    push    rcx                         # save user RIP
    push    r11                         # save user RFLAGS
    push    r9                          # save arg5

    mov     r9,  r8
    mov     r8,  r10
    mov     rcx, rdx
    mov     rdx, rsi
    mov     rsi, rdi
    mov     rdi, rax

    call    syscall_dispatch_extern     # rax = syscall return value

    add     rsp, 8                      # discard arg5
    pop     r11                         # user RFLAGS
    pop     rcx                         # user RIP

    # ── Signal handler delivery ───────────────────────────────────────────
    # Pass user context to maybe_deliver_signal so it can push a sigframe.
    # Returns signum (>0) if delivering to a handler, 0 otherwise.
    push    rax                         # preserve syscall return value
    mov     rdi, rcx                    # arg0: user_rip
    mov     rsi, r11                    # arg1: user_rflags
    mov     rdx, qword ptr gs:24        # arg2: user_rsp
    mov     rcx, [rsp]                  # arg3: saved_rax (syscall return, for sigframe)
    call    maybe_deliver_signal        # rax = signum (0 = no delivery)
    test    rax, rax
    jz      .Lcheck_override
    mov     rdi, rax                    # rdi = signum → handler's first arg

.Lcheck_override:
    # Apply any pending context override (signal delivery or sigreturn).
    # signal_new_rip (gs:40) is set by maybe_deliver_signal and sys_rt_sigreturn.
    mov     r10, qword ptr gs:40
    test    r10, r10
    jz      .Lno_override
    mov     rcx, r10                    # override user RIP
    mov     r10, qword ptr gs:48
    mov     qword ptr gs:24, r10        # override user RSP
    mov     r11, qword ptr gs:56        # override user RFLAGS
    xor     r10, r10
    mov     qword ptr gs:40, r10        # clear signal_new_rip for next syscall
.Lno_override:
    pop     rax                         # restore syscall return value
    # ── End signal delivery ───────────────────────────────────────────────

    mov     rsp, qword ptr gs:24        # restore user RSP
    swapgs
    sysretq
"#);

// ── Initialise SYSCALL MSRs ───────────────────────────────────────────────────

/// Initialise GS base for the boot CPU.
/// Must be called BEFORE the LAPIC timer is started (interrupts::init) so that
/// the timer handler's `gs:[fpu_off]` read hits a valid per-CPU struct.
pub fn init_gs_base() {
    unsafe {
        let stack_top = BOOT_SYSCALL_STACK.0.as_ptr().add(SYSCALL_STACK_SIZE) as u64;
        BOOT_PERCPU.self_ptr     = &raw const BOOT_PERCPU as u64;
        BOOT_PERCPU.cpu_id       = 0;
        BOOT_PERCPU.kernel_rsp   = stack_top;
        BOOT_PERCPU.user_rsp     = 0;
        BOOT_PERCPU.ticks_per_ms = 0;
        BOOT_PERCPU.fpu_ptr      = 0; // no FPU context for the idle task
        hal::arch_x86_64::set_gs_base(&raw mut BOOT_PERCPU);
    }
}

/// Set up the SYSCALL/SYSRET fast path.
///
/// Must be called after `gdt::init()`. GS base must already be set (call
/// `init_gs_base()` before `interrupts::init()`).
pub fn init_lstar() {
    unsafe {
        // GS base already set by init_gs_base(); just configure SYSCALL MSRs.

        // ── STAR ─────────────────────────────────────────────────────────────
        // bits[47:32] = kernel CS selector → SS = that + 8 = kernel DS
        // bits[63:48] = base for SYSRET 64-bit:
        //   CS = bits[63:48] + 16 | 3 = user_code
        //   SS = bits[63:48] + 8  | 3 = user_data
        // Given our GDT layout (null, k-code 0x08, k-data 0x10, u-data 0x18, u-code 0x20):
        //   kernel CS = 0x08, kernel DS base for SYSRET = 0x10
        let kernel_cs = crate::gdt::kernel_cs().0 as u64;   // 0x0008
        let kernel_ds = crate::gdt::kernel_ds().0 as u64;   // 0x0010
        let star = (kernel_ds << 48) | (kernel_cs << 32);
        wrmsr(MSR_IA32_STAR, star);

        // ── LSTAR — syscall handler entry point ───────────────────────────────
        extern "C" { fn _syscall_entry(); }
        wrmsr(MSR_IA32_LSTAR, _syscall_entry as *const () as u64);

        // ── FMASK — clear IF (interrupts) on SYSCALL so kernel is uninterruptible
        //            at entry until we enable them explicitly.
        wrmsr(MSR_IA32_FMASK, 0x200);  // bit 9 = IF

        // ── EFER — set SCE (System Call Enable, bit 0) ────────────────────────
        let efer = rdmsr(MSR_IA32_EFER);
        wrmsr(MSR_IA32_EFER, efer | 1);
    }

    crate::klog!(INFO, "SYSCALL: LSTAR={:#x} STAR={:#x} FMASK=0x200",
        unsafe { rdmsr(MSR_IA32_LSTAR) },
        unsafe { rdmsr(MSR_IA32_STAR)  },
    );
}

// ── Dispatch function (called from asm) ──────────────────────────────────────

/// C-ABI syscall dispatch routine called by `_syscall_entry`.
///
/// # Safety
/// Called exclusively from ring-0 kernel context after a SYSCALL instruction.
/// User pointer arguments (buf_ptr, etc.) are validated before use.
#[no_mangle]
pub unsafe extern "C" fn syscall_dispatch_extern(
    nr:   u64,
    arg0: u64,
    arg1: u64,
    arg2: u64,
    arg3: u64,
    arg4: u64,
    arg5: u64,
) -> i64 {
    SYSCALL_COUNT.fetch_add(1, Ordering::Relaxed);
    let pid = crate::scheduler::current_pid();
    // Per-task histogram, anomaly detection, and transformer context (all O(1)).
    crate::syscall_stats::record(pid, nr);
    crate::transformer_sched::record_syscall(pid, nr);
    let (alert, score) = crate::anomaly::observe(pid, nr);
    if alert {
        crate::klog!(WARN, "ANOMALY: pid={} score={:.3} nr={}", pid, score, nr);
        // Feed into AI security pipeline.
        ai_subsystem::event_bus::publish(
            ai_subsystem::event_bus::KernelEvent::SyscallIssued { pid, syscall_nr: nr });
    }
    let result = match nr {
        // ── Phase 11 core ──────────────────────────────────────────────────
        nr::READ          => sys_read(arg0, arg1, arg2),
        nr::WRITE         => sys_write(arg0, arg1, arg2),
        nr::OPEN          => sys_open(arg0, arg1, arg2),
        nr::CLOSE         => sys_close(arg0),
        nr::STAT          => sys_stat(arg0, arg1, arg2),   // arg2 = buf (stat by path)
        nr::FSTAT         => sys_fstat(arg0, arg1),
        nr::LSTAT         => sys_stat(arg0, arg1, arg2),   // same as stat for now
        nr::LSEEK         => sys_lseek(arg0, arg1, arg2),
        nr::MMAP          => sys_mmap(arg0, arg1, arg2, arg3, arg4 as i64, arg5),
        nr::MPROTECT      => sys_mprotect(arg0, arg1, arg2),
        nr::MUNMAP        => sys_munmap(arg0, arg1),
        nr::GETPID        => sys_getpid(),
        nr::GETPPID       => sys_getppid(),
        nr::EXIT | nr::EXIT_GROUP => sys_exit(arg0 as i32),
        nr::GETDENTS64    => sys_getdents64(arg0, arg1, arg2),
        // ── Phase 18 process management ───────────────────────────────────
        nr::EXECVE        => sys_execve(arg0, arg1, arg2),
        nr::FORK          => sys_fork(),
        nr::WAIT4         => sys_wait4(arg0 as i32, arg1, arg2),
        nr::CLONE         => sys_clone(arg0, arg1, arg2, arg3, arg4),
        nr::KILL          => sys_kill(arg0 as i32, arg1 as i32),
        nr::TGKILL        => sys_kill(arg1 as i32, arg2 as i32), // same routing
        // ── Phase 18 memory ───────────────────────────────────────────────
        nr::BRK           => sys_brk(arg0),
        // ── Phase 18 I/O ──────────────────────────────────────────────────
        nr::READV         => sys_readv(arg0, arg1, arg2),
        nr::WRITEV        => sys_writev(arg0, arg1, arg2),
        nr::PREAD64       => sys_pread64(arg0, arg1, arg2, arg3),
        nr::PWRITE64      => sys_pwrite64(arg0, arg1, arg2, arg3),
        nr::IOCTL         => sys_ioctl(arg0, arg1, arg2),
        nr::FCNTL         => sys_fcntl(arg0, arg1, arg2),
        nr::DUP           => sys_dup(arg0),
        nr::DUP2          => sys_dup2(arg0, arg1),
        nr::DUP3          => sys_dup2(arg0, arg1), // flags ignored
        nr::PIPE          => sys_pipe2(arg0, 0),
        nr::PIPE2         => sys_pipe2(arg0, arg1),
        // ── Phase 18 signals ──────────────────────────────────────────────
        nr::RT_SIGACTION  => sys_rt_sigaction(arg0, arg1, arg2),
        nr::RT_SIGPROCMASK=> sys_rt_sigprocmask(arg0, arg1, arg2),
        nr::SIGALTSTACK   => sys_sigaltstack(arg0, arg1),
        // RT_SIGRETURN is dispatched later in the match (full implementation)
        // ── Phase 18 credentials ─────────────────────────────────────────
        nr::GETUID | nr::GETEUID => sys_getuid(),
        nr::GETGID | nr::GETEGID => sys_getgid(),
        nr::SETUID        => 0,  // always root for now
        nr::SETGID        => 0,
        nr::SETPGID       => 0,
        nr::SETSID        => sys_getpid(),
        nr::GETPGID       => sys_getpid(),
        nr::UMASK         => 0o022,
        // ── Phase 18/19 time ─────────────────────────────────────────────
        nr::NANOSLEEP     => sys_nanosleep(arg0, arg1),
        nr::CLOCK_GETTIME => sys_clock_gettime(arg0, arg1),
        nr::GETTIMEOFDAY  => sys_gettimeofday(arg0, arg1),
        // ── Phase 18/19 misc ─────────────────────────────────────────────
        nr::UNAME         => sys_uname(arg0),
        nr::PRCTL         => sys_prctl(arg0, arg1, arg2),
        nr::ARCH_PRCTL    => sys_arch_prctl(arg0, arg1),
        nr::GETRANDOM     => sys_getrandom(arg0, arg1, arg2),
        nr::SET_TID_ADDRESS => sys_getpid(), // return tid (same as pid) and store addr (ignore)
        nr::GETTID        => sys_getpid(),
        nr::SYSINFO       => sys_sysinfo(arg0),
        nr::GETRLIMIT | nr::PRLIMIT64 => sys_getrlimit(arg0, arg1),
        nr::SETRLIMIT     => 0,
        nr::MADVISE | nr::MINCORE | nr::MSYNC | nr::MLOCK | nr::MUNLOCK => 0, // safe no-ops
        nr::STATFS | nr::FSTATFS => sys_statfs(arg0, arg1),
        // ── Phase 19 futex / threading ───────────────────────────────────
        nr::FUTEX         => sys_futex(arg0, arg1 as i32, arg2 as u32, arg3),
        nr::SET_ROBUST_LIST => sys_set_robust_list(arg0, arg1 as usize),
        nr::GET_ROBUST_LIST => 0, // not needed for musl hello world
        // ── Phase 19 epoll / event I/O ───────────────────────────────────
        nr::EPOLL_CREATE1 => sys_epoll_create1(arg0),
        nr::EPOLL_CTL     => sys_epoll_ctl(arg0, arg1 as i32, arg2 as i32, arg3),
        nr::EPOLL_WAIT    => sys_epoll_wait(arg0, arg1, arg2 as i32, arg3 as i32),
        nr::EVENTFD2      => sys_eventfd2(arg0, arg1 as i32),
        nr::POLL          => sys_poll(arg0, arg1, arg2 as i32),
        nr::SELECT        => sys_select(arg0 as i32, arg1, arg2, arg3, arg4),
        // ── Phase 19 socket ──────────────────────────────────────────────
        nr::SOCKET        => sys_socket(arg0, arg1, arg2),
        nr::CONNECT       => sys_connect(arg0, arg1, arg2),
        nr::ACCEPT | nr::ACCEPT4 => sys_accept(arg0, arg1, arg2),
        nr::BIND          => sys_bind(arg0, arg1, arg2),
        nr::LISTEN        => sys_listen(arg0, arg1),
        nr::SENDTO        => sys_sendto(arg0, arg1, arg2, arg3, arg4, arg5),
        nr::RECVFROM      => sys_recvfrom(arg0, arg1, arg2, arg3, arg4, arg5),
        nr::SHUTDOWN      => 0,
        nr::GETSOCKNAME   => sys_getsockname(arg0, arg1, arg2),
        nr::GETPEERNAME   => sys_getpeername(arg0, arg1, arg2),
        nr::SETSOCKOPT    => sys_setsockopt(arg0, arg1, arg2, arg3, arg4),
        nr::GETSOCKOPT    => sys_getsockopt(arg0, arg1, arg2, arg3, arg4),
        // ── FS helpers ───────────────────────────────────────────────────
        nr::GETCWD        => sys_getcwd(arg0, arg1),
        nr::CHDIR         => sys_chdir(arg0, arg1),
        nr::MKDIR         => sys_mkdir(arg0, arg1, arg2),
        nr::MKDIRAT       => sys_mkdir(arg2, arg3, arg4),  // dirfd ignored
        nr::RMDIR         => sys_rmdir(arg0, arg1),
        nr::UNLINK        => sys_unlink(arg0, arg1),
        nr::UNLINKAT      => sys_unlink(arg1, arg2),       // dirfd ignored
        nr::RENAME        => sys_rename(arg0, arg1, arg2, arg3),
        nr::RENAMEAT | nr::RENAMEAT2 => sys_rename(arg1, arg2, arg3, arg4),
        nr::LINK          => 0, // stub — hard links not fully supported
        nr::SYMLINK       => 0, // stub
        nr::SYMLINKAT     => 0,
        nr::READLINK      => sys_readlink(arg0, arg1, arg2),
        nr::READLINKAT    => sys_readlink(arg1, arg2, arg3), // dirfd ignored
        nr::CREAT         => sys_open(arg0, arg1, 0o666),
        nr::OPENAT        => sys_open(arg1, arg2, arg3),   // dirfd ignored
        nr::FSTATAT       => sys_stat(arg1, arg2, arg3),   // dirfd ignored
        nr::FACCESSAT     => sys_access(arg1, arg2),
        nr::ACCESS        => sys_access(arg0, arg1),
        nr::STATX         => 0,
        nr::TRUNCATE | nr::FTRUNCATE => sys_ftruncate(arg0, arg1),
        nr::CHMOD | nr::FCHMOD | nr::CHOWN | nr::FCHOWN | nr::LCHOWN => 0,
        nr::MKNOD         => 0,
        nr::SENDFILE      => sys_sendfile(arg0, arg1, arg2, arg3),
        nr::MEMFD_CREATE  => EINVAL,
        nr::TIMES         => 0,
        nr::SETTIMEOFDAY  => 0,
        nr::ALARM         => 0,
        nr::PAUSE         => { crate::scheduler::yield_cpu(); -4 } // EINTR
        nr::GETGROUPS | nr::SETGROUPS => 0,
        // ── Phase 24 timerfd ─────────────────────────────────────────────
        nr::TIMERFD_CREATE  => sys_timerfd_create(arg0, arg1),
        nr::TIMERFD_SETTIME => sys_timerfd_settime(arg0, arg1, arg2, arg3),
        nr::TIMERFD_GETTIME => sys_timerfd_gettime(arg0, arg1),
        // ── Phase 24 inotify / signalfd (stubs) ──────────────────────────
        nr::INOTIFY_INIT1 | nr::INOTIFY_ADD_WATCH | nr::INOTIFY_RM_WATCH => {
            let pid = crate::scheduler::current_pid();
            alloc_fd(pid) as i64
        }
        nr::SIGNALFD4     => { let pid=crate::scheduler::current_pid(); alloc_fd(pid) as i64 }
        nr::CLONE3        => sys_fork(), // simplified clone3 — just fork
        // ── AI subsystem ─────────────────────────────────────────────────
        nr::RT_SIGRETURN  => sys_rt_sigreturn(),
        nr::AI_QUERY      => sys_ai_query(arg0, arg1),
        nr::AI_LOG        => sys_ai_log(arg0, arg1, arg2),
        nr::SYS_INTENT    => sys_intent(arg0, arg1),
        _                 => {
            crate::klog!(WARN, "SYSCALL: unimplemented nr={}", nr);
            ENOSYS
        },
    };

    // ── Signal delivery on syscall return ────────────────────────────────────
    // Check for pending unmasked signals before returning to user space.
    // Signals with user-space handlers require user-stack modification
    // (sigframe push) and are handled here for default-action signals only.
    // Handler-based delivery (push sigframe, redirect RIP) is wired
    // below via the asm wrapper once user RIP/RSP are accessible.
    deliver_pending_signals_default();

    result
}

/// Sigframe layout on the user stack (growing down from user_rsp).
/// Pushed by maybe_deliver_signal; popped by sys_rt_sigreturn.
///
/// [new_rsp + 0]:  VDSO_ADDR (restorer — handler's `ret` lands here)
/// [new_rsp + 8]:  signum
/// [new_rsp + 16]: saved rax (syscall return value, restored by sigreturn)
/// [new_rsp + 24]: saved user RIP
/// [new_rsp + 32]: saved user RFLAGS
/// [new_rsp + 40]: original user RSP (before sigframe push)
///
/// Total: 48 bytes.  new_rsp is chosen so (new_rsp % 16) == 8 for SysV alignment.
const SIGFRAME_SLOTS: usize = 6;

/// Called from _syscall_entry (naked asm) just before SYSRETQ.
/// Receives the user context that was current at syscall time.
/// If a user-space signal handler is pending, pushes a sigframe, sets the
/// percpu override fields, and returns the signum (> 0).
/// For default-action signals, terminates the process and never returns.
/// Returns 0 if no signal pending.
#[no_mangle]
pub unsafe extern "C" fn maybe_deliver_signal(
    user_rip:    u64,
    user_rflags: u64,
    user_rsp:    u64,
    saved_rax:   u64,
) -> u64 {
    let pid = crate::scheduler::current_pid();
    let (signum, handler) = match crate::scheduler::take_pending_signal(pid) {
        Some(s) => s,
        None    => return 0,
    };

    if handler == 0 {
        let exit_code = match signum {
            9  => { crate::klog!(INFO, "SIGKILL pid={}", pid); 128 + 9 }
            15 => { crate::klog!(INFO, "SIGTERM pid={}", pid); 128 + 15 }
            11 => { crate::klog!(WARN, "SIGSEGV pid={}", pid); 139 }
            8  => { crate::klog!(WARN, "SIGFPE  pid={}", pid);  136 }
            6  => { crate::klog!(WARN, "SIGABRT pid={}", pid);  134 }
            _  => return 0, // SIGCHLD, SIGURG etc. — silently drop
        };
        crate::scheduler::exit_current_direct(pid, exit_code);
    }

    // Build the sigframe on the user stack.
    // Align new_rsp so that (new_rsp % 16) == 8 (SysV: RSP before `push` of return addr).
    let frame_bytes = (SIGFRAME_SLOTS * 8) as u64;
    // Round down to 16-byte boundary, then subtract 8 so RSP%16 == 8 at handler entry
    // (SysV ABI: RSP is 16n-8 at a call site, because `call` pushes 8 bytes).
    let new_rsp = ((user_rsp - frame_bytes) & !15u64).wrapping_sub(8);

    let f = new_rsp as *mut u64;
    f.add(0).write(crate::memory::VDSO_ADDR); // restorer (ret target)
    f.add(1).write(signum as u64);             // signum
    f.add(2).write(saved_rax);                 // saved rax
    f.add(3).write(user_rip);                  // saved user RIP
    f.add(4).write(user_rflags);               // saved user RFLAGS
    f.add(5).write(user_rsp);                  // original user RSP

    // Set percpu override fields — picked up by _syscall_entry asm.
    let cpu = hal::arch_x86_64::gs_cpu_data();
    (*cpu).signal_new_rip    = handler;
    (*cpu).signal_new_rsp    = new_rsp;
    (*cpu).signal_new_rflags = 0x0202; // IF=1 for handler entry
    (*cpu).signal_signum     = signum as u64;

    crate::klog!(DEBUG, "signal {}: pid={} handler={:#x} sigframe={:#x}",
        signum, pid, handler, new_rsp);

    signum as u64
}

/// sys_rt_sigreturn — restores process state after a signal handler returns.
/// The handler's `ret` jumped to the vDSO trampoline, which issued this syscall.
/// At entry: user_rsp (from gs:24) = original new_rsp + 8 (ret consumed the restorer).
unsafe fn sys_rt_sigreturn() -> i64 {
    let cpu = hal::arch_x86_64::gs_cpu_data();
    // Recover sigframe base: user_rsp at syscall entry = new_rsp + 8.
    let user_rsp_now = (*cpu).user_rsp;
    let frame_base   = user_rsp_now - 8; // new_rsp (where frame starts)
    let f = frame_base as *const u64;

    // [+0] restorer (already consumed), [+1] signum, [+2] saved_rax,
    // [+3] saved_rip, [+4] saved_rflags, [+5] saved_user_rsp
    let saved_rax    = f.add(2).read();
    let saved_rip    = f.add(3).read();
    let saved_rflags = f.add(4).read();
    let saved_rsp    = f.add(5).read();

    // Restore the interrupted context via the same percpu override mechanism.
    (*cpu).signal_new_rip    = saved_rip;
    (*cpu).signal_new_rsp    = saved_rsp;
    (*cpu).signal_new_rflags = saved_rflags;

    crate::klog!(DEBUG, "sigreturn: rip={:#x} rsp={:#x}", saved_rip, saved_rsp);

    // Return the saved rax so the interrupted syscall's return value is preserved.
    saved_rax as i64
}

/// Handle default-action signals at syscall return (called from deliver_pending_signals_default).
/// Now a thin wrapper — real delivery happens in maybe_deliver_signal.
unsafe fn deliver_pending_signals_default() {
    // maybe_deliver_signal now handles both default and user-handler delivery.
    // This path is a no-op; signals are delivered by the _syscall_entry asm hook.
}

// ── Syscall implementations ───────────────────────────────────────────────────

/// Validate that a user-space pointer range is safe to dereference.
/// Returns `Err(EFAULT)` if the range overlaps kernel address space.
///
/// Convention: user-space addresses are below 0x0000_8000_0000_0000
/// (canonical user-space in x86_64 with 48-bit VA).
#[inline]
unsafe fn validate_user_ptr(ptr: u64, len: u64) -> Result<*const u8, i64> {
    const USER_END: u64 = 0x0000_8000_0000_0000;
    if ptr == 0 || ptr.saturating_add(len) > USER_END {
        return Err(EFAULT);
    }
    Ok(ptr as *const u8)
}

#[inline]
unsafe fn validate_user_ptr_mut(ptr: u64, len: u64) -> Result<*mut u8, i64> {
    validate_user_ptr(ptr, len).map(|p| p as *mut u8)
}

// ── sys_read ─────────────────────────────────────────────────────────────────

unsafe fn sys_read(fd: u64, buf_ptr: u64, len: u64) -> i64 {
    if len == 0 { return 0; }
    let safe_len = len.min(65536) as usize;
    let buf = match validate_user_ptr_mut(buf_ptr, safe_len as u64) {
        Ok(p) => p,
        Err(e) => return e,
    };

    match fd {
        0 => {
            // stdin: poll PS/2 keyboard
            if let Some(ev) = drivers::input::poll_event() {
                if let Some(ch) = ev.ascii {
                    core::ptr::write(buf, ch as u8);
                    return 1;
                }
            }
            0   // no data available
        }
        _ => {
            // Get handle pointer and drop the table lock before doing I/O
            // (avoids holding the lock across a potentially blocking VFS read).
            let pid = crate::scheduler::current_pid();
            let handle_ptr: usize = {
                let mut table = FD_TABLE.lock();
                match table.get_mut(&(pid, fd)) {
                    Some(h) => h as *mut alloc::boxed::Box<dyn crate::vfs::FileHandle> as usize,
                    None    => return EBADF,
                }
            };
            let h = &mut *(handle_ptr as *mut alloc::boxed::Box<dyn crate::vfs::FileHandle>);
            let mut tmp = alloc::vec![0u8; safe_len];
            match h.read(&mut tmp) {
                Ok(n) => {
                    core::ptr::copy_nonoverlapping(tmp.as_ptr(), buf, n);
                    // Syscall readahead: if this task is in an I/O-heavy cluster,
                    // tell intel_storage to prefetch the next window of this file.
                    // Zero-latency: queued for the next storage tick, not blocking.
                    if let Some((_, profile, _)) = crate::fingerprint::classify_task(pid) {
                        if profile.prefault_pages > 0 {
                            if let Some(path) = FD_PATH_TABLE.lock().get(&(pid, fd)).cloned() {
                                crate::intel_storage::readahead_for_cluster(
                                    &path, profile.prefault_pages);
                            }
                        }
                    }
                    n as i64
                }
                Err(_) => EINVAL,
            }
        }
    }
}

// ── sys_write ────────────────────────────────────────────────────────────────

unsafe fn sys_write(fd: u64, buf_ptr: u64, len: u64) -> i64 {
    if len == 0 { return 0; }
    let safe_len = len.min(65536) as usize;
    let buf = match validate_user_ptr(buf_ptr, safe_len as u64) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let slice = core::slice::from_raw_parts(buf, safe_len);

    match fd {
        1 | 2 => {
            // stdout / stderr → kernel log (serial + VGA)
            for &byte in slice {
                crate::logger::write_byte(byte);
            }
            safe_len as i64
        }
        _ => {
            let pid = crate::scheduler::current_pid();
            let handle_ptr: usize = {
                let mut table = FD_TABLE.lock();
                match table.get_mut(&(pid, fd)) {
                    Some(h) => h as *mut alloc::boxed::Box<dyn crate::vfs::FileHandle> as usize,
                    None    => return EBADF,
                }
            };
            let h = &mut *(handle_ptr as *mut alloc::boxed::Box<dyn crate::vfs::FileHandle>);
            match h.write(slice) {
                Ok(n)  => n as i64,
                Err(_) => EINVAL,
            }
        }
    }
}

// ── sys_getpid ───────────────────────────────────────────────────────────────

unsafe fn sys_getpid() -> i64 {
    // Return 1 (init process) until the scheduler exposes current PID.
    crate::scheduler::current_pid() as i64
}

// ── sys_exit ─────────────────────────────────────────────────────────────────

unsafe fn sys_exit(code: i32) -> i64 {
    crate::klog!(INFO, "sys_exit({}) — task terminating", code);
    crate::scheduler::exit_current(code);
    // exit_current does not return; return 0 to satisfy the type.
    0
}

// ── sys_ai_query ─────────────────────────────────────────────────────────────

// sys_ai_query query types:
//   0 = QUERY_STATUS:       returns AI subsystem health (audit count)
//   1 = QUERY_ANOMALY:      returns current pid's anomaly score × 1000 (i32 fixed-point)
//   2 = QUERY_SET_TUNABLE:  arg = ptr to null-terminated "name=value" string; applies live tunable
//   3 = QUERY_CLUSTER:      returns current pid's behavioral cluster ID [0-7]
//                           upper 8 bits: cluster nice_adjust (as u8, cast to i8 for sign)
//                           lower 8 bits: cluster label
//   4 = QUERY_WAKER:        returns the PID that most recently woke this task (or 0)
unsafe fn sys_ai_query(query_type: u64, arg: u64) -> i64 {
    match query_type {
        0 => ai_subsystem::audit::entry_count() as i64,
        1 => {
            let score = crate::anomaly::score(crate::scheduler::current_pid());
            (score * 1000.0) as i64
        }
        3 => {
            let pid = crate::scheduler::current_pid();
            match crate::fingerprint::classify_task(pid) {
                Some((cluster, profile, _)) => {
                    // Pack: [31:16] = cluster id, [15:8] = nice_adjust as u8, [7:0] = label
                    ((cluster as i64) << 16)
                        | (((profile.nice_adjust as u8) as i64) << 8)
                        | (profile.label as i64)
                }
                None => -1,
            }
        }
        4 => crate::causal::last_waker(crate::scheduler::current_pid()).unwrap_or(0) as i64,
        2 => {
            // Parse "name=value" from user pointer.
            let s = match read_user_cstr(arg, 256) {
                Some(s) => s,
                None    => return EFAULT,
            };
            if let Some(eq) = s.find('=') {
                let name  = &s[..eq];
                let value = s[eq+1..].parse::<i64>().unwrap_or(0);
                match crate::tunables::apply(name, value) {
                    Ok(v)  => { crate::klog!(INFO, "tunable {} = {}", name, v); v }
                    Err(e) => { crate::klog!(WARN, "tunable error: {}", e); EINVAL }
                }
            } else {
                EINVAL
            }
        }
        _ => {
            crate::klog!(DEBUG, "sys_ai_query(type={}) unknown", query_type);
            0
        }
    }
}

// ── sys_intent ───────────────────────────────────────────────────────────────
//
// sys_intent(intent_type, hint_value) — NodeAI-specific.
// Userspace declares its scheduling/I-O intent so the kernel can route resources
// optimally without guessing from syscall patterns alone.
//
// Intent types:
//   0 = INTENT_DEFAULT        (remove any previously set intent)
//   1 = INTENT_LATENCY        (latency-sensitive server: max priority, min quantum)
//   2 = INTENT_BATCH          (batch job: lower priority, larger quantum, greedy I/O)
//   3 = INTENT_INTERACTIVE    (interactive UI: boost after I/O wait)
//   4 = INTENT_IO_SEQUENTIAL  (hint: prefetch next pages; hint_value = stride bytes)
//   5 = INTENT_IO_RANDOM      (random I/O: don't prefetch)
//   6 = INTENT_MEMORY_LARGE   (will allocate large working set; hint_value = est. bytes)
//   7 = INTENT_CPU_BOUND      (no I/O wait expected; full quantum)

unsafe fn sys_intent(intent_type: u64, hint_value: u64) -> i64 {
    let pid = crate::scheduler::current_pid();
    crate::scheduler::set_intent(pid, intent_type as u8, hint_value);
    crate::fingerprint::label_from_intent(pid, intent_type as u8);
    crate::klog!(DEBUG, "sys_intent: pid={} type={} hint={}", pid, intent_type, hint_value);
    0
}

// ── sys_ai_log ───────────────────────────────────────────────────────────────

unsafe fn sys_ai_log(buf_ptr: u64, len: u64, _flags: u64) -> i64 {
    if len == 0 { return 0; }
    let safe_len = len.min(65536) as usize;
    let buf = match validate_user_ptr_mut(buf_ptr, safe_len as u64) {
        Ok(p)  => p,
        Err(e) => return e,
    };
    // Copy AI audit log entries into user buffer.
    // For now, write the count of audit entries as an ASCII string.
    let count = ai_subsystem::audit::entry_count();
    let msg = alloc::format!("ai_log_entries={}\n", count);
    let to_copy = msg.len().min(safe_len);
    core::ptr::copy_nonoverlapping(msg.as_ptr(), buf, to_copy);
    to_copy as i64
}

// ── sys_open ─────────────────────────────────────────────────────────────────

unsafe fn sys_open(path_ptr: u64, path_len: u64, _flags: u64) -> i64 {
    let safe_len = path_len.min(4096) as usize;
    let path_buf = match validate_user_ptr(path_ptr, safe_len as u64) {
        Ok(p)  => p,
        Err(e) => return e,
    };
    let path_bytes = core::slice::from_raw_parts(path_buf, safe_len);
    let path_str   = match core::str::from_utf8(path_bytes) {
        Ok(s)  => s.trim_end_matches('\0'),
        Err(_) => return EINVAL,
    };

    let node = match crate::vfs::lookup(path_str) {
        Ok(n)  => n,
        Err(_) => return ENOENT,
    };
    let handle = match node.open() {
        Ok(h)  => h,
        Err(_) => return ENOENT,
    };
    let pid = crate::scheduler::current_pid();
    let fd  = alloc_fd(pid);
    // For directories: also store the VfsNode for getdents64 readdir calls.
    let is_dir = node.stat().map(|s| s.is_dir).unwrap_or(false);
    if is_dir {
        DIR_NODES.lock().insert((pid, fd), node);
    }
    FD_TABLE.lock().insert((pid, fd), handle);
    FD_PATH_TABLE.lock().insert((pid, fd), alloc::string::String::from(path_str));
    fd as i64
}

// ── sys_close ────────────────────────────────────────────────────────────────

unsafe fn sys_close(fd: u64) -> i64 {
    let pid = crate::scheduler::current_pid();
    DIR_NODES.lock().remove(&(pid, fd));
    FD_PATH_TABLE.lock().remove(&(pid, fd));
    EPOLL_TABLE.lock().remove(&(pid, fd)); // no-op if not an epoll fd
    if FD_TABLE.lock().remove(&(pid, fd)).is_some() { 0 } else { EBADF }
}

// ── sys_fstat ────────────────────────────────────────────────────────────────
//
// Writes a compact NodeAI stat structure (64 bytes) to the user pointer:
//   offset  0: ino     (u64)
//   offset  8: size    (u64)
//   offset 16: mode    (u32) — 0x41FF = dir, 0x81B6 = regular
//   offset 20: nlink   (u32)
//   remaining bytes padded to 64 bytes.

unsafe fn sys_fstat(fd: u64, stat_ptr: u64) -> i64 {
    // stdin/stdout/stderr have a synthetic stat
    let pid = crate::scheduler::current_pid();
    let stat_bytes = match validate_user_ptr_mut(stat_ptr, 64) {
        Ok(p)  => p,
        Err(e) => return e,
    };
    // Zero the stat buffer first
    core::ptr::write_bytes(stat_bytes, 0, 64);

    let stat = if fd < 3 {
        crate::vfs::Stat { ino: fd + 1, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666 }
    } else {
        let mut table = FD_TABLE.lock();
        match table.get_mut(&(pid, fd)) {
            Some(h) => match h.stat() {
                Ok(s)  => s,
                Err(_) => return EBADF,
            },
            None => return EBADF,
        }
    };

    // Write fields at fixed offsets
    let ino_ptr   = stat_bytes as *mut u64;
    let size_ptr  = stat_bytes.add(8) as *mut u64;
    let mode_ptr  = stat_bytes.add(16) as *mut u32;
    let nlink_ptr = stat_bytes.add(20) as *mut u32;
    ino_ptr.write_unaligned(stat.ino);
    size_ptr.write_unaligned(stat.size);
    mode_ptr.write_unaligned(if stat.is_dir { 0x41FF } else { 0x81B6 });
    nlink_ptr.write_unaligned(stat.nlink);
    0
}

// ── sys_lseek ────────────────────────────────────────────────────────────────

unsafe fn sys_lseek(fd: u64, offset: u64, _whence: u64) -> i64 {
    let pid = crate::scheduler::current_pid();
    let mut table = FD_TABLE.lock();
    match table.get_mut(&(pid, fd)) {
        Some(h) => match h.seek(offset) {
            Ok(pos) => pos as i64,
            Err(_)  => EINVAL,
        },
        None => EBADF,
    }
}

// ── sys_mmap ─────────────────────────────────────────────────────────────────
//
// Anonymous mmap only (fd == -1 or ignored).  Allocates zeroed pages in user VA.
// Returns the mapped virtual address on success, or a negative errno.

const MMAP_PROT_WRITE: u64 = 0x2;
const MMAP_PROT_EXEC:  u64 = 0x4;

// Next anonymous mmap base address (below 128 TiB, well under kernel space).
static MMAP_NEXT_ADDR: AtomicU64 = AtomicU64::new(0x0000_1000_0000_0000u64);

unsafe fn sys_mmap(addr: u64, len: u64, prot: u64, flags: u64, _fd: i64, _offset: u64) -> i64 {
    if len == 0 { return EINVAL; }
    // Round len up to page size
    let page_size: u64 = 4096;
    let len_aligned = (len + page_size - 1) & !(page_size - 1);

    let writable   = prot & MMAP_PROT_WRITE != 0;
    let executable = prot & MMAP_PROT_EXEC  != 0;

    // MAP_FIXED: map at exact address requested
    const MAP_FIXED: u64 = 0x10;
    let vaddr = if flags & MAP_FIXED != 0 && addr != 0 {
        addr
    } else {
        // Choose a free address (ignore addr hint for simplicity)
        MMAP_NEXT_ADDR.fetch_add(len_aligned + page_size, Ordering::Relaxed)
    };

    match crate::memory::map_user_range(vaddr, len_aligned, writable, executable) {
        Ok(()) => {
            // Predictive prefault: use the caller's fingerprint cluster to
            // determine how many extra pages to pre-fault beyond the requested
            // range. This eliminates page-fault latency on first access for
            // I/O-heavy and batch workloads.
            let pid = crate::scheduler::current_pid();
            if let Some((_, profile, _)) = crate::fingerprint::classify_task(pid) {
                let extra = profile.prefault_pages as u64;
                if extra > 0 {
                    let prefault_base  = vaddr + len_aligned;
                    let prefault_bytes = extra * page_size;
                    // Extend the mapping — ignore errors (best-effort).
                    let _ = crate::memory::map_user_range(
                        prefault_base, prefault_bytes, writable, false);
                    crate::klog!(TRACE,
                        "mmap prefault: pid={} base={:#x} +{}p cluster",
                        pid, prefault_base, extra);
                }
            }
            vaddr as i64
        }
        Err(_) => EINVAL,
    }
}

// ── sys_munmap ───────────────────────────────────────────────────────────────

unsafe fn sys_munmap(addr: u64, len: u64) -> i64 {
    if len == 0 { return 0; }
    use x86_64::structures::paging::{Page, Size4KiB};
    use x86_64::VirtAddr;

    let page_sz = crate::memory::PAGE_SIZE;
    let addr_aligned = addr & !(page_sz - 1);
    let pages = (len + page_sz - 1) / page_sz;

    for i in 0..pages {
        let virt = addr_aligned + i * page_sz;
        let page: Page<Size4KiB> = Page::containing_address(VirtAddr::new(virt));
        if let Ok(frame) = crate::memory::unmap_page(page) {
            // Free the backing physical frame.
            unsafe { crate::memory::free_frame(frame.start_address().as_u64()); }
        }
        // If unmap_page fails the page was never mapped — not an error (POSIX says so).
    }
    0
}

// ── sys_getdents64 ───────────────────────────────────────────────────────────
//
// linux_dirent64 layout (packed, variable-length records):
//   u64  d_ino     (8 bytes)
//   i64  d_off     (8 bytes — offset of NEXT entry, or u64::MAX for last)
//   u16  d_reclen  (2 bytes — total record size including name + null + padding)
//   u8   d_type    (1 byte  — DT_REG=8, DT_DIR=4, DT_UNKNOWN=0)
//   char d_name[]  (null-terminated, padded to 8-byte alignment)
//   Total header before name: 19 bytes.
unsafe fn sys_getdents64(fd: u64, buf_ptr: u64, buf_len: u64) -> i64 {
    let pid = crate::scheduler::current_pid();
    // Look up the VfsNode stored at open() time.
    let node = {
        let nodes = DIR_NODES.lock();
        nodes.get(&(pid, fd)).cloned()
    };
    let node = match node {
        Some(n) => n,
        None    => return EBADF,
    };
    let entries = match node.readdir() {
        Ok(e)  => e,
        Err(_) => return ENOTDIR,
    };

    let buf = match validate_user_ptr_mut(buf_ptr, buf_len) {
        Ok(p)  => p,
        Err(e) => return e,
    };
    let buf_end = buf_ptr + buf_len;
    let mut pos: u64 = buf_ptr;
    let mut total: i64 = 0;

    for (idx, entry) in entries.iter().enumerate() {
        let name_bytes = entry.name.as_bytes();
        let name_len   = name_bytes.len();
        // reclen = 8+8+2+1 + name_len+1, rounded up to next 8-byte boundary
        let raw_len = 19usize + name_len + 1;
        let reclen  = (raw_len + 7) & !7;
        if pos + reclen as u64 > buf_end { break; }

        let p = pos as *mut u8;
        // d_ino (8)
        (p as *mut u64).write_unaligned(entry.ino);
        // d_off (8) — offset of next record
        ((p as u64 + 8) as *mut i64).write_unaligned(
            (idx + 1) as i64 * reclen as i64
        );
        // d_reclen (2)
        ((p as u64 + 16) as *mut u16).write_unaligned(reclen as u16);
        // d_type (1)
        *p.add(18) = if entry.is_dir { 4 } else { 8 };
        // d_name (null-terminated, zero-padded to reclen)
        let name_dst = p.add(19);
        core::ptr::write_bytes(name_dst, 0, reclen - 19);
        core::ptr::copy_nonoverlapping(name_bytes.as_ptr(), name_dst, name_len);

        pos   += reclen as u64;
        total += reclen as i64;
    }
    total
}

const ENOTDIR: i64 = -20;

// ─────────────────────────────────────────────────────────────────────────────
//  Phase 18 — Ring-3 Process Launch
// ─────────────────────────────────────────────────────────────────────────────

// ── Helper: read null-terminated string from user space ──────────────────────
unsafe fn read_user_cstr(ptr: u64, max: usize) -> Option<alloc::string::String> {
    if ptr == 0 { return None; }
    let bytes = core::slice::from_raw_parts(ptr as *const u8, max);
    let len   = bytes.iter().position(|&b| b == 0).unwrap_or(max);
    core::str::from_utf8(&bytes[..len]).ok().map(|s| alloc::string::String::from(s))
}

// ── Helper: jump to ring-3 via IRETQ ─────────────────────────────────────────
/// Transfer execution to user space.  Never returns.
unsafe fn ring3_jump(entry: u64, user_rsp: u64) -> ! {
    // user segment selectors (RPL=3)
    let user_cs: u64 = (crate::gdt::user_cs().0 | 3) as u64;
    let user_ss: u64 = (crate::gdt::user_ds().0 | 3) as u64;
    let rflags:  u64 = 0x0000_0202;   // IF=1, reserved bit 1

    crate::klog!(INFO, "ring3_jump: entry={:#x} rsp={:#x}", entry, user_rsp);

    core::arch::asm!(
        // Undo the swapgs done in _syscall_entry so the user sees GS=0.
        "swapgs",
        // Build IRETQ frame: [SS, RSP, RFLAGS, CS, RIP]
        "push {ss}",
        "push {rsp}",
        "push {rflags}",
        "push {cs}",
        "push {rip}",
        "iretq",
        ss     = in(reg) user_ss,
        rsp    = in(reg) user_rsp,
        rflags = in(reg) rflags,
        cs     = in(reg) user_cs,
        rip    = in(reg) entry,
        options(noreturn)
    );
}

// ── sys_execve ───────────────────────────────────────────────────────────────
const USER_STACK_TOP:  u64 = 0x0000_7FFF_FFFF_F000;
const USER_STACK_SIZE: u64 = 8 * 1024 * 1024;

unsafe fn sys_execve(path_ptr: u64, argv_ptr: u64, envp_ptr: u64) -> i64 {
    // 1. Read path
    let path = match read_user_cstr(path_ptr, 4096) {
        Some(p) => p,
        None    => return EFAULT,
    };
    crate::klog!(INFO, "sys_execve: loading '{}'", path);

    // 2. Read ELF from VFS
    let node = match crate::vfs::lookup(&path) {
        Ok(n)  => n,
        Err(_) => return ENOENT,
    };
    let mut handle = match node.open() {
        Ok(h)  => h,
        Err(_) => return ENOENT,
    };
    let mut elf_data: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        match handle.read(&mut tmp) {
            Ok(0) => break,
            Ok(n) => elf_data.extend_from_slice(&tmp[..n]),
            Err(_) => return EINVAL,
        }
    }

    // 3a. Allocate a fresh address space for this process.
    let new_cr3 = match crate::memory::alloc_user_cr3() {
        Some(cr3) => cr3,
        None      => return EINVAL,
    };
    // Switch to the new CR3 so ELF segment mappings land in this process's space.
    core::arch::asm!("mov cr3, {}", in(reg) new_cr3, options(nomem, nostack));
    // Update task's CR3 field.
    let pid = crate::scheduler::current_pid();
    crate::scheduler::set_task_cr3(pid, new_cr3);
    crate::klog!(INFO, "execve: new address space cr3={:#x}", new_cr3);

    // 3b. Parse + load ELF into the new address space.
    let image = match crate::elf::parse(&elf_data) {
        Ok(img) => img,
        Err(e)  => { crate::klog!(WARN, "execve: ELF parse error {:?}", e); return EINVAL; }
    };
    if let Err(e) = crate::elf::load_image(&image) {
        crate::klog!(WARN, "execve: ELF load error {:?}", e);
        return EINVAL;
    }
    let entry = image.entry;

    // 4. Map user stack in the new address space.
    let stack_base = USER_STACK_TOP - USER_STACK_SIZE;
    if let Err(_) = crate::memory::map_user_range(stack_base, USER_STACK_SIZE, true, false) {
        return EINVAL;
    }

    // 5. Collect argv strings
    let mut argv_strs: alloc::vec::Vec<alloc::vec::Vec<u8>> = alloc::vec::Vec::new();
    {
        let mut v = alloc::vec::Vec::from(path.as_bytes());
        v.push(0);
        argv_strs.push(v);
    }
    if argv_ptr != 0 {
        let arr = argv_ptr as *const u64;
        let mut i = 1usize;
        loop {
            if i > 256 { break; }
            let ptr = *arr.add(i);
            if ptr == 0 { break; }
            if let Some(s) = read_user_cstr(ptr, 4096) {
                let mut v = alloc::vec::Vec::from(s.as_bytes());
                v.push(0);
                argv_strs.push(v);
            }
            i += 1;
        }
    }

    // 6. Collect envp strings
    let mut envp_strs: alloc::vec::Vec<alloc::vec::Vec<u8>> = alloc::vec::Vec::new();
    if envp_ptr != 0 {
        let arr = envp_ptr as *const u64;
        let mut i = 0usize;
        loop {
            if i > 256 { break; }
            let ptr = *arr.add(i);
            if ptr == 0 { break; }
            if let Some(s) = read_user_cstr(ptr, 4096) {
                let mut v = alloc::vec::Vec::from(s.as_bytes());
                v.push(0);
                envp_strs.push(v);
            }
            i += 1;
        }
    }
    if envp_strs.is_empty() {
        envp_strs.push(b"PATH=/bin:/usr/bin:/sbin:/usr/sbin\0".to_vec());
        envp_strs.push(b"HOME=/root\0".to_vec());
        envp_strs.push(b"TERM=vt100\0".to_vec());
    }

    // 7. Build SysV AMD64 user stack
    let mut sp = USER_STACK_TOP;

    // 7a. Write string data (descending)
    let mut argv_ptrs: alloc::vec::Vec<u64> = alloc::vec::Vec::new();
    for s in &argv_strs {
        sp -= s.len() as u64;
        core::ptr::copy_nonoverlapping(s.as_ptr(), sp as *mut u8, s.len());
        argv_ptrs.push(sp);
    }
    let mut envp_ptrs: alloc::vec::Vec<u64> = alloc::vec::Vec::new();
    for s in &envp_strs {
        sp -= s.len() as u64;
        core::ptr::copy_nonoverlapping(s.as_ptr(), sp as *mut u8, s.len());
        envp_ptrs.push(sp);
    }

    // 7b. 16 random bytes for AT_RANDOM
    sp -= 16;
    sp &= !15u64;
    let rand_ptr = sp;
    let tsc: u64;
    core::arch::asm!("rdtsc; shl rdx, 32; or rax, rdx",
        out("rax") tsc, out("rdx") _, options(nomem, nostack));
    *(rand_ptr as *mut u64)       = tsc;
    *((rand_ptr + 8) as *mut u64) = tsc ^ 0xDEAD_BEEF_CAFE_BABEu64;

    sp &= !7u64;

    let n_pointers = 1 + argv_ptrs.len() + 1 + envp_ptrs.len() + 1 + 10;
    if (sp.wrapping_sub((n_pointers as u64) * 8)) & 0xF != 0 {
        sp -= 8;
    }

    macro_rules! push64 {
        ($v:expr) => {{
            sp -= 8;
            *(sp as *mut u64) = $v as u64;
        }};
    }

    // auxv
    push64!(0u64); push64!(0u64);          // AT_NULL
    push64!(0u64); push64!(16u64);         // AT_HWCAP
    push64!(rand_ptr); push64!(25u64);     // AT_RANDOM
    push64!(entry); push64!(9u64);         // AT_ENTRY
    push64!(4096u64); push64!(6u64);       // AT_PAGESZ

    push64!(0u64);  // envp null
    for &ptr in envp_ptrs.iter().rev() { push64!(ptr); }
    push64!(0u64);  // argv null
    for &ptr in argv_ptrs.iter().rev() { push64!(ptr); }
    push64!(argv_strs.len() as u64);  // argc

    crate::klog!(INFO, "execve: entry={:#x} rsp={:#x} argc={}",
        entry, sp, argv_strs.len());

    // 8. Reset brk
    let pid = crate::scheduler::current_pid();
    crate::scheduler::set_user_brk(pid, 0);

    // 9. Jump to user mode
    ring3_jump(entry, sp);
}

// ── sys_fork ─────────────────────────────────────────────────────────────────
unsafe fn sys_fork() -> i64 {
    let parent_pid = crate::scheduler::current_pid();
    match crate::scheduler::fork_task(parent_pid) {
        Some(child_pid) => child_pid as i64,
        None            => EINVAL,
    }
}

// ── sys_wait4 ────────────────────────────────────────────────────────────────
unsafe fn sys_wait4(_pid: i32, wstatus_ptr: u64, options: u64) -> i64 {
    const WNOHANG: u64 = 1;
    let parent_pid = crate::scheduler::current_pid();

    // Check immediately for a zombie child.
    if let Some((child_pid, code)) = crate::scheduler::reap_zombie_child(parent_pid) {
        if wstatus_ptr != 0 {
            if let Ok(p) = validate_user_ptr_mut(wstatus_ptr, 4) {
                let status: i32 = (code & 0xFF) << 8;
                *(p as *mut i32) = status;
            }
        }
        return child_pid as i64;
    }

    // WNOHANG: don't block, return 0 immediately if no zombie.
    if options & WNOHANG != 0 {
        return 0;
    }

    // Sleep until a child exits and wakes us.
    crate::scheduler::sleep_current();
    crate::scheduler::yield_cpu();

    // Re-check after wakeup.
    match crate::scheduler::reap_zombie_child(parent_pid) {
        Some((child_pid, code)) => {
            if wstatus_ptr != 0 {
                if let Ok(p) = validate_user_ptr_mut(wstatus_ptr, 4) {
                    let status: i32 = (code & 0xFF) << 8;
                    *(p as *mut i32) = status;
                }
            }
            child_pid as i64
        }
        None => -10, // ECHILD (no children)
    }
}

// ── sys_clone ────────────────────────────────────────────────────────────────
const CLONE_VM:      u64 = 0x00000100; // share address space
const CLONE_THREAD:  u64 = 0x00010000; // same thread group
const CLONE_SETTLS:  u64 = 0x00080000; // set TLS from argument

unsafe fn sys_clone(flags: u64, stack_ptr: u64, _ptid: u64, tls: u64, _ctid: u64) -> i64 {
    let parent_pid = crate::scheduler::current_pid();
    if (flags & CLONE_VM) != 0 {
        // Thread: shares address space (CR3 not copied), gets new stack.
        let settls = (flags & CLONE_SETTLS) != 0;
        match crate::scheduler::spawn_user_thread(parent_pid, stack_ptr, tls, settls) {
            Some(tid) => tid as i64,
            None      => EINVAL,
        }
    } else {
        sys_fork()
    }
}

// ── sys_kill ─────────────────────────────────────────────────────────────────
unsafe fn sys_kill(pid: i32, sig: i32) -> i64 {
    crate::klog!(DEBUG, "sys_kill(pid={}, sig={})", pid, sig);
    if sig <= 0 || sig > 64 { return EINVAL; }
    let target = if pid > 0 {
        pid as u64
    } else if pid == 0 {
        crate::scheduler::current_pid() // send to self's process group (simplified: self)
    } else {
        return EINVAL; // broadcast/negative pid groups not implemented
    };
    crate::scheduler::send_signal(target, sig as u8);
    0
}

// ── sys_brk ──────────────────────────────────────────────────────────────────
const BRK_START: u64 = 0x0000_0040_0000_0000;

unsafe fn sys_brk(addr: u64) -> i64 {
    let pid      = crate::scheduler::current_pid();
    let cur_brk  = crate::scheduler::get_user_brk(pid);
    let base_brk = if cur_brk == 0 { BRK_START } else { cur_brk };

    if addr == 0 || addr <= base_brk {
        return base_brk as i64;
    }

    let page_size: u64 = 4096;
    let new_aligned = (addr + page_size - 1) & !(page_size - 1);
    let size = new_aligned - base_brk;

    match crate::memory::map_user_range(base_brk, size, true, false) {
        Ok(())  => { crate::scheduler::set_user_brk(pid, new_aligned); new_aligned as i64 }
        Err(_)  => base_brk as i64,
    }
}

// ── sys_mprotect ─────────────────────────────────────────────────────────────
unsafe fn sys_mprotect(_addr: u64, _len: u64, _prot: u64) -> i64 { 0 }

// ── sys_getppid ──────────────────────────────────────────────────────────────
unsafe fn sys_getppid() -> i64 {
    crate::scheduler::get_parent_pid(crate::scheduler::current_pid()) as i64
}

// ── sys_getuid / getgid ──────────────────────────────────────────────────────
unsafe fn sys_getuid() -> i64 {
    let (uid, _, _, _) = crate::scheduler::get_credentials(crate::scheduler::current_pid());
    uid as i64
}
unsafe fn sys_getgid() -> i64 {
    let (_, gid, _, _) = crate::scheduler::get_credentials(crate::scheduler::current_pid());
    gid as i64
}

// ── sys_uname ────────────────────────────────────────────────────────────────
unsafe fn sys_uname(buf_ptr: u64) -> i64 {
    const ULEN: usize = 65;
    let total: u64 = 6 * ULEN as u64;
    let p = match validate_user_ptr_mut(buf_ptr, total) {
        Ok(p)  => p,
        Err(e) => return e,
    };
    core::ptr::write_bytes(p, 0, total as usize);
    let fields: [&[u8]; 6] = [
        b"NodeAI", b"nodeai", b"1.0.0-nodeai",
        b"NodeAI Kernel 1.0.0", b"x86_64", b"nodeai.local",
    ];
    for (i, field) in fields.iter().enumerate() {
        let off = i * ULEN;
        let len = field.len().min(ULEN - 1);
        core::ptr::copy_nonoverlapping(field.as_ptr(), p.add(off), len);
    }
    0
}

// ── sys_nanosleep ─────────────────────────────────────────────────────────────
// Previously busy-spun — now yields the CPU per tick so other tasks run.
unsafe fn sys_nanosleep(req_ptr: u64, _rem_ptr: u64) -> i64 {
    if req_ptr == 0 { return EFAULT; }
    let secs  = *(req_ptr as *const u64);
    let nsecs = *((req_ptr + 8) as *const u64);
    let ms    = secs * 1000 + nsecs / 1_000_000;
    if ms == 0 { return 0; }
    let deadline = crate::scheduler::uptime_ms() + ms;
    while crate::scheduler::uptime_ms() < deadline {
        crate::scheduler::yield_cpu(); // hand CPU to other tasks while sleeping
    }
    0
}

// ── sys_clock_gettime ────────────────────────────────────────────────────────
unsafe fn sys_clock_gettime(_clk_id: u64, tp_ptr: u64) -> i64 {
    if tp_ptr == 0 { return EFAULT; }
    let ms = crate::scheduler::uptime_ms();
    *(tp_ptr as *mut u64)       = ms / 1000;
    *((tp_ptr + 8) as *mut u64) = (ms % 1000) * 1_000_000;
    0
}

// ── sys_gettimeofday ─────────────────────────────────────────────────────────
unsafe fn sys_gettimeofday(tv_ptr: u64, _tz: u64) -> i64 {
    if tv_ptr == 0 { return 0; }
    let ms = crate::scheduler::uptime_ms();
    *(tv_ptr as *mut u64)         = ms / 1000;
    *((tv_ptr + 8) as *mut u64)   = (ms % 1000) * 1000;
    0
}

// ── sys_getrandom ────────────────────────────────────────────────────────────
unsafe fn sys_getrandom(buf_ptr: u64, len: u64, _flags: u64) -> i64 {
    let safe_len = len.min(256) as usize;
    let p = match validate_user_ptr_mut(buf_ptr, safe_len as u64) {
        Ok(p)  => p,
        Err(e) => return e,
    };
    for i in (0..safe_len).step_by(8) {
        let mut rand: u64 = 0;
        let mut ok: u8    = 0;
        core::arch::asm!(
            "rdrand {val}", "setc {ok}",
            val = out(reg) rand, ok = out(reg_byte) ok,
            options(nomem, nostack)
        );
        let val = if ok != 0 {
            rand
        } else {
            let tsc: u64;
            core::arch::asm!("rdtsc; shl rdx, 32; or rax, rdx",
                out("rax") tsc, out("rdx") _, options(nomem, nostack));
            tsc ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        };
        let remaining = (safe_len - i).min(8);
        core::ptr::copy_nonoverlapping(
            &val as *const u64 as *const u8, p.add(i), remaining);
    }
    safe_len as i64
}

// ── sys_ioctl ────────────────────────────────────────────────────────────────
const TIOCGWINSZ: u64 = 0x5413;
const TCGETS:     u64 = 0x5401;
const TCSETS:     u64 = 0x5402;
// OSS /dev/dsp ioctls
const SNDCTL_DSP_RESET:    u64 = 0x00005000;
const SNDCTL_DSP_SYNC:     u64 = 0x00005001;
const SNDCTL_DSP_SPEED:    u64 = 0xC0045002;
const SNDCTL_DSP_STEREO:   u64 = 0xC0045003;
const SNDCTL_DSP_GETBLKSIZE: u64 = 0xC0045004;
const SNDCTL_DSP_SETFMT:   u64 = 0xC0045005;
const SNDCTL_DSP_CHANNELS: u64 = 0xC0045006;
const SNDCTL_DSP_POST:     u64 = 0x00005008;
const SNDCTL_DSP_SETFRAGMENT: u64 = 0xC004500A;
const SNDCTL_DSP_GETFMTS:  u64 = 0x8004500B;
const SNDCTL_DSP_GETOSPACE: u64 = 0x800C5012;
const SOUND_MIXER_WRITE_VOLUME: u64 = 0xC0044D00;
const SOUND_MIXER_READ_VOLUME:  u64 = 0x80044D00;

unsafe fn sys_ioctl(_fd: u64, request: u64, arg: u64) -> i64 {
    use crate::desktop::{
        COMPOSER_CREATE_WINDOW, COMPOSER_DESTROY_WINDOW, COMPOSER_FLIP,
        COMPOSER_MOVE, COMPOSER_RESIZE, COMPOSER_SET_TITLE,
        wm_create_window, wm_destroy_window, wm_flip,
        wm_composite,
    };
    match request {
        // ── Composer IOCTLs ────────────────────────────────────────────────────
        COMPOSER_CREATE_WINDOW => {
            // arg → *const ComposerCreateArgs  { x: i32, y: i32, w: u32, h: u32,
            //                                    title_ptr: u64, title_len: u64 }
            if arg == 0 { return -22; } // EINVAL
            let p = arg as *const i32;
            let x = *p;
            let y = *p.add(1);
            let w = *(p.add(2) as *const u32);
            let h = *(p.add(3) as *const u32);
            let title_ptr = *(p.add(4) as *const u64);
            let title_len = *(p.add(5) as *const u64) as usize;
            let title = if title_ptr != 0 && title_len > 0 {
                let sl = core::slice::from_raw_parts(title_ptr as *const u8, title_len.min(63));
                core::str::from_utf8(sl).unwrap_or("window")
            } else { "window" };
            let id = wm_create_window(x, y, w, h, title);
            wm_composite();
            id as i64
        }
        COMPOSER_DESTROY_WINDOW => {
            wm_destroy_window(arg as u32);
            0
        }
        COMPOSER_FLIP => {
            wm_flip(arg as u32);
            0
        }
        COMPOSER_MOVE => {
            if arg == 0 { return -22; }
            // arg → *const [u32/i32; 3]  { id: u32, x: i32, y: i32 }
            let id = *(arg as *const u32);
            let x  = *(arg as *const i32).add(1);
            let y  = *(arg as *const i32).add(2);
            crate::desktop::compositor::with_wm_pub(|s| {
                if let Some(w) = s.windows.get_mut(&id) { w.x = x; w.y = y; }
            });
            wm_composite();
            0
        }
        COMPOSER_RESIZE => {
            if arg == 0 { return -22; }
            let id = *(arg as *const u32);
            let nw = *(arg as *const u32).add(1);
            let nh = *(arg as *const u32).add(2);
            crate::desktop::compositor::with_wm_pub(|s| {
                if let Some(w) = s.windows.get_mut(&id) {
                    w.w = nw; w.h = nh;
                    w.pixels.resize((nw * nh) as usize, 0xFF_22_22_2A);
                }
            });
            wm_composite();
            0
        }
        COMPOSER_SET_TITLE => {
            if arg == 0 { return -22; }
            // arg → *const [u32; 2 + len/4]  { id: u32, title_len: u32, title_bytes }
            let id  = *(arg as *const u32);
            let len = *(arg as *const u32).add(1) as usize;
            let ptr = (arg as *const u8).add(8);
            let s   = core::slice::from_raw_parts(ptr, len.min(63));
            let title = core::str::from_utf8(s).unwrap_or("?");
            crate::desktop::wm_set_title(id, title);
            0
        }
        // ── Terminal IOCTLs ────────────────────────────────────────────────────
        TIOCGWINSZ => {
            if arg != 0 {
                let p = arg as *mut u16;
                *p        = 24; *p.add(1) = 80;
                *p.add(2) = 0;  *p.add(3) = 0;
            }
            0
        }
        TCGETS => {
            if arg != 0 {
                core::ptr::write_bytes(arg as *mut u8, 0, 60);
                let p = arg as *mut u32;
                *p          = 0x500;   // ICRNL|IXON
                *p.add(1)   = 0x5;     // OPOST|ONLCR
                *p.add(2)   = 0xBF;    // CS8|CREAD|HUPCL|CLOCAL
                *p.add(3)   = 0x8A3B;  // ECHO|ECHOE|ICANON|ISIG|IEXTEN
            }
            0
        }
        TCSETS => 0,
        // ── OSS audio ioctls ───────────────────────────────────────────────────
        SNDCTL_DSP_RESET => { 0 }
        SNDCTL_DSP_SYNC  => { 0 }
        SNDCTL_DSP_POST  => { 0 }
        SNDCTL_DSP_SPEED => {
            // write-back the accepted sample rate
            if arg != 0 { *(arg as *mut u32) = 48000; }
            0
        }
        SNDCTL_DSP_STEREO => {
            if arg != 0 { *(arg as *mut u32) = 1; }  // stereo
            0
        }
        SNDCTL_DSP_CHANNELS => {
            if arg != 0 { *(arg as *mut u32) = 2; }  // 2 channels
            0
        }
        SNDCTL_DSP_SETFMT => {
            if arg != 0 { *(arg as *mut u32) = 0x10; } // AFMT_S16_LE
            0
        }
        SNDCTL_DSP_GETFMTS => {
            if arg != 0 { *(arg as *mut u32) = 0x10; } // AFMT_S16_LE
            0
        }
        SNDCTL_DSP_GETBLKSIZE => {
            if arg != 0 { *(arg as *mut u32) = 4096; }
            4096
        }
        SNDCTL_DSP_SETFRAGMENT => 0,
        SNDCTL_DSP_GETOSPACE => {
            // audio_buf_info: fragments, fragstotal, fragsize, bytes
            if arg != 0 {
                let p = arg as *mut u32;
                *p          = 8;   // fragments available
                *p.add(1)   = 32;  // total fragments
                *p.add(2)   = 4096; // fragsize
                *p.add(3)   = 32768; // bytes available
            }
            0
        }
        SOUND_MIXER_WRITE_VOLUME => {
            if arg != 0 {
                let v = *(arg as *const u32);
                let pct = (v & 0xFF) as u8; // left channel 0-100
                crate::audio::set_volume(pct);
            }
            0
        }
        SOUND_MIXER_READ_VOLUME => {
            let pct = crate::audio::get_volume() as u32;
            if arg != 0 { *(arg as *mut u32) = pct | (pct << 8); }
            0
        }
        _      => -25, // ENOTTY
    }
}

// ── sys_fcntl ────────────────────────────────────────────────────────────────
const F_DUPFD:    u64 = 0;
const F_GETFD:    u64 = 1;
const F_SETFD:    u64 = 2;
const F_GETFL:    u64 = 3;
const F_SETFL:    u64 = 4;
const FD_CLOEXEC: u64 = 1;

unsafe fn sys_fcntl(fd: u64, cmd: u64, _arg: u64) -> i64 {
    match cmd {
        F_GETFD  => FD_CLOEXEC as i64,
        F_SETFD  => 0,
        F_GETFL  => 0o2,   // O_RDWR
        F_SETFL  => 0,
        F_DUPFD  => sys_dup(fd),
        _        => EINVAL,
    }
}

// ── sys_dup / sys_dup2 ───────────────────────────────────────────────────────
unsafe fn sys_dup(oldfd: u64) -> i64 {
    let pid = crate::scheduler::current_pid();
    let newfd = alloc_fd(pid);
    if oldfd < 3 { return newfd as i64; }
    let has = FD_TABLE.lock().contains_key(&(pid, oldfd));
    if has { newfd as i64 } else { EBADF }
}

unsafe fn sys_dup2(oldfd: u64, newfd: u64) -> i64 {
    if oldfd == newfd { return oldfd as i64; }
    let pid = crate::scheduler::current_pid();
    FD_TABLE.lock().remove(&(pid, newfd));
    if oldfd < 3 || newfd < 3 { return newfd as i64; }
    newfd as i64
}

// ── sys_pipe2 ────────────────────────────────────────────────────────────────
//
// Pipe state is shared via an Arc<Mutex<PipeState>> between the read and write
// ends. When the write end is closed, writer_count drops to 0 and reads on an
// empty pipe return 0 (EOF) rather than blocking forever.

struct PipeState {
    buf:          alloc::vec::Vec<u8>,
    writer_count: u32, // number of open write fds for this pipe
    reader_count: u32, // number of open read fds
}
impl PipeState {
    fn new() -> Self { Self { buf: alloc::vec::Vec::new(), writer_count: 1, reader_count: 1 } }
}

static PIPE_TABLE: spin::Mutex<alloc::collections::BTreeMap<usize, alloc::sync::Arc<spin::Mutex<PipeState>>>>
    = spin::Mutex::new(alloc::collections::BTreeMap::new());
static NEXT_PIPE: AtomicU64 = AtomicU64::new(0);

struct PipeHandle { pipe_id: usize, is_write: bool }
impl crate::vfs::FileHandle for PipeHandle {
    fn bytes_available(&self) -> usize {
        if self.is_write { return 0; }
        PIPE_TABLE.lock().get(&self.pipe_id).map(|p| p.lock().buf.len()).unwrap_or(0)
    }
    fn read(&mut self, buf: &mut [u8]) -> crate::vfs::VfsResult<usize> {
        if self.is_write { return Err(crate::vfs::VfsError::PermissionDenied); }
        let state_arc = match PIPE_TABLE.lock().get(&self.pipe_id).cloned() {
            Some(a) => a,
            None    => return Ok(0),
        };
        let mut state = state_arc.lock();
        if state.buf.is_empty() {
            // EOF if all writers closed; EAGAIN-equivalent (0 bytes) otherwise.
            return Ok(0);
        }
        let n = buf.len().min(state.buf.len());
        buf[..n].copy_from_slice(&state.buf[..n]);
        state.buf.drain(..n);
        Ok(n)
    }
    fn write(&mut self, buf: &[u8]) -> crate::vfs::VfsResult<usize> {
        if !self.is_write { return Err(crate::vfs::VfsError::PermissionDenied); }
        let state_arc = match PIPE_TABLE.lock().get(&self.pipe_id).cloned() {
            Some(a) => a,
            None    => return Err(crate::vfs::VfsError::Io),
        };
        state_arc.lock().buf.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn seek(&mut self, _: u64) -> crate::vfs::VfsResult<u64> { Ok(0) }
    fn stat(&self) -> crate::vfs::VfsResult<crate::vfs::Stat> {
        Ok(crate::vfs::Stat { ino: 0, size: 0, is_dir: false, nlink: 2,
                              uid: 0, gid: 0, mode: 0o622 })
    }
}
impl Drop for PipeHandle {
    fn drop(&mut self) {
        let mut tbl = PIPE_TABLE.lock();
        if let Some(arc) = tbl.get(&self.pipe_id).cloned() {
            let mut state = arc.lock();
            if self.is_write { state.writer_count = state.writer_count.saturating_sub(1); }
            else              { state.reader_count = state.reader_count.saturating_sub(1); }
            let dead = state.writer_count == 0 && state.reader_count == 0;
            drop(state);
            if dead { tbl.remove(&self.pipe_id); }
        }
    }
}

unsafe fn sys_pipe2(fds_ptr: u64, _flags: u64) -> i64 {
    let p = match validate_user_ptr_mut(fds_ptr, 8) {
        Ok(p)  => p as *mut u32,
        Err(e) => return e,
    };
    let pipe_id = NEXT_PIPE.fetch_add(1, Ordering::Relaxed) as usize;
    PIPE_TABLE.lock().insert(pipe_id, alloc::sync::Arc::new(spin::Mutex::new(PipeState::new())));
    let pid = crate::scheduler::current_pid();
    let rfd = alloc_fd(pid);
    let wfd = alloc_fd(pid);
    {
        let mut table = FD_TABLE.lock();
        table.insert((pid, rfd), alloc::boxed::Box::new(PipeHandle { pipe_id, is_write: false }));
        table.insert((pid, wfd), alloc::boxed::Box::new(PipeHandle { pipe_id, is_write: true  }));
    }
    *p        = rfd as u32;
    *p.add(1) = wfd as u32;
    0
}

// ── sys_writev ───────────────────────────────────────────────────────────────
unsafe fn sys_writev(fd: u64, iov_ptr: u64, iovcnt: u64) -> i64 {
    let count = iovcnt.min(1024) as usize;
    let mut total = 0i64;
    for i in 0..count {
        let base = iov_ptr + (i as u64) * 16;
        let iov_base = *(base as *const u64);
        let iov_len  = *((base + 8) as *const u64);
        if iov_len == 0 { continue; }
        let n = sys_write(fd, iov_base, iov_len);
        if n < 0 { return n; }
        total += n;
    }
    total
}

// ── sys_readv ────────────────────────────────────────────────────────────────
unsafe fn sys_readv(fd: u64, iov_ptr: u64, iovcnt: u64) -> i64 {
    let count = iovcnt.min(1024) as usize;
    let mut total = 0i64;
    for i in 0..count {
        let base = iov_ptr + (i as u64) * 16;
        let iov_base = *(base as *const u64);
        let iov_len  = *((base + 8) as *const u64);
        if iov_len == 0 { continue; }
        let n = sys_read(fd, iov_base, iov_len);
        if n < 0 { return n; }
        total += n;
        if (n as u64) < iov_len { break; }
    }
    total
}

// ── sys_pread64 / sys_pwrite64 ───────────────────────────────────────────────
unsafe fn sys_pread64(fd: u64, buf: u64, len: u64, off: u64) -> i64 {
    sys_lseek(fd, off, 0);
    sys_read(fd, buf, len)
}
unsafe fn sys_pwrite64(fd: u64, buf: u64, len: u64, off: u64) -> i64 {
    sys_lseek(fd, off, 0);
    sys_write(fd, buf, len)
}

// ── sys_set_robust_list ───────────────────────────────────────────────────────
// musl pthread calls this to register a per-thread robust futex list.
// On thread death, the kernel walks the list and marks locked futexes as OWNER_DIED.
unsafe fn sys_set_robust_list(head: u64, len: usize) -> i64 {
    if len != core::mem::size_of::<u64>() * 3 {
        return EINVAL; // Linux checks for exactly 24 bytes
    }
    let pid = crate::scheduler::current_pid();
    crate::scheduler::set_robust_list(pid, head, len);
    0
}

// ── sys_sigaltstack ──────────────────────────────────────────────────────────
// musl __init_tls calls sigaltstack(NULL, &old_ss) to query.
// We return SS_DISABLE (no alt stack active) which is the correct initial state.
const SS_DISABLE: i32 = 4;
unsafe fn sys_sigaltstack(ss_ptr: u64, old_ss_ptr: u64) -> i64 {
    // struct stack_t: { void *ss_sp, int ss_flags, size_t ss_size } = 24 bytes
    if old_ss_ptr != 0 {
        if let Ok(p) = validate_user_ptr_mut(old_ss_ptr, 24) {
            core::ptr::write_bytes(p, 0, 24);
            // ss_flags at offset 8: SS_DISABLE
            (p.add(8) as *mut i32).write(SS_DISABLE);
        }
    }
    let _ = ss_ptr; // setting a new alt stack is ignored for now
    0
}

// ── sys_rt_sigaction ─────────────────────────────────────────────────────────
unsafe fn sys_rt_sigaction(signum: u64, act_ptr: u64, oldact_ptr: u64) -> i64 {
    if oldact_ptr != 0 {
        if let Ok(p) = validate_user_ptr_mut(oldact_ptr, 32) {
            core::ptr::write_bytes(p, 0, 32);
        }
    }
    if act_ptr != 0 && signum < 64 {
        if let Ok(p) = validate_user_ptr(act_ptr, 8) {
            let handler = *(p as *const u64);
            let pid = crate::scheduler::current_pid();
            crate::scheduler::set_signal_handler(pid, signum as usize, handler);
        }
    }
    0
}

// ── sys_rt_sigprocmask ───────────────────────────────────────────────────────
unsafe fn sys_rt_sigprocmask(_how: u64, _set: u64, oldset: u64) -> i64 {
    if oldset != 0 {
        if let Ok(p) = validate_user_ptr_mut(oldset, 8) {
            *(p as *mut u64) = 0;
        }
    }
    0
}

// ── sys_prctl ────────────────────────────────────────────────────────────────
const PR_SET_NAME: u64 = 15;
const PR_GET_NAME: u64 = 16;

unsafe fn sys_prctl(op: u64, arg2: u64, _arg3: u64) -> i64 {
    match op {
        PR_SET_NAME => { let _ = read_user_cstr(arg2, 16); 0 }
        PR_GET_NAME => {
            if arg2 != 0 {
                let p = arg2 as *mut u8;
                let n = b"nodeai-proc\0";
                core::ptr::copy_nonoverlapping(n.as_ptr(), p, n.len());
            }
            0
        }
        _ => 0,
    }
}

// ── sys_arch_prctl ───────────────────────────────────────────────────────────
const ARCH_SET_FS: u64 = 0x1002;
const ARCH_GET_FS: u64 = 0x1003;
const ARCH_SET_GS: u64 = 0x1001;
const MSR_FS_BASE: u32 = 0xC000_0100;
const MSR_GS_BASE: u32 = 0xC000_0101;

unsafe fn sys_arch_prctl(code: u64, addr: u64) -> i64 {
    match code {
        ARCH_SET_FS => {
            core::arch::asm!("wrmsr",
                in("ecx") MSR_FS_BASE,
                in("eax") addr as u32,
                in("edx") (addr >> 32) as u32,
                options(nomem, nostack));
            let pid = crate::scheduler::current_pid();
            crate::scheduler::set_fs_base(pid, addr);
            0
        }
        ARCH_GET_FS => {
            if addr != 0 {
                let pid = crate::scheduler::current_pid();
                *(addr as *mut u64) = crate::scheduler::get_fs_base(pid);
            }
            0
        }
        ARCH_SET_GS => {
            core::arch::asm!("wrmsr",
                in("ecx") MSR_GS_BASE,
                in("eax") addr as u32,
                in("edx") (addr >> 32) as u32,
                options(nomem, nostack));
            0
        }
        _ => EINVAL,
    }
}

// ── sys_sysinfo ──────────────────────────────────────────────────────────────
unsafe fn sys_sysinfo(ptr: u64) -> i64 {
    if ptr == 0 { return 0; }
    let p = ptr as *mut u64;
    *p          = crate::scheduler::uptime_ms() / 1000;
    *p.add(1)   = 0; *p.add(2) = 0; *p.add(3) = 0;  // loads
    *p.add(4)   = crate::scheduler::total_ram_pages() * 4096;
    *p.add(5)   = crate::scheduler::free_mb() * 1024 * 1024;
    *p.add(6)   = 0; *p.add(7) = 0; *p.add(8) = 0; *p.add(9) = 0;
    *(ptr.wrapping_add(80) as *mut u16) = crate::scheduler::task_count() as u16;
    *(ptr.wrapping_add(108) as *mut u32) = 1;
    0
}

// ── sys_getrlimit ────────────────────────────────────────────────────────────
unsafe fn sys_getrlimit(_resource: u64, rlim_ptr: u64) -> i64 {
    if rlim_ptr == 0 { return 0; }
    let p = rlim_ptr as *mut u64;
    *p = u64::MAX; *p.add(1) = u64::MAX;
    0
}

// ── sys_statfs ───────────────────────────────────────────────────────────────
unsafe fn sys_statfs(_path: u64, buf_ptr: u64) -> i64 {
    if buf_ptr == 0 { return 0; }
    let blk = 4096u64;
    let total = crate::scheduler::total_ram_pages();
    let free  = crate::scheduler::free_mb() * 1024 * 1024 / blk;
    let p = buf_ptr as *mut u64;
    *p = 0x0102_1994; *p.add(1) = blk; *p.add(2) = total;
    *p.add(3) = free; *p.add(4) = free;
    *p.add(5) = 1_000_000; *p.add(6) = 999_999;
    *p.add(7) = 0; *p.add(8) = 255; *p.add(9) = blk; *p.add(10) = 0;
    0
}

// ── sys_stat ─────────────────────────────────────────────────────────────────
unsafe fn sys_stat(path_ptr: u64, stat_ptr: u64, _unused: u64) -> i64 {
    let path = match read_user_cstr(path_ptr, 4096) {
        Some(p) => p,
        None    => return EFAULT,
    };
    let stat_bytes = match validate_user_ptr_mut(stat_ptr, 144) {
        Ok(p)  => p,
        Err(e) => return e,
    };
    core::ptr::write_bytes(stat_bytes, 0, 144);

    let node = match crate::vfs::lookup(&path) {
        Ok(n)  => n,
        Err(_) => return ENOENT,
    };
    let st = match node.stat() {
        Ok(s)  => s,
        Err(_) => return EINVAL,
    };
    let p = stat_bytes as *mut u64;
    *p        = st.ino;
    *p.add(1) = st.nlink as u64;
    *(stat_bytes.add(16) as *mut u32) = if st.is_dir { 0o40755 } else { st.mode as u32 | 0o100000 };
    *(stat_bytes.add(20) as *mut u32) = st.uid;
    *(stat_bytes.add(24) as *mut u32) = st.gid;
    *(stat_bytes.add(48) as *mut u64) = st.size;
    *(stat_bytes.add(56) as *mut u64) = 4096;
    *(stat_bytes.add(64) as *mut u64) = (st.size + 511) / 512;
    0
}

// ── sys_getcwd ───────────────────────────────────────────────────────────────
unsafe fn sys_getcwd(buf_ptr: u64, size: u64) -> i64 {
    if size < 2 { return EINVAL; }
    let p = match validate_user_ptr_mut(buf_ptr, size) {
        Ok(p) => p, Err(e) => return e,
    };
    let cwd = b"/\0";
    core::ptr::copy_nonoverlapping(cwd.as_ptr(), p, cwd.len());
    buf_ptr as i64
}

// ── sys_chdir / mkdir / rmdir / unlink / rename ──────────────────────────────
unsafe fn sys_chdir(_path: u64, _len: u64) -> i64 { 0 }

unsafe fn sys_mkdir(path_ptr: u64, _len: u64, _mode: u64) -> i64 {
    let path = match read_user_cstr(path_ptr, 4096) { Some(p) => p, None => return EFAULT };
    let (parent_path, name) = crate::vfs::path::split_parent(&path);
    let parent = match crate::vfs::lookup(parent_path) { Ok(n) => n, Err(_) => return ENOENT };
    match parent.mkdir(name) { Ok(_) => 0, Err(_) => EINVAL }
}

unsafe fn sys_rmdir(path_ptr: u64, _len: u64) -> i64 {
    let path = match read_user_cstr(path_ptr, 4096) { Some(p) => p, None => return EFAULT };
    let (parent_path, name) = crate::vfs::path::split_parent(&path);
    let parent = match crate::vfs::lookup(parent_path) { Ok(n) => n, Err(_) => return ENOENT };
    match parent.unlink(name) { Ok(()) => 0, Err(_) => EINVAL }
}

unsafe fn sys_unlink(path_ptr: u64, _len: u64) -> i64 {
    let path = match read_user_cstr(path_ptr, 4096) { Some(p) => p, None => return EFAULT };
    let (parent_path, name) = crate::vfs::path::split_parent(&path);
    let parent = match crate::vfs::lookup(parent_path) { Ok(n) => n, Err(_) => return ENOENT };
    match parent.unlink(name) { Ok(()) => 0, Err(_) => ENOENT }
}

unsafe fn sys_rename(old_ptr: u64, _ol: u64, new_ptr: u64, _nl: u64) -> i64 {
    let _old = read_user_cstr(old_ptr, 4096);
    let _new = read_user_cstr(new_ptr, 4096);
    ENOSYS
}

// ── sys_sendfile ─────────────────────────────────────────────────────────────
unsafe fn sys_sendfile(out_fd: u64, in_fd: u64, _offset: u64, count: u64) -> i64 {
    let mut buf = [0u8; 4096];
    let mut total = 0i64;
    let mut remaining = count;
    while remaining > 0 {
        let to_read = remaining.min(4096) as usize;
        let pid = crate::scheduler::current_pid();
        let n = {
            let mut table = FD_TABLE.lock();
            if let Some(h) = table.get_mut(&(pid, in_fd)) {
                match h.read(&mut buf[..to_read]) { Ok(n) => n, Err(_) => break }
            } else { break }
        };
        if n == 0 { break; }
        let wrote = sys_write(out_fd, buf.as_ptr() as u64, n as u64);
        if wrote < 0 { break; }
        total += n as i64;
        remaining -= n as u64;
    }
    total
}

// ─────────────────────────────────────────────────────────────────────────────
//  Epoll (stub), Futex (real wait queue), Sockets (real bind/listen/accept)
// ─────────────────────────────────────────────────────────────────────────────

// ── sys_futex ────────────────────────────────────────────────────────────────
const FUTEX_WAIT:    i32 = 0;
const FUTEX_WAKE:    i32 = 1;
const FUTEX_PRIVATE: i32 = 128;

/// Futex wait queue: uaddr → list of sleeping PIDs.
static FUTEX_WAITERS: spin::Mutex<alloc::collections::BTreeMap<u64, alloc::vec::Vec<u64>>>
    = spin::Mutex::new(alloc::collections::BTreeMap::new());

unsafe fn sys_futex(uaddr: u64, op: i32, val: u32, _timeout: u64) -> i64 {
    let op_code = op & !FUTEX_PRIVATE;
    match op_code {
        FUTEX_WAIT => {
            if uaddr == 0 { return EFAULT; }
            // Atomically check *uaddr == val; if not, return EAGAIN.
            let cur = core::ptr::read_volatile(uaddr as *const u32);
            if cur != val { return EAGAIN; }
            // Register this task in the futex wait queue then sleep.
            let pid = crate::scheduler::current_pid();
            FUTEX_WAITERS.lock().entry(uaddr).or_default().push(pid);
            crate::scheduler::sleep_current();
            crate::scheduler::yield_cpu();
            0
        }
        FUTEX_WAKE => {
            // Wake up to `val` waiters on this address.
            let to_wake = val as usize;
            let pids: alloc::vec::Vec<u64> = {
                let mut wq = FUTEX_WAITERS.lock();
                if let Some(list) = wq.get_mut(&uaddr) {
                    let woke: alloc::vec::Vec<u64> = list.drain(..list.len().min(to_wake)).collect();
                    if list.is_empty() { wq.remove(&uaddr); }
                    woke
                } else {
                    alloc::vec::Vec::new()
                }
            };
            let n = pids.len() as i64;
            let waker = crate::scheduler::current_pid();
            for pid in &pids {
                crate::causal::record_wakeup(waker, *pid);
            }
            for pid in pids { crate::scheduler::wake_pid(pid); }
            n
        }
        _ => ENOSYS,
    }
}

// ── sys_epoll ────────────────────────────────────────────────────────────────
//
// Level-triggered epoll. Edge-triggered (EPOLLET) is accepted without error but
// falls back to level-triggered behaviour — correct for all well-written callers
// that drain the fd before re-arming.
//
// struct epoll_event layout (Linux x86_64, __packed__):
//   offset 0: events u32
//   offset 4: data   u64  (union — we treat as raw u64)
//   total: 12 bytes
unsafe fn sys_epoll_create1(_flags: u64) -> i64 {
    let pid = crate::scheduler::current_pid();
    let epfd = alloc_fd(pid);
    EPOLL_TABLE.lock().insert((pid, epfd), EpollInstance::new());
    epfd as i64
}

unsafe fn sys_epoll_ctl(epfd: u64, op: i32, watched_fd: i32, event_ptr: u64) -> i64 {
    let pid = crate::scheduler::current_pid();
    // CTL_DEL does not require a valid event pointer.
    let interest = if op != EPOLL_CTL_DEL && event_ptr != 0 {
        match validate_user_ptr(event_ptr, 12) {
            Ok(p) => {
                let events = core::ptr::read_unaligned(p as *const u32);
                let data   = core::ptr::read_unaligned(p.add(4) as *const u64);
                EpollInterest { events: events & !EPOLLET, data }
            }
            Err(e) => return e,
        }
    } else {
        EpollInterest { events: 0, data: 0 }
    };

    let mut table = EPOLL_TABLE.lock();
    let inst = match table.get_mut(&(pid, epfd)) {
        Some(i) => i,
        None    => return EBADF,
    };
    match op {
        EPOLL_CTL_ADD | EPOLL_CTL_MOD => { inst.interests.insert(watched_fd, interest); 0 }
        EPOLL_CTL_DEL                 => { inst.interests.remove(&watched_fd); 0 }
        _                             => EINVAL,
    }
}

unsafe fn sys_epoll_wait(epfd: u64, events_out: u64, maxevents: i32, timeout_ms: i32) -> i64 {
    if maxevents <= 0 { return EINVAL; }
    let max = maxevents as usize;
    // Validate the output buffer (12 bytes per epoll_event).
    let out_buf_len = (max as u64).saturating_mul(12);
    let out_ptr = match validate_user_ptr_mut(events_out, out_buf_len) {
        Ok(p)  => p,
        Err(e) => return e,
    };
    let pid = crate::scheduler::current_pid();
    let deadline = if timeout_ms < 0 {
        u64::MAX
    } else {
        crate::scheduler::uptime_ms() + timeout_ms as u64
    };

    loop {
        // Snapshot interest list (drop EPOLL_TABLE lock before touching FD_TABLE).
        let interests: alloc::vec::Vec<(i32, EpollInterest)> = {
            let table = EPOLL_TABLE.lock();
            match table.get(&(pid, epfd)) {
                Some(inst) => inst.interests.iter().map(|(&fd, &ev)| (fd, ev)).collect(),
                None       => return EBADF,
            }
        };

        let mut n_ready: usize = 0;
        // Hold FD_TABLE for the entire poll round — prevents TOCTOU between
        // bytes_available() and contains_key() on the same fd.
        let mut fd_tbl = FD_TABLE.lock();
        for (watched_fd, interest) in &interests {
            if n_ready >= max { break; }
            let wfd = *watched_fd as u64;

            let mut revents: u32 = 0;
            if wfd == 1 || wfd == 2 {
                if interest.events & EPOLLOUT != 0 { revents |= EPOLLOUT; }
            } else {
                match fd_tbl.get_mut(&(pid, wfd)) {
                    Some(h) => {
                        let available = h.bytes_available();
                        if interest.events & EPOLLIN  != 0 && available > 0 { revents |= EPOLLIN; }
                        if interest.events & EPOLLOUT != 0                    { revents |= EPOLLOUT; }
                    }
                    None => { revents |= EPOLLHUP | EPOLLERR; }
                }
            }

            if revents != 0 {
                let slot = out_ptr.add(n_ready * 12);
                core::ptr::write_unaligned(slot as *mut u32, revents);
                core::ptr::write_unaligned(slot.add(4) as *mut u64, interest.data);
                n_ready += 1;
            }
        }
        drop(fd_tbl);

        if n_ready > 0 || crate::scheduler::uptime_ms() >= deadline {
            return n_ready as i64;
        }
        crate::scheduler::yield_cpu();
    }
}

/// Format /proc/epoll — all active epoll instances, their interest lists, and readiness state.
pub fn format_epoll_table() -> alloc::vec::Vec<u8> {
    use alloc::string::String;
    let table = EPOLL_TABLE.lock();
    if table.is_empty() {
        return b"no active epoll instances\n".to_vec();
    }
    let mut out = String::from("PID   EPFD  WATCHED_FD  EVENTS\n");
    out.push_str("----  ----  ----------  ------\n");
    for (&(pid, epfd), inst) in table.iter() {
        for (&fd, &ev) in inst.interests.iter() {
            let mut flags = String::new();
            if ev.events & EPOLLIN  != 0 { flags.push_str("IN "); }
            if ev.events & EPOLLOUT != 0 { flags.push_str("OUT "); }
            if ev.events & EPOLLERR != 0 { flags.push_str("ERR "); }
            if ev.events & EPOLLHUP != 0 { flags.push_str("HUP "); }
            out.push_str(&alloc::format!("{:<6}{:<6}{:<12}{}\n",
                pid, epfd, fd, flags.trim_end()));
        }
    }
    out.into_bytes()
}

// ── sys_eventfd2 ─────────────────────────────────────────────────────────────
unsafe fn sys_eventfd2(_initval: u64, _flags: i32) -> i64 {
    let pid = crate::scheduler::current_pid();
    alloc_fd(pid) as i64
}

// ── sys_poll ─────────────────────────────────────────────────────────────────
// struct pollfd: fd(i32)+events(i16)+revents(i16) = 8 bytes, but compiler may pad.
// Linux packs it tightly; we use byte offsets.
const POLLIN:  u16 = 0x0001;
const POLLOUT: u16 = 0x0004;
const POLLERR: u16 = 0x0008;

unsafe fn sys_poll(fds_ptr: u64, nfds: u64, timeout_ms: i32) -> i64 {
    if fds_ptr == 0 || nfds == 0 { return 0; }
    let struct_sz: u64 = 8; // sizeof(struct pollfd)
    let buf_len = nfds.saturating_mul(struct_sz);
    let buf = match validate_user_ptr_mut(fds_ptr, buf_len) {
        Ok(p)  => p,
        Err(e) => return e,
    };
    let pid = crate::scheduler::current_pid();
    let deadline = if timeout_ms < 0 {
        u64::MAX
    } else {
        crate::scheduler::uptime_ms() + timeout_ms as u64
    };

    loop {
        let mut ready = 0i64;
        for i in 0..nfds {
            let entry = buf.add((i * struct_sz) as usize);
            let fd    = core::ptr::read_unaligned(entry as *const i32) as u64;
            let events= core::ptr::read_unaligned(entry.add(4) as *const u16);
            // Clear revents
            core::ptr::write_unaligned(entry.add(6) as *mut u16, 0);
            if (fd as i32) < 0 { continue; }

            let mut revents: u16 = 0;
            if fd == 1 || fd == 2 {
                // stdout/stderr: always writable.
                if events & POLLOUT != 0 { revents |= POLLOUT; }
            } else {
                // Use bytes_available() — non-destructive, no data consumed.
                let available = {
                    let mut table = FD_TABLE.lock();
                    table.get_mut(&(pid, fd)).map(|h| h.bytes_available()).unwrap_or(0)
                };
                if events & POLLIN  != 0 && available > 0 { revents |= POLLIN;  }
                if events & POLLOUT != 0                   { revents |= POLLOUT; }
            }
            if revents != 0 {
                core::ptr::write_unaligned(entry.add(6) as *mut u16, revents);
                ready += 1;
            }
        }
        if ready > 0 || crate::scheduler::uptime_ms() >= deadline { return ready; }
        // Yield and retry.
        crate::scheduler::yield_cpu();
    }
}
unsafe fn sys_select(_n: i32, _r: u64, _w: u64, _e: u64, tv: u64) -> i64 {
    if tv != 0 {
        let ms = *(tv as *const u64) * 1000 + *((tv+8) as *const u64) / 1000;
        let start = crate::scheduler::uptime_ms();
        while crate::scheduler::uptime_ms().wrapping_sub(start) < ms.min(10) { core::hint::spin_loop(); }
    }
    0
}

// ── Socket stubs ─────────────────────────────────────────────────────────────
const AF_INET:     u64 = 2;
const SOCK_STREAM: u64 = 1;

/// Listening socket fd — placeholder in FD_TABLE; bound port lives in SOCKET_PORTS.
struct SocketHandle;
impl crate::vfs::FileHandle for SocketHandle {
    fn read(&mut self,  b: &mut [u8]) -> crate::vfs::VfsResult<usize> { let _ = b; Ok(0) }
    fn write(&mut self, b: &[u8])     -> crate::vfs::VfsResult<usize> { Ok(b.len()) }
    fn seek(&mut self,  _: u64)       -> crate::vfs::VfsResult<u64>   { Ok(0) }
    fn stat(&self) -> crate::vfs::VfsResult<crate::vfs::Stat> {
        Ok(crate::vfs::Stat { ino: 0, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o600 })
    }
}

/// Connected socket fd — wraps a TcpSocketKey for an Established connection.
struct ConnectedSocketHandle {
    local_port:  u16,
    remote_ip:   [u8; 4],
    remote_port: u16,
}
impl crate::vfs::FileHandle for ConnectedSocketHandle {
    fn bytes_available(&self) -> usize {
        crate::net::tcp::rx_buf_len(self.local_port, self.remote_ip, self.remote_port)
    }
    fn read(&mut self, buf: &mut [u8]) -> crate::vfs::VfsResult<usize> {
        Ok(crate::net::tcp::recv(self.local_port, self.remote_ip, self.remote_port, buf))
    }
    fn write(&mut self, buf: &[u8]) -> crate::vfs::VfsResult<usize> {
        let n = crate::net::tcp::send(self.local_port, self.remote_ip, self.remote_port, buf);
        Ok(n)
    }
    fn seek(&mut self, _: u64) -> crate::vfs::VfsResult<u64> { Ok(0) }
    fn stat(&self) -> crate::vfs::VfsResult<crate::vfs::Stat> {
        Ok(crate::vfs::Stat { ino: 0, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o600 })
    }
}

unsafe fn sys_socket(domain: u64, sock_type: u64, _proto: u64) -> i64 {
    if domain == AF_INET && (sock_type & 0xF) == SOCK_STREAM {
        let pid = crate::scheduler::current_pid();
        let fd  = alloc_fd(pid);
        FD_TABLE.lock().insert((pid, fd), alloc::boxed::Box::new(SocketHandle));
        return fd as i64;
    }
    ENOSYS
}

unsafe fn sys_connect(_sockfd: u64, addr_ptr: u64, _alen: u64) -> i64 {
    if addr_ptr == 0 { return EFAULT; }
    0 // stub: outbound connect not yet implemented
}

/// Parse port from sockaddr_in: family(u16) + port(u16 big-endian) at offset 2.
unsafe fn sys_bind(sockfd: u64, addr_ptr: u64, alen: u64) -> i64 {
    if addr_ptr == 0 || alen < 4 { return EFAULT; }
    let p = match validate_user_ptr(addr_ptr, 4) {
        Ok(p)  => p,
        Err(e) => return e,
    };
    let port_be = core::ptr::read_unaligned(p.add(2) as *const u16);
    let port = u16::from_be(port_be);
    let pid = crate::scheduler::current_pid();
    // Verify the fd exists before recording the port.
    if !FD_TABLE.lock().contains_key(&(pid, sockfd)) { return EBADF; }
    SOCKET_PORTS.lock().insert((pid, sockfd), port);
    0
}

unsafe fn sys_listen(sockfd: u64, _backlog: u64) -> i64 {
    let pid = crate::scheduler::current_pid();
    let port = SOCKET_PORTS.lock().get(&(pid, sockfd)).copied();
    match port {
        Some(p) => { crate::net::tcp::listen(p); 0 }
        None    => EINVAL,
    }
}

/// Non-blocking accept: returns EAGAIN if no Established connection is ready.
/// TODO: add accept queue for O(1) lookup when connection count grows.
unsafe fn sys_accept(sockfd: u64, _addr: u64, _alen: u64) -> i64 {
    let pid = crate::scheduler::current_pid();
    let port = SOCKET_PORTS.lock().get(&(pid, sockfd)).copied();
    let port = match port { Some(p) => p, None => return EINVAL };
    match crate::net::tcp::accept(port) {
        None => EAGAIN,
        Some(key) => {
            let conn = ConnectedSocketHandle {
                local_port:  key.local_port,
                remote_ip:   key.remote_ip,
                remote_port: key.remote_port,
            };
            let new_fd = alloc_fd(pid);
            FD_TABLE.lock().insert((pid, new_fd), alloc::boxed::Box::new(conn));
            crate::klog!(INFO, "sys_accept: new conn fd={} port {}↔{}", new_fd, port, key.remote_port);
            new_fd as i64
        }
    }
}

unsafe fn sys_sendto(s: u64, b: u64, l: u64, _f: u64, _a: u64, _al: u64) -> i64 { sys_write(s, b, l) }
unsafe fn sys_recvfrom(s: u64, b: u64, l: u64, _f: u64, _a: u64, _al: u64) -> i64 { sys_read(s, b, l) }

// getsockname / getpeername — return stub AF_INET 0.0.0.0:0 address
unsafe fn sys_getsockname(_s: u64, addr_ptr: u64, addrlen_ptr: u64) -> i64 {
    if addr_ptr == 0 { return EFAULT; }
    // struct sockaddr_in: u16 family, u16 port, u32 addr, [8]pad
    const SA_LEN: u64 = 16;
    if let Ok(p) = validate_user_ptr_mut(addr_ptr, SA_LEN) {
        core::ptr::write_bytes(p, 0, SA_LEN as usize);
        *(p as *mut u16) = 2u16.to_be(); // AF_INET in native byte order = 2
        if addrlen_ptr != 0 {
            if let Ok(lp) = validate_user_ptr_mut(addrlen_ptr, 4) {
                *(lp as *mut u32) = SA_LEN as u32;
            }
        }
    }
    0
}
unsafe fn sys_getpeername(s: u64, addr_ptr: u64, addrlen_ptr: u64) -> i64 {
    sys_getsockname(s, addr_ptr, addrlen_ptr)
}

const SO_REUSEADDR: u64 = 2;
const SO_KEEPALIVE: u64 = 9;
const SO_ERROR:     u64 = 4;
const SO_TYPE:      u64 = 3;
const SOL_SOCKET:   u64 = 1;

unsafe fn sys_setsockopt(_s: u64, _lvl: u64, _opt: u64, _val: u64, _vlen: u64) -> i64 {
    0 // accept all socket options silently
}
unsafe fn sys_getsockopt(_s: u64, level: u64, optname: u64, val_ptr: u64, len_ptr: u64) -> i64 {
    if val_ptr == 0 { return EINVAL; }
    let want_len: u32 = if len_ptr != 0 { *(len_ptr as *const u32) } else { 4 };
    if want_len < 4 { return EINVAL; }
    let out = match (level, optname) {
        (SOL_SOCKET, SO_TYPE)      => 1i32,   // SOCK_STREAM
        (SOL_SOCKET, SO_ERROR)     => 0i32,   // no error
        (SOL_SOCKET, SO_REUSEADDR) => 1i32,
        (SOL_SOCKET, SO_KEEPALIVE) => 0i32,
        _                          => 0i32,
    };
    if let Ok(p) = validate_user_ptr_mut(val_ptr, 4) {
        *(p as *mut i32) = out;
        if len_ptr != 0 { *(len_ptr as *mut u32) = 4; }
    }
    0
}

// ── Phase 24: timerfd ────────────────────────────────────────────────────────

/// Per-timer state stored in the global timer table.
struct TimerHandle {
    /// Absolute expiry time in milliseconds since boot.
    expiry_ms:    u64,
    /// Interval for repeating timers (0 = one-shot).
    interval_ms:  u64,
    /// Number of expirations not yet read.
    expirations:  u64,
}

impl crate::vfs::FileHandle for TimerHandle {
    fn read(&mut self, b: &mut [u8]) -> crate::vfs::VfsResult<usize> {
        if b.len() < 8 { return Ok(0); }
        let now = crate::scheduler::uptime_ms();
        if self.expiry_ms > 0 && now >= self.expiry_ms {
            let elapsed = now - self.expiry_ms;
            let extra = if self.interval_ms > 0 { elapsed / self.interval_ms } else { 0 };
            self.expirations += 1 + extra;
            if self.interval_ms > 0 {
                self.expiry_ms = now + self.interval_ms;
            } else {
                self.expiry_ms = 0;
            }
        }
        let count = self.expirations;
        self.expirations = 0;
        let bytes = count.to_ne_bytes();
        b[..8].copy_from_slice(&bytes);
        Ok(8)
    }
    fn write(&mut self, _b: &[u8]) -> crate::vfs::VfsResult<usize> { Ok(0) }
    fn seek(&mut self, _: u64) -> crate::vfs::VfsResult<u64> { Ok(0) }
    fn stat(&self) -> crate::vfs::VfsResult<crate::vfs::Stat> {
        Ok(crate::vfs::Stat { ino: 0, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o600 })
    }
}

// Timer file descriptor fd → TimerHandle stored in FD_TABLE alongside others.
unsafe fn sys_timerfd_create(_clockid: u64, _flags: u64) -> i64 {
    let pid = crate::scheduler::current_pid();
    let fd = alloc_fd(pid);
    FD_TABLE.lock().insert((pid, fd), alloc::boxed::Box::new(TimerHandle {
        expiry_ms: 0,
        interval_ms: 0,
        expirations: 0,
    }));
    fd as i64
}

unsafe fn sys_timerfd_settime(fd: u64, _flags: u64, new_value_ptr: u64, _old_ptr: u64) -> i64 {
    // struct itimerspec: { timespec it_interval, timespec it_value }
    // timespec: { i64 tv_sec, i64 tv_nsec }  (16 bytes each → 32 bytes total)
    if new_value_ptr == 0 { return EFAULT; }
    let p = match validate_user_ptr(new_value_ptr, 32) { Ok(p) => p, Err(e) => return e };
    let iv_sec  = *(p as *const i64);
    let iv_nsec = *((p as *const i64).add(1));
    let val_sec = *((p as *const i64).add(2));
    let val_nsec= *((p as *const i64).add(3));
    let interval_ms = (iv_sec as u64) * 1000 + (iv_nsec.max(0) as u64) / 1_000_000;
    let value_ms    = (val_sec as u64) * 1000 + (val_nsec.max(0) as u64) / 1_000_000;
    let pid = crate::scheduler::current_pid();
    let mut table = FD_TABLE.lock();
    if let Some(h) = table.get_mut(&(pid, fd as u64)) {
        let h_bytes = h as *mut alloc::boxed::Box<dyn crate::vfs::FileHandle> as *mut u8;
        let _ = h_bytes; // just update via downcast trick: store params in a side table
        // Simpler: replace the handle entirely
        let now = crate::scheduler::uptime_ms();
        *h = alloc::boxed::Box::new(TimerHandle {
            expiry_ms:   if value_ms > 0 { now + value_ms } else { 0 },
            interval_ms,
            expirations: 0,
        });
        return 0;
    }
    EBADF
}

unsafe fn sys_timerfd_gettime(fd: u64, curr_value_ptr: u64) -> i64 {
    if curr_value_ptr == 0 { return EFAULT; }
    let _ = fd;
    // Write zero itimerspec (simplified)
    if let Ok(p) = validate_user_ptr_mut(curr_value_ptr, 32) {
        core::ptr::write_bytes(p, 0, 32);
    }
    0
}

// ── Phase 24: readlink / access / ftruncate ──────────────────────────────────

unsafe fn sys_readlink(path_ptr: u64, buf_ptr: u64, buf_len: u64) -> i64 {
    // Most symlinks in our kernel are proc/self pseudo-links
    let path = match read_user_cstr(path_ptr, 4096) { Some(p) => p, None => return EFAULT };
    let target: &[u8] = match path.as_str() {
        "/proc/self/exe" | "/proc/self/cwd" => b"/",
        "/proc/self/fd"  => b"/proc/self/fd",
        _ => return -22, // EINVAL — not a symlink
    };
    let copy = target.len().min(buf_len as usize);
    if copy == 0 { return EINVAL; }
    if let Ok(p) = validate_user_ptr_mut(buf_ptr, buf_len) {
        core::ptr::copy_nonoverlapping(target.as_ptr(), p, copy);
    }
    copy as i64
}

unsafe fn sys_access(path_ptr: u64, _mode: u64) -> i64 {
    // Return 0 (OK) if file exists, ENOENT otherwise
    let path = match read_user_cstr(path_ptr, 4096) { Some(p) => p, None => return EFAULT };
    match crate::vfs::lookup(&path) {
        Ok(_)  => 0,
        Err(_) => ENOENT,
    }
}

unsafe fn sys_ftruncate(fd: u64, _length: u64) -> i64 {
    // Stub: we don't support real truncation yet
    let _ = fd;
    0
}

// ── kernel_exec_entry ─────────────────────────────────────────────────────────
/// Called from kernel shell `exec` command to jump a pre-loaded ELF to ring 3.
/// Allocates and maps the user stack, then does IRETQ to the given entry point.
pub unsafe fn kernel_exec_entry(entry: u64, _entry2: u64) {
    let _ = crate::memory::map_user_range(
        USER_STACK_TOP - USER_STACK_SIZE,
        USER_STACK_SIZE,
        true,   // writable
        false,  // not executable
    );
    ring3_jump(entry, USER_STACK_TOP - 16);
}
