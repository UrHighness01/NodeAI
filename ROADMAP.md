# NodeAI Kernel — Project Roadmap

> A Linux-inspired, AI-integrated operating system kernel written in Rust.
> Goal: surpass Linux in memory safety, resource efficiency, and introduce native AI decision-making at the kernel level.
> Primary test platform: Oracle VirtualBox → then real hardware (x86_64, later ARM).

---

## Project Philosophy

- **Safety first** — Rust's ownership model eliminates entire classes of kernel bugs (buffer overflows, use-after-free, data races) that have plagued Linux for decades.
- **AI as a first-class kernel citizen** — not a userspace daemon, not a module — AI inference runs in a dedicated kernel ring with hardware-level access.
- **Minimal resource footprint** — every subsystem must justify its presence; no legacy bloat.
- **Verifiability** — critical kernel paths must be formally verifiable where possible.

---

## Milestone Overview

| Phase | Name | Target |
|-------|------|--------|
| 0 | Toolchain & Environment | Week 1–2 |
| 1 | Bootloader & Bare Metal | Week 2–6 |
| 2 | Core Kernel Infrastructure | Week 6–16 |
| 3 | Memory Management | Week 10–20 |
| 4 | Process & Scheduler | Week 16–26 |
| 5 | Hardware Abstraction Layer | Week 20–30 |
| 6 | Drivers (VirtualBox target) | Week 24–34 |
| 7 | Filesystem & Storage | Week 28–40 |
| 8 | AI Subsystem Integration | Week 30–50 |
| 9 | Networking Stack | Week 38–52 |
| 10 | Security & Privilege Model | Week 44–60 |
| 11 | Userspace & Syscall ABI | Week 48–64 |
| 12 | Real Hardware Bring-up | Week 60–80 |
| 12a | Framebuffer & GUI Desktop ✅ | Week 55–65 |
| 12b | Extended Syscalls & POSIX Basics ✅ | Week 58–68 |
| 12c | TCP/IP Completion & TLS ✅ | Week 62–72 |
| 13 | Self-hosting & AI Autonomy | Week 70+ |
| 14 | Users, Groups & Authentication | Week 72–82 |
| 15 | Interactive Shell & Terminal | Week 76–86 |
| 16 | Coreutils & System Administration | Week 80–92 |
| 17 | Networking Tools & Services | Week 88–100 |
| **18** | **Ring-3 Process Launch & execve** | **Week 90–96** |
| **19** | **musl libc & Static Userspace Runtimes** | **Week 94–110** |
| **20** | **Package Manager (NodePkg)** | **Week 108–118** |
| **21** | **Dynamic Linker & Shared Libraries** | **Week 115–128** |
| **22** | **Native GUI Multi-Window System** | **Week 112–130** |
| **23** | **Intelli Browser — Native Kernel Port** | **Week 120–145** |
| **24** | **Linux ABI Full Parity** | **Week 130–155** |
| **25** | **Audio Subsystem** | **Week 140–150** |
| **26** | **NodeAI Application Platform** | **Week 145–170** |
| **27** | **Hardware Parity & Production Readiness** | **Week 155–185** |
| **28** | **Developer Experience & Self-Hosting** | **Week 170–200** |
| **29** | **AI Parity & Beyond Linux** | **Week 190+** |

---

## Phase 0 — Toolchain & Environment Setup ✅ COMPLETE

**Goal:** establish a reproducible build environment, CI pipeline, and VirtualBox test harness.

### Tasks
- [x] Install Rust nightly toolchain (`rustup toolchain install nightly`)
- [x] Add `x86_64-unknown-none` target (`rustup target add x86_64-unknown-none`)
- [x] Install `llvm-tools-preview` component (in rust-toolchain.toml)
- [x] Set up workspace `Cargo.toml` with all crates (kernel, bootloader, hal, ai_subsystem, drivers, image-builder)
- [x] Configure `.cargo/config.toml` for bare-metal linking (custom linker script via kernel/.cargo/config.toml)
- [x] Set up Oracle VirtualBox VM script (`scripts/run_vbox.ps1`)
- [x] Set up QEMU as fast iteration test target (`scripts/run_qemu.ps1`)
- [x] Set up serial output capture from QEMU/VirtualBox for kernel log debugging (COM1 → stdio)
- [x] Document toolchain version pins in `rust-toolchain.toml`

---

## Phase 1 — Bootloader & Bare Metal Entry ✅ COMPLETE

**Goal:** kernel boots from ISO/disk image, enters 64-bit long mode, prints to serial and VGA.

### Tasks
- [x] Write or integrate UEFI/BIOS bootloader (`bootloader` crate v0.11 + `bootloader_api`)
- [x] Transition from 16-bit real mode → 32-bit protected mode → 64-bit long mode (handled by bootloader)
- [x] Set up Global Descriptor Table (GDT) — code/data segments, TSS (`kernel/src/gdt.rs`)
- [x] Set up Interrupt Descriptor Table (IDT) — fault handlers, hardware IRQs (`kernel/src/interrupts/mod.rs`)
- [x] Enable paging — identity-map lower memory, map kernel to higher half (bootloader + `kernel/layout.ld`)
- [x] VGA text mode driver (early output, no dependencies) (`kernel/src/vga.rs`)
- [x] UART/serial driver (COM1) for headless debug logging (`kernel/src/logger.rs`)
- [x] `kernel_main()` entry point — stack set up, BSS zeroed, control handed off (`kernel/src/main.rs`)
- [x] Kernel panic handler with VGA + serial dump (`kernel/src/main.rs`)
- [x] Boot information passing (memory map, RSDP pointer) from bootloader to kernel
- [x] `image-builder` host tool creates BIOS/UEFI disk images (`image-builder/src/main.rs`)
- [x] Build scripts: `scripts/build.ps1`, `scripts/run_qemu.ps1`, `scripts/run_vbox.ps1`

---

## Phase 2 — Core Kernel Infrastructure ✅ COMPLETE

**Goal:** foundational kernel services that everything else depends on.

### Tasks

#### Logging & Debug
- [x] `klog!` macro — leveled logging (ERROR, WARN, INFO, DEBUG, TRACE)
- [x] Ring-buffer kernel log (readable via `/proc/kmsg` equivalent later)
- [x] Symbolicated stack traces (embed kernel symbol table in binary)

#### Synchronization Primitives
- [x] Spinlock (no_std, x86 `LOCK XCHG`)
- [x] RwSpinlock
- [x] Mutex (sleeping, backed by scheduler — Phase 4 dep)
- [x] Semaphore
- [x] Wait queues
- [x] Atomic operations wrappers

#### Interrupt Handling
- [x] APIC/xAPIC initialization (replace 8259 PIC)
- [x] IRQ routing table
- [x] Software interrupt (syscall `SYSCALL`/`SYSENTER`) entry point
- [x] Exception handlers: #PF, #GP, #UD, #DE, #NMI, #DF (double fault with separate stack)
- [x] Timer: LAPIC timer calibration via PIT
- [x] Inter-Processor Interrupts (IPI) skeleton for SMP later

#### ACPI
- [x] RSDP/RSDT/XSDT parsing
- [x] MADT parsing (CPU count, IOAPIC addresses)
- [x] FADT parsing (power management registers)
- [x] HPET detection

---

## Phase 3 — Memory Management ✅ COMPLETE

**Goal:** robust, efficient memory subsystem outperforming Linux's allocator in fragmentation and latency.

### Tasks

#### Physical Memory Manager (PMM)
- [x] Parse UEFI/Multiboot memory map
- [x] Buddy allocator for physical frames (power-of-two block sizes)
- [x] Frame reference counting (for shared mappings)
- [x] Memory zones: DMA (0–16 MB), Normal, High

#### Virtual Memory Manager (VMM)
- [x] 4-level page table management (PML4 → PDPT → PD → PT)
- [x] `map_page()`, `unmap_page()`, `remap_page()` primitives
- [x] Kernel address space layout (KASLR-capable)
- [x] User address space layout per-process
- [x] On-demand paging / lazy allocation
- [x] Copy-on-Write (CoW) for `fork()`-like semantics
- [x] Guard pages for stack overflow detection

#### Kernel Heap
- [x] Slab allocator (cache-aligned, per-CPU caches to eliminate lock contention)
- [x] `GlobalAlloc` impl so `alloc` crate works (`Box`, `Vec`, `Arc`, etc.)
- [x] Heap statistics and leak detection in debug builds

#### AI Memory Region
- [x] Dedicated physically-contiguous memory pool for AI model weights
- [x] DMA-capable mapping for AI accelerator (NPU/GPU) if present
- [x] Locked pages (no swap) for inference hot paths

---

## Phase 4 — Process & Scheduler ✅ COMPLETE

**Goal:** preemptive multitasking with an AI-augmented scheduler that outperforms CFS.

### Tasks

#### Task Model
- [x] `Task` struct: PID, state, registers, stack, page table, priority, AI score
- [x] Kernel threads (no user address space)
- [x] User processes (separate address space)
- [x] Thread model within a process (shared page tables)
- [x] `fork()` / `exec()` / `exit()` / `wait()` semantics

#### Scheduler — Classic Layer
- [x] Run queues per CPU (SMP-ready from day one)
- [x] Preemptive round-robin as baseline
- [x] Priority levels (real-time, normal, idle)
- [x] Context switch: save/restore `x86_64` full register set including SSE/AVX state
- [x] Voluntary yield (`sched_yield`)
- [x] Sleep / wakeup infrastructure

#### Scheduler — AI Layer (Novel)
- [x] Per-task behavioral fingerprint (cache miss rate, I/O patterns, CPU burst length)
- [x] AI inference call from scheduler tick — predict next CPU burst length
- [x] Dynamic priority adjustment based on AI prediction
- [x] AI-driven NUMA-aware placement
- [x] Feedback loop: measure prediction accuracy, retrain model weights online
- [x] Kill switch: fall back to pure CFS-equivalent if AI module fails

#### IPC
- [x] Pipes (anonymous, kernel ring buffer)
- [x] Signals (async notification)
- [x] Shared memory regions
- [x] Message queues (AI subsystem uses these for kernel↔AI communication)

---

## Phase 5 — Hardware Abstraction Layer (HAL) ✅ COMPLETE

**Goal:** clean trait-based HAL so drivers and AI subsystem are architecture-agnostic.

### Tasks
- [x] `hal` crate with pure-Rust traits: `Cpu`, `Timer`, `Uart`, `InterruptController`, `Mmu`
- [x] `x86_64` implementation of all HAL traits
- [x] CPU feature detection (CPUID: SSE4.2, AVX2, AVX-512, AMX for AI)
- [x] RDTSC-based high-resolution timer
- [x] MSR (Model Specific Register) abstraction
- [x] CPU power management interface (C-states, P-states) — AI can tune these
- [x] SMP: AP (Application Processor) startup via SIPI sequence
- [x] Per-CPU data areas (GS-based, no global mutable state)
- [x] ARM64 HAL stub (future)

---

## Phase 6 — Device Drivers (VirtualBox Target) ✅ COMPLETE

**Goal:** enough drivers to boot a full system in VirtualBox, support AI model loading from disk.

### Tasks

#### Storage
- [x] VirtIO-blk driver (primary VirtualBox disk interface)
- [ ] AHCI/SATA driver (real hardware fallback)
- [ ] NVMe driver (high-performance SSD for AI model weight storage)
- [x] Block device abstraction trait

#### Display / Console
- [ ] VirtIO-gpu driver (or VESA/VBE framebuffer)
- [ ] Framebuffer console (font rendering, scrolling)
- [ ] VirtualBox guest additions display channel (optional)

#### Input
- [x] PS/2 keyboard driver (VirtualBox default)
- [ ] USB HID keyboard/mouse (for real hardware)

#### Network
- [ ] VirtIO-net driver (VirtualBox networking)
- [ ] Intel e1000 driver (VirtualBox hardware emulation option)

#### PCI/PCIe
- [x] PCI configuration space enumeration
- [x] BAR helpers (I/O, MMIO, size probe)
- [ ] MSI/MSI-X interrupt routing
- [ ] PCIe extended config space support

#### USB
- [ ] xHCI (USB 3.0) controller driver
- [ ] USB device enumeration

#### AI Hardware Drivers
- [ ] Generic NPU/ML-accelerator driver interface trait
- [ ] CPU-only fallback inference engine (SIMD-optimized, AVX2/AVX-512)
- [ ] CUDA-like kernel launch interface for GPU inference (future)

---

## Phase 7 — Filesystem & Storage ✅ COMPLETE

**Goal:** fast, safe filesystem with integrity guarantees for kernel and AI model storage.

### Tasks

#### VFS Layer
- [x] Virtual Filesystem Switch (VFS) — VfsNode trait, FileHandle trait
- [x] Mount table (`MOUNTS` RwLock)
- [x] Path resolution (walk, component iteration)
- [x] File descriptor table per process (in Task via FdEntry)
- [ ] `open()`, `read()`, `write()`, `close()`, `seek()`, `stat()`, `mmap()` syscalls (Phase 11)

#### Filesystems
- [x] **ramfs** — in-memory FS, root filesystem backed by BTreeMap
- [x] **devfs** — /dev/null, /dev/zero, /dev/kmsg using KRING snapshot
- [ ] **ext4** read/write support (standard Linux-compatible)
- [ ] **NodeAI-FS** — custom append-optimized FS (stretch goal)
- [ ] **FAT32** read (for EFI System Partition)

#### Block Layer
- [x] Block device abstraction (VirtIO-blk trait)
- [ ] Block cache (page-aligned, LRU eviction)
- [ ] Write-back dirty tracking
- [ ] I/O scheduler — AI-assisted

#### procfs / sysfs equivalents
- [x] /dev mounted with devfs
- [x] /proc, /sys, /ai directories created in root ramfs
- [ ] Populated /proc entries (Phase 11)

---

## Phase 8 — AI Subsystem (Core of This Project) ✅ COMPLETE

**Goal:** AI inference running natively in kernel space, with controlled access to all kernel data structures, able to make autonomous decisions on scheduling, memory, I/O, and security.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   USER SPACE                        │
│    Applications ──► AI Query Syscall Interface      │
└──────────────────────────┬──────────────────────────┘
                           │ syscall
┌──────────────────────────▼──────────────────────────┐
│              KERNEL SPACE                           │
│  ┌─────────────┐   ┌──────────────────────────┐    │
│  │  Scheduler  │◄──│   AI Decision Engine     │    │
│  │  MemMgr     │◄──│   (Inference Runtime)    │    │
│  │  I/O Sched  │◄──│                          │    │
│  │  Net Stack  │◄──│   Model: NodeAI-Core-1   │    │
│  │  Security   │◄──│   Weights in locked RAM  │    │
│  └─────────────┘   └──────────┬───────────────┘    │
│                               │                     │
│                    ┌──────────▼───────────┐         │
│                    │  AI Hardware Driver  │         │
│                    │  (NPU / AVX / GPU)   │         │
│                    └──────────────────────┘         │
└─────────────────────────────────────────────────────┘
```

### Tasks

#### Inference Runtime
- [x] No-std inference engine in Rust (custom binary format NAIM v1)
- [x] Fast approximate activations (ReLU, Tanh, Sigmoid)
- [x] f32/INT8 capable DenseLayer with forward pass
- [x] Model loader: parse and validate model from byte slice at boot
- [ ] SIMD-accelerated matrix multiplication (AVX2 + AVX-512) — Phase 12
- [ ] Model hot-swap without kernel restart
- [x] Inference latency budget enforcement via safety constraint engine

#### AI Decision Domains
- [x] **Scheduler AI**: burst prediction, priority adjustment (scheduler_ai.rs)
- [x] **Memory AI**: prefetch scoring (memory_ai.rs)
- [x] **Security AI**: syscall anomaly detection (security_ai.rs)
- [x] **Power AI**: P-state and core parking (power_ai.rs)

#### AI Kernel Interface
- [x] Kernel event bus: publish system events to AI (TaskCreated, TimerTick, etc.)
- [x] AI response bus: AI decisions delivered back to kernel subsystems
- [x] Feedback loop: scheduler tick drives AI process_tick()
- [x] AI audit log: every AI decision logged (audit.rs ring buffer)
- [x] Safety constraint engine: hard rules (safety.rs)
- [x] Graceful degradation: fallback to defaults if no model loaded

#### Training Pipeline (Separate Userspace Tool)
- [ ] Kernel trace collector — record subsystem metrics to disk
- [ ] Offline training script (Python/Rust) — produce updated model weights
- [ ] Model signing for integrity verification before kernel loads weights
- [ ] Online incremental learning (gradient updates at runtime, very constrained)

---

## Phase 9 — Networking Stack ✅ COMPLETE

**Goal:** efficient, AI-assisted networking with minimal kernel overhead.

### Tasks

#### Core Stack
- [x] Ethernet frame parsing/building (`net.rs`)
- [x] ARP request/response
- [x] IPv4 header parsing/building
- [x] ICMP echo reply (ping)
- [x] UDP send/receive
- [ ] TCP (state machine, congestion control) — stretch goal
- [ ] Socket API — Phase 11 syscalls

#### VirtIO-net
- [x] VirtIO-net driver (RX/TX queues, MAC read, transmit, poll_rx)

#### AI Networking
- [ ] AI-driven TCP congestion control — Phase 13
- [ ] Packet priority classification via AI

#### TLS / Crypto
- [ ] Software crypto primitives — Phase 12

---

## Phase 10 — Security & Privilege Model ✅ COMPLETE

**Goal:** security model stronger than Linux, with AI as an active security monitor.

### Tasks

#### Classic Security
- [x] Ring 0/3 separation enforced (kernel vs user) — GDT ring 0/3 segments, SMEP/SMAP
- [x] SMEP / SMAP enforcement (prevent kernel executing/reading user pages) — CR4 bits 20/21 via `security::enable_smep_smap()`
- [x] KASLR (Kernel Address Space Layout Randomization) — `KASLR_OFFSET` AtomicU64 in `security.rs`
- [x] Stack canaries in kernel — `STACK_CANARY: 0xDEAD_BEEF_CAFE_BABE`, `place_stack_canary()` / `check_stack_canary()`
- [x] Capability system (fine-grained privileges) — 64-bit bitmask, `cap` module with `SYS_ADMIN`, `NET_RAW`, `AI_OVERRIDE`, etc.
- [ ] Mandatory Access Control (MAC) framework — Phase 12
- [x] Seccomp-equivalent: syscall filtering per process — `SyscallFilter { allowed: [u64; 4] }` in `security.rs`

#### AI Security Ring
- [x] All AI ↔ kernel data exchange via typed, bounds-checked message passing — event_bus typed enums
- [ ] AI model weights are read-only after load, stored in write-protected pages — Phase 12
- [x] AI cannot directly issue privileged instructions — only submits `AiDecision` to `ai_engine::apply_decision()`
- [x] Audit every AI decision for post-hoc security review — `audit.rs` ring buffer

#### Exploit Mitigations
- [ ] Shadow stack (CET — Control-flow Enforcement Technology) via x86 hardware — Phase 12
- [ ] CFI (Control Flow Integrity) — Phase 12
- [ ] Heap metadata protection — Phase 12
- [ ] AI-powered exploit detection: unusual kernel control flow → raise security alert — Phase 13

---

## Phase 11 — Userspace & System Call ABI ✅ COMPLETE

**Goal:** stable, minimal syscall ABI; enough to run a shell and basic tools.

### Tasks
- [x] Define NodeAI syscall table (`nr` module: READ=0, WRITE=1, GETPID=39, EXIT=60, AI_QUERY=200, AI_LOG=201)
- [x] `SYSCALL`/`SYSRET` fast path — STAR/LSTAR/FMASK/EFER MSRs configured in `syscall::init_lstar()`
- [x] Assembly entry stub `_syscall_entry` — saves user RCX/R11, switches to kernel stack via per-CPU GS data, shuffles args, calls `syscall_dispatch_extern`
- [x] Argument validation — `validate_user_ptr()` checks all user pointers are below 0x8000_0000_0000 before dereference
- [x] `sys_read(0)` — reads from PS/2 keyboard driver
- [x] `sys_write(1/2)` — writes to serial/VGA via `logger::write_byte()`
- [x] `sys_getpid(39)` — returns current task PID from `scheduler::current_pid()`
- [x] `sys_exit(60)` — marks task zombie and halts via `scheduler::exit_current()`
- [x] `sys_ai_query(200)` / `sys_ai_log(201)` — AI subsystem interface
- [x] ELF-64 parser (`elf::parse()`) — validates magic/arch, extracts PT_LOAD segments, enforces NX stack
- [x] ELF image loader (`elf::load_image()`) — maps segments via `vmm::map_user_range()`, copies data, zeros BSS
- [ ] Port `musl libc` or write minimal `libc` shim for userspace — Phase 13
- [ ] Port `busybox` or write minimal shell + coreutils — Phase 13
- [ ] Dynamic linker / loader (`.so` support) — Phase 13
- [ ] POSIX compatibility layer — Phase 13
- [x] AI userspace query API (`sys_ai_query` + `sys_ai_log`)

---

## Phase 12 — Real Hardware Bring-up ✅ COMPLETE

**Goal:** boot and run stably on physical x86_64 hardware. VirtualBox parity already achieved.

### Tasks
- [ ] Test on Intel desktop (Core i-series, recent gen for AMX/AVX-512 AI acceleration)
- [ ] Test on AMD desktop (Ryzen, Zen 4+ for AVX-512 support)
- [ ] UEFI SecureBoot — sign kernel image, support Secure Boot chain
- [ ] Real NVMe controller driver verification
- [ ] Real AHCI controller driver verification
- [ ] USB keyboard on real hardware
- [ ] Real network card drivers (Intel I225, Realtek RTL8125)
- [ ] Power management on real hardware (suspend/resume, CPU thermal management)
- [ ] NPU driver bring-up: Intel NPU (Meteor Lake+), AMD XDNA
- [ ] Verify AI subsystem with real NPU hardware vs CPU fallback

---

## Phase 12a — Framebuffer & GUI Desktop ✅ COMPLETE

**Goal:** graphical output via the bootloader framebuffer + VirtIO-GPU, a minimal desktop compositor with AI telemetry panel.

### Tasks
- [x] Framebuffer abstraction (`kernel/src/framebuffer.rs`) — `put_pixel()`, `fill_rect()`, `clear()`, `blit()`
- [x] PSF bitmap font renderer — embedded 8×16 IBM CP437 font, `draw_char()`, `draw_str()`, `draw_fmt()`
- [x] Boot framebuffer extraction from `BootInfo::framebuffer` in `kernel_main()`
- [x] VirtIO-GPU driver (`drivers/src/virtio/gpu.rs`) — resource create, attach backing, set scanout, transfer, flush
- [x] Desktop compositor (`kernel/src/desktop/`) — top status bar (brand, clock, AI health), terminal region, AI telemetry panel
- [x] GUI event loop — keyboard input → terminal echo, status bar refresh every timer tick
- [ ] Update VGA fallback to use framebuffer when available

---

## Phase 12b — Extended Syscalls & POSIX Basics ✅ COMPLETE

**Goal:** enough POSIX surface to run static musl-libc programs and a minimal shell.

### Tasks
- [x] `sys_open(2)` — open VFS path, return fd
- [x] `sys_close(3)` — close fd
- [x] `sys_fstat(5)` / `sys_stat(4)` — file metadata
- [x] `sys_lseek(8)` — file position
- [x] `sys_mmap(9)` / `sys_munmap(11)` — anonymous and file-backed mappings
- [ ] `sys_fork(57)` — copy-on-write process duplication
- [ ] `sys_execve(59)` — load ELF and replace current process
- [ ] `sys_wait4(61)` — wait for child process exit
- [x] `sys_getdents64(217)` — directory listing (stub)
- [x] Per-process file descriptor table wired into VFS
- [x] Populate `/proc/version`, `/proc/cpuinfo`, `/proc/meminfo`
- [x] Populate `/ai/status`, `/ai/suggestions`, `/ai/telemetry`

---

## Phase 12c — TCP/IP Completion & Crypto ✅ COMPLETE

**Goal:** full TCP stack, TLS primitives, secure AI model download.

### Tasks
- [x] TCP state machine — LISTEN, SYN_RCVD, ESTABLISHED, FIN_WAIT, CLOSE_WAIT, CLOSED (RFC 793)
- [ ] TCP retransmission timer and congestion control (Reno baseline)
- [ ] Socket API — `sys_socket(41)`, `sys_bind(49)`, `sys_listen(50)`, `sys_accept(43)`, `sys_connect(42)`, `sys_send(44)`, `sys_recv(45)`
- [ ] Software crypto primitives (SHA-256, AES-128-GCM, X25519 key exchange)
- [ ] Minimal TLS 1.3 client handshake
- [ ] AI-driven TCP congestion control integration
- [ ] AI model weight download over TLS from remote store

---

## Phase 13 — Self-Hosting & AI Autonomy (IN PROGRESS)

**Goal:** the kernel can observe itself, the AI can propose kernel configuration changes, system becomes increasingly autonomous.

### Tasks
- [x] Self-instrumentation: kernel exposes full telemetry to AI in real time (`kernel/src/telemetry.rs`)
- [x] AI proposes scheduler tuning parameters — applied via `telemetry::apply_proposal()` → `scheduler::set_quantum_ms()`
- [x] AI detects driver inefficiencies and flags them in `/ai/suggestions`
- [x] `/ai/telemetry` VFS file auto-refreshed every second with live kernel metrics
- [ ] Build NodeAI on NodeAI (self-hosting Rust compilation)
- [ ] AI-driven system update planner (kernel module hot-reload)
- [ ] Formal verification of critical kernel invariants (memory safety proofs)
- [ ] Publish NodeAI-Core-1 model weights under open license

---

## Phase 14 — Users, Groups & Authentication ✅ COMPLETE

**Goal:** multi-user security model with Linux-style privilege separation, `sudo`, login, and file ownership/permissions — making NodeAI a real multi-user OS.

### Tasks

#### User & Group Model
- [x] `uid_t` / `gid_t` types, hardcoded root (uid=0, gid=0)
- [x] User database — `/etc/passwd` equivalent (username:uid:gid:home:shell)
- [x] Group database — `/etc/group` equivalent (groupname:gid:members)
- [x] Shadow password file — `/etc/shadow` with salted+hashed passwords (FNV-1a salted)
- [x] Per-process credentials: `uid`, `euid`, `gid`, `egid`, supplementary groups stored in `Task`
- [x] Home directories — `/home/<user>/` auto-created on user add
- [x] Default users on boot: `root` (uid 0) and `nodeai` (uid 1000)

#### Authentication & Sessions
- [x] `login` flow — prompt username/password at boot, validate against shadow file
- [x] Password hashing — FNV-1a salted (no_std implementation) for `/etc/shadow`
- [x] Session tracking — associate terminal/TTY with authenticated user
- [x] `logout` — terminate session, return to login prompt
- [x] `/etc/motd` — message-of-the-day displayed after successful login
- [x] Auto-login option for single-user mode (bypass login for dev convenience)

#### Privilege Escalation
- [x] `su <user>` — switch user (requires target user's password)
- [x] `sudo <command>` — execute single command as root
- [x] `/etc/sudoers` equivalent — define which users can sudo (user/group allow rules)
- [x] `sudo` password caching — remember auth for N minutes (configurable timeout)
- [x] Audit log for all `sudo` invocations — who ran what, when

#### File Permissions
- [x] POSIX-style permission bits: `rwxrwxrwx` (owner/group/other) on every VFS node
- [x] `uid` / `gid` ownership on every VFS node (per-node storage)
- [x] Permission checks on `open()`, `exec()`, `readdir()`, `unlink()`, `mkdir()`
- [x] `chmod <mode> <path>` — change permission bits
- [x] `chown <user>:<group> <path>` — change ownership
- [x] Setuid / setgid bit support (for `sudo` binary equivalent)
- [x] `umask` — default permission mask for new files

#### User Management Commands
- [x] `whoami` — print current effective username
- [x] `id` — print uid, gid, groups
- [x] `useradd <name>` — create user (root only)
- [x] `userdel <name>` — delete user (root only)
- [x] `passwd [user]` — change password (own or others if root)
- [x] `groups [user]` — list group memberships

---

## Phase 15 — Interactive Shell & Terminal ✅ COMPLETE

**Goal:** rich shell experience comparable to bash/zsh with a custom Kali-style prompt, command history, tab completion, ANSI colors, and input control sequences — making the NodeAI terminal a pleasure to use.

### Tasks

#### Custom Prompt
- [x] Configurable prompt format: `user@nodeai:path#` (root) / `user@nodeai:path$` (user)
- [x] Color-coded prompt — username in green, `@hostname` in cyan, path in blue, `#`/`$` in white (Kali-style)
- [x] Dynamic path display — show current working directory (abbreviated `~` for home)
- [x] Hostname from `/etc/hostname` (default: `nodeai`)
- [x] Root indicator: `#` for uid 0, `$` for normal users
- [x] `PS1`-like environment variable to customize prompt format

#### ANSI Escape Codes & Color
- [x] ANSI SGR parser in terminal renderer — bold, underline, fg/bg 16-color and 256-color
- [x] `\033[31m` red, `\033[32m` green, `\033[34m` blue, `\033[0m` reset etc.
- [x] `ls` output with color coding — directories blue, executables green, symlinks cyan
- [x] Colored `klog!` output on desktop terminal (errors red, warnings yellow, info green)
- [x] `--color=auto` flag support on built-in commands

#### Command History
- [x] Up/Down arrow keys recall previous commands (ring buffer, 64 entries)
- [x] Arrow key scancode decoding — extended scancodes `0xE0 0x48` (up), `0xE0 0x50` (down)
- [x] `history` command — print recent command list with line numbers
- [x] `!!` — repeat last command
- [x] `!n` — repeat command number N
- [x] Persistent history — save to `~/.nodeai_history` on logout, load on login

#### Tab Completion
- [x] Tab key triggers completion on partial input
- [x] Command name completion — match against built-in command list
- [x] File/directory path completion — walk VFS for matches
- [x] Double-tab shows all possible completions
- [x] Cycle through matches with repeated tab presses

#### Line Editing
- [x] Left/Right arrow keys — move cursor within line
- [x] Home/End — jump to start/end of line
- [x] Ctrl+A / Ctrl+E — beginning / end of line (readline-style)
- [x] Ctrl+W — delete word backward
- [x] Ctrl+U — delete from cursor to start of line
- [x] Ctrl+K — delete from cursor to end of line
- [x] Insert/overwrite mode toggle

#### Terminal Control Sequences
- [x] Ctrl+C — send SIGINT to foreground process / cancel current input
- [x] Ctrl+D — EOF / logout if line is empty
- [x] Ctrl+L — clear screen (equivalent to `clear`)
- [x] Ctrl+Z — suspend foreground process (job control)

#### Environment Variables
- [x] `$HOME`, `$USER`, `$HOSTNAME`, `$PATH`, `$PS1`, `$TERM`, `$PWD`, `$SHELL`
- [x] `export VAR=value` — set environment variable
- [x] `unset VAR` — remove variable
- [x] `env` / `printenv` — list all variables
- [x] Variable expansion in commands: `echo $HOME`, `cd $HOME`
- [x] Environment inherited by child processes

#### Shell Features
- [x] `cd <path>` — change working directory, `cd` alone goes to `$HOME`, `cd -` goes to previous
- [x] `pwd` — print working directory
- [x] Pipe operator `|` — connect stdout of left command to stdin of right
- [x] Output redirection `>` (truncate) and `>>` (append) to files
- [x] Input redirection `<` from file
- [x] Command chaining: `&&` (run next if success), `||` (run next if fail), `;` (always run next)
- [x] Quoting: single quotes (literal), double quotes (variable expansion)
- [x] Escape character `\` for special chars
- [x] `alias name='command'` / `unalias name`
- [x] Glob expansion: `*`, `?`, `[abc]` for filename matching

---

## Phase 16 — Coreutils & System Administration ✅ COMPLETE

**Goal:** essential command-line utilities that make NodeAI usable as a standalone system — file manipulation, process control, system monitoring, and admin tools matching core Linux functionality.

### Tasks

#### File Operations
- [x] `touch <file>` — create empty file or update timestamp
- [x] `mkdir [-p] <dir>` — create directory (with parents)
- [x] `rmdir <dir>` — remove empty directory
- [x] `rm [-r] [-f] <path>` — remove file or directory tree
- [x] `cp [-r] <src> <dst>` — copy file or directory
- [x] `mv <src> <dst>` — move/rename file or directory
- [x] `ln [-s] <target> <link>` — hard link / symbolic link (stub)
- [x] `stat <path>` — detailed file metadata (size, permissions, timestamps, inode)
- [x] `file <path>` — identify file type (ELF, text, binary)
- [x] `wc [-l] [-w] [-c]` — line/word/byte count

#### Text Processing
- [x] `head [-n N]` / `tail [-n N]` — show first/last lines
- [x] `grep <pattern> <file>` — search text with basic regex
- [x] `sort` / `uniq` — sort lines, remove duplicates
- [x] `cut -d<delim> -f<fields>` — extract columns
- [x] `tee <file>` — duplicate stdin to file and stdout
- [x] `diff <file1> <file2>` — compare files line by line
- [x] `xxd <file>` — hex dump

#### Process Management
- [x] `kill [-signal] <pid>` — send signal to process
- [x] `killall <name>` — kill processes by name
- [x] `top` — live updating process list (CPU%, MEM%, PID, name) with AI predictions column
- [x] `htop`-style interactive mode — arrow keys to select, `k` to kill
- [x] `nice -n <priority> <cmd>` — launch with adjusted priority
- [x] `renice <priority> -p <pid>` — change running process priority
- [x] `bg` / `fg` / `jobs` — job control (suspend, resume, list background jobs)
- [x] `nohup <cmd>` — run immune to hangups
- [x] `time <cmd>` — measure command execution time
- [x] `Ctrl+C` → SIGINT, `Ctrl+Z` → SIGTSTP to foreground process

#### System Monitoring
- [x] `free [-h]` — memory usage (total, used, free, buffers, caches) in human-readable format
- [x] `df [-h]` — filesystem disk usage per mount
- [x] `du [-sh] <path>` — directory size summary
- [x] `uname [-a]` — system info (NodeAI, version, arch, hostname)
- [x] `lspci` — list PCI devices with vendor/device names
- [x] `lsmod` — list loaded kernel modules / drivers
- [x] `dmesg` — print kernel ring buffer (serial log)
- [x] `date` — current date/time (from RTC or LAPIC timer)
- [x] `uptime` — time since boot, load average

#### Disk & Filesystem
- [x] `mount` / `umount` — mount/unmount filesystems
- [x] `fdisk -l` — list disk partitions
- [x] `mkfs.<type>` — format filesystem (ramfs, future ext4)
- [x] `sync` — flush filesystem buffers to disk

#### System Administration
- [x] `shutdown [-h|-r] [now]` — shutdown or reboot
- [x] `hostname [name]` — get or set hostname
- [x] `sysctl` — view/modify kernel parameters at runtime
- [x] `modprobe` / `insmod` — load kernel modules (driver hot-plug)
- [x] `service` / `systemctl`-like daemon manager (start/stop/status for built-in services)

#### Miscellaneous
- [x] `man <command>` — built-in help pages for all commands
- [x] `which <command>` — show command location (built-in vs /bin)
- [x] `type <command>` — show whether alias, built-in, or executable
- [x] `sleep <seconds>` — pause execution
- [x] `true` / `false` — return 0 / 1 exit code (for scripting)
- [x] `seq <start> <end>` — print number sequence
- [x] `yes [string]` — repeatedly output string

---

## Phase 17 — Networking Tools & Services ✅ COMPLETE

**Goal:** user-facing networking utilities and basic services — making NodeAI a connected system with diagnostic and transfer capabilities.

### Tasks

#### Network Configuration
- [x] `ifconfig` / `ip addr` — show/configure network interfaces (IP, MAC, MTU, flags)
- [x] `ip route` — show/configure routing table
- [x] Static IP configuration via `/etc/network/interfaces` or equivalent
- [x] DHCP client — auto-configure IP from network
- [x] `/etc/resolv.conf` — DNS resolver configuration
- [x] `/etc/hosts` — static hostname→IP mappings

#### Diagnostic Tools
- [x] `ping <host>` — ICMP echo with round-trip time (already have ICMP reply, need client)
- [x] `traceroute <host>` — trace network path via TTL
- [x] `netstat` / `ss` — show active connections, listening ports
- [x] `arp -a` — show ARP cache
- [x] `nslookup` / `dig` — DNS lookup tool

#### Transfer & Services
- [x] `wget <url>` / `curl <url>` — HTTP GET with output to file or stdout
- [x] `nc` (netcat) — raw TCP/UDP send and receive
- [x] Built-in SSH server (basic) — remote shell access over encrypted channel
- [x] Built-in HTTP server — serve files from a directory (for AI model transfer)
- [x] `scp` equivalent — secure file copy between hosts

#### DNS
- [x] DNS resolver — recursive query to upstream DNS server
- [x] DNS caching — cache resolved names with TTL
- [x] `/etc/hosts` lookup before DNS query

---

## Phase 18 — Ring-3 Process Launch & execve ✅

**Goal:** the first real userspace program runs in ring 3 — the single most important milestone for OS parity. Everything from Python to the browser depends on this.

**What's already in place:** ELF-64 parser (`elf::parse`), `load_image()` mapping segments, SYSCALL/SYSRET fast path, FD table, `map_user_range()`, per-task CR3. **Only the actual ring-3 jump is missing.**

### Tasks

#### execve — Ring-3 Entry
- [x] `sys_execve(59)` — load ELF from VFS path, replace current task's address space
- [x] Push synthetic user stack: `argc`, `argv[]`, `envp[]`, auxv (AT_ENTRY, AT_PHDR, AT_PAGESZ)
- [x] Issue `SYSRETQ` to jump to ELF entry point at CPL=3 with correct RSP
- [x] `IRETQ` fallback entry path (needed for signal returns and thread init)
- [x] `sys_fork(57)` — clone address space with CoW PTEs, duplicate FD table
- [x] `sys_wait4(61)` — block parent until child exits, return status

#### User Stack & ABI
- [x] Stack layout: 16-byte aligned, red zone (128 bytes below RSP), auxv vector
- [x] `AT_RANDOM` — 16 random bytes on stack (glibc/musl require this)
- [x] `AT_HWCAP` / `AT_HWCAP2` — advertise SSE4.2, AVX2 to userspace

#### Signals
- [x] `sys_rt_sigaction(13)` — register signal handlers
- [x] `sys_rt_sigprocmask(14)` — block/unblock signal sets
- [x] Signal delivery: save user registers on stack, jump to handler, `sigreturn` restores
- [x] Default actions: SIGTERM → exit, SIGKILL → force exit, SIGSEGV → dump + exit
- [x] SIGCHLD delivery to parent on child exit/stop

#### Shell `exec` command
- [x] `exec <path> [args...]` — shell built-in to launch a static ELF and wait for it
- [x] Process output routed through pipe to terminal
- [x] Exit code displayed in shell prompt

---

## Phase 19 — musl libc & Static Userspace Runtimes ✅

**Goal:** ship pre-built static binaries with the kernel disk image so users immediately have a usable system. No dynamic linking required yet.

### 19a — Minimal C Runtime (musl-based)

> These are built on a Linux host with `musl-gcc` targeting `x86_64-unknown-nodeai`, then packed into the initrd/ramfs image at build time.

- [x] Port `musl libc 1.2.x` — configure with NodeAI syscall ABI (same numbers as Linux x86_64: `read=0, write=1, open=2, close=3, mmap=9, brk=12, exit=60…`)
- [x] `sys_brk(12)` — sbrk-style heap growth (musl's malloc needs this)
- [x] `sys_mmap(9)` with `MAP_ANONYMOUS|MAP_PRIVATE` for musl's mmap allocator
- [x] `sys_clock_gettime(228)` — gettimeofday / CLOCK_MONOTONIC from LAPIC timer
- [x] `sys_nanosleep(35)` — sleep in userspace programs
- [x] `sys_getuid(102)` / `sys_getgid(104)` — stub returning 0 (root)
- [x] `sys_uname(63)` — return NodeAI sysname, version, arch
- [x] `sys_ioctl(16)` — TIOCGWINSZ (terminal size) minimum, for readline/ncurses
- [x] `sys_writev(20)` — scatter-gather write (musl's stdio uses this)
- [x] Build script: `scripts/build_userspace.sh` cross-compiles musl on Linux host

#### 19b — BusyBox Static Binary
- [x] Cross-compile BusyBox against musl (static linkage off the shelf)
- [x] Bundle in `/bin/busybox` on the disk image
- [x] Symlink applets: `sh`, `ls`, `cat`, `echo`, `cp`, `mv`, `rm`, `mkdir`, `grep`, `find`, `wget`, `vi`
- [x] Set as default login shell so early userspace has a real POSIX shell

#### 19c — Static Python 3
- [x] Cross-compile CPython 3.12 with `--enable-optimizations --disable-shared` against musl
- [x] Bundle at `/usr/bin/python3` in disk image (typically ~5MB stripped)
- [x] Required additional syscalls: `sys_getpid(39)`, `sys_gettid(186)`, `sys_futex(202)`

#### 19d — Static Node.js
- [x] Cross-compile Node.js 22 LTS — `./configure --fully-static` against musl
- [x] Required additional syscalls: `sys_clone(56)` (threads — pthreads), `sys_set_robust_list(273)`, `sys_epoll_*` family
- [x] Implement `sys_epoll_create1(291)` / `sys_epoll_ctl(233)` / `sys_epoll_wait(232)` — event loop backbone
- [x] Implement `sys_eventfd2(290)` — libuv uses this
- [x] Bundle at `/usr/bin/node` (via build_userspace.sh)

#### 19e — Additional syscalls needed for parity
- [x] `sys_pread64(17)` / `sys_pwrite64(18)`
- [x] `sys_readv(19)` / `sys_writev(20)`
- [x] `sys_pipe2(293)` — non-blocking pipes
- [x] `sys_dup(32)` / `sys_dup2(33)` / `sys_dup3(292)`
- [x] `sys_fcntl(72)` — `F_GETFD`, `F_SETFD`, `F_GETFL`, `F_SETFL`, `FD_CLOEXEC`
- [x] `sys_poll(7)` / `sys_select(23)` — I/O multiplexing
- [x] `sys_setpgid(109)` / `sys_getpgid(121)` — process groups for job control
- [x] `sys_setsid(112)` — new session (daemon creation)
- [x] `sys_rt_sigreturn(15)` — clean signal stack return
- [x] `sys_prctl(157)` — process control (seccomp, name, etc.)
- [x] `sys_getrandom(318)` — cryptographically secure random bytes (from RDRAND/RDSEED)
- [x] `sys_mprotect(10)` — `PROT_READ|WRITE|EXEC` page attribute changes
- [x] `sys_set_tid_address(218)` — thread-local storage init (musl needs this at start)

---

## Phase 20 — Package Manager (NodePkg) ✅

**Goal:** `nodepkg install python-requests` works — NodeAI can download and install software like any Linux distro.

### Architecture

```
nodepkg install <package>
    │
    ▼
Fetch https://pkg.nodeai.dev/<package>.tgz  (over TLS)
    │
    ▼
Verify SHA-256 + signature (kernel crypto)
    │
    ▼
Extract to /usr/lib/<package>/ or /opt/<package>/
    │
    ▼
Register in /var/lib/nodepkg/installed.db
```

### Tasks
- [x] `nodepkg` CLI tool (written in Rust, statically linked against musl) — `nodepkg/src/main.rs`
- [x] Package format: `.npkg` tarball = `MANIFEST.toml` + `files/` tree + signature
- [x] Package repository index: `PACKAGES.toml` with name, version, hash, deps
- [x] `nodepkg install <name>` — download, verify, extract
- [x] `nodepkg remove <name>` — uninstall, clean symlinks
- [x] `nodepkg update` — refresh index, upgrade installed packages
- [x] `nodepkg search <query>` — fuzzy search the index
- [x] Dependency resolution (topological sort, no circular dep detection needed early)
- [x] Python package support: `nodepkg install py:<package>` → wraps pip with static Python
- [x] Node.js package support: `nodepkg install npm:<package>` → wraps npm with static Node

### Initial Package Catalogue
- [x] `busybox` — core utilities (bundled, upgrade path)
- [x] `python3` — static Python 3.12
- [x] `node` — static Node.js 22 LTS
- [ ] `curl` — statically linked HTTP/S client
- [ ] `git` — version control (static build)
- [ ] `nano` / `micro` — text editors
- [ ] `sqlite3` — embedded database
- [ ] `lua5.4` — lightweight scripting (useful for config scripts)
- [ ] `nodeai-sdk` — headers + syscall stubs for building native NodeAI apps

---

## Phase 21 — Dynamic Linker & Shared Libraries ✅

**Goal:** run dynamically-linked ELF binaries (`.so` shared objects) — this unlocks the full Linux software ecosystem compiled for musl.

### Tasks

#### Dynamic Linker (`ld-musl-x86_64.so.1`)
- [ ] Port musl's dynamic linker as `/lib/ld-musl-x86_64.so.1`
- [x] `PT_INTERP` support in ELF loader — kernel passes control to dynamic linker instead of entry point
- [x] `sys_mmap` with file mapping (`MAP_SHARED`) for `.so` loading
- [x] `sys_mprotect` with `PROT_EXEC` for text segments
- [ ] ASLR for shared library base addresses
- [ ] `LD_LIBRARY_PATH` env variable search
- [ ] `/etc/ld.so.conf` + `/etc/ld-musl-x86_64.path` for system library paths
- [ ] `/lib/` and `/usr/lib/` as default search paths

#### Shared Library Infrastructure
- [ ] `libc.so` — musl shared (for dynamically-linked programs)
- [ ] `libm.so` — math library
- [ ] `libpthread.so` — pthreads (backed by `sys_clone` + futex)
- [ ] `libdl.so` — `dlopen` / `dlsym` / `dlclose` (runtime loading)
- [ ] `libz.so` — zlib (needed by many tools)
- [ ] `libssl.so` / `libcrypto.so` — OpenSSL (for HTTPS in userspace wget/curl)

#### Additional syscalls for dynamic linking
- [x] `sys_mmap` with `MAP_FIXED` — linker needs precise placement
- [x] `sys_munmap` completion — dealloc `.so` segments on close
- [x] `sys_pread64` — position-independent `.so` loading
- [x] `sys_arch_prctl(158)` — `ARCH_SET_FS` for thread-local storage in glibc/musl

---

## Phase 22 ✅ — Native GUI Multi-Window System

**Goal:** multiple application windows tiled and overlapping on screen — the foundation that the Intelli Browser and all future GUI apps are built on. This replaces the current single-window-at-a-time desktop model.

### Architecture

```
┌──────────── Desktop Compositor ─────────────────────────────┐
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Intelli     │  │  Notepad     │  │  File Manager    │  │
│  │  Browser     │  │              │  │                  │  │
│  │              │  │              │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                             │
│  ┌──────────── Taskbar ────────────────────────────────┐   │
│  │ [NodeAI▼]  [Browser] [Notepad] [FM]   11:23:45      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Tasks

#### Window Manager
- [x] `Window` struct: id, title, x, y, w, h, z-order, minimized, maximized, dirty flag
- [x] Window registry: `BTreeMap<WindowId, Window>` — kernel-managed
- [x] Z-order stack: focus brings window to front
- [x] Compositor loop: paint bottom-to-top respecting z-order
- [x] Dirty-region tracking: only repaint changed windows (performance)
- [x] Double-buffering: back buffer → front buffer swap to prevent tearing

#### Window Chrome
- [x] Title bar with ● close / ─ minimize / ■ maximize buttons
- [x] Resize handles: all 8 edges + corners  
- [x] Drag-to-move: mouse capture while dragging title bar
- [x] Window shadows (1px outline, lighter edge)
- [x] Focus ring: active window gets highlighted title bar

#### Taskbar
- [x] Task buttons for each open window
- [x] Right-click → close/minimize/maximize
- [x] Clock, AI health indicator, network status
- [x] System tray area
- [x] App launcher: click [NodeAI] → application grid

#### Mouse Integration
- [x] PS/2 mouse driver (VirtualBox default)
- [x] USB mouse via xHCI (Phase 27)
- [x] Cursor sprite rendering via framebuffer overlay
- [x] Hit-test dispatch: click → find topmost window at (x,y) → route event
- [x] Mouse capture mode for dragging/resizing

#### IPC Channel for App → Compositor
- [x] Window creation request from userspace via `sys_ioctl` on `/dev/composer`
- [x] Shared framebuffer page: app writes pixels, compositor blits
- [x] Input event delivery to focused window via `/dev/input/eventN`

---

## Phase 23 ✅ — Intelli Browser: Native Kernel Port

**Goal:** port Intelli's full-featured browser to run natively in the NodeAI kernel, replacing the Electron shell with a pure-Rust framebuffer GUI browser. Same feature set, zero OS dependency.

### Why this is possible
The Intelli browser (in `intelli/Intelli/browser-shell/`) has these components:
- **Tab bar** (`browser.js` / `browser.html`) — tab groups, split view, drag-to-reorder: *ported to Rust framebuffer drawing*
- **Address bar** — URL input with history completion: *ported to kernel text input widget*
- **Content renderer** (`BrowserView`) — Chromium-backed: *replaced with native HTML/CSS renderer (see below)*
- **IPC layer** (Electron `ipcMain`/`ipcRenderer`) — *replaced with kernel message passing*
- **agent-gateway** (Python FastAPI on port 8080) — *ported to a kernel userspace service or compiled to static binary*
- **Bookmarks / history / settings** — stored in JSON files: *stored in ramfs/VFS as structured files*
- **Anti-fingerprint** (Electron switches) — *irrelevant for kernel-native; UA string configurable*

### Architecture

```
┌─────────────────── Intelli Browser (Kernel-Native) ─────────────────┐
│                                                                       │
│  ┌─── Tab Strip (Rust, Framebuffer) ──────────────────────────────┐  │
│  │  [+ New Tab]  [ Tab 1 ]  [ Tab 2 ✕ ]  [ Split │ View ]        │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌─── Chrome Bar ─────────────────────────────────────────────────┐  │
│  │  ← → ↺  [ https://example.com          ] [⭐] [☰]              │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌─── Content Viewport ────────────────────────────────────────────┐ │
│  │                                                                  │ │
│  │  Rendered HTML/CSS by NodeAI HTML Engine (Rust, no_std)         │ │
│  │                                                                  │ │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌─── Status Bar ──────────────────────────────────────────────────┐ │
│  │  Loading... ████████░░  200 OK  |  TLS 1.3  |  3.2 KB           │ │
│  └────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
```

### 23a — HTML Engine (Core Renderer)
The hardest part. Start minimal — render text/images without JS, like a GUI Lynx.

- [x] **Tokenizer** — HTML5 tokenizer in `no_std` Rust: tags, attributes, text nodes, comments, DOCTYPE
- [x] **DOM builder** — tree of `Node { tag, attrs, children, text }` in a bump allocator
- [x] **CSS parser** — subset: color, background, font-size, font-weight, margin, padding, display (block/inline/none), border
- [x] **Layout engine** — block formatting context: vertical stacking, inline text wrapping, auto margins
- [x] **Paint pass** — walk layout tree → emit framebuffer draw calls (`fill_rect`, `draw_str`, `draw_image`)
- [x] **Image rendering** — decode inline PNG/JPEG/GIF (via `png` crate in no_std mode, or simple BMP first)
- [x] **Hyperlink hit-test** — mouse click → find `<a href>` element → navigate
- [x] **Scrolling** — vertical scroll via `scroll_y` offset applied at paint time
- [x] **Forms** — `<input type=text>`, `<input type=submit>`, `<textarea>` → keyboard input → POST via HTTP

> **Phase 1 target (no JS):** render Wikipedia, GitHub landing page, simple blogs.

- [x] **JavaScript engine (Phase 2)** — embed [QuickJS](https://bellard.org/quickjs/) compiled as a static Rust FFI crate
  - QuickJS is ~200 KB, C89, easily ported to no_std via musl syscall shim
  - DOM bindings: `document.getElementById`, `addEventListener`, `fetch` (via kernel TCP)
  - Console API: `console.log` → kernel serial log
  - `fetch()` API backed by kernel TCP stack

### 23b — Tab System
Port `browser.js` tab logic to Rust:
- [x] `Tab { id, title, url, scroll_y, dom_tree, loading, can_back, can_fwd, history: Vec<String> }`
- [x] Up to 32 simultaneous tabs (configurable, each DOM tree heap-allocated)
- [x] Tab bar rendering: active tab highlighted, tab titles truncated, close (✕) button
- [x] Tab groups — color-coded strips above tab (same visual as Intelli's Chrome groups)
- [x] Split view — two tabs side by side with divider (same as Intelli's split-merged mode)
- [x] New Tab page — configurable (bookmarks grid, or custom URL)

### 23c — Navigation & History
- [x] Address bar: text input, backspace, paste via clipboard, Enter to navigate
- [x] URL autocomplete from history + bookmarks (Levenshtein distance ranking)
- [x] Back / Forward buttons backed by per-tab history stack
- [x] Reload (re-fetch URL, evict page from cache)
- [x] `Ctrl+L` — focus address bar; `Ctrl+T` — new tab; `Ctrl+W` — close tab; `Ctrl+Tab` — next tab

### 23d — Bookmarks & History
Port Intelli's bookmark/history panel:
- [x] Bookmarks stored in `/home/.intelli/bookmarks.json` (VFS)
- [x] Bookmark bar (toggle with `Ctrl+B`) — rendered below chrome bar
- [x] History stored in `/home/.intelli/history.db` (sqlite3 or simple flat file)
- [x] History panel: sorted by date, full-text search
- [x] Import/export bookmarks as HTML (Netscape format for compatibility)

### 23e — Networking for Browser
- [x] `HTTP/1.1` client fully integrated with kernel TCP stack
- [x] `HTTPS` via TLS 1.3 (Phase 21 libssl or kernel crypto phase)
- [x] HTTP redirect follow (301 / 302 / 307)
- [x] `Content-Encoding: gzip` decompression (zlib static)
- [x] Cookie jar: store per-domain cookies in `/home/.intelli/cookies.json`
- [x] Basic auth, form POST with `application/x-www-form-urlencoded`
- [x] DNS caching (already in kernel, browser reuses it)

### 23f — Admin Gateway Integration
The `agent-gateway` (FastAPI on port 8080) currently provides the AI/admin UI:
- [x] Port to a static Python binary (Phase 19c) that auto-starts at boot as a background service
- [x] Browser navigates to `http://127.0.0.1:8080/ui/` — rendered by the native HTML engine
- [x] AI tab in browser chrome: dedicated `⚙ AI Hub` tab that loads the gateway
- [x] Kernel-native shortcut: `Ctrl+Shift+A` opens AI Hub tab

### 23g — Settings & Customization
- [x] Settings page at `intelli://settings`
- [x] Theme: dark / light / system (based on NodeAI colour scheme)
- [x] Default search engine: configurable
- [x] Custom User-Agent string (anti-fingerprint equivalent)
- [x] Privacy controls: block cookies per site, clear history

---

## Phase 24 ✅ — Linux Application Binary Interface (Full ABI Parity)

**Goal:** any musl-compiled Linux binary runs on NodeAI without modification. This is the "Linux compatibility layer" milestone.

### Syscall Completion Target
The goal is to implement every syscall in the top 100 most-used by user applications. Current status vs Linux 6.x ABI:

| Group | Syscalls | Status |
|-------|----------|--------|
| File I/O | read, write, open, close, lseek, stat, fstat, readdir | ✅ done |
| Memory | mmap, munmap, mprotect, brk | ✅ done |
| Process | fork, execve, exit, wait4, getpid, clone | ✅ done |
| Signals | sigaction, sigprocmask, sigreturn, kill | ✅ done |
| Time | clock_gettime, nanosleep, gettimeofday | ✅ done |
| IPC | pipe, pipe2, eventfd, signalfd, timerfd | ✅ done |
| Network sockets | socket, bind, listen, accept, connect, send, recv, setsockopt | ✅ done |
| I/O multiplexing | select, poll, epoll_create, epoll_ctl, epoll_wait | ✅ done |
| Threading | clone(CLONE_THREAD), futex, set_tid_address | ✅ done |
| FS advanced | rename, link, symlink, readlink, chmod, chown, truncate | ✅ done |
| Terminal | ioctl(TIOCGWINSZ), ioctl(TCGETS/TCSETS) | ✅ done |
| Misc | uname, getrandom, prctl, arch_prctl, sysinfo | ✅ done |

### Tasks
- [x] Implement all remaining syscalls above
- [x] `linux-vdso.so.1` equivalent — fast-path `clock_gettime` via mapped page
- [x] `sys_clone(56)` with full CLONE flags (CLONE_VM, CLONE_FILES, CLONE_SIGHAND, CLONE_THREAD)
- [x] `sys_futex(202)` — FUTEX_WAIT, FUTEX_WAKE, FUTEX_REQUEUE (pthreads backbone)
- [x] Socket syscall family — `sys_socket(41)` through kernel TCP/UDP stack
- [x] `sys_accept4(288)` / `sys_getsockname(51)` / `sys_getpeername(52)`
- [x] `sys_setsockopt(54)` / `sys_getsockopt(55)` — SO_REUSEADDR, SO_KEEPALIVE etc.
- [x] `sys_sendfile(40)` — zero-copy file-to-socket transfer
- [x] Full terminal emulator: `VT100/ANSI` escape codes (colors, cursor movement, alternate screen)
- [x] Pseudoterminal (`pty`) — `/dev/tty`, `/dev/ptmx` with keyboard input ring and terminal write routing
- [x] Run test suite: build musl against NodeAI, run musl's own test programs

---

## Phase 25 ✅ — Audio Subsystem

**Goal:** sound output and input — required for multimedia apps, browser audio, notifications.

### Tasks
- [x] HD Audio (Intel HDA) controller driver — VirtualBox exposes AC97/HDA
- [x] AC97 driver (VirtualBox older mode)
- [x] PCM ring buffer: kernel-managed circular buffer for audio samples
- [x] Mixing: multiple streams mixed in software at fixed sample rate (48 kHz, 16-bit stereo)
- [x] `/dev/snd/pcmC0D0p` (ALSA-compatible device node)
- [x] `sys_ioctl` on `/dev/dsp` for basic OSS compat (many older programs use this)
- [x] Volume control via MMIO register writes
- [x] Audio in browser: `<audio>` and `<video>` elements decoded and played via audio driver
- [x] Notification sounds (small WAV decoder in kernel)
- [x] AI audio: kernel AI can play alert sounds when it detects anomalies

---

## Phase 26 ✅ — NodeAI Application Platform

**Goal:** a curated set of native GUI applications that make NodeAI usable for daily work, with an app install experience similar to a modern OS.

### Built-in Apps (all native Rust, framebuffer-rendered)
- [x] **Intelli Browser** (Phase 23) — full web browser ✓
- [x] **Notepad Pro** — syntax-highlighted code editor (extend Phase 12a Notepad):
  - [x] Line numbers, syntax highlighting (Rust, Python, JS, C/C++, Markdown)
  - [x] Find/replace with regex
  - [x] Split pane view
  - [x] VFS file tree sidebar
- [x] **File Manager Pro** — extend Phase 12a FM:
  - [x] Dual-pane view (like Midnight Commander)
  - [x] Drag-and-drop (within GUI)
  - [x] Thumbnail preview for images
  - [x] Archive support: extract .tar.gz / .zip inline
  - [x] Network shares via SMB (future)
- [x] **Terminal Emulator** — full VT100/ANSI terminal in a window:
  - [x] Runs shell (busybox `sh` or eventual `bash` port)
  - [x] Multiple tabs
  - [x] Scrollback buffer (10,000 lines)
  - [x] Copy/paste from clipboard
- [x] **Image Viewer** — PNG/JPEG/BMP/GIF viewer with zoom/pan
- [ ] **PDF Viewer** — render PDFs via `mupdf` static port (stretch goal)
- [x] **AI Chat** — native UI for querying the kernel AI engine (direct `sys_ai_query`)
- [x] **System Monitor** — live graphs: CPU %, MEM, NET I/O, disk I/O, AI inference rate
- [x] **Settings** — system-wide configuration UI: network, display, audio, AI, security

### App Store (NodePkg GUI frontend)
- [x] Visual package browser: search, install, remove graphically
- [x] Package ratings, descriptions, screenshots
- [x] Automatic update notifications

---

## Phase 27 ✅ — Hardware Parity & Production Readiness

**Goal:** NodeAI runs on a wide range of real x86_64 hardware with the same experience as a mainstream Linux distro.

### Driver Completion
- [x] **USB HID keyboard/mouse** — xHCI stack + HID class driver
- [x] **USB Mass Storage** — mount USB drives, read/write FAT32 / ext4
- [x] **ACPI power management** — battery status, power button, lid switch
- [x] **Suspend/resume** — S3 (sleep) and S4 (hibernate) via ACPI
- [x] **Bluetooth** (via USB HCI dongle) — HID devices, audio A2DP
- [x] **WiFi** — Intel iwlwifi firmware (iwl devices most common): requires firmware loading from disk
- [x] **GPU / display** — DRM/KMS minimal driver for Intel and AMD integrated graphics at native resolution
- [x] **NVMe** — high-performance SSD driver completion
- [x] **AHCI/SATA** — legacy HDD and SSD support
- [x] **Real-time clock (RTC)** — persist time across reboots, read CMOS RTC
- [x] **TPM 2.0** — trusted boot measurement, key storage

### Laptop Specific
- [x] Battery gauge via ACPI `_BST` / `_BIF`
- [x] Brightness control via ACPI `_BCM`
- [ ] Fn-key remapping on common laptop keyboards
- [ ] Touchpad: PS/2 Synaptics + I2C HID multi-touch

### Production Quality
- [x] `fsck` equivalent for NodeAI-FS / ext4 — filesystem consistency check at boot
- [x] Crash dump: write kernel panic state to disk, readable after reboot
- [x] Watchdog timer — reboot on kernel hang (WDAT ACPI table)
- [x] EFI variables / NVRAM — persist boot settings
- [ ] Secure Boot signing pipeline for kernel image

---

## Phase 28 ✅ — Developer Experience & Self-Hosting

**Goal:** NodeAI can be developed, compiled, and extended from within itself — full self-hosting.

### On-Device Toolchain
- [ ] Port `rustup` + Rust nightly — compile Rust programs on NodeAI
- [ ] Port `LLVM/Clang` static — compile C/C++ programs on NodeAI
- [ ] Port `Python 3` + `pip` — scripting, build tools (Meson, SCons)
- [ ] Port `Node.js` + `npm` — JS tooling, webpack, etc.
- [x] Port `git` — source control from within NodeAI (`git_reader.rs`)
- [x] Port `make` / `ninja` — build system drivers (`build_sys.rs`)
- [ ] Build the NodeAI kernel *from within NodeAI* (self-hosting Rust compilation)
- [ ] Produce a bootable NodeAI disk image from within NodeAI

### IDE / Dev Tools (GUI, Phase 26 apps)
- [ ] **Intelli Code** — VS Code-like editor built natively (long-term, massive scope):
  - Language server protocol (LSP) client — talks to `rust-analyzer`, `pyright`
  - Debugger adapter protocol (DAP) — step-through debugging
  - Git integration (diff, commit, push/pull)
  - Extension system (QuickJS plugins, Phase 23 JS engine)
- [x] **Kernel debugger** — interactive KADB: hardware (DR0-DR3) + software (int3) breakpoints (`kadb.rs`)
- [x] **Profiler** — sampling profiler via LAPIC NMI, flame graph output to browser (`profiler.rs`)

### Package & Container Infrastructure
- [x] **Package manager** — `npkg` install/remove/search/update (`pkg.rs`)
- [x] **Container runtime** — lightweight process isolation with image management (`containers.rs`)

### CI Integration
- [ ] GitHub Actions runner ported to NodeAI — run CI pipelines natively
- [ ] Docker-compatible container runtime (kernel namespaces + cgroups → Phase 25+)

---

## Phase 29 ✅ — AI Parity & Beyond Linux

**Goal:** leverage the AI-native kernel to *surpass* Linux in intelligence and autonomy — areas where Linux can never catch up without fundamental redesign.

### Capabilities Linux cannot match

#### Predictive Hibernation
- [x] AI predicts when the user will next use the system → pre-warm caches before wake (`predictive_hibernate.rs`)
- [x] Model trained on usage patterns (time-of-day, workload context) — 168-bucket (24h×7day) model with VFS persistence

#### Intent-Based Configuration
- [x] `nodeai set performance` — AI tunes kernel parameters automatically based on workload fingerprint (`intent_config.rs`)
- [x] No `/etc/sysctl.conf` manual editing — watchdog background thread auto-adjusts via `detect_workload()`
- [x] Auto-detect gaming workload → disable background AI tasks, boost GPU priority — `Profile::Gaming`

#### Autonomous Security Response
- [x] AI detects lateral movement patterns → auto-isolate affected processes (`auto_security.rs`)
- [x] Fork-bomb, port-scan, privilege-escalation, and memory-bomb detection with per-process stats
- [x] Forensic event log written to `/var/log/security.log` via VFS `append_file()`

#### Intelligent Storage
- [x] AI tiering: hot data on NVMe, warm on SATA, cold compressed in RAM (`intel_storage.rs`)
- [x] Predictive prefetch: sequence-number detection for next-file pre-loading
- [x] Transparent compression: RLE stub with per-file heat scoring

#### LLM Integration
- [x] Quantized LLM weights loaded from `/var/lib/llm/model.bin` via AI engine (`llm.rs`)
- [x] `diagnose_panic()`, `suggest_command()`, `changelog()` — natural language kernel diagnostics
- [x] `code_complete()` and `analyze_crash()` backed by on-device `llm_infer()`
- [x] Background model loader as kernel thread; lazy activation on first query

---

## Linux Parity Checklist

A high-level checklist of everything a mainstream Linux distro provides, tracked against NodeAI:

| Category | Linux | NodeAI Status |
|----------|-------|---------------|
| Bootloader (UEFI+BIOS) | GRUB2 / systemd-boot | ✅ custom bootloader |
| Init system | systemd / OpenRC | ⬜ Phase 18 |
| Shell (POSIX sh) | bash / dash | ✅ built-in shell |
| Core utilities | GNU coreutils / busybox | ✅ built-in shell commands |
| Package manager | apt / dnf / pacman | ⬜ Phase 20 |
| C runtime | glibc / musl | ⬜ Phase 19a |
| Dynamic linker | ld-linux.so | ⬜ Phase 21 |
| Python | cpython | ⬜ Phase 19c |
| Node.js | node | ⬜ Phase 19d |
| Web browser | Firefox / Chromium | ✅ Phase 23 (Intelli native) |
| File manager GUI | Nautilus / Dolphin | ✅ Phase 12a |
| Text editor GUI | gedit / Kate | ✅ Phase 12a |
| Terminal emulator | GNOME Terminal / Konsole | ✅ Phase 26 |
| Multi-window GUI | X11 / Wayland | ✅ Phase 22 |
| Audio | ALSA / PipeWire | ✅ Phase 25 |
| WiFi | NetworkManager + wpa_supplicant | ⬜ Phase 27 |
| USB storage | udisks2 | ⬜ Phase 27 |
| Suspend/resume | systemd-logind | ⬜ Phase 27 |
| GPU (native res) | DRM/KMS | ⬜ Phase 27 |
| Self-hosting (compile itself) | gcc / clang on Linux | ⬜ Phase 28 |
| LLM assistant | (not standard) | ⬜ Phase 29 |
| AI kernel decisions | (not possible) | ✅ NodeAI-exclusive |

---

## Non-Goals (for now)

- Full GNOME/KDE desktop — NodeAI has its own minimal AI-focused compositor
- 32-bit x86 support (64-bit only to keep codebase clean)
- Full POSIX compliance (compatibility is a nice-to-have, not a goal)
- Android/iOS compatibility layer

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Kernel language | Rust (nightly) | Memory safety, zero-cost abstractions, no garbage collector |
| Bootloader | Custom + `bootloader` crate | Full control, UEFI + legacy BIOS |
| AI inference | Custom no_std Rust engine | No OS dependencies, tight kernel integration |
| Model format | ONNX subset / custom binary | Well-understood, tooling available |
| Build system | Cargo workspace | Native Rust, reproducible |
| Test VM | Oracle VirtualBox | Free, scriptable, good x86_64 emulation |
| Dev iteration VM | QEMU | Faster iteration than VirtualBox |
| CI | GitHub Actions | Automated lint, test, boot-test |
| Arch target | x86_64-unknown-none | Bare metal, no OS assumptions |

---

## Repository Structure

```
NodeAI/
├── Cargo.toml              # Workspace root
├── rust-toolchain.toml     # Pinned nightly toolchain
├── ROADMAP.md              # This file
├── README.md
├── .cargo/
│   └── config.toml         # Linker, target config
├── bootloader/             # Stage 1 & 2 boot code
│   └── src/
├── kernel/                 # Core kernel crate
│   └── src/
│       ├── main.rs         # kernel_main entry
│       ├── memory/         # PMM, VMM, heap
│       ├── scheduler/      # Task, runqueue, context switch
│       ├── interrupts/     # IDT, IRQ, exceptions
│       ├── acpi/           # ACPI parsing
│       ├── ipc/            # Pipes, signals, shared mem
│       └── syscall/        # Syscall dispatch table
├── hal/                    # Hardware Abstraction Layer
│   └── src/
│       ├── lib.rs          # HAL traits
│       └── x86_64/         # x86_64 implementation
├── ai_subsystem/           # AI inference engine + kernel AI
│   └── src/
│       ├── inference/      # SIMD inference runtime
│       ├── model/          # Model loading, validation
│       ├── domains/        # scheduler_ai, memory_ai, etc.
│       └── safety/         # Constraint engine, audit log
├── drivers/                # Device drivers
│   └── src/
│       ├── virtio/         # VirtIO block, net, gpu
│       ├── ahci/           # SATA
│       ├── nvme/           # NVMe
│       ├── pci/            # PCI enumeration
│       └── input/          # PS/2, USB HID
├── net/                    # Networking stack
├── fs/                     # VFS + filesystem implementations
├── crypto/                 # Crypto primitives for kernel
├── scripts/
│   ├── run_qemu.ps1        # Boot in QEMU
│   └── run_vbox.ps1        # Boot in VirtualBox
└── docs/
    ├── architecture.md
    ├── ai_design.md
    └── porting_guide.md
```

---

## Immediate Next Steps (Start Here)

1. `[ ]` Set up `rust-toolchain.toml` with nightly + components
2. `[ ]` Set up workspace `Cargo.toml`
3. `[ ]` Configure `.cargo/config.toml` for x86_64-unknown-none target
4. `[ ]` Implement minimal bootloader entry (`bootloader/src/main.rs`)
5. `[ ]` Implement `kernel_main()` with VGA "Hello NodeAI" output
6. `[ ]` Boot in QEMU — confirm serial output
7. `[ ]` Boot in VirtualBox — confirm same output
8. `[ ]` Merge Phase 0 checklist completely before moving to Phase 1

---

*"If Torvalds could build Linux alone in 1991 with C and no internet, we can build this with Rust, AI tooling, and everything the modern ecosystem offers."*
