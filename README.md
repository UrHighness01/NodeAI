# NodeAI Kernel

An AI-integrated operating system kernel written in Rust, built from scratch with:

- **Memory safety** via Rust's ownership model — no buffer overflows, no use-after-free, no data races by construction
- **Native AI at the kernel level** — inference runs in kernel space, directly influencing scheduling, memory, I/O, power, and security decisions
- **Minimal resource footprint** — every subsystem earns its place; no legacy bloat

## Status

**Active development — Phase 29 (AI integration complete, self-hosting target next).**

The kernel boots, runs preemptively, serves HTTP from userspace, and applies AI decisions to real scheduling. Tested on Oracle VirtualBox and QEMU.

### What works today

| Subsystem | Status |
|-----------|--------|
| Boot (BIOS/UEFI, multiboot2) | ✅ Working |
| Memory (buddy allocator, 4-level page tables, per-process CR3) | ✅ Working |
| Preemptive scheduler (naked timer handler, full context switch) | ✅ Working |
| Process model (fork with own address space, execve, wait4 sleep/wake) | ✅ Working |
| Signals (SIGKILL/SIGTERM/SIGSEGV default actions, SIGCHLD to parent) | ✅ Working |
| Syscall fast-path (SYSCALL/SYSRET, 60+ syscalls) | ✅ Working |
| VFS (ramfs root, devfs, procfs, `/ai` filesystem) | ✅ Working |
| Block storage (AHCI SATA, NVMe → `/dev/sdX`, `/dev/nvmeX`) | ✅ Working |
| Networking (VirtIO-net, ARP, DNS, TCP state machine, HTTP) | ✅ Working |
| Userspace networking (bind/listen/accept backlog, send/recv) | ✅ Working |
| Framebuffer + desktop (800×600, launcher, window manager) | ✅ Working |
| AI subsystem (scheduler AI decisions applied, LLM weight loader) | ✅ Working |
| Demand paging (heap + stack, SIGSEGV on invalid address) | ✅ Working |
| futex (real wait queue, FUTEX_WAIT/WAKE) | ✅ Working |

### Known limitations

- **FPU/SSE state not saved across context switches.** The timer handler saves the 15 integer GPRs but not XMM0–15, MXCSR, or AVX state. Tasks using SSE (musl `memcpy`, float operations, SIMD) can silently corrupt each other's FPU registers. Fix: add `xsave`/`xrstor` to the timer handler. Until then, single-threaded workloads are unaffected.
- **fork copies pages but not CoW** — both parent and child get independent copies at fork time. Correct, but uses more memory than lazy CoW would.
- **epoll is a stub** — `poll()` works and is used instead.
- **Signal handlers**: user-space handlers work; nested signal delivery (signal while handling signal) is not yet supported.
- **TCP**: no congestion control, no TLS.
- **Self-hosting**: on-device Rust compilation is the next major milestone.

## Quick Start

### Prerequisites

```bash
# Rust nightly + bare-metal target
rustup toolchain install nightly
rustup target add x86_64-unknown-none
rustup component add rust-src llvm-tools-preview

# QEMU for fast iteration
apt install qemu-system-x86  # Linux
# or: https://www.qemu.org/download/
```

### Build and run (QEMU)

```bash
cargo build --package nodeai-kernel
./scripts/run_qemu.sh
```

### Run in VirtualBox

```powershell
# Windows (PowerShell)
.\scripts\run_vbox.ps1 -Create   # first time
.\scripts\run_vbox.ps1 -Gui      # subsequent runs
```

## Architecture

```
NodeAI/
├── bootloader/      — BIOS/UEFI boot chain (multiboot2)
├── kernel/src/
│   ├── main.rs          — boot sequence, subsystem init
│   ├── scheduler/       — preemptive round-robin + AI priority
│   ├── memory/          — buddy PMM, 4-level VMM, per-process CR3
│   ├── interrupts/      — IDT, LAPIC, naked timer handler (context switch)
│   ├── syscall/         — SYSCALL/SYSRET fast-path, 60+ handlers
│   ├── vfs/             — ramfs, devfs, procfs, blockdev layer
│   ├── net.rs           — ARP/IP/TCP/DNS/HTTP stack
│   ├── ai_engine.rs     — AI ↔ kernel bridge (decisions, LLM)
│   ├── ahci.rs          — AHCI/SATA driver
│   ├── nvme.rs          — NVMe driver
│   └── shell.rs         — in-kernel shell
├── ai_subsystem/    — no_std inference engine (DenseLayer, SequentialModel)
├── drivers/         — VirtIO-net, PS/2, USB, audio
├── hal/             — hardware abstraction traits
└── scripts/         — QEMU and VirtualBox launch scripts
```

## Context switch implementation

The timer interrupt uses a `#[naked]` Rust function that saves all 15 GPRs onto the current task's kernel stack, calls `schedule_from_interrupt(old_rsp) -> new_rsp`, switches the stack pointer, restores GPRs from the new stack, and executes `iretq`. Each task's kernel stack holds a complete 160-byte saved interrupt frame that is the canonical context representation — no separate context save struct.

Per-process page tables: each `execve` allocates a fresh L4 via `alloc_user_cr3()` (kernel half copied, user half empty), switches CR3, and maps ELF segments in isolation. CR3 and TSS.RSP0 are updated on every context switch.

## AI integration

The AI subsystem runs entirely in kernel space. On each timer tick:
1. `ai_engine::process_tick()` drains the event bus
2. `scheduler_ai::predict()` produces priority adjustments per task
3. `apply_decision()` calls `scheduler::adjust_priority(pid, delta)`
4. Security alerts above threshold demote suspect tasks

The LLM loader reads `/var/lib/llm/model.bin` (NLLM format) at boot, parses weight matrices into a `SequentialModel`, and makes it available via `sys_ai_query`.

## Design principles

1. Safety first — if it compiles without `unsafe`, it cannot crash the kernel
2. AI as a first-class citizen — not a plugin, not a daemon; part of the kernel
3. Hard constraints on AI — every AI decision passes through the safety engine
4. Graceful degradation — every AI component has a deterministic fallback
5. Observable — every AI decision is audited and readable from `/ai/log`

## License

MIT OR Apache-2.0
