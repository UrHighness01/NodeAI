# NodeAI Kernel

An AI-integrated operating system kernel written in Rust, inspired by Linux but designed from the ground up with:

- **Memory safety** via Rust's ownership and borrow checker — eliminating buffer overflows, use-after-free, and data races at compile time
- **Native AI at the kernel level** — inference runs in kernel space with full system visibility, influencing scheduling, memory management, I/O, power, and security
- **Minimal resource footprint** — no legacy baggage, every subsystem earns its place

## Status

Early development — Phase 0 (toolchain setup).

See [ROADMAP.md](ROADMAP.md) for the complete development plan.

## Quick Start

### Prerequisites

```powershell
# Install Rust nightly
rustup toolchain install nightly
rustup target add x86_64-unknown-none
rustup component add rust-src llvm-tools-preview

# Install QEMU for fast iteration testing
# Download from: https://www.qemu.org/download/
```

### Build

```powershell
cargo build --package nodeai-kernel
```

### Run in QEMU

```powershell
.\scripts\run_qemu.ps1
```

### Run in VirtualBox

```powershell
# First time: create the VM
.\scripts\run_vbox.ps1 -Create

.\scripts\run_vbox.ps1 -Gui

# Subsequent runs
.\scripts\run_vbox.ps1 -Gui
```

## Architecture

```
NodeAI/
├── bootloader/      — BIOS/UEFI boot chain
├── kernel/          — Core kernel (memory, scheduler, interrupts, syscalls)
├── hal/             — Hardware Abstraction Layer (architecture-agnostic traits)
├── ai_subsystem/    — AI inference engine + scheduler/memory/security AI
├── drivers/         — VirtIO, PCI, PS/2, AHCI, NVMe
├── scripts/         — QEMU and VirtualBox automation
└── docs/            — Architecture documents
```

## Target Hardware

- **Development**: QEMU (fast iteration), Oracle VirtualBox (integration testing)
- **Phase 12+**: Physical x86_64 hardware (Intel/AMD desktop, Intel NPU for AI acceleration)

## Design Principles

1. Safety first — if it compiles without `unsafe`, it cannot crash the kernel
2. AI as a first-class citizen — not a plugin, not a daemon; part of the kernel
3. Hard constraints on AI — AI decisions always pass through the safety engine before being applied
4. Graceful degradation — every AI component has a deterministic fallback
5. Observable — every AI decision is audited and readable from `/ai/log`

## License

MIT OR Apache-2.0
