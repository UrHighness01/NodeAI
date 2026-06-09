#!/usr/bin/env bash
# Test script for git bisect.
# Builds the kernel and checks for heartbeat via QEMU.
# Returns 0 (good) if heartbeat found, 1 (bad) if not.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Build the kernel
cargo build -Zbuild-std=core,alloc -p nodeai-kernel --release 2>&1 | tail -3 || exit 125

KERNEL_ELF="$ROOT/target/x86_64-unknown-none/release/nodeai-kernel"
if [[ ! -f "$KERNEL_ELF" ]]; then
    exit 125
fi

# Create boot image
IMG_DIR="$ROOT/target/images"
mkdir -p "$IMG_DIR"
"$ROOT/target/x86_64-unknown-linux-gnu/debug/image-builder" "$KERNEL_ELF" "$IMG_DIR" 2>&1

BIOS_IMG="$IMG_DIR/nodeai-bios.img"
if [[ ! -f "$BIOS_IMG" ]]; then
    exit 125
fi

# Run QEMU with 25-second timeout
QEMU_ARGS=(
    -machine q35
    -cpu qemu64,+avx2,+rdrand,+rdseed
    -m 512M
    -serial stdio
    -display none
    -no-reboot
    -no-shutdown
    -drive "format=raw,file=$BIOS_IMG"
)
if [[ -w /dev/kvm ]]; then
    QEMU_ARGS+=(-enable-kvm)
fi

# Capture output and check for heartbeat
timeout 25 qemu-system-x86_64 "${QEMU_ARGS[@]}" > /tmp/qemu-bisect-out.txt 2>&1 || true

if grep -q "NodeAI alive" /tmp/qemu-bisect-out.txt; then
    exit 0  # good — heartbeat found
fi

if grep -q "KERNEL PANIC" /tmp/qemu-bisect-out.txt; then
    exit 1  # bad — panic
fi

exit 1  # bad — no heartbeat
