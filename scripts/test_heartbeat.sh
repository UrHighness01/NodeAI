#!/usr/bin/env bash
# Test if the kernel boots and produces a heartbeat within 20 seconds.
# Returns exit code 0 if heartbeat found, 1 if not.
# Must be run from the workspace root.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 1. Build the kernel
echo "==> Building kernel..."
cargo build -Zbuild-std=core,alloc -p nodeai-kernel --release 2>&1 | tail -5

KERNEL_ELF="$ROOT/target/x86_64-unknown-none/release/nodeai-kernel"
if [[ ! -f "$KERNEL_ELF" ]]; then
    echo "ERROR: Kernel ELF not found"
    exit 1
fi

# 2. Create boot image
echo "==> Creating disk image..."
IMG_DIR="$ROOT/target/images"
mkdir -p "$IMG_DIR"
"$ROOT/target/x86_64-unknown-linux-gnu/debug/image-builder" "$KERNEL_ELF" "$IMG_DIR"

BIOS_IMG="$IMG_DIR/nodeai-bios.img"
if [[ ! -f "$BIOS_IMG" ]]; then
    echo "ERROR: BIOS image not found"
    exit 1
fi

# 3. Run QEMU with 20-second timeout, capture serial output
echo "==> Running QEMU (20s timeout)..."
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

# Run QEMU with timeout, capture output
timeout 20 qemu-system-x86_64 "${QEMU_ARGS[@]}" 2>&1 | tee /tmp/qemu-test-output.txt | tail -20

# 4. Check for heartbeat signal
if grep -q "NodeAI alive" /tmp/qemu-test-output.txt; then
    echo "=== HEARTBEAT FOUND ==="
    grep "NodeAI alive" /tmp/qemu-test-output.txt
    exit 0
fi

if grep -q "KERNEL PANIC" /tmp/qemu-test-output.txt; then
    echo "=== KERNEL PANIC ==="
    grep -a "KERNEL PANIC" /tmp/qemu-test-output.txt | tail -5
    exit 2
fi

echo "=== NO HEARTBEAT ==="
tail -20 /tmp/qemu-test-output.txt
exit 1
