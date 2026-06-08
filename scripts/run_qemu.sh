#!/usr/bin/env bash
# Run NodeAI kernel in QEMU on Linux.
#
# Build pipeline:
#   1. cargo build --package nodeai-kernel   (bare-metal ELF, x86_64-unknown-none)
#   2. cargo build --package image-builder   (host tool)
#   3. image-builder <kernel.elf> <out-dir>  (creates BIOS/UEFI .img)
#   4. qemu-system-x86_64 -drive file=nodeai-bios.img ...
#
# Usage: ./scripts/run_qemu.sh [--debug] [--release] [--uefi] [--memory 512]
#
# Dependencies (install once):
#   sudo apt install qemu-system-x86 curl
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
#   source "$HOME/.cargo/env"
#   rustup toolchain install nightly
#   rustup component add rust-src llvm-tools-preview --toolchain nightly
#   rustup target add x86_64-unknown-none --toolchain nightly

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MEMORY=512
DEBUG=0
RELEASE=0
UEFI=0
GUI=0

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)   DEBUG=1 ;;
        --release) RELEASE=1 ;;
        --uefi)    UEFI=1 ;;
        --gui)     GUI=1 ;;
        --memory)  MEMORY="$2"; shift ;;
        --memory=*)MEMORY="${1#*=}" ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

# ── Sanity checks ─────────────────────────────────────────────────────────────
if ! command -v qemu-system-x86_64 &>/dev/null; then
    echo "ERROR: qemu-system-x86_64 not found."
    echo "Install with: sudo apt install qemu-system-x86"
    exit 1
fi

if ! command -v cargo &>/dev/null; then
    echo "ERROR: cargo not found."
    echo "Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "Then: source \"\$HOME/.cargo/env\""
    exit 1
fi

cd "$ROOT"

# ── 1. Build the kernel ───────────────────────────────────────────────────────
echo "==> Building NodeAI kernel (x86_64-unknown-none)..."
BUILD_ARGS=(build --package nodeai-kernel)
[[ $RELEASE -eq 1 ]] && BUILD_ARGS+=(--release)
cargo "${BUILD_ARGS[@]}"

PROFILE="debug"
[[ $RELEASE -eq 1 ]] && PROFILE="release"
KERNEL_ELF="$ROOT/target/x86_64-unknown-none/$PROFILE/nodeai-kernel"

if [[ ! -f "$KERNEL_ELF" ]]; then
    echo "ERROR: Kernel ELF not found at $KERNEL_ELF"
    exit 1
fi
echo "  Kernel: $KERNEL_ELF ($(du -sh "$KERNEL_ELF" | cut -f1))"

# ── 2. Build the image-builder host tool ─────────────────────────────────────
# Must pass --target explicitly: .cargo/config.toml defaults to x86_64-unknown-none
echo "==> Building image-builder..."
cargo build --package image-builder --target x86_64-unknown-linux-gnu
IMAGE_BUILDER="$ROOT/target/x86_64-unknown-linux-gnu/debug/image-builder"

# ── 3. Create bootable disk images ────────────────────────────────────────────
echo "==> Creating disk images..."
IMG_DIR="$ROOT/target/images"
mkdir -p "$IMG_DIR"
"$IMAGE_BUILDER" "$KERNEL_ELF" "$IMG_DIR"

BIOS_IMG="$IMG_DIR/nodeai-bios.img"
UEFI_IMG="$IMG_DIR/nodeai-uefi.img"

# ── 4. Run QEMU ───────────────────────────────────────────────────────────────
echo "==> Starting QEMU (${MEMORY} MiB RAM)..."

QEMU_ARGS=(
    -machine q35
    -cpu qemu64,+avx2
    -m "${MEMORY}M"
    -serial stdio
    -no-reboot
    -no-shutdown
)

if [[ $GUI -eq 0 ]]; then
    QEMU_ARGS+=(-display none -monitor unix:qemu-monitor.sock,server,nowait) # headless — all output via serial/stdio
fi

# KVM acceleration if available (10-50x faster)
if [[ -w /dev/kvm ]]; then
    QEMU_ARGS+=(-enable-kvm)
    echo "  KVM acceleration: enabled"
else
    echo "  KVM acceleration: unavailable (add user to kvm group: sudo usermod -aG kvm \$USER)"
fi

if [[ $UEFI -eq 1 ]]; then
    # Find OVMF firmware
    OVMF=""
    for f in /usr/share/OVMF/OVMF_CODE.fd \
              /usr/share/ovmf/OVMF.fd \
              /usr/share/qemu/OVMF.fd; do
        [[ -f "$f" ]] && OVMF="$f" && break
    done
    if [[ -z "$OVMF" ]]; then
        echo "WARNING: OVMF firmware not found. Install: sudo apt install ovmf"
        echo "Falling back to BIOS boot."
        UEFI=0
    else
        QEMU_ARGS+=(-bios "$OVMF" -drive "format=raw,file=$UEFI_IMG")
        echo "  Boot: UEFI via $UEFI_IMG"
    fi
fi

if [[ $UEFI -eq 0 ]]; then
    QEMU_ARGS+=(-drive "format=raw,file=$BIOS_IMG")
    echo "  Boot: BIOS via $BIOS_IMG"
fi

if [[ $DEBUG -eq 1 ]]; then
    QEMU_ARGS+=(-s -S)
    echo "  GDB stub: localhost:1234 (QEMU paused at entry)"
    echo "  Connect with:"
    echo "    gdb $KERNEL_ELF"
    echo "    (gdb) target remote localhost:1234"
    echo "    (gdb) continue"
fi

echo ""
echo "  Serial output → stdout below"
echo "  Press Ctrl+A X to quit QEMU"
echo "─────────────────────────────────────────────────────────────────────────"
exec qemu-system-x86_64 "${QEMU_ARGS[@]}"
