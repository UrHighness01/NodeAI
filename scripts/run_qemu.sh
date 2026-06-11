#!/usr/bin/env bash
# Run NodeAI kernel in QEMU on Linux.
#
# Build pipeline:
#   1. cargo build --package nodeai-kernel   (bare-metal ELF, x86_64-unknown-none)
#   2. cargo build --package image-builder   (host tool)
#   3. image-builder <kernel.elf> <out-dir>  (creates BIOS/UEFI .img)
#   4. qemu-system-x86_64 -drive file=nodeai-bios.img ...
#
# Usage: ./scripts/run_qemu.sh [--debug] [--release] [--uefi] [--nographic] [--gui] [--memory 512] [--wifi]
#
# --nographic  Terminal mode — keyboard mapped to serial port.
#              Type commands in this terminal. Ctrl+A then X to quit.
# --gui        SDL window + serial log. Keyboard goes to SDL window (PS/2).
#              Run 'tail -f target/qemu_serial.log' in another terminal for logs.
#              This is the BEST mode for interactive use with the kernel desktop.
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
MEMORY=2048
DEBUG=0
RELEASE=1
UEFI=0
GUI=0
WIFI=0
NOGRAPHIC=0

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)      DEBUG=1 ;;
        --release)    RELEASE=1 ;;
        --uefi)       UEFI=1 ;;
        --gui)        GUI=1 ;;
        --nographic)  NOGRAPHIC=1 ;;
        --wifi)       WIFI=1 ;;
        --memory)     MEMORY="$2"; shift ;;
        --memory=*)   MEMORY="${1#*=}" ;;
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
    -cpu Haswell
    -m "${MEMORY}M"
    -no-reboot
    -no-shutdown
)

if [[ $NOGRAPHIC -eq 1 ]]; then
    # Full terminal mode: keyboard → UART serial → kernel UART input.
    # NOTE: Kernel must also read from serial (COM1) — PS/2 IRQ is NOT
    # available in -nographic because QEMU maps terminal to serial.
    QEMU_ARGS+=(-nographic)
    echo "  Display: nographic (terminal = serial port)"
    echo "  Type commands in this terminal. Press Ctrl+A then X to quit."
elif [[ $GUI -eq 1 ]]; then
    # SDL window: keyboard → PS/2 IRQ1 → kernel shell.
    # Serial goes to a log file so terminal keystrokes aren't eaten.
    SERIAL_LOG="${ROOT}/target/qemu_serial.log"
    QEMU_ARGS+=(-serial file:"$SERIAL_LOG" -display sdl)
    echo "  Display: SDL window — click the window to focus it, then type commands"
    echo "  Serial logs → tail -f $SERIAL_LOG"
    echo "  Keyboard input goes to the SDL window (PS/2)."
else
    # Headless: serial routed to stdio, no window. Output visible, no typing.
    QEMU_ARGS+=(-serial stdio -display none -monitor unix:qemu-monitor.sock,server,nowait)
    echo "  Display: headless (serial → stdout, read-only)"
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

# Qwen3.5 weight disk (second AHCI drive, index 1)
QWEN35_BIN="${ROOT}/../models/lm_qwen35.bin"
QWEN35_IMG="${ROOT}/target/qwen35_weights.img"
if [[ -f "$QWEN35_BIN" ]]; then
    # Wrap raw binary in a raw disk image (no filesystem — kernel reads raw bytes)
    if [[ ! -f "$QWEN35_IMG" ]] || [[ "$QWEN35_BIN" -nt "$QWEN35_IMG" ]]; then
        echo "  Building Qwen3.5 weight disk image..."
        cp "$QWEN35_BIN" "$QWEN35_IMG"
    fi
    QWEN35_SIZE=$(stat -c %s "$QWEN35_IMG")
    QEMU_ARGS+=(-drive "format=raw,file=$QWEN35_IMG,if=ide,index=1")
    echo "  Qwen3.5: weight disk attached ($((QWEN35_SIZE / 1048576))MB)"
else
    echo "  Qwen3.5: weight binary not found at $QWEN35_BIN — Qwen3.5 will be unavailable"
    echo "    Run: python3 scripts/convert_qwen35_kernel.py"
fi

if [[ $WIFI -eq 1 ]]; then
    # USB passthrough for AR9271 WiFi dongle (TP-Link TL-WN722N v1 or similar)
    # Requires the dongle plugged into the host and user in 'plugdev' group:
    #   sudo usermod -aG plugdev $USER && sudo udevadm trigger
    QEMU_ARGS+=(-device usb-ehci,id=usb0)
    QEMU_ARGS+=(-device usb-host,bus=usb0.0,vendorid=0x0cf3,productid=0x9271)
    echo "  WiFi: AR9271 USB passthrough enabled (vendorid=0x0cf3 productid=0x9271)"
    echo "  Make sure dongle is plugged in and you are in the plugdev group"
fi

if [[ $DEBUG -eq 1 ]]; then
    QEMU_ARGS+=(-s -S)
    echo "  GDB stub: localhost:1234 (QEMU paused at entry)"
    echo "  Connect with:"
    echo "    gdb $KERNEL_ELF"
    echo "    (gdb) target remote localhost:1234"
    echo "    (gdb) continue"
fi

echo "─────────────────────────────────────────────────────────────────────────"

if [[ $NOGRAPHIC -eq 1 ]]; then
    # Run QEMU in foreground but trap Ctrl+C so it kills QEMU cleanly.
    # Without this trap, -nographic swallows SIGINT and the process hangs.
    qemu-system-x86_64 "${QEMU_ARGS[@]}" &
    QPID=$!
    trap 'kill $QPID 2>/dev/null; wait $QPID 2>/dev/null; exit 0' INT TERM
    wait $QPID
elif [[ $GUI -eq 1 ]]; then
    SERIAL_LOG="${ROOT}/target/qemu_serial.log"
    # Ensure log file exists
    touch "$SERIAL_LOG"
    # Start QEMU in background
    qemu-system-x86_64 "${QEMU_ARGS[@]}" &
    QPID=$!
    trap 'kill $QPID 2>/dev/null; wait $QPID 2>/dev/null; exit 0' INT TERM
    echo "QEMU PID=$QPID — tails serial log below (Ctrl+C to quit):"
    echo "─────────────────────────────────────────────────────────────────────────"
    tail -f --pid=$QPID "$SERIAL_LOG"
else
    exec qemu-system-x86_64 "${QEMU_ARGS[@]}"
fi
