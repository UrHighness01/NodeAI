#!/usr/bin/env bash
# Run NodeAI kernel in QEMU on Linux.
#
# Build pipeline:
#   1. cargo build --package nodeai-kernel   (bare-metal ELF, x86_64-unknown-none)
#   2. cargo build --package image-builder   (host tool)
#   3. image-builder <kernel.elf> <out-dir>  (creates BIOS/UEFI .img)
#   4. qemu-system-x86_64 -drive file=nodeai-bios.img ...
#
# Usage: ./scripts/run_qemu.sh [--debug] [--release] [--uefi] [--nographic] [--gui] [--memory 512]
#
# --nographic  Full interactive terminal mode — keyboard input goes to kernel shell.
#              Type commands like 'consc hello', 'help', 'mem'. Press Ctrl+A X to quit.
#              This is the ONLY mode that allows typing commands into the kernel.
# --gui        SDL window + serial output to terminal. Keyboard goes to SDL window,
#              NOT the kernel serial shell — you cannot type commands this way.
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
    # Full terminal mode: keyboard → serial → kernel shell.
    # Ctrl+C kills QEMU cleanly from the host side.
    QEMU_ARGS+=(-nographic)
    echo "  Display: nographic (terminal = serial console)"
    echo "  Type commands directly. Press Ctrl+C to quit."
elif [[ $GUI -eq 1 ]]; then
    # SDL window: keyboard → PS/2 IRQ1 → kernel shell. Type commands in the SDL window.
    # Serial output still goes to this terminal so you can see kernel logs.
    QEMU_ARGS+=(-serial stdio -display sdl,grab-on-hover=off)
    echo "  Display: SDL window — click the window to focus it, then type commands"
    echo "  Serial logs appear in this terminal"
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
else
    exec qemu-system-x86_64 "${QEMU_ARGS[@]}"
fi
