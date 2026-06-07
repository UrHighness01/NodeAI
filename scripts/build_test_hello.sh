#!/usr/bin/env bash
# Build and test a musl-static hello world on NodeAI.
# Usage: ./scripts/build_test_hello.sh [--qemu | --vbox]
#
# Prerequisites:
#   musl-gcc (apt install musl-tools)
#   QEMU or VirtualBox
#   NodeAI built: cargo build --package nodeai-kernel

set -e
KERNEL_BIN=${KERNEL_BIN:-target/x86_64-unknown-none/debug/nodeai-kernel}
HELLO_SRC=/tmp/nodeai_hello.c
HELLO_BIN=/tmp/nodeai_hello

# 1. Write a minimal C hello world.
cat > "$HELLO_SRC" << 'EOF'
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    printf("Hello from NodeAI userspace!\n");
    return 0;
}
EOF

# 2. Compile with musl-gcc (static, no dynamic linker).
echo "[+] Compiling with musl-gcc..."
musl-gcc -static -Os -o "$HELLO_BIN" "$HELLO_SRC"
echo "    Binary: $(ls -lh "$HELLO_BIN" | awk '{print $5}')"
file "$HELLO_BIN"

# 3. Inject into NodeAI VFS image (place at /usr/bin/hello in disk image).
echo "[+] Injecting into NodeAI disk image..."
# TODO: use a proper disk image tool (mcopy, debugfs, etc.)
# For now, print the binary path for manual placement.
echo "    Place $HELLO_BIN at /usr/bin/hello in NodeAI's initramfs or disk image."
echo "    Then from NodeAI shell: execve /usr/bin/hello"

# 4. Run in QEMU with serial capture.
if [[ "${1:-}" == "--qemu" ]]; then
    echo "[+] Booting NodeAI in QEMU..."
    qemu-system-x86_64 \
        -m 512M \
        -serial stdio \
        -display none \
        -kernel "$KERNEL_BIN" \
        -append "console=ttyS0" \
        2>&1 | tee /tmp/nodeai_boot.log
    echo "[+] Boot log saved to /tmp/nodeai_boot.log"
fi

echo "[+] Done. Expected output from NodeAI: 'Hello from NodeAI userspace!'"
