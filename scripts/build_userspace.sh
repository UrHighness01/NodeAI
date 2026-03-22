#!/usr/bin/env bash
# NodeAI Phase 19 — Userspace Build Script
# Run on a Linux host (Debian/Ubuntu recommended) with:
#   sudo apt-get install -y musl-tools build-essential wget curl git python3-dev
#
# This script cross-compiles the NodeAI userspace toolchain:
#   - musl libc 1.2.x (19a)
#   - BusyBox (19b)
#   - CPython 3.12 static (19c)
#   - Node.js 22 LTS static (19d)
#
# All binaries are placed into output/initrd/ which is then packed
# into a CPIO archive by build_initrd.sh and embedded in the disk image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$ROOT_DIR/output"
INITRD_DIR="$OUTPUT_DIR/initrd"
BUILD_DIR="$OUTPUT_DIR/build"
SYSROOT="$OUTPUT_DIR/sysroot"

NPROC=$(nproc 2>/dev/null || echo 4)

mkdir -p "$INITRD_DIR"/{bin,lib,usr/bin,usr/lib,etc,dev,proc,sys,tmp,var/lib/nodepkg}
mkdir -p "$BUILD_DIR"
mkdir -p "$SYSROOT"

log() { echo -e "\033[1;36m[NodeAI Build]\033[0m $*"; }
die() { echo -e "\033[1;31m[FATAL]\033[0m $*" >&2; exit 1; }

# ── Phase 19a — musl libc ────────────────────────────────────────────────────
build_musl() {
    log "Building musl libc 1.2.5..."
    local ver="1.2.5"
    local src="$BUILD_DIR/musl-$ver"
    if [ ! -d "$src" ]; then
        wget -q "https://musl.libc.org/releases/musl-$ver.tar.gz" -O "$BUILD_DIR/musl-$ver.tar.gz"
        tar -xf "$BUILD_DIR/musl-$ver.tar.gz" -C "$BUILD_DIR"
    fi
    cd "$src"
    ./configure \
        --prefix="$SYSROOT" \
        --syslibdir="$SYSROOT/lib" \
        --enable-static \
        --disable-shared \
        CFLAGS="-O2 -fPIC"
    make -j"$NPROC"
    make install
    # create musl-gcc wrapper
    cat > "$SYSROOT/bin/musl-gcc" << 'EOF'
#!/bin/sh
exec gcc "$@" -specs /dev/stdin <<SPEC
%rename cc1 cc1_orig
*cc1:
%(cc1_orig) -isystem SYSROOT/include
*link_libgcc:
-L SYSROOT/lib -lc
SPEC
EOF
    chmod +x "$SYSROOT/bin/musl-gcc"
    log "musl libc built  ✓"
}

# ── Phase 19b — BusyBox ──────────────────────────────────────────────────────
build_busybox() {
    log "Building BusyBox 1.36.1..."
    local ver="1.36.1"
    local src="$BUILD_DIR/busybox-$ver"
    if [ ! -d "$src" ]; then
        wget -q "https://busybox.net/downloads/busybox-$ver.tar.bz2" -O "$BUILD_DIR/busybox-$ver.tar.bz2"
        tar -xf "$BUILD_DIR/busybox-$ver.tar.bz2" -C "$BUILD_DIR"
    fi
    cd "$src"
    make defconfig
    # Enable static linking against musl
    sed -i 's/# CONFIG_STATIC is not set/CONFIG_STATIC=y/' .config
    sed -i "s|CONFIG_SYSROOT=\"\"|CONFIG_SYSROOT=\"$SYSROOT\"|" .config
    # Disable features not supported yet
    sed -i 's/CONFIG_FEATURE_INETD_RPC=y/# CONFIG_FEATURE_INETD_RPC is not set/' .config
    make -j"$NPROC" CC="$SYSROOT/bin/musl-gcc"
    cp busybox "$INITRD_DIR/bin/busybox"
    strip "$INITRD_DIR/bin/busybox"
    # create symlinks for common applets
    cd "$INITRD_DIR/bin"
    for applet in sh ls cat echo cp mv rm mkdir grep find wget vi tar gzip; do
        ln -sf busybox "$applet" 2>/dev/null || true
    done
    log "BusyBox built  ✓"
}

# ── Phase 19c — CPython 3.12 static ─────────────────────────────────────────
build_python() {
    log "Building CPython 3.12.x (static, musl)..."
    local ver="3.12.3"
    local src="$BUILD_DIR/Python-$ver"
    if [ ! -d "$src" ]; then
        wget -q "https://www.python.org/ftp/python/$ver/Python-$ver.tar.xz" -O "$BUILD_DIR/Python-$ver.tar.xz"
        tar -xf "$BUILD_DIR/Python-$ver.tar.xz" -C "$BUILD_DIR"
    fi
    cd "$src"
    # Patch Setup.local to exclude modules requiring external libs
    cat > Modules/Setup.local << 'EOF'
*disabled*
_tkinter
readline
curses
_curses
_curses_panel
EOF
    CC="$SYSROOT/bin/musl-gcc" \
    LDFLAGS="-static" \
    ./configure \
        --prefix="$SYSROOT/python" \
        --disable-shared \
        --enable-optimizations \
        --with-ensurepip=no \
        --without-readline \
        --without-curses \
        ac_cv_func_setuid=yes \
        ac_cv_func_getuid=yes
    make -j"$NPROC" LDFLAGS="-static"
    make install
    cp "$SYSROOT/python/bin/python3" "$INITRD_DIR/usr/bin/python3"
    strip "$INITRD_DIR/usr/bin/python3" 2>/dev/null || true
    ln -sf python3 "$INITRD_DIR/usr/bin/python"
    log "CPython built  ✓"
}

# ── Phase 19d — Node.js 22 LTS static ───────────────────────────────────────
build_nodejs() {
    log "Building Node.js 22 LTS (static, musl)..."
    local ver="22.2.0"
    local src="$BUILD_DIR/node-v$ver"
    if [ ! -d "$src" ]; then
        wget -q "https://nodejs.org/dist/v$ver/node-v$ver.tar.xz" -O "$BUILD_DIR/node-v$ver.tar.xz"
        tar -xf "$BUILD_DIR/node-v$ver.tar.xz" -C "$BUILD_DIR"
    fi
    cd "$src"
    CC="$SYSROOT/bin/musl-gcc" \
    CXX="g++ --specs=/dev/null" \
    ./configure \
        --fully-static \
        --without-npm \
        --without-inspector \
        --without-intl \
        --dest-cpu=x64
    make -j"$NPROC"
    cp out/Release/node "$INITRD_DIR/usr/bin/node"
    strip "$INITRD_DIR/usr/bin/node" 2>/dev/null || true
    log "Node.js built  ✓"
}

# ── /etc/passwd, /etc/group, /init ───────────────────────────────────────────
write_rootfs_skeleton() {
    log "Writing rootfs skeleton..."
    cat > "$INITRD_DIR/etc/passwd" << 'EOF'
root:x:0:0:Root:/root:/bin/sh
nobody:x:65534:65534:Nobody:/:/bin/false
EOF
    cat > "$INITRD_DIR/etc/group" << 'EOF'
root:x:0:root
nobody:x:65534:
EOF
    cat > "$INITRD_DIR/etc/hostname" << 'EOF'
nodeai
EOF
    cat > "$INITRD_DIR/etc/hosts" << 'EOF'
127.0.0.1  localhost nodeai
::1        localhost
EOF
    # Minimal /init that execs the NodeAI kernel shell (PID 1 stub)
    cat > "$INITRD_DIR/init" << 'EOF'
#!/bin/sh
# NodeAI init (Phase 19 stub — replaced by systemd-like init in Phase 26)
mount -t proc proc /proc 2>/dev/null || true
mount -t sysfs sysfs /sys 2>/dev/null || true
exec /bin/sh
EOF
    chmod +x "$INITRD_DIR/init"
    log "rootfs skeleton written  ✓"
}

# ── Main ─────────────────────────────────────────────────────────────────────
main() {
    log "NodeAI Phase 19 userspace build starting..."
    log "Output: $INITRD_DIR"

    build_musl
    build_busybox
    build_python
    build_nodejs
    write_rootfs_skeleton

    log ""
    log "Phase 19 build complete. Binaries:"
    ls -lh "$INITRD_DIR/bin/busybox"   2>/dev/null || true
    ls -lh "$INITRD_DIR/usr/bin/python3" 2>/dev/null || true
    ls -lh "$INITRD_DIR/usr/bin/node"  2>/dev/null || true
    log ""
    log "Next: run scripts/build_initrd.sh to pack into CPIO + embed in disk image."
}

main "$@"
