#!/usr/bin/env bash
# NodeAI Phase 19 — Build CPIO initrd from output/initrd/
# Run after build_userspace.sh to pack FS into a CPIO archive
# then append it to the NodeAI disk image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$ROOT_DIR/output"
INITRD_DIR="$OUTPUT_DIR/initrd"
INITRD_IMG="$OUTPUT_DIR/initrd.cpio.gz"

log() { echo -e "\033[1;36m[initrd]\033[0m $*"; }

[ -d "$INITRD_DIR" ] || { echo "Run build_userspace.sh first"; exit 1; }

log "Packing $INITRD_DIR into $INITRD_IMG..."
cd "$INITRD_DIR"
find . | cpio -o -H newc 2>/dev/null | gzip -9 > "$INITRD_IMG"

SIZE=$(du -sh "$INITRD_IMG" | cut -f1)
log "initrd.cpio.gz: $SIZE  ✓"
log "Embed path: $INITRD_IMG"
log "Pass to QEMU: -initrd $INITRD_IMG"
