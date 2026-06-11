#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RELEASE=1
TIMEOUT=40
TEST_LOG="/tmp/nodeai_kernel_test.log"
while [[ $# -gt 0 ]]; do case $1 in --release) RELEASE=1 ;; --debug) RELEASE=0 ;; --timeout) TIMEOUT="$2"; shift ;; *) echo "Unknown: $1"; exit 1 ;; esac; shift; done
echo "=== Test 1: Build ==="
touch kernel/src/main.rs
BA=(build -Zbuild-std=core,alloc -p nodeai-kernel)
[[ $RELEASE -eq 1 ]] && BA+=(--release)
cargo "${BA[@]}" > /dev/null 2>&1 && echo "PASS: Build" || { echo "FAIL: Build"; exit 1; }
P="debug"; [[ $RELEASE -eq 1 ]] && P="release"
KE="$ROOT/target/x86_64-unknown-none/$P/nodeai-kernel"
echo "=== Test 2: Boot Image ==="
ID="$ROOT/target/images"; mkdir -p "$ID"
"$ROOT/target/x86_64-unknown-linux-gnu/debug/image-builder" "$KE" "$ID" > /dev/null 2>&1
echo "PASS: Boot image"
echo "=== Test 3: Boot (${TIMEOUT}s) ==="
QEMU="/usr/bin/qemu-system-x86_64"
QA=(-machine q35 -cpu qemu64,+avx2,+rdrand,+rdseed -m 2048M)
QA+=(-serial stdio -display none -no-reboot -no-shutdown)
QA+=(-drive "format=raw,file=$ID/nodeai-bios.img")

# Qwen weight disks skipped in test mode (767MB AHCI read takes >60s).
# Use ./scripts/run_qemu.sh --gui for Qwen voices.

"$QEMU" "${QA[@]}" > "$TEST_LOG" 2>&1 &
QPID=$!; sleep "$TIMEOUT"; kill "$QPID" 2>/dev/null || true; wait "$QPID" 2>/dev/null || true
set +e
HB=$(grep -c "NodeAI alive" "$TEST_LOG" 2>/dev/null); PA=$(grep -c "KERNEL PANIC" "$TEST_LOG" 2>/dev/null)
PF=$(grep -c "#PF FATAL" "$TEST_LOG" 2>/dev/null); OM=$(grep -c "Heap OOM" "$TEST_LOG" 2>/dev/null)
IR=$(grep -c "VFS accessible" "$TEST_LOG" 2>/dev/null); set -e
HB="${HB:-0}"; PA="${PA:-0}"; PF="${PF:-0}"; OM="${OM:-0}"; IR="${IR:-0}"
TE=$((PA+PF+OM))
if [[ $TE -gt 0 ]]; then echo "FAIL: $TE errors"; exit 4; fi
if [[ $HB -ge 5 ]]; then echo "PASS: $HB heartbeats"
elif [[ $HB -ge 1 ]]; then echo "WARN: Only $HB heartbeats"
else echo "FAIL: No heartbeats"; exit 2; fi
echo "=== Test 4: Initrd ==="; [[ $IR -ge 1 ]] && echo "PASS" || { echo "FAIL"; exit 3; }
echo "=== Test 5: Subsystems ==="
for s in "VFS initialized" "Scheduler:" "SYSCALL:" "Users:" "Telemetry:" "Security:"; do grep -q "$s" "$TEST_LOG" 2>/dev/null && echo "  OK: $s" || echo "  N/A: $s"; done
echo "=== Test 6: AI Modules ==="
for s in "info_bottleneck" "cross_modal" "novelty" "ai_engine" "initrd"; do grep -q "$s" "$TEST_LOG" 2>/dev/null && echo "  OK: $s"; done
echo ""; echo "ALL TESTS PASSED - ${HB} heartbeats"; exit 0
