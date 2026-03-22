# Run NodeAI kernel in QEMU for fast development iteration.
# Build pipeline:
#   1. cargo build --package nodeai-kernel           (bare-metal ELF)
#   2. cargo build --package image-builder           (host tool, wraps ELF in disk image)
#   3. image-builder <kernel.elf> <out-dir>          (creates BIOS/UEFI .img)
#   4. qemu-system-x86_64 -drive file=nodeai-bios.img
#
# Requires: qemu-system-x86_64 on PATH
# Usage:    .\scripts\run_qemu.ps1 [-Debug] [-Memory 512] [-Release] [-Uefi]

param(
    [switch]$Debug,        # Attach GDB stub on localhost:1234, pause at start
    [switch]$Release,      # Build in release mode
    [switch]$Uefi,         # Boot via UEFI instead of BIOS
    [int]$Memory = 512     # RAM in MiB
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot

# ── Ensure rustup shims take priority over any system-installed Rust ──────────
# This matters when a standalone Rust is installed (e.g. "Rust stable MSVC 1.x")
# ahead of %USERPROFILE%\.cargo\bin on PATH.
$cargoBin = "$env:USERPROFILE\.cargo\bin"
if (Test-Path $cargoBin) {
    $env:PATH = "$cargoBin;$env:PATH"
}

# ── 1. Build the kernel ───────────────────────────────────────────────────────
Write-Host "==> Building NodeAI kernel (x86_64-unknown-none)..." -ForegroundColor Cyan
$buildArgs = @("build", "--package", "nodeai-kernel")
if ($Release) { $buildArgs += "--release" }

Push-Location $ROOT
& cargo @buildArgs
if ($LASTEXITCODE -ne 0) { Write-Error "Kernel build failed."; exit 1 }

$profile = if ($Release) { "release" } else { "debug" }
$kernelELF = "$ROOT\target\x86_64-unknown-none\$profile\nodeai-kernel"
if (!(Test-Path $kernelELF)) {
    Write-Error "Kernel ELF not found at $kernelELF"
    exit 1
}
Write-Host "  Kernel ELF: $kernelELF" -ForegroundColor Green

# ── 2. Build image-builder (host tool) ────────────────────────────────────────
Write-Host "==> Building image-builder (host tool)..." -ForegroundColor Cyan
$hostTarget = "x86_64-pc-windows-msvc"
& cargo build --package image-builder --target $hostTarget
if ($LASTEXITCODE -ne 0) { Write-Error "image-builder build failed."; exit 1 }

$imageBuilder = "$ROOT\target\$hostTarget\debug\image-builder.exe"
if (!(Test-Path $imageBuilder)) {
    Write-Error "image-builder.exe not found at $imageBuilder"
    exit 1
}

# ── 3. Create bootable disk images ────────────────────────────────────────────
Write-Host "==> Creating disk images..." -ForegroundColor Cyan
$imgDir = "$ROOT\target\images"
New-Item -ItemType Directory -Force -Path $imgDir | Out-Null

& $imageBuilder $kernelELF $imgDir
if ($LASTEXITCODE -ne 0) { Write-Error "Image creation failed."; exit 1 }

$biosImg = "$imgDir\nodeai-bios.img"
$uefiImg = "$imgDir\nodeai-uefi.img"

# ── 4. Run QEMU ───────────────────────────────────────────────────────────────
Write-Host "==> Starting QEMU ($Memory MiB RAM)..." -ForegroundColor Cyan

$qemuArgs = @(
    "-machine", "q35",
    "-cpu", "qemu64,+avx2",
    "-m", "${Memory}M",
    "-serial", "stdio",       # COM1 → stdout (klog output appears here)
    "-no-reboot",
    "-no-shutdown"
)

if ($Uefi) {
    # UEFI boot — requires OVMF firmware (install qemu-ovmf or similar)
    $ovmf = "C:\Program Files\qemu\share\OVMF.fd"
    if (!(Test-Path $ovmf)) {
        Write-Warning "OVMF not found at $ovmf — trying fallback locations"
        $ovmf = (Get-Command ovmf -ErrorAction SilentlyContinue)?.Source
    }
    if ($ovmf) {
        $qemuArgs += @("-bios", $ovmf)
    } else {
        Write-Warning "OVMF firmware not found — UEFI boot may fail"
    }
    $qemuArgs += @("-drive", "format=raw,file=$uefiImg")
    Write-Host "  Boot: UEFI via $uefiImg"
} else {
    $qemuArgs += @("-drive", "format=raw,file=$biosImg")
    Write-Host "  Boot: BIOS via $biosImg"
}

if ($Debug) {
    $qemuArgs += @("-s", "-S")   # GDB server on :1234, pause at entry
    Write-Host "  GDB stub listening on localhost:1234" -ForegroundColor Yellow
    Write-Host "  Connect with: gdb target/x86_64-unknown-none/$profile/nodeai-kernel" -ForegroundColor Yellow
    Write-Host "    (gdb) target remote localhost:1234" -ForegroundColor Yellow
}

Pop-Location
& qemu-system-x86_64 @qemuArgs

