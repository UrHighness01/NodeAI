# NodeAI build helper — builds the kernel and creates bootable disk images.
# Usage: .\scripts\build.ps1 [-Release] [-Images]

param(
    [switch]$Release,  # Build with optimisations
    [switch]$Images    # Also build disk images via image-builder
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot

# Ensure rustup shims take priority over any system-installed Rust
$cargoBin = "$env:USERPROFILE\.cargo\bin"
if (Test-Path $cargoBin) { $env:PATH = "$cargoBin;$env:PATH" }

Push-Location $ROOT

# ── Kernel ────────────────────────────────────────────────────────────────────
Write-Host "==> Building kernel (x86_64-unknown-none)..." -ForegroundColor Cyan
$args = @("build", "--package", "nodeai-kernel")
if ($Release) { $args += "--release" }
& cargo @args
if ($LASTEXITCODE -ne 0) { Write-Error "Kernel build failed." ; exit 1 }

$profile = if ($Release) { "release" } else { "debug" }
$kernelELF = "$ROOT\target\x86_64-unknown-none\$profile\nodeai-kernel"
Write-Host "  ✓ $kernelELF" -ForegroundColor Green

# ── Disk images (optional) ────────────────────────────────────────────────────
if ($Images) {
    Write-Host "==> Building image-builder (host)..." -ForegroundColor Cyan
    # Build for the host (Windows). We override .cargo/config.toml's default target
    # by passing the host triple explicitly so cargo doesn't cross-compile to bare metal.
    $hostTriple = "x86_64-pc-windows-msvc"
    & cargo build --package image-builder --target $hostTriple
    if ($LASTEXITCODE -ne 0) {
        # Fallback: try without explicit target (lets cargo pick the host triple)
        Write-Warning "Build with explicit target failed, retrying without --target..."
        & cargo build --package image-builder
        if ($LASTEXITCODE -ne 0) { Write-Error "image-builder build failed."; exit 1 }
        $imageBuilder = "$ROOT\target\debug\image-builder.exe"
    } else {
        $imageBuilder = "$ROOT\target\$hostTriple\debug\image-builder.exe"
    }

    $imgDir = "$ROOT\target\images"
    New-Item -ItemType Directory -Force -Path $imgDir | Out-Null

    Write-Host "==> Creating bootable disk images..." -ForegroundColor Cyan
    & $imageBuilder $kernelELF $imgDir
    if ($LASTEXITCODE -ne 0) { Write-Error "Image creation failed."; exit 1 }

    Write-Host "  ✓ $imgDir\nodeai-bios.img" -ForegroundColor Green
    Write-Host "  ✓ $imgDir\nodeai-uefi.img" -ForegroundColor Green
}

Pop-Location
Write-Host "Build complete." -ForegroundColor Green
