# Run NodeAI kernel in Oracle VirtualBox
# Automatically builds the kernel, converts the disk image to VDI, and boots the VM.
#
# Usage:
#   .\scripts\run_vbox.ps1              — build + boot (creates VM on first run)
#   .\scripts\run_vbox.ps1 -Create      — force-recreate the VM from scratch
#   .\scripts\run_vbox.ps1 -Release     — use a release-mode kernel build
#   .\scripts\run_vbox.ps1 -Gui         — launch with GUI window instead of headless

param(
    [string]$VmName  = "NodeAI-Dev",
    [int]$Memory     = 512,
    [switch]$Create,   # Force-recreate the VM from scratch
    [switch]$Release,  # Build kernel with --release
    [switch]$Gui       # Start VM with a GUI window (default: headless)
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot

# ── Locate VBoxManage (check PATH first, then default install dir) ────────────
$VBoxManage = "VBoxManage"
if (!(Get-Command $VBoxManage -ErrorAction SilentlyContinue)) {
    $defaultPath = "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe"
    if (Test-Path $defaultPath) {
        $VBoxManage = $defaultPath
        $env:PATH = "C:\Program Files\Oracle\VirtualBox;$env:PATH"
    } else {
        Write-Error "VBoxManage not found. Install Oracle VirtualBox."
        exit 1
    }
}

# Ensure rustup shims take priority over any system-installed Rust
$cargoBin = "$env:USERPROFILE\.cargo\bin"
if (Test-Path $cargoBin) { $env:PATH = "$cargoBin;$env:PATH" }

function Assert-VBoxManage {
    # Already resolved above; this is a no-op kept for clarity.
}

function New-NodeAIVm {
    Write-Host "Creating VirtualBox VM: $VmName" -ForegroundColor Cyan

    & $VBoxManage createvm --name $VmName --ostype "Linux_64" --register

    & $VBoxManage modifyvm $VmName `
        --memory $Memory `
        --vram 16 `
        --cpus 2 `
        --firmware bios `
        --boot1 disk `
        --boot2 none `
        --nic1 nat `
        --nictype1 virtio `
        --uart1 0x3F8 4 --uartmode1 file "$ROOT\nodeai_serial.log"

    # Storage controllers — VirtIO for disk (matches our driver), IDE kept for CD
    & $VBoxManage storagectl $VmName --name "VirtIO Controller" --add virtio
    & $VBoxManage storagectl $VmName --name "IDE Controller"    --add ide

    Write-Host "  VM shell created (no disk attached yet — will be added during build step)." -ForegroundColor Green
}

function Build-Images {
    Write-Host "==> Building kernel (x86_64-unknown-none)..." -ForegroundColor Cyan
    $cargoArgs = @("build", "--package", "nodeai-kernel")
    if ($Release) { $cargoArgs += "--release" }
    & cargo @cargoArgs
    if ($LASTEXITCODE -ne 0) { Write-Error "Kernel build failed."; exit 1 }

    Write-Host "==> Building image-builder (host)..." -ForegroundColor Cyan
    $hostTriple = "x86_64-pc-windows-msvc"
    & cargo build --package image-builder --target $hostTriple
    if ($LASTEXITCODE -ne 0) { Write-Error "image-builder build failed."; exit 1 }

    $profile      = if ($Release) { "release" } else { "debug" }
    $kernelELF    = "$ROOT\target\x86_64-unknown-none\$profile\nodeai-kernel"
    $imageBuilder = "$ROOT\target\$hostTriple\debug\image-builder.exe"
    $imgDir       = "$ROOT\target\images"
    New-Item -ItemType Directory -Force -Path $imgDir | Out-Null

    Write-Host "==> Creating bootable disk images..." -ForegroundColor Cyan
    & $imageBuilder $kernelELF $imgDir
    if ($LASTEXITCODE -ne 0) { Write-Error "Image creation failed."; exit 1 }
    Write-Host "  OK  $imgDir\nodeai-bios.img" -ForegroundColor Green
    Write-Host "  OK  $imgDir\nodeai-uefi.img" -ForegroundColor Green
}

function Convert-ImageToVdi {
    $imgPath  = "$ROOT\target\images\nodeai-bios.img"
    $vdiPath  = "$ROOT\nodeai-boot.vdi"

    if (!(Test-Path $imgPath)) {
        Write-Error "Disk image not found at $imgPath — did the build succeed?"
        exit 1
    }

    # Step 1: Detach the old medium from the VM (must happen before closing/deleting)
    & $VBoxManage storageattach $VmName `
        --storagectl "VirtIO Controller" --port 0 --device 0 `
        --type hdd --medium none 2>$null

    # Step 2: Unregister from VirtualBox media registry (no --delete so the file path
    #         is also resolved correctly; we delete it ourselves next)
    & $VBoxManage closemedium disk $vdiPath 2>$null

    # Step 3: Remove the physical file (closemedium without --delete keeps the file)
    if (Test-Path $vdiPath) { Remove-Item $vdiPath -Force }

    Write-Host "==> Converting BIOS image -> VDI..." -ForegroundColor Cyan
    & $VBoxManage convertfromraw $imgPath $vdiPath --format VDI
    if ($LASTEXITCODE -ne 0) { Write-Error "VDI conversion failed."; exit 1 }
    Write-Host "  OK $vdiPath" -ForegroundColor Green
    return $vdiPath
}

function Attach-BootDisk ($vdiPath) {
    & $VBoxManage storageattach $VmName `
        --storagectl "VirtIO Controller" --port 0 --device 0 `
        --type hdd --medium $vdiPath
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to attach boot disk."; exit 1 }
    Write-Host "  OK Boot disk attached." -ForegroundColor Green
}

# ── Main ───────────────────────────────────────────────────────────────────────

Push-Location $ROOT

# Check if the VM exists
$vmList = & $VBoxManage list vms
$vmExists = $vmList -match [regex]::Escape("`"$VmName`"")

# Destroy and re-create if -Create was requested
if ($Create -and $vmExists) {
    Write-Host "Destroying existing VM: $VmName" -ForegroundColor Yellow
    & $VBoxManage controlvm $VmName poweroff 2>$null
    Start-Sleep 1
    & $VBoxManage unregistervm $VmName --delete
    $vmExists = $false
}

if (!$vmExists) {
    New-NodeAIVm
}

# Always rebuild + refresh the boot disk on each run
Build-Images
$vdi = Convert-ImageToVdi
Attach-BootDisk $vdi

# Boot the VM
$vmType = if ($Gui) { "gui" } else { "headless" }
Write-Host "==> Starting '$VmName' ($vmType)..." -ForegroundColor Cyan
& $VBoxManage startvm $VmName --type $vmType
if ($LASTEXITCODE -ne 0) { Write-Error "Failed to start VM."; exit 1 }

Write-Host ""
Write-Host "NodeAI is running." -ForegroundColor Green
Write-Host "  Serial log : $ROOT\nodeai_serial.log"
Write-Host "  Stop VM    : VBoxManage controlvm $VmName poweroff"
Write-Host "  GUI access : VBoxManage startvm $VmName --type gui  (if running headless)"

Pop-Location
