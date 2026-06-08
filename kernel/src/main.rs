//! NodeAI Kernel — Main Entry Point
//!
//! Architecture: x86_64-unknown-none (bare metal)
//! This is where the bootloader hands control to the kernel.

#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]
#![feature(alloc_error_handler)]

extern crate alloc;
extern crate libm;

use bootloader_api::{entry_point, BootInfo, BootloaderConfig};
use bootloader_api::config::Mapping;

mod acpi;
mod ai_engine;
pub mod audio;
mod desktop;
mod elf;
mod framebuffer;
mod gdt;
mod interrupts;
pub mod kring;
mod logger;
mod memory;
mod net;
mod scheduler;
mod security;
mod shell;
pub mod syscall;
mod telemetry;
pub mod users;
pub mod vfs;
pub mod vga;

// ── Phase 27 — Hardware Parity ────────────────────────────────────────────────
mod ahci;
mod crash_dump;
mod efi_vars;
mod fsck;
mod gpu;
mod nvme;
mod power;
mod rtc;
mod tpm;
mod usb;
mod watchdog;
mod wifi;

// ── Phase 28 — Developer Experience & Self-Hosting ──────────────────────────
mod build_sys;
mod containers;
mod git_reader;
mod kadb;
mod pkg;
mod profiler;

// ── Phase 29 — AI Parity & Beyond Linux ──────────────────────────────────────
mod auto_security;
mod intel_storage;
mod intent_config;
mod llm;
mod predictive_hibernate;
pub mod syscall_stats;  // per-task syscall histograms
pub mod anomaly;        // causal anomaly detector
pub mod tunables;       // live AI-adjustable kernel parameters
pub mod fingerprint;    // behavioral cluster classifier
pub mod causal;         // live causal process wakeup DAG
pub mod transformer_sched; // transformer-based scheduling policy

/// Bootloader configuration — tells the bootloader to map all physical memory
/// at a dynamic virtual offset so we can access physical frames by VA.
const BOOTLOADER_CONFIG: BootloaderConfig = {
    let mut cfg = BootloaderConfig::new_default();
    cfg.mappings.physical_memory = Some(Mapping::Dynamic);
    cfg
};

entry_point!(kernel_main, config = &BOOTLOADER_CONFIG);

/// Kernel entry point called by the bootloader.
/// At this point we are in 64-bit long mode with paging enabled by the bootloader.
fn kernel_main(boot_info: &'static mut BootInfo) -> ! {
    // ── Phase 0: Early serial logging (no allocator, no paging changes needed) ──
    logger::init();
    klog!(INFO, "NodeAI Kernel v{} starting", env!("CARGO_PKG_VERSION"));

    // ── Phase 1: GDT + TSS ────────────────────────────────────────────────────
    gdt::init();

    // ── Phase 1: Memory subsystem ─────────────────────────────────────────────
    // Extract RSDP and framebuffer pointers BEFORE memory::init consumes boot_info.
    let rsdp_addr = boot_info.rsdp_addr.into_option();
    // Capture raw framebuffer pointer (virtual addr already mapped by bootloader).
    let fb_setup: Option<(*mut u8, usize, usize, usize, usize, framebuffer::PixelFormat)> =
        boot_info.framebuffer.as_mut().map(|fb| {
            let info = fb.info();
            let ptr  = fb.buffer_mut().as_mut_ptr();
            let fmt  = match info.pixel_format {
                bootloader_api::info::PixelFormat::Rgb => framebuffer::PixelFormat::Rgb,
                bootloader_api::info::PixelFormat::Bgr => framebuffer::PixelFormat::Bgr,
                _                                      => framebuffer::PixelFormat::Unknown,
            };
            (ptr, info.width, info.height, info.stride, info.bytes_per_pixel, fmt)
        });
    let phys_offset = memory::init(boot_info);

    // ── Phase 1: VGA console (remapped after phys_offset is known) ───────────
    vga::init(phys_offset);
    klog!(INFO, "VGA console ready");

    // ── Phase 12a: Framebuffer + Desktop ─────────────────────────────────────
    if let Some((ptr, w, h, stride, bpp, fmt)) = fb_setup {
        framebuffer::init(ptr, w, h, stride, bpp, fmt);
        desktop::init();
        klog!(INFO, "Desktop: {}×{} framebuffer up", w, h);
    }

    // ── Phase 1: Remap APIC to virtual address ────────────────────────────────
    interrupts::apic::remap_to_virtual(phys_offset);

    // ── Phase 1: IDT + APIC ───────────────────────────────────────────────────
    interrupts::init();

    // ── Phase 1: I/O APIC — route IRQ1 (keyboard) to vector 0x21 ────────────
    interrupts::io_apic::init(phys_offset);

    // ── Phase 2: ACPI ─────────────────────────────────────────────────────────
    if let Some(rsdp) = rsdp_addr {
        acpi::init(rsdp, phys_offset);
    } else {
        klog!(WARN, "No RSDP address from bootloader — ACPI unavailable");
    }

    // ── Phase 4: Scheduler ────────────────────────────────────────────────────
    scheduler::init();

    // ── Phase 6: PCI device scan + VirtIO-blk init ───────────────────────────
    {
        use drivers::pci;
        use drivers::virtio::blk::{VirtioBlk, VIRTIO_VENDOR, VIRTIO_BLK_DEVICE, VIRTIO_BLK_DEVICE2};
        use drivers::virtio::gpu::{VirtioGpu, VIRTIO_GPU_VENDOR, VIRTIO_GPU_DEVICE};
        use drivers::virtio::net::{VirtioNet, VIRTIO_NET_VENDOR, VIRTIO_NET_DEVICE, VIRTIO_NET_DEVICE2};
        let devices = pci::enumerate();
        klog!(INFO, "PCI: {} device(s) found", devices.len());
        for addr in &devices {
            let id = addr.id();
            klog!(DEBUG, "  PCI {:02x}:{:02x}.{} vendor={:#06x} device={:#06x}",
                addr.bus, addr.device, addr.function, id.vendor_id, id.device_id);
            if id.vendor_id == VIRTIO_VENDOR
                && (id.device_id == VIRTIO_BLK_DEVICE || id.device_id == VIRTIO_BLK_DEVICE2)
            {
                addr.enable_bus_master();
                // SAFETY: phys_offset is valid for all of physical RAM mapping
                if let Some(blk) = unsafe { VirtioBlk::init(*addr, phys_offset) } {
                    klog!(INFO, "VirtIO-blk: {} sectors ({} MiB)", blk.sector_count(),
                        blk.sector_count() / 2048);
                }
            }
            // VirtIO-GPU (Phase 12a): probe only when bootloader FB unavailable
            if id.vendor_id == VIRTIO_GPU_VENDOR && id.device_id == VIRTIO_GPU_DEVICE {
                addr.enable_bus_master();
                if let Some(mut gpu) = unsafe { VirtioGpu::init(*addr) } {
                    if !framebuffer::is_available() {
                        if let Some(fb_ptr) = unsafe { gpu.setup_framebuffer(1024, 768) } {
                            framebuffer::init(fb_ptr, 1024, 768, 1024, 4,
                                framebuffer::PixelFormat::Unknown);
                            desktop::init();
                        }
                    }
                    klog!(INFO, "VirtIO-GPU: device at {:02x}:{:02x}.{}",
                        addr.bus, addr.device, addr.function);
                }
            }
            // VirtIO-Net (Phase 17): network interface
            if id.vendor_id == VIRTIO_NET_VENDOR
                && (id.device_id == VIRTIO_NET_DEVICE || id.device_id == VIRTIO_NET_DEVICE2)
                && addr.class_code() == 0x02  // Network controller
            {
                if let Some(nic) = unsafe { VirtioNet::init(*addr, memory::phys_offset(), memory::alloc_frames) } {
                    klog!(INFO, "VirtIO-net: device at {:02x}:{:02x}.{} MAC {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                        addr.bus, addr.device, addr.function,
                        nic.mac[0], nic.mac[1], nic.mac[2], nic.mac[3], nic.mac[4], nic.mac[5]);
                    net::init_nic(nic);
                }
            }
            // AC97 Audio (Phase 25): Intel ICH AC97 controller
            if id.vendor_id == audio::AC97_VENDOR
                && matches!(id.device_id,
                    audio::AC97_DEV_ICH | audio::AC97_DEV_ICH0 |
                    audio::AC97_DEV_ICH2 | audio::AC97_DEV_ICH3)
            {
                addr.enable_bus_master();
                // BAR0 = NAM (Mixer), BAR1 = NABM (Bus Master); both I/O ports
                let nam  = addr.bar_io_base(0);
                let nabm = addr.bar_io_base(1);
                audio::init_if_present(id.vendor_id, id.device_id, nam, nabm);
            }
        }

        // Init PS/2 keyboard
        drivers::input::init();
        klog!(INFO, "PS/2 keyboard initialized");
    }

    // ── Phase 27: Hardware parity drivers ────────────────────────────────────
    rtc::init();
    crash_dump::init(phys_offset);
    ahci::init(phys_offset);
    nvme::init(phys_offset);
    usb::init(phys_offset);
    gpu::init(phys_offset);
    wifi::init(phys_offset);
    watchdog::init(300); // 5-minute hardware watchdog timeout
    tpm::init(phys_offset);
    power::init();
    // Check for crash dump from previous boot
    if let Some(entry) = crash_dump::check_previous_crash() {
        klog!(WARN, "Previous boot crashed at RIP={:#x}: {}", entry.rip, entry.message);
        crash_dump::clear_crash_record();
    }
    // EFI variables (RT services VA — set to 0 if bootloader doesn't expose it)
    efi_vars::init(0);

    // ── Phase 28: Developer experience & self-hosting ────────────────────────
    kadb::init();
    profiler::init(phys_offset);
    pkg::init();
    containers::init();
    // build_sys and git_reader are demand-loaded (no startup cost)
    klog!(INFO, "Phase 28: dev tools ready");

    // ── Phase 29: AI-native intelligence beyond Linux ─────────────────────────
    predictive_hibernate::init();
    intent_config::init();
    auto_security::init();
    intel_storage::init();
    llm::init();
    klog!(INFO, "Phase 29: AI-parity features active");

    // ── Phase 7: VFS ─────────────────────────────────────────────────────────
    vfs::init();

    // ── Phase 17: Network configuration ──────────────────────────────────────
    net::init_routes();
    net::init_hosts();

    // ── Phase 8: AI subsystem ────────────────────────────────────────────────
    ai_engine::init();
    fingerprint::init();
    transformer_sched::init();

    // ── Phase 12b: Populate /proc and /ai virtual filesystem entries ──────────
    vfs::procfs::init();

    // ── Phase 14: Users & authentication ─────────────────────────────────────
    users::init();

    // ── Phase 12c: Kernel shell (after users so prompt shows username) ────────
    shell::init();
    // ── Phase 13: Self-instrumentation telemetry ──────────────────────────────
    telemetry::init();
    // ── Phase 10: Security hardening ─────────────────────────────────────────
    security::init();

    // ── Phase 11: Syscall fast-path (LSTAR/STAR/FMASK MSRs) ──────────────────
    syscall::init_lstar();
    klog!(INFO, "SYSCALL: fast-path active");

    klog!(INFO, "NodeAI Kernel boot complete — entering idle loop");
    vga_println!("NodeAI boot complete. AI kernel online.");

    // All subsystems initialized — enable hardware interrupts now so the
    // APIC timer can fire safely (scheduler, AI engine, telemetry all ready).
    x86_64::instructions::interrupts::enable();
    klog!(INFO, "Interrupts enabled");

    idle_loop()
}

fn idle_loop() -> ! {
    loop {
        // Poll the NIC for incoming frames
        net::poll();
        // Service HTTP/SSH if running
        net::http_server_poll();
        net::ssh_server_poll();
        // Execute any browser fetch that was queued from the keyboard IRQ
        // (runs here so interrupts remain enabled during the blocking fetch)
        crate::desktop::browser_fetch_tick();
        x86_64::instructions::interrupts::enable_and_hlt();
    }
}

/// Kernel panic handler — prints info to serial and VGA, then halts.
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    x86_64::instructions::interrupts::disable();

    // Try serial (most likely to work since it requires no paging)
    logger::log(
        logger::Level::ERROR,
        "panic",
        0,
        format_args!("KERNEL PANIC: {}", info),
    );

    // Try VGA if it's set up
    use core::fmt::Write;
    if let Some(mut w) = vga::WRITER.try_lock() {
        w.set_colour(vga::Colour::LightRed, vga::Colour::Black);
        let _ = write!(w, "\nKERNEL PANIC: {}\n", info);
    }

    loop {
        x86_64::instructions::hlt();
    }
}

/// Called by the global allocator when allocation fails.
#[alloc_error_handler]
fn alloc_error(layout: core::alloc::Layout) -> ! {
    panic!("Kernel heap OOM: {:?}", layout);
}

// ── Logging macro ─────────────────────────────────────────────────────────────

#[macro_export]
macro_rules! klog {
    ($level:ident, $($arg:tt)*) => {
        $crate::logger::log(
            $crate::logger::Level::$level,
            file!(),
            line!(),
            format_args!($($arg)*),
        )
    };
}

