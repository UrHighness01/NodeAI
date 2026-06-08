//! Local APIC (xAPIC MMIO mode) — replaces legacy 8259 PIC.
//!
//! Provides:
//!  - Disabling the 8259 PIC
//!  - Enabling the local APIC
//!  - Configuring the APIC timer to deliver periodic interrupts
//!  - EOI (End-Of-Interrupt) acknowledgment

/// xAPIC MMIO base (can be relocated via IA32_APIC_BASE MSR, but 0xFEE00000 is the default).
pub const LOCAL_APIC_BASE: u64 = 0xFEE0_0000;

/// IRQ vector assignments (above 0x20 to avoid CPU exceptions).
pub const TIMER_VECTOR:    u8 = 0x20;
pub const KEYBOARD_VECTOR: u8 = 0x21;
pub const MOUSE_VECTOR:    u8 = 0x2C;
pub const SPURIOUS_VECTOR: u8 = 0xFF;

// ── APIC register offsets ─────────────────────────────────────────────────────
const REG_ID:           u32 = 0x020;
const REG_VERSION:      u32 = 0x030;
const REG_SPURIOUS:     u32 = 0x0F0;
const REG_EOI:          u32 = 0x0B0;
const REG_TIMER_LVT:    u32 = 0x320;
const REG_TIMER_ICR:    u32 = 0x380; // Initial Count Register
const REG_TIMER_CCR:    u32 = 0x390; // Current Count Register
const REG_TIMER_DCR:    u32 = 0x3E0; // Divide Configuration Register

// ── APIC register access ──────────────────────────────────────────────────────

/// Physical address of the local APIC MMIO region.
/// After higher-half remapping, this is updated to its virtual equivalent.
static mut APIC_VIRT_BASE: u64 = LOCAL_APIC_BASE;

unsafe fn apic_read(reg: u32) -> u32 {
    let addr = (APIC_VIRT_BASE + reg as u64) as *const u32;
    core::ptr::read_volatile(addr)
}

unsafe fn apic_write(reg: u32, val: u32) {
    let addr = (APIC_VIRT_BASE + reg as u64) as *mut u32;
    core::ptr::write_volatile(addr, val);
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Remap APIC registers to the virtual address space after VMM is set up.
pub fn remap_to_virtual(phys_mem_offset: u64) {
    unsafe {
        APIC_VIRT_BASE = phys_mem_offset + LOCAL_APIC_BASE;
    }
}

/// Initialise and enable the local APIC.
/// # Safety
/// Must be called with interrupts disabled. The VMM must be initialised before
/// this function is called so that the APIC MMIO region can be explicitly mapped.
pub unsafe fn init_apic() {
    // ── Diagnostic: check IA32_APIC_BASE MSR bit 11 (APIC Global Enable) ──
    let apic_base_msr: u32;
    let apic_base_msr_hi: u32;
    core::arch::asm!("rdmsr", in("ecx") 0x1Bu32, out("eax") apic_base_msr, out("edx") apic_base_msr_hi);
    let apic_base = (apic_base_msr_hi as u64) << 32 | apic_base_msr as u64;
    let enabled = (apic_base_msr >> 11) & 1;
    let base_addr = apic_base_msr & 0xFFFFF000u32; // bits 12-31
    crate::klog!(INFO, "IA32_APIC_BASE={:#x} enabled={} addr={:#05x}",
        apic_base, enabled, base_addr);

    // If APIC is globally disabled, enable it.
    if enabled == 0 {
        crate::klog!(WARN, "APIC globally disabled in IA32_APIC_BASE — enabling");
        core::arch::asm!("wrmsr", in("ecx") 0x1Bu32,
            in("eax") apic_base_msr | (1 << 11),
            in("edx") apic_base_msr_hi);
    }

    // Explicitly map the LAPIC MMIO region into the virtual address space.
    // The bootloader's Mapping::Dynamic already maps this VA but with WB
    // caching (APIC writes would buffer and never reach the device).
    // map_mmio unmaps the WB entry first and creates a UC+WT mapping.
    crate::memory::map_mmio(LOCAL_APIC_BASE, APIC_VIRT_BASE, 0x1000);

    // Disable legacy 8259 PIC by masking all IRQs.
    disable_pic();

    // Enable APIC via SPURIOUS register bit 8.
    let spiv = apic_read(REG_SPURIOUS);
    apic_write(REG_SPURIOUS, spiv | (1 << 8) | SPURIOUS_VECTOR as u32);

    // Calibrate LAPIC timer using PIT channel 2 as reference.
    let ticks_per_ms = calibrate_timer();
    crate::klog!(INFO, "LAPIC timer calibrated: {} ticks/ms", ticks_per_ms);

    // Configure periodic timer at 100 Hz (10 ms tick) for scheduler.
    // Firing at 1 kHz caused excessive VM-exits in VirtualBox → host CPU at 100%.
    apic_write(REG_TIMER_DCR, 0x3);                               // divisor = 16
    apic_write(REG_TIMER_LVT, (1 << 17) | TIMER_VECTOR as u32);  // periodic mode
    apic_write(REG_TIMER_ICR, ticks_per_ms * 10);                 // 10 ms period → 100 Hz

    // ── Diagnostic: readback REG_TIMER_LVT to confirm write stuck ──────────
    let lvt_readback = apic_read(REG_TIMER_LVT);
    if lvt_readback != ((1 << 17) | TIMER_VECTOR as u32) {
        crate::klog!(ERROR, "TIMER LVT write did NOT stick! wrote={:#x} readback={:#x}",
            (1 << 17) | TIMER_VECTOR as u32, lvt_readback);
    } else {
        crate::klog!(INFO, "TIMER LVT readback OK — value={:#x}", lvt_readback);
    }

    let version = apic_read(REG_VERSION);
    crate::klog!(INFO, "LAPIC enabled — version={:#x}", version & 0xFF);
}

/// Calibrate the LAPIC timer against the PIT.
/// Returns LAPIC ticks per millisecond.
unsafe fn calibrate_timer() -> u32 {
    use x86_64::instructions::port::Port;

    // PIT channel 2, mode 0 (one-shot), binary.
    // Count down from 0xFFFF at 1.193182 MHz ≈ 838 ns / tick.
    // We measure how many LAPIC ticks happen in 10 ms of PIT time.

    const PIT_HZ: u64 = 1_193_182;
    const MEASURE_MS: u64 = 10;
    const PIT_TICKS: u16 = ((PIT_HZ * MEASURE_MS) / 1000) as u16; // ≈ 11932

    let mut cmd: Port<u8> = Port::new(0x43);
    let mut ch2: Port<u8> = Port::new(0x42);
    let mut port61: Port<u8> = Port::new(0x61);

    // Gate PIT ch2 off, enable speaker gate
    let p61 = port61.read();
    port61.write((p61 & 0xFD) | 0x01); // disable gate, keep speaker on

    // PIT channel 2: mode 0 (one-shot), access mode lobyte/hibyte
    cmd.write(0xB0);
    ch2.write((PIT_TICKS & 0xFF) as u8);
    ch2.write((PIT_TICKS >> 8) as u8);

    // Set divisor and start LAPIC counter at max
    apic_write(REG_TIMER_DCR, 0x3); // divide by 16
    apic_write(REG_TIMER_ICR, 0xFFFF_FFFF);

    // Gate PIT ch2 on — starts counting
    let p61 = port61.read();
    port61.write(p61 | 0x01);

    // Wait for PIT output to go high (channel counted to zero)
    loop {
        let v = port61.read();
        if v & 0x20 != 0 { break; } // OUT pin high = done
    }

    // Stop LAPIC timer
    apic_write(REG_TIMER_LVT, 1 << 16); // masked

    let lapic_remaining = apic_read(REG_TIMER_CCR);
    let elapsed = 0xFFFF_FFFF - lapic_remaining;

    // elapsed ticks measured over MEASURE_MS milliseconds
    let per_ms = elapsed / MEASURE_MS as u32;
    per_ms.max(1) // guard against calibration failure
}

/// Send EOI (End-Of-Interrupt) to the local APIC.
/// Must be called at the end of every hardware interrupt handler.
/// # Safety
/// Caller must be in an interrupt context.
pub unsafe fn eoi() {
    apic_write(REG_EOI, 0);
}

/// Send an Inter-Processor Interrupt (IPI) to a specific APIC ID.
/// `vector` is the IDT vector to deliver (32–255).
/// # Safety
/// Target CPU must be online and have its IDT entry configured.
pub unsafe fn send_ipi(apic_id: u8, vector: u8) {
    // ICR high: destination APIC ID in bits 24-31
    apic_write(0x310, (apic_id as u32) << 24);
    // ICR low: fixed delivery, edge triggered, assert, vector
    apic_write(0x300, (1 << 14) | vector as u32);
    // Wait for delivery to clear (Delivery Status bit 12)
    while apic_read(0x300) & (1 << 12) != 0 { core::hint::spin_loop(); }
}

/// Broadcast an INIT IPI to all other CPUs (for SMP bring-up).
/// # Safety
/// SMP init path must be carefully sequenced.
pub unsafe fn send_init_ipi_all_excluding_self() {
    // All excluding self | INIT | Level Trigger | Assert
    apic_write(0x310, 0); // upper: broadcast uses ICR low flags
    apic_write(0x300, 0x000C_4500); // shorthand = 11 (all excl self), INIT
    while apic_read(0x300) & (1 << 12) != 0 { core::hint::spin_loop(); }
}

/// Send STARTUP IPI (SIPI) to all other CPUs with the given start page.
/// `start_page`: physical address >> 12 (must fit in 8 bits, i.e. below 1 MiB).
/// # Safety
/// AP trampoline must be at the given physical page.
pub unsafe fn send_sipi_all_excluding_self(start_page: u8) {
    apic_write(0x310, 0);
    apic_write(0x300, 0x000C_4600 | start_page as u32);
    while apic_read(0x300) & (1 << 12) != 0 { core::hint::spin_loop(); }
}

// ── Legacy PIC disable ────────────────────────────────────────────────────────

/// Mask all IRQs on both 8259 PIC chips, effectively disabling them.
unsafe fn disable_pic() {
    use x86_64::instructions::port::Port;

    // Send cascade init sequence so the PIC is in known state, then mask all.
    // ICW1: initialise + ICW4 needed
    Port::<u8>::new(0x20).write(0x11);
    Port::<u8>::new(0xA0).write(0x11);
    // ICW2: remap IRQ 0-7 to 0xA0, IRQ 8-15 to 0xA8 (above our APIC vectors)
    Port::<u8>::new(0x21).write(0xA0);
    Port::<u8>::new(0xA1).write(0xA8);
    // ICW3: cascade configuration
    Port::<u8>::new(0x21).write(0x04);
    Port::<u8>::new(0xA1).write(0x02);
    // ICW4: 8086 mode
    Port::<u8>::new(0x21).write(0x01);
    Port::<u8>::new(0xA1).write(0x01);
    // OCW1: mask ALL IRQs
    Port::<u8>::new(0x21).write(0xFF);
    Port::<u8>::new(0xA1).write(0xFF);
}
