//! ACPI Power Management — Phase 27.
//!
//! Provides:
//!   - System suspend (S3 sleep) and power-off (S5) via PM1 control registers
//!   - Battery status via EC / ACPI _BST data (if available)
//!   - Screen backlight brightness control via ACPI _BCM or direct I/O fallback
//!   - Power-button SCI handling

use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use x86_64::instructions::port::Port;

// ── PM1 register access (obtained from FADT) ──────────────────────────────────

static PM1A_PORT:  core::sync::atomic::AtomicU16  = core::sync::atomic::AtomicU16::new(0);
static PM1B_PORT:  core::sync::atomic::AtomicU16  = core::sync::atomic::AtomicU16::new(0);
/// Sleep type for S3 (suspend) — guessed value; real value from DSDT _S3_.
static SLP_TYP_S3: core::sync::atomic::AtomicU16  = core::sync::atomic::AtomicU16::new(5);
/// Sleep type for S5 (power off) — from FADT's ACPI slp_typa.
static SLP_TYP_S5: core::sync::atomic::AtomicU16  = core::sync::atomic::AtomicU16::new(0);
static POWER_INIT: AtomicBool = AtomicBool::new(false);

// ── Battery EC ports (typical ACPI EC at 0x62/0x66) ──────────────────────────
const EC_DATA: u16 = 0x62;
const EC_CMD:  u16 = 0x66;

// Standard SMBus battery registers (0-based, via EC)
const EC_CMD_READ: u8 = 0x80;  // Read EC byte
const EC_BAT_STA:  u8 = 0x01;  // Battery status
const EC_BAT_RATE: u8 = 0x14;  // Current (mA, signed 16-bit) at offset 0x14
const EC_BAT_REM:  u8 = 0x16;  // Remaining capacity (mWh)
const EC_BAT_FULL: u8 = 0x18;  // Full charge capacity (mWh)

// ── Backlight — attempt Intel GMBUS / simple ACPI I/O fallback ───────────────
static BRIGHTNESS: AtomicU8 = AtomicU8::new(100);

// ── Initialise from ACPI FADT data ────────────────────────────────────────────

pub fn init() {
    if let Some(pm) = crate::acpi::pm_info() {
        PM1A_PORT.store(pm.pm1a_ctrl as u16, Ordering::Relaxed);
        if pm.pm1b_ctrl != 0 {
            PM1B_PORT.store(pm.pm1b_ctrl as u16, Ordering::Relaxed);
        }
        // slp_typa from FADT is the S5 (power-off) sleep type
        SLP_TYP_S5.store(pm.slp_typa, Ordering::Relaxed);
    }
    POWER_INIT.store(true, Ordering::Relaxed);
    crate::klog!(INFO, "power: ACPI power management ready");
}

// ── Sleep / shutdown ──────────────────────────────────────────────────────────

/// Suspend to RAM (S3). Writes SLP_TYP | SLP_EN to PM1 control register.
/// On real hardware this causes the machine to sleep; never returns on success.
pub fn suspend_s3() {
    let port_a = PM1A_PORT.load(Ordering::Relaxed);
    if port_a == 0 {
        crate::klog!(WARN, "power: no PM1a port, cannot suspend");
        return;
    }
    let typ = SLP_TYP_S3.load(Ordering::Relaxed);
    let val_a: u16 = (typ << 10) | (1 << 13); // SLP_EN
    unsafe {
        let mut p: Port<u16> = Port::new(port_a);
        p.write(val_a);

        let port_b = PM1B_PORT.load(Ordering::Relaxed);
        if port_b != 0 {
            let mut pb: Port<u16> = Port::new(port_b);
            pb.write(val_a); // same type / SLP_EN for B
        }
    }
    // If we return here the platform does not support S3
    crate::klog!(WARN, "power: S3 suspend returned — platform may not support it");
}

/// System power-off (S5). Writes SLP_TYP(S5) | SLP_EN.  Does not return.
pub fn power_off() -> ! {
    let port_a = PM1A_PORT.load(Ordering::Relaxed);
    if port_a != 0 {
        let typ = SLP_TYP_S5.load(Ordering::Relaxed);
        let val_a: u16 = (typ << 10) | (1 << 13);
        unsafe {
            let mut p: Port<u16> = Port::new(port_a);
            p.write(val_a);
        }
    }
    // QEMU / Bochs fallback: write 0x2000 to port 0x604
    unsafe {
        let mut p: Port<u16> = Port::new(0x604);
        p.write(0x2000u16);
    }
    // If still running, halt
    loop { x86_64::instructions::hlt(); }
}

/// Warm reboot via keyboard controller.
pub fn reboot() -> ! {
    unsafe {
        // Flush 8042 output buffer then pulse reset line
        let mut cmd_port: Port<u8> = Port::new(0x64);
        let mut data_port: Port<u8> = Port::new(0x60);
        let deadline = crate::scheduler::uptime_ms() + 200;
        while crate::scheduler::uptime_ms() < deadline {
            let st: u8 = cmd_port.read();
            if st & 0x01 != 0 { let _: u8 = data_port.read(); }
            if st & 0x02 == 0 { break; }
            core::hint::spin_loop();
        }
        cmd_port.write(0xFEu8); // pulse reset line
    }
    loop { x86_64::instructions::hlt(); }
}

// ── Battery ────────────────────────────────────────────────────────────────────

/// Read a single byte from the Embedded Controller.
unsafe fn ec_read(reg: u8) -> Option<u8> {
    let mut cmd: Port<u8> = Port::new(EC_CMD);
    let mut data: Port<u8> = Port::new(EC_DATA);

    // Wait for IBF = 0 (input buffer empty)
    let deadline = crate::scheduler::uptime_ms() + 100;
    loop {
        if crate::scheduler::uptime_ms() >= deadline { return None; }
        let st: u8 = cmd.read();
        if st & 0x02 == 0 { break; }
        core::hint::spin_loop();
    }
    cmd.write(EC_CMD_READ);

    // Wait for IBF = 0
    let deadline = crate::scheduler::uptime_ms() + 100;
    loop {
        if crate::scheduler::uptime_ms() >= deadline { return None; }
        let st: u8 = cmd.read();
        if st & 0x02 == 0 { break; }
        core::hint::spin_loop();
    }
    data.write(reg);

    // Wait for OBF = 1 (output buffer full)
    let deadline = crate::scheduler::uptime_ms() + 100;
    loop {
        if crate::scheduler::uptime_ms() >= deadline { return None; }
        let st: u8 = cmd.read();
        if st & 0x01 != 0 { break; }
        core::hint::spin_loop();
    }
    Some(data.read())
}

unsafe fn ec_read16(reg: u8) -> Option<u16> {
    let lo = ec_read(reg)?;
    let hi = ec_read(reg + 1)?;
    Some(lo as u16 | ((hi as u16) << 8))
}

/// Returns battery charge percentage (0–100) if a battery is present.
pub fn battery_percent() -> Option<u8> {
    unsafe {
        let sta = ec_read(EC_BAT_STA)?;
        if sta & 0x01 == 0 { return None; } // battery absent

        let remaining = ec_read16(EC_BAT_REM)? as u32;
        let full      = ec_read16(EC_BAT_FULL)? as u32;
        if full == 0 { return None; }
        let pct = (remaining * 100 / full).min(100) as u8;
        Some(pct)
    }
}

/// Returns battery charging state: `true` = charging.
pub fn battery_charging() -> bool {
    unsafe {
        ec_read(EC_BAT_STA)
            .map(|s| s & 0x02 != 0)
            .unwrap_or(false)
    }
}

// ── Brightness ────────────────────────────────────────────────────────────────

/// Set screen brightness 0–100. Attempts ACPI _BCM, falls back to port 0x61.
pub fn set_brightness(level: u8) {
    let clamped = level.min(100);
    BRIGHTNESS.store(clamped, Ordering::Relaxed);
    // TODO: invoke ACPI AML _BCM method when AML interpreter is available.
    // For now just record the value; display driver picks it up via `brightness()`.
    crate::klog!(DEBUG, "power: brightness set to {}%", clamped);
}

/// Current brightness setting.
pub fn brightness() -> u8 { BRIGHTNESS.load(Ordering::Relaxed) }

// ── SCI handler (called from interrupt context) ───────────────────────────────

static POWER_BUTTON_PRESSED: AtomicBool = AtomicBool::new(false);

/// Handle ACPI SCI interrupt.  Should be called from the SCI IRQ handler.
pub fn handle_sci() {
    // Read PM1_STS (PM1a_EVT_BLK = PM1a_CNT_BLK - 4 typically, but varies by FADT)
    // Simple heuristic: check power-button status via ACPI PM1 event
    let port_a = PM1A_PORT.load(Ordering::Relaxed);
    if port_a == 0 { return; }
    let sts_port = port_a.wrapping_sub(4); // PM1a_STS is 4 bytes before PM1a_CNT
    unsafe {
        let mut p: Port<u16> = Port::new(sts_port);
        let sts: u16 = p.read();
        if sts & (1 << 8) != 0 {
            // Power button pressed
            POWER_BUTTON_PRESSED.store(true, Ordering::Relaxed);
            // Clear status by writing 1
            p.write(1u16 << 8);
        }
    }
}

/// Returns true if the power button was pressed since last check (clears flag).
pub fn power_button_pressed() -> bool {
    POWER_BUTTON_PRESSED.swap(false, Ordering::Relaxed)
}

// ── Phase 29 additions ────────────────────────────────────────────────────────

/// Prepare for hibernation (S4).  Currently falls back to S3 suspend
/// until a full hibernation image writer is implemented.
pub fn prepare_hibernate() {
    crate::klog!(INFO, "power: preparing hibernate (S4) — suspending to S3 for now");
    suspend_s3();
}

/// Set the CPU P-state/frequency scaling governor.
/// 0 = powersave, 1 = ondemand/balanced, 2 = performance.
pub fn set_cpu_governor(level: u8) {
    // Hint via ACPI _PPC or MSR IA32_PERF_CTL; stub for now.
    crate::klog!(DEBUG, "power: cpu_governor={}", level);
}
