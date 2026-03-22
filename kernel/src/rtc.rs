//! CMOS Real-Time Clock (RTC) driver.
//!
//! Reads the x86 CMOS RTC via I/O ports 0x70/0x71.
//! Provides: `init()`, `current_time()`, `boot_epoch()`.

use core::sync::atomic::{AtomicU64, Ordering};
use x86_64::instructions::port::Port;

// CMOS I/O ports
const CMOS_ADDR: u16 = 0x70;
const CMOS_DATA: u16 = 0x71;

// CMOS register indices
const RTC_SECONDS: u8 = 0x00;
const RTC_MINUTES: u8 = 0x02;
const RTC_HOURS:   u8 = 0x04;
const RTC_DAY:     u8 = 0x07;
const RTC_MONTH:   u8 = 0x08;
const RTC_YEAR:    u8 = 0x09;
const RTC_CENTURY: u8 = 0x32;
const RTC_STAT_A:  u8 = 0x0A;
const RTC_STAT_B:  u8 = 0x0B;

/// A snapshot of the wall-clock time.
#[derive(Debug, Clone, Copy, Default)]
pub struct RtcTime {
    pub second: u8,
    pub minute: u8,
    pub hour:   u8,
    pub day:    u8,
    pub month:  u8,
    pub year:   u16,  // fully expanded (e.g. 2024)
}

impl RtcTime {
    /// Seconds since the Unix epoch 1970-01-01 00:00:00 (approximate — ignores leap years).
    pub fn unix_approx(&self) -> u64 {
        let y = self.year as u64;
        let days = (y - 1970) * 365
            + (y - 1969) / 4       // leap years (simplified)
            + self.month as u64 * 30
            + self.day as u64;
        days * 86400
            + self.hour   as u64 * 3600
            + self.minute as u64 * 60
            + self.second as u64
    }
}

// ── I/O helpers ───────────────────────────────────────────────────────────────

unsafe fn cmos_read(reg: u8) -> u8 {
    Port::<u8>::new(CMOS_ADDR).write(reg & 0x7F); // keep NMI bit clear
    Port::<u8>::new(CMOS_DATA).read()
}

fn bcd_to_bin(v: u8) -> u8 { (v & 0x0F) + (v >> 4) * 10 }

fn update_in_progress() -> bool {
    unsafe { cmos_read(RTC_STAT_A) & 0x80 != 0 }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Read the current wall-clock time from CMOS.
pub fn current_time() -> RtcTime {
    // Spin until no update-in-progress
    while update_in_progress() { core::hint::spin_loop(); }

    unsafe {
        let stat_b = cmos_read(RTC_STAT_B);
        let bcd    = stat_b & 0x04 == 0;
        let h24    = stat_b & 0x02 != 0;

        let mut sec  = cmos_read(RTC_SECONDS);
        let mut min  = cmos_read(RTC_MINUTES);
        let mut hour = cmos_read(RTC_HOURS);
        let mut day  = cmos_read(RTC_DAY);
        let mut mon  = cmos_read(RTC_MONTH);
        let mut yr   = cmos_read(RTC_YEAR);
        let cent     = cmos_read(RTC_CENTURY);

        if bcd {
            sec  = bcd_to_bin(sec);
            min  = bcd_to_bin(min);
            hour = bcd_to_bin(hour & 0x7F);
            day  = bcd_to_bin(day);
            mon  = bcd_to_bin(mon);
            yr   = bcd_to_bin(yr);
        }

        // Convert 12-hour to 24-hour
        if !h24 && hour & 0x80 != 0 {
            hour = ((hour & 0x7F) + 12) % 24;
        }

        let century = if cent > 0 { bcd_to_bin(cent) as u16 } else { 20 };
        let full_year = century * 100 + yr as u16;

        RtcTime { second: sec, minute: min, hour, day, month: mon, year: full_year }
    }
}

/// Unix timestamp captured at kernel boot (set during `init()`).
static BOOT_UNIX: AtomicU64 = AtomicU64::new(0);

/// Initialise the RTC driver: read & log the current time, store boot timestamp.
pub fn init() {
    let t = current_time();
    crate::klog!(INFO,
        "RTC: {:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC",
        t.year, t.month, t.day, t.hour, t.minute, t.second);
    BOOT_UNIX.store(t.unix_approx(), Ordering::Relaxed);
}

/// Return the Unix timestamp at the time `init()` was called.
pub fn boot_epoch() -> u64 { BOOT_UNIX.load(Ordering::Relaxed) }

/// Current wall-clock seconds (approximate — adds uptime_ms to boot epoch).
pub fn wall_clock_secs() -> u64 {
    boot_epoch() + crate::scheduler::uptime_ms() / 1000
}
