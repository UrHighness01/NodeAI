//! Interrupt Coalescing Statistics — safe, non-blocking monitoring.
//!
//! THIS MODULE IS DISABLED FOR INTERRUPTS IN THE TIMER'S PRIORITY CLASS.
//! Keyboard (0x21), mouse (0x2C), and timer (0x20) all share APIC priority
//! class 2.  Deferring EOI for any of these suppresses ALL interrupts at
//! equal or lower priority — the timer stops firing, uptime freezes, and
//! the heartbeat dies.  This is not a software bug, it is xAPIC hardware
//! behaviour: the ISR (In-Service Register) masks all vectors ≤ its own
//! priority until EOI is written.
//!
//! Keyboard and mouse handlers in `mod.rs` use DIRECT `apic::eoi()` for
//! this reason.  This module exists to:
//!
//!   1. Provide a home for coalescing statistics counters.
//!   2. Document the APIC priority-class constraint for future developers.
//!   3. Be available for IPI coalescing on SMP systems, where IPI vectors
//!      sit at priority class 6+ (0xE0-0xFF) — above the timer — making
//!      deferral safe.
//!
//! Usage (future SMP IPI coalescing only):
//!   ```ignore
//!   // SAFE because IPI vector 0xFE > timer's 0x20 in priority.
//!   // Deferred EOI only blocks lower-priority interrupts (which is fine).
//!   if should_coalesce {
//!       PENDING_EOI.fetch_add(1, Ordering::Relaxed);
//!   } else {
//!       flush_pending();
//!   }
//!   ```

use core::sync::atomic::{AtomicU8, Ordering};

/// Number of coalesced EOIs not yet flushed (tracked for statistics only).
/// Actual EOI must ALWAYS be sent before deferral — see module-level docs.
static PENDING_EOI: AtomicU8 = AtomicU8::new(0);

/// Flush any pending coalesced EOI counter (for use in periodic timer tick).
/// Since actual EOIs are sent immediately by the interrupt handlers, this
/// simply resets the counter.  No APIC write is performed.
pub unsafe fn flush_pending() {
    PENDING_EOI.store(0, Ordering::Relaxed);
}

/// Record one coalesced interrupt for statistics.
/// Safe to call from any interrupt handler AFTER the real apic::eoi().
pub fn record_coalesced() {
    PENDING_EOI.fetch_add(1, Ordering::Relaxed);
}

/// Return the number of coalesced interrupts since last flush (debug/stat).
pub fn pending_count() -> u8 {
    PENDING_EOI.load(Ordering::Relaxed)
}
