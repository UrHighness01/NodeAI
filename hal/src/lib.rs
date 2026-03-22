//! HAL — Hardware Abstraction Layer
//!
//! Defines pure traits for hardware interactions so the kernel and AI subsystem
//! remain architecture-agnostic. The x86_64 module provides the concrete implementation.

#![no_std]

pub mod arch_x86_64;

// ── Core Traits ───────────────────────────────────────────────────────────────

/// CPU control: halt, NOP, barrier, feature query.
pub trait Cpu: Sized {
    /// Halt the current CPU until the next interrupt.
    fn halt();
    /// Memory fence — prevent instruction reordering across this point.
    fn memory_fence();
    /// Returns true if the CPU supports AVX2 (needed for AI SIMD paths).
    fn has_avx2() -> bool;
    /// Returns true if the CPU supports AVX-512 (optimal AI path).
    fn has_avx512f() -> bool;
    /// Returns true if the CPU has an Intel AMX (matrix math) unit.
    fn has_amx() -> bool;
}

/// High-resolution monotonic timer.
pub trait Timer {
    /// Returns current timestamp in nanoseconds since boot.
    fn now_ns() -> u64;
    /// Returns TSC frequency in Hz (calibrated at boot).
    fn tsc_freq_hz() -> u64;
}

/// Byte-oriented UART for early debug output.
pub trait Uart {
    fn write_byte(byte: u8);
    fn write_str(s: &str) {
        for b in s.bytes() {
            Self::write_byte(b);
        }
    }
}

/// Interrupt controller abstraction.
pub trait InterruptController {
    /// Acknowledge end-of-interrupt for the given IRQ vector.
    fn eoi(vector: u8);
    /// Mask (disable) a hardware IRQ line.
    fn mask(irq: u8);
    /// Unmask (enable) a hardware IRQ line.
    fn unmask(irq: u8);
}

/// CPU frequency / power management.
pub trait PowerManagement {
    /// Request a P-state (performance level 0 = max, higher = lower freq).
    fn set_pstate(level: u8);
    /// Enter a C-state hint (0 = active, 1 = halt, 2+ = deeper sleep).
    fn request_cstate(level: u8);
}
