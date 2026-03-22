//! Serial UART (COM1) driver — used for early debug output and kernel log.
//! Also used by QEMU/VirtualBox to capture kernel output from the host.

use x86_64::instructions::port::Port;

const COM1: u16 = 0x3F8;

pub struct Serial;

impl Serial {
    /// Send a single byte via COM1 (blocking until transmit register empty).
    pub fn write_byte(byte: u8) {
        unsafe {
            // Wait until Transmit Holding Register Empty (THRE) bit is set
            while Port::<u8>::new(COM1 + 5).read() & 0x20 == 0 {}
            Port::<u8>::new(COM1).write(byte);
        }
    }

    pub fn write_str(s: &str) {
        for b in s.bytes() {
            // Translate \n to \r\n for terminal emulators
            if b == b'\n' { Self::write_byte(b'\r'); }
            Self::write_byte(b);
        }
    }
}
