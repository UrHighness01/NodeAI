//! Kernel logger — writes to COM1 serial port and VGA text buffer.
//! No allocator dependency; uses only static ring buffer.

use core::fmt::{self, Write};
use spin::Mutex;

// ── Log level ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Level {
    TRACE = 0,
    DEBUG = 1,
    INFO  = 2,
    WARN  = 3,
    ERROR = 4,
}

impl Level {
    fn as_str(self) -> &'static str {
        match self {
            Level::TRACE => "TRACE",
            Level::DEBUG => "DEBUG",
            Level::INFO  => " INFO",
            Level::WARN  => " WARN",
            Level::ERROR => "ERROR",
        }
    }
}

// ── Serial COM1 writer ─────────────────────────────────────────────────────────

const COM1: u16 = 0x3F8;

struct SerialWriter;

impl SerialWriter {
    fn write_byte(&self, byte: u8) {
        // Spin until transmit holding register is empty
        unsafe {
            while x86_64::instructions::port::PortReadOnly::<u8>::new(COM1 + 5).read() & 0x20 == 0 {}
            x86_64::instructions::port::Port::<u8>::new(COM1).write(byte);
        }
    }
}

impl fmt::Write for SerialWriter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.bytes() {
            self.write_byte(byte);
        }
        Ok(())
    }
}

static SERIAL: Mutex<SerialWriter> = Mutex::new(SerialWriter);

// ── Public API ─────────────────────────────────────────────────────────────────

const MIN_LEVEL: Level = Level::DEBUG;

/// Write a single byte directly to the serial console (used by sys_write).
pub fn write_byte(byte: u8) {
    SERIAL.lock().write_byte(byte);
}

pub fn init() {
    // Initialize COM1: 115200 baud, 8N1
    unsafe {
        use x86_64::instructions::port::Port;
        Port::<u8>::new(COM1 + 1).write(0x00); // Disable interrupts
        Port::<u8>::new(COM1 + 3).write(0x80); // Enable DLAB
        Port::<u8>::new(COM1 + 0).write(0x01); // Divisor low  (115200 baud)
        Port::<u8>::new(COM1 + 1).write(0x00); // Divisor high
        Port::<u8>::new(COM1 + 3).write(0x03); // 8 bits, no parity, 1 stop
        Port::<u8>::new(COM1 + 2).write(0xC7); // Enable FIFO, clear, 14-byte threshold
        Port::<u8>::new(COM1 + 4).write(0x0B); // IRQs enabled, RTS/DSR set
    }
}

pub fn log(level: Level, file: &str, line: u32, args: fmt::Arguments<'_>) {
    if level < MIN_LEVEL {
        return;
    }

    // Format into a stack buffer (no heap needed)
    let mut buf = [0u8; 256];
    let msg_len;
    {
        let mut cursor = Cursor { buf: &mut buf, pos: 0 };
        let _ = write!(cursor, "[{}] {}:{} \u{2014} {}\r\n", level.as_str(), file, line, args);
        msg_len = cursor.pos;
    }
    let msg = &buf[..msg_len];

    // Write to serial
    {
        let mut w = SERIAL.lock();
        for &b in msg {
            w.write_byte(b);
        }
    }

    // Also push to the kernel ring buffer
    crate::kring::KRING.lock().push(level as u8, msg);
}

// ── Stack-only cursor for formatting without alloc ────────────────────────────

struct Cursor<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl fmt::Write for Cursor<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let bytes = s.as_bytes();
        let n = bytes.len().min(self.buf.len() - self.pos);
        self.buf[self.pos..self.pos + n].copy_from_slice(&bytes[..n]);
        self.pos += n;
        Ok(())
    }
}
