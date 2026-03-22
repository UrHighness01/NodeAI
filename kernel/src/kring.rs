//! Kernel ring-buffer log — stores last N log entries in a fixed static buffer.
//! Readable from `/proc/kmsg` equivalent in a later phase.
//! All serial output from `logger.rs` also flows here.

use spin::Mutex;

pub const RING_SIZE: usize = 256;
pub const ENTRY_LEN: usize = 128;

#[derive(Clone, Copy)]
pub struct LogEntry {
    pub level: u8,     // 0=TRACE..4=ERROR
    pub len: usize,
    pub data: [u8; ENTRY_LEN],
}

impl LogEntry {
    const EMPTY: Self = Self { level: 0, len: 0, data: [0u8; ENTRY_LEN] };

    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.data[..self.len]).unwrap_or("<utf8-err>")
    }
}

pub struct KernelRing {
    buf: [LogEntry; RING_SIZE],
    head: usize, // next write position
    count: usize,
}

impl KernelRing {
    pub const fn new() -> Self {
        Self {
            buf: [LogEntry::EMPTY; RING_SIZE],
            head: 0,
            count: 0,
        }
    }

    /// Push a formatted log entry into the ring.
    pub fn push(&mut self, level: u8, msg: &[u8]) {
        let entry = &mut self.buf[self.head];
        entry.level = level;
        let n = msg.len().min(ENTRY_LEN);
        entry.data[..n].copy_from_slice(&msg[..n]);
        entry.len = n;

        self.head = (self.head + 1) % RING_SIZE;
        if self.count < RING_SIZE {
            self.count += 1;
        }
    }

    /// Iterate over entries oldest-first.
    pub fn iter(&self) -> impl Iterator<Item = &LogEntry> {
        let start = if self.count == RING_SIZE {
            self.head // oldest entry when ring is full
        } else {
            0
        };
        let count = self.count;
        (0..count).map(move |i| {
            let idx = (start + i) % RING_SIZE;
            &self.buf[idx]
        })
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

pub static KRING: Mutex<KernelRing> = Mutex::new(KernelRing::new());
