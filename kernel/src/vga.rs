//! VGA text-mode driver — 80×25, 16 colours.
//!
//! Provides a `Writer` that can be used as the `core::fmt::Write` target,
//! enabling on-screen kernel output via the `klog!` macro.
//! Memory-mapped at the standard VGA buffer address 0xB8000.
//!
//! VGA text-mode console driver (physical 0xB8000, mapped via phys_offset).

use core::fmt;
use spin::Mutex;

// ── Colour ────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Colour {
    Black        = 0,
    Blue         = 1,
    Green        = 2,
    Cyan         = 3,
    Red          = 4,
    Magenta      = 5,
    Brown        = 6,
    LightGray    = 7,
    DarkGray     = 8,
    LightBlue    = 9,
    LightGreen   = 10,
    LightCyan    = 11,
    LightRed     = 12,
    Pink         = 13,
    Yellow       = 14,
    White        = 15,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
struct ColourCode(u8);

impl ColourCode {
    const fn new(fg: Colour, bg: Colour) -> Self {
        ColourCode((bg as u8) << 4 | (fg as u8))
    }
}

// ── Screen cell ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
struct ScreenChar {
    ascii: u8,
    colour: ColourCode,
}

const BUFFER_HEIGHT: usize = 25;
const BUFFER_WIDTH: usize = 80;

/// Raw VGA text buffer mapped at 0xB8000.
/// `repr(transparent)` ensure no extra padding between cells.
#[repr(transparent)]
struct Buffer {
    chars: [[ScreenChar; BUFFER_WIDTH]; BUFFER_HEIGHT],
}

// ── Writer ────────────────────────────────────────────────────────────────────

/// VGA cursor and write state.
pub struct Writer {
    col: usize,
    colour: ColourCode,
    /// Safety: the pointer is valid as long as VGA MMIO is mapped (forever in our kernel).
    buffer: *mut Buffer,
}

// Safety: VGA buffer at 0xB8000 is a static device register — safe to share across CPUs
// once protected by the Mutex below.
unsafe impl Send for Writer {}

impl Writer {
    pub fn write_byte(&mut self, byte: u8) {
        match byte {
            b'\n' => self.newline(),
            b'\r' => { self.col = 0; }
            byte => {
                if self.col >= BUFFER_WIDTH {
                    self.newline();
                }
                let row = BUFFER_HEIGHT - 1;
                let col = self.col;
                self.write_cell(row, col, ScreenChar { ascii: byte, colour: self.colour });
                self.col += 1;
            }
        }
    }

    pub fn write_string(&mut self, s: &str) {
        for byte in s.bytes() {
            match byte {
                // Printable ASCII range + newline/return
                0x20..=0x7e | b'\n' | b'\r' => self.write_byte(byte),
                // Non-printable: write '■' placeholder
                _ => self.write_byte(0xFE),
            }
        }
    }

    fn newline(&mut self) {
        // Scroll all rows up by one
        for row in 1..BUFFER_HEIGHT {
            for col in 0..BUFFER_WIDTH {
                let ch = self.read_cell(row, col);
                self.write_cell(row - 1, col, ch);
            }
        }
        self.clear_row(BUFFER_HEIGHT - 1);
        self.col = 0;
    }

    fn clear_row(&mut self, row: usize) {
        let blank = ScreenChar { ascii: b' ', colour: self.colour };
        for col in 0..BUFFER_WIDTH {
            self.write_cell(row, col, blank);
        }
    }

    #[inline]
    fn write_cell(&mut self, row: usize, col: usize, sc: ScreenChar) {
        // Safety: row/col are bounds-checked by callers.
        unsafe { (*self.buffer).chars[row][col] = sc; }
    }

    #[inline]
    fn read_cell(&self, row: usize, col: usize) -> ScreenChar {
        unsafe { (*self.buffer).chars[row][col] }
    }

    /// Change the active foreground/background colour.
    pub fn set_colour(&mut self, fg: Colour, bg: Colour) {
        self.colour = ColourCode::new(fg, bg);
    }

    /// Clear the entire screen.
    pub fn clear(&mut self) {
        for row in 0..BUFFER_HEIGHT {
            self.clear_row(row);
        }
    }
}

impl fmt::Write for Writer {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.write_string(s);
        Ok(())
    }
}

// ── Global Writer ─────────────────────────────────────────────────────────────

/// VGA_BUFFER_ADDR — the physical address of the VGA text buffer.
/// Accessible via phys_offset + 0xB8000 after memory::init sets up the higher-half mapping.
const VGA_BUFFER_ADDR: usize = 0xB8000;

pub static WRITER: Mutex<Writer> = Mutex::new(Writer {
    col: 0,
    colour: ColourCode::new(Colour::LightGreen, Colour::Black),
    buffer: VGA_BUFFER_ADDR as *mut Buffer,
});

// ── Public API ────────────────────────────────────────────────────────────────

/// Print a string followed by a newline to the VGA buffer.
#[macro_export]
macro_rules! vga_println {
    () => ($crate::vga::_print(format_args!("\n")));
    ($($arg:tt)*) => ($crate::vga::_print(format_args!("{}\n", format_args!($($arg)*))));
}

/// Print a string to the VGA buffer (no newline).
#[macro_export]
macro_rules! vga_print {
    ($($arg:tt)*) => ($crate::vga::_print(format_args!($($arg)*)));
}

#[doc(hidden)]
pub fn _print(args: fmt::Arguments) {
    use core::fmt::Write;
    WRITER.lock().write_fmt(args).unwrap();
}

/// Initialise VGA: remaps the buffer pointer to the physical-memory-offset mapping
/// provided by the bootloader, then clears the screen.
pub fn init(phys_mem_offset: u64) {
    let vga_virt = phys_mem_offset + VGA_BUFFER_ADDR as u64;
    {
        let mut w = WRITER.lock();
        w.buffer = vga_virt as *mut Buffer;
        w.clear();
    }
    vga_println!("NodeAI Kernel — VGA initialised");
}
