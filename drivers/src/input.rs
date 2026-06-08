//! PS/2 keyboard driver — Phase 6.
//!
//! Handles PS/2 controller initialisation, scancode set 1 decoding,
//! and a small keyboard event FIFO usable from IRQ context.

use x86_64::instructions::port::Port;
use spin::Mutex;
use core::sync::atomic::{AtomicBool, Ordering};
use alloc::collections::VecDeque;

// ── Port addresses ────────────────────────────────────────────────────────────
const DATA_PORT:   u16 = 0x60; // Read: data, Write: command
const STATUS_PORT: u16 = 0x64; // Read: status
const CMD_PORT:    u16 = 0x64; // Write: controller command

// ── Status register bits ──────────────────────────────────────────────────────
const STATUS_OUTPUT_FULL: u8  = 0x01;
const STATUS_INPUT_FULL:  u8  = 0x02;
/// Bit 5 of status: output buffer contains aux (mouse) data.
const STATUS_AUX:         u8  = 0x20;

// ── Event queue (ring buffer, 64 entries) ─────────────────────────────────────
static KEY_QUEUE: Mutex<VecDeque<KeyEvent>> = Mutex::new(VecDeque::new());
const KEY_QUEUE_CAP: usize = 64;

/// Extended key codes for non-ASCII keys (arrow keys, Home, End, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SpecialKey {
    Up     = 1,
    Down   = 2,
    Left   = 3,
    Right  = 4,
    Home   = 5,
    End    = 6,
    Delete = 7,
    PageUp = 8,
    PageDown = 9,
    Insert = 10,
    F1  = 11, F2  = 12, F3  = 13, F4  = 14, F5  = 15,
    F6  = 16, F7  = 17, F8  = 18, F9  = 19, F10 = 20,
    F11 = 21, F12 = 22,
}

/// A decoded mouse movement / button event.
#[derive(Debug, Clone, Copy)]
pub struct MouseEvent {
    /// Horizontal delta: positive = right.
    pub dx:    i16,
    /// Vertical delta: positive = down (PS/2 sign already inverted).
    pub dy:    i16,
    pub left:  bool,
    pub right: bool,
}

/// A decoded keyboard event.
#[derive(Debug, Clone, Copy)]
pub struct KeyEvent {
    pub scancode: u8,
    pub pressed:  bool,
    pub ascii:    Option<char>,
    pub special:  Option<SpecialKey>,
}

// Track whether we're in an 0xE0 extended sequence
static EXTENDED: spin::Mutex<bool> = spin::Mutex::new(false);
// Track Shift key state (left or right shift held)
static SHIFT: AtomicBool = AtomicBool::new(false);

// ── PS/2 Mouse state ─────────────────────────────────────────────────────────
static MOUSE_QUEUE: Mutex<VecDeque<MouseEvent>> = Mutex::new(VecDeque::new());
const MOUSE_QUEUE_CAP: usize = 32;
static mut MOUSE_BUF: [u8; 3] = [0u8; 3];
static mut MOUSE_IDX: usize   = 0;

/// Initialise the PS/2 controller, keyboard, and mouse.
pub fn init() {
    // Pre-allocate queues so interrupt handlers never trigger the allocator (which deadlocks if HEAP.lock is held).
    KEY_QUEUE.lock().reserve_exact(KEY_QUEUE_CAP);
    MOUSE_QUEUE.lock().reserve_exact(MOUSE_QUEUE_CAP);

    unsafe {
        // Flush output buffer
        while Port::<u8>::new(STATUS_PORT).read() & STATUS_OUTPUT_FULL != 0 {
            Port::<u8>::new(DATA_PORT).read();
        }

        // Disable both PS/2 devices during init
        ps2_cmd(0xAD); // disable keyboard
        ps2_cmd(0xA7); // disable mouse

        // Read controller config, enable both keyboard and aux interrupts
        ps2_cmd(0x20); // read configuration byte
        wait_output();
        let mut config = Port::<u8>::new(DATA_PORT).read();
        config |= 0x03;  // bit0 = keyboard IRQ enable, bit1 = mouse IRQ enable
        config &= !0x30; // bit4 = keyboard disable, bit5 = mouse disable  (clear both)
        ps2_cmd(0x60);   // write configuration byte
        wait_input();
        Port::<u8>::new(DATA_PORT).write(config);

        // Controller self-test
        ps2_cmd(0xAA);
        wait_output();
        let result = Port::<u8>::new(DATA_PORT).read();
        if result != 0x55 {
            crate::serial::Serial::write_str("PS/2: controller self-test failed\n");
        }

        // ── Keyboard ────────────────────────────────────────────────────────────
        ps2_cmd(0xAE); // enable keyboard
        Port::<u8>::new(DATA_PORT).write(0xFF); // reset
        wait_output();
        let _ = Port::<u8>::new(DATA_PORT).read(); // ACK

        // ── Mouse ────────────────────────────────────────────────────────────────
        ps2_cmd(0xA8); // enable auxiliary (mouse) port

        // Set mouse defaults
        send_mouse(0xF6);
        wait_output();
        let _ = Port::<u8>::new(DATA_PORT).read(); // ACK 0xFA

        // Enable mouse data reporting
        send_mouse(0xF4);
        wait_output();
        let _ = Port::<u8>::new(DATA_PORT).read(); // ACK 0xFA
    }
}

/// Send a command byte to the auxiliary (mouse) device via the PS/2 controller.
unsafe fn send_mouse(byte: u8) {
    ps2_cmd(0xD4); // tell controller: next write goes to aux device
    wait_input();
    Port::<u8>::new(DATA_PORT).write(byte);
}

unsafe fn ps2_cmd(cmd: u8) {
    wait_input();
    Port::<u8>::new(CMD_PORT).write(cmd);
}

unsafe fn wait_output() {
    let mut tries = 100_000u32;
    while Port::<u8>::new(STATUS_PORT).read() & STATUS_OUTPUT_FULL == 0 {
        core::hint::spin_loop();
        tries -= 1;
        if tries == 0 { break; }
    }
}

unsafe fn wait_input() {
    let mut tries = 100_000u32;
    while Port::<u8>::new(STATUS_PORT).read() & STATUS_INPUT_FULL != 0 {
        core::hint::spin_loop();
        tries -= 1;
        if tries == 0 { break; }
    }
}

/// Read a scancode from the PS/2 data port (polling — keyboard only).
/// Returns `Some(scancode)` if keyboard data is ready (bit 5 must be 0 — not aux).
/// Process a single byte of mouse data.
unsafe fn process_mouse_byte(byte: u8) {
    // Byte 0 sync guard: bit 3 must always be set in a valid first byte.
    if MOUSE_IDX == 0 && byte & 0x08 == 0 { return; }

    MOUSE_BUF[MOUSE_IDX] = byte;
    MOUSE_IDX += 1;
    if MOUSE_IDX == 3 {
        MOUSE_IDX = 0;
        let b0 = MOUSE_BUF[0];
        let b1 = MOUSE_BUF[1];
        let b2 = MOUSE_BUF[2];
        // Discard packet if overflow bits are set
        if b0 & 0xC0 != 0 { return; }
        // 9-bit signed X: sign bit = bit4 of b0
        let dx = (b1 as i16) + if b0 & 0x10 != 0 { -256i16 } else { 0 };
        // 9-bit signed Y: sign bit = bit5 of b0; negate (PS/2 Y positive = up)
        let dy_ps2 = (b2 as i16) + if b0 & 0x20 != 0 { -256i16 } else { 0 };
        let dy = -dy_ps2;
        let ev = MouseEvent {
            dx,
            dy,
            left:  b0 & 0x01 != 0,
            right: b0 & 0x02 != 0,
        };
        let mut q = MOUSE_QUEUE.lock();
        if q.len() < MOUSE_QUEUE_CAP {
            q.push_back(ev);
        }
    }
}

/// Read a scancode from the PS/2 data port (polling — keyboard only).
pub fn read_scancode() -> Option<u8> {
    let status: u8 = unsafe { Port::<u8>::new(STATUS_PORT).read() };
    if status & STATUS_OUTPUT_FULL != 0 {
        let sc = unsafe { Port::<u8>::new(DATA_PORT).read() };
        if status & STATUS_AUX == 0 {
            return Some(sc);
        }
    }
    None
}

/// Pop the oldest mouse event from the queue.
pub fn poll_mouse_event() -> Option<MouseEvent> {
    x86_64::instructions::interrupts::without_interrupts(|| {
        MOUSE_QUEUE.lock().pop_front()
    })
}

/// IRQ12 handler — called from IDT when PS/2 mouse interrupt fires.
pub fn mouse_irq_handler() {
    let status: u8 = unsafe { Port::<u8>::new(STATUS_PORT).read() };
    if status & STATUS_OUTPUT_FULL == 0 { return; }
    let byte = unsafe { Port::<u8>::new(DATA_PORT).read() };
    
    if status & STATUS_AUX != 0 {
        unsafe { process_mouse_byte(byte); }
    } else {
        // Was keyboard data! Route it to keyboard handler.
        process_keyboard_byte(byte);
    }
}

/// IRQ1 handler — called from IDT when keyboard interrupt fires.
pub fn keyboard_irq_handler() {
    let status: u8 = unsafe { Port::<u8>::new(STATUS_PORT).read() };
    if status & STATUS_OUTPUT_FULL == 0 { return; }
    let byte = unsafe { Port::<u8>::new(DATA_PORT).read() };
    
    if status & STATUS_AUX != 0 {
        // Was mouse data! Route it to mouse handler.
        unsafe { process_mouse_byte(byte); }
    } else {
        process_keyboard_byte(byte);
    }
}

/// Process a single byte of keyboard data.
fn process_keyboard_byte(sc: u8) {
    // Handle 0xE0 prefix (extended scancodes)
    if sc == 0xE0 {
        *EXTENDED.lock() = true;
        return;
    }
    let is_ext = {
        let mut ext = EXTENDED.lock();
        let was = *ext;
        *ext = false;
        was
    };
    let event = if is_ext {
        decode_extended(sc)
    } else {
        decode_scancode_set1(sc)
    };
    let mut q = KEY_QUEUE.lock();
    if q.len() < KEY_QUEUE_CAP {
        q.push_back(event);
    }
}

/// Pop the oldest keyboard event from the queue.
pub fn poll_event() -> Option<KeyEvent> {
    x86_64::instructions::interrupts::without_interrupts(|| {
        KEY_QUEUE.lock().pop_front()
    })
}

// ── Scancode Set 1 decoder ────────────────────────────────────────────────────

fn decode_scancode_set1(sc: u8) -> KeyEvent {
    let released = sc & 0x80 != 0;
    let base = sc & 0x7F;

    // Update shift state: 0x2A = Left Shift, 0x36 = Right Shift
    if base == 0x2A || base == 0x36 {
        SHIFT.store(!released, Ordering::Relaxed);
    }

    let special = match base {
        0x3B => Some(SpecialKey::F1),
        0x3C => Some(SpecialKey::F2),
        0x3D => Some(SpecialKey::F3),
        0x3E => Some(SpecialKey::F4),
        0x3F => Some(SpecialKey::F5),
        0x40 => Some(SpecialKey::F6),
        0x41 => Some(SpecialKey::F7),
        0x42 => Some(SpecialKey::F8),
        0x43 => Some(SpecialKey::F9),
        0x44 => Some(SpecialKey::F10),
        0x57 => Some(SpecialKey::F11),
        0x58 => Some(SpecialKey::F12),
        _ => None,
    };

    let shift = SHIFT.load(Ordering::Relaxed);
    let ascii = if shift {
        SCANCODE_TABLE_SHIFT.get(base as usize).copied().flatten()
    } else {
        SCANCODE_TABLE.get(base as usize).copied().flatten()
    };

    KeyEvent {
        scancode: sc,
        pressed:  !released,
        ascii,
        special,
    }
}

/// Decode an extended (0xE0-prefixed) scancode.
fn decode_extended(sc: u8) -> KeyEvent {
    let released = sc & 0x80 != 0;
    let base = sc & 0x7F;
    let special = match base {
        0x48 => Some(SpecialKey::Up),
        0x50 => Some(SpecialKey::Down),
        0x4B => Some(SpecialKey::Left),
        0x4D => Some(SpecialKey::Right),
        0x47 => Some(SpecialKey::Home),
        0x4F => Some(SpecialKey::End),
        0x53 => Some(SpecialKey::Delete),
        0x49 => Some(SpecialKey::PageUp),
        0x51 => Some(SpecialKey::PageDown),
        0x52 => Some(SpecialKey::Insert),
        _ => None,
    };
    KeyEvent {
        scancode: sc,
        pressed: !released,
        ascii: None,
        special,
    }
}

// Basic US QWERTY layout, scancode set 1 (indices 0x00–0x58), unshifted.
const SCANCODE_TABLE: [Option<char>; 87] = [
    None,       // 0x00
    None,       // 0x01 Esc
    Some('1'),  Some('2'),  Some('3'),  Some('4'),  Some('5'),
    Some('6'),  Some('7'),  Some('8'),  Some('9'),  Some('0'),
    Some('-'),  Some('='),
    None,       // 0x0E Backspace
    Some('\t'), // 0x0F Tab
    Some('q'),  Some('w'),  Some('e'),  Some('r'),  Some('t'),
    Some('y'),  Some('u'),  Some('i'),  Some('o'),  Some('p'),
    Some('['),  Some(']'),
    Some('\n'), // 0x1C Enter
    None,       // 0x1D Left Ctrl
    Some('a'),  Some('s'),  Some('d'),  Some('f'),  Some('g'),
    Some('h'),  Some('j'),  Some('k'),  Some('l'),
    Some(';'),  Some('\''),
    Some('`'),
    None,       // 0x2A Left Shift
    Some('\\'),
    Some('z'),  Some('x'),  Some('c'),  Some('v'),  Some('b'),
    Some('n'),  Some('m'),
    Some(','),  Some('.'),  Some('/'),
    None,       // 0x36 Right Shift
    Some('*'),  // 0x37 Num *
    None,       // 0x38 Left Alt
    Some(' '),  // 0x39 Space
    None,       // 0x3A Caps Lock
    // F1–F10: indices 0x3B–0x44
    None, None, None, None, None, None, None, None, None, None,
    None,       // 0x45 Num Lock
    None,       // 0x46 Scroll Lock
    None, None, None, None, None, None, None, None, None, // Numpad 7-9, -, 4-6, +, 1-3
    None,       // 0x52 Numpad 0
    None,       // 0x53 Numpad .
    // 0x54-0x58
    None, None, None, None, None,
];

// Shifted US QWERTY layout (Shift held), scancode set 1.
const SCANCODE_TABLE_SHIFT: [Option<char>; 87] = [
    None,       // 0x00
    None,       // 0x01 Esc
    Some('!'),  Some('@'),  Some('#'),  Some('$'),  Some('%'),
    Some('^'),  Some('&'),  Some('*'),  Some('('),  Some(')'),
    Some('_'),  Some('+'),
    None,       // 0x0E Backspace
    Some('\t'), // 0x0F Tab (unchanged)
    Some('Q'),  Some('W'),  Some('E'),  Some('R'),  Some('T'),
    Some('Y'),  Some('U'),  Some('I'),  Some('O'),  Some('P'),
    Some('{'),  Some('}'),
    Some('\n'), // 0x1C Enter
    None,       // 0x1D Left Ctrl
    Some('A'),  Some('S'),  Some('D'),  Some('F'),  Some('G'),
    Some('H'),  Some('J'),  Some('K'),  Some('L'),
    Some(':'),  Some('"'),
    Some('~'),
    None,       // 0x2A Left Shift
    Some('|'),
    Some('Z'),  Some('X'),  Some('C'),  Some('V'),  Some('B'),
    Some('N'),  Some('M'),
    Some('<'),  Some('>'),  Some('?'),
    None,       // 0x36 Right Shift
    Some('*'),  // 0x37 Num *
    None,       // 0x38 Left Alt
    Some(' '),  // 0x39 Space
    None,       // 0x3A Caps Lock
    // F1–F10: indices 0x3B–0x44
    None, None, None, None, None, None, None, None, None, None,
    None,       // 0x45 Num Lock
    None,       // 0x46 Scroll Lock
    None, None, None, None, None, None, None, None, None, // Numpad 7-9, -, 4-6, +, 1-3
    None,       // 0x52 Numpad 0
    None,       // 0x53 Numpad .
    // 0x54-0x58
    None, None, None, None, None,
];

