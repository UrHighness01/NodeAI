//! USB HID class driver — keyboard/mouse input via USB.
//!
//! Implements:
//!   - Boot-protocol keyboard (8-byte reports)
//!   - Boot-protocol mouse (3+ byte reports)
//!
//! The driver is called from `usb/mod.rs` after a HID interface is enumerated.
//! Decoded key events are injected into the kernel event queue.

use alloc::vec::Vec;
use spin::Mutex;

// ── Modifier bitmask (byte 0 of keyboard report) ──────────────────────────────
const MOD_LCTRL:  u8 = 1 << 0;
const MOD_LSHIFT: u8 = 1 << 1;
const MOD_LALT:   u8 = 1 << 2;
const MOD_LGUI:   u8 = 1 << 3;
const MOD_RCTRL:  u8 = 1 << 4;
const MOD_RSHIFT: u8 = 1 << 5;
const MOD_RALT:   u8 = 1 << 6;
const MOD_RGUI:   u8 = 1 << 7;

// ── HID Usage → ASCII (simplistic boot-protocol map) ─────────────────────────
#[rustfmt::skip]
static USAGE_MAP: [u8; 84] = [
    0,   0,   0,   0,
    b'a',b'b',b'c',b'd',b'e',b'f',b'g',b'h',b'i',b'j',b'k',b'l',b'm',
    b'n',b'o',b'p',b'q',b'r',b's',b't',b'u',b'v',b'w',b'x',b'y',b'z',
    b'1',b'2',b'3',b'4',b'5',b'6',b'7',b'8',b'9',b'0',
    b'\n', 0x1B, 0x08, b'\t', b' ',
    b'-',b'=',b'[',b']',b'\\',0,b';',b'\'',b'`',b',',b'.',b'/',
    0,  // CapsLock
    // F1-F12
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,  // PrintScreen
    0,  // ScrollLock
    0,  // Pause
    0,  // Insert
    0,  // Home
    0,  // PageUp
    0x7F, // Delete
    0,  // End
    0,  // PageDown
    0,0,0,0, // Arrow keys (Right, Left, Down, Up)
    0,  // NumLock
];

#[rustfmt::skip]
static USAGE_MAP_SHIFT: [u8; 80] = [
    0,   0,   0,   0,
    b'A',b'B',b'C',b'D',b'E',b'F',b'G',b'H',b'I',b'J',b'K',b'L',b'M',
    b'N',b'O',b'P',b'Q',b'R',b'S',b'T',b'U',b'V',b'W',b'X',b'Y',b'Z',
    b'!',b'@',b'#',b'$',b'%',b'^',b'&',b'*',b'(',b')',
    b'\n', 0x1B, 0x08, b'\t', b' ',
    b'_',b'+',b'{',b'}',b'|',0,b':',b'"',b'~',b'<',b'>',b'?',
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0x7F,0,0,0,0,0,0,0,
];

// ── Key event ring (simple circular buffer)  ────────────────────────────────

const KEY_RING_SIZE: usize = 64;

struct KeyRing {
    buf:  [u8; KEY_RING_SIZE],
    head: usize,
    tail: usize,
}
impl KeyRing {
    const fn new() -> Self { Self { buf: [0; KEY_RING_SIZE], head: 0, tail: 0 } }
    fn push(&mut self, ch: u8) {
        let next = (self.tail + 1) % KEY_RING_SIZE;
        if next != self.head { self.buf[self.tail] = ch; self.tail = next; }
    }
    fn pop(&mut self) -> Option<u8> {
        if self.head == self.tail { return None; }
        let ch = self.buf[self.head];
        self.head = (self.head + 1) % KEY_RING_SIZE;
        Some(ch)
    }
}

static KEY_RING: Mutex<KeyRing> = Mutex::new(KeyRing::new());

// Track pressed keys to detect releases
static PREV_KEYS: Mutex<[u8; 6]> = Mutex::new([0u8; 6]);

// ── Mouse state ────────────────────────────────────────────────────────────────

pub struct MouseState {
    pub buttons: u8,
    pub dx: i16,
    pub dy: i16,
}
static MOUSE: Mutex<MouseState> = Mutex::new(MouseState { buttons: 0, dx: 0, dy: 0 });

// ── Public API ────────────────────────────────────────────────────────────────

/// Process a boot-protocol keyboard report (8 bytes).
pub fn process_keyboard_report(report: &[u8]) {
    if report.len() < 8 { return; }
    let mods = report[0];
    let shift = mods & (MOD_LSHIFT | MOD_RSHIFT) != 0;
    let keys = &report[2..8];

    let mut prev = PREV_KEYS.lock();
    for &usage in keys {
        if usage == 0 || usage == 1 { continue; }
        // Only fire if this usage code wasn't present last report
        if prev.iter().any(|&p| p == usage) { continue; }
        let map: &[u8] = if shift { &USAGE_MAP_SHIFT } else { &USAGE_MAP };
        if (usage as usize) < map.len() {
            let ch = map[usage as usize];
            if ch != 0 {
                KEY_RING.lock().push(ch);
                // Inject into desktop keyboard handler
                crate::desktop::terminal_input(ch);
            }
        }
    }
    prev.copy_from_slice(keys);
}

/// Process a boot-protocol mouse report (3+ bytes).
pub fn process_mouse_report(report: &[u8]) {
    if report.len() < 3 { return; }
    let mut m = MOUSE.lock();
    m.buttons = report[0] & 0x07;
    m.dx = report[1] as i8 as i16;
    m.dy = report[2] as i8 as i16;
    crate::desktop::mouse_event(m.dx, m.dy, m.buttons & 1 != 0, m.buttons & 2 != 0);
}

/// Pop one character from the keyboard ring. Safe to call from interrupt context.
pub fn read_key() -> Option<u8> { KEY_RING.lock().pop() }

/// Current mouse state snapshot.
pub fn mouse_state() -> (u8, i16, i16) {
    let m = MOUSE.lock();
    (m.buttons, m.dx, m.dy)
}
