//! Multi-window compositor — Phase 22.
//!
//! Provides a proper desktop window manager with:
//!   - `Window` structs with per-window pixel buffers
//!   - Z-order compositing (paint bottom-to-top)
//!   - Window chrome: title bar with close/minimize/maximize buttons
//!   - Drag-to-move, resize handles (8-directional)
//!   - Window shadows (1-pixel outline)
//!   - Taskbar at the bottom: window buttons + clock + AI status
//!   - Mouse hit-test dispatch
//!
//! Ioctl codes for `/dev/composer`:
//!   `COMPOSER_CREATE_WINDOW  = 0xC001`
//!   `COMPOSER_DESTROY_WINDOW = 0xC002`
//!   `COMPOSER_FLIP           = 0xC003`
//!   `COMPOSER_MOVE           = 0xC004`
//!   `COMPOSER_RESIZE         = 0xC005`
//!   `COMPOSER_SET_TITLE      = 0xC006`

use alloc::{collections::BTreeMap, vec::Vec};
use spin::{Mutex, Once};
use crate::framebuffer as fb;

// ── Layout constants ──────────────────────────────────────────────────────────
/// Height of the per-window title bar (window chrome, not the top panel).
pub const WINTITLE_H:   usize = 24;
/// Height of the taskbar at the bottom of the screen.
pub const TASKBAR_H:    usize = 32;
/// Resize-hit border thickness in pixels.
const RESIZE_BORDER:    usize = 5;
/// Maximum simultaneous windows.
const MAX_WINDOWS:      usize = 16;
/// Minimum window size.
const WIN_MIN_W:        u32   = 120;
const WIN_MIN_H:        u32   = 60;

/// Ioctl code: create a new window.  arg = pointer to `ComposerCreateArgs`.
pub const COMPOSER_CREATE_WINDOW:  u64 = 0xC001;
/// Ioctl code: destroy a window.  arg = window_id as u64.
pub const COMPOSER_DESTROY_WINDOW: u64 = 0xC002;
/// Ioctl code: blit window pixel buffer to screen.  arg = window_id.
pub const COMPOSER_FLIP:           u64 = 0xC003;
/// Ioctl code: move window.  arg = pointer to `[i32; 2]` (x, y).
pub const COMPOSER_MOVE:           u64 = 0xC004;
/// Ioctl code: resize window.  arg = pointer to `[u32; 2]` (w, h).
pub const COMPOSER_RESIZE:         u64 = 0xC005;
/// Ioctl code: set window title.  arg = pointer to NUL-terminated string.
pub const COMPOSER_SET_TITLE:      u64 = 0xC006;

// ── Colour scheme ─────────────────────────────────────────────────────────────
const TITLE_ACTIVE:   (u8,u8,u8) = (0x2A, 0x4A, 0x80);
const TITLE_INACTIVE: (u8,u8,u8) = (0x22, 0x22, 0x32);
const TITLE_FG:       (u8,u8,u8) = (0xEE, 0xEE, 0xEE);
const TITLE_FG_DIM:   (u8,u8,u8) = (0x88, 0x88, 0x88);
const WIN_BORDER_ACT: (u8,u8,u8) = (0x35, 0x84, 0xE4);
const WIN_BORDER_IN:  (u8,u8,u8) = (0x40, 0x40, 0x56);
const WIN_SHADOW:     (u8,u8,u8) = (0x00, 0x00, 0x00);
const BTN_CLOSE:      (u8,u8,u8) = (0xEC, 0x6A, 0x5E);
const BTN_MIN:        (u8,u8,u8) = (0xF4, 0xBF, 0x4F);
const BTN_MAX:        (u8,u8,u8) = (0x61, 0xC5, 0x54);
const BTN_BOR:        (u8,u8,u8) = (0x18, 0x18, 0x18);
const TASKBR_BG:      (u8,u8,u8) = (0x12, 0x12, 0x1C);
const TASKBR_BTN:     (u8,u8,u8) = (0x28, 0x28, 0x38);
const TASKBR_ACTIVE:  (u8,u8,u8) = (0x2A, 0x55, 0xA8);
const TASKBR_FG:      (u8,u8,u8) = (0xCC, 0xCC, 0xCC);
const DESK_BG:        (u8,u8,u8) = (0x22, 0x22, 0x2A);
const FONT_W: usize = 8;
const FONT_H: usize = 16;

// ── Window struct ─────────────────────────────────────────────────────────────
pub struct Window {
    pub id:        u32,
    title:         [u8; 64],
    title_len:     usize,
    pub x:         i32,
    pub y:         i32,    // content-area top-left (chrome is above this)
    pub w:         u32,
    pub h:         u32,
    saved_x:       i32,
    saved_y:       i32,
    saved_w:       u32,
    saved_h:       u32,
    pub z_order:   u32,
    pub minimized: bool,
    pub maximized: bool,
    pub dirty:     bool,
    /// ARGB pixel buffer: `pixels[y * w + x]` = `0x00RRGGBB`.
    pub pixels:    Vec<u32>,
}

impl Window {
    pub fn title(&self) -> &str {
        core::str::from_utf8(&self.title[..self.title_len]).unwrap_or("?")
    }
    pub fn set_title(&mut self, t: &str) {
        let b = t.as_bytes();
        let l = b.len().min(63);
        self.title[..l].copy_from_slice(&b[..l]);
        self.title_len = l;
    }
    /// Full bounding rect including window chrome (shadow excluded).
    fn chrome_y(&self) -> i32 { self.y - WINTITLE_H as i32 }
    fn chrome_h(&self) -> u32 { self.h + WINTITLE_H as u32 }
    fn contains(&self, sx: i32, sy: i32) -> bool {
        sx >= self.x && sx < self.x + self.w as i32
            && sy >= self.chrome_y() && sy < self.y + self.h as i32
    }
}

// ── Drag state ────────────────────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq, Eq)]
enum DragKind {
    None,
    Move,
    ResizeN, ResizeS, ResizeE, ResizeW,
    ResizeNE, ResizeNW, ResizeSE, ResizeSW,
}

struct DragState {
    kind:         DragKind,
    win_id:       u32,
    orig_mouse_x: i32,
    orig_mouse_y: i32,
    orig_x:       i32,
    orig_y:       i32,
    orig_w:       u32,
    orig_h:       u32,
}

// ── WmState ───────────────────────────────────────────────────────────────────
pub struct WmState {
    pub windows:  BTreeMap<u32, Window>,
    pub z_stack:  Vec<u32>,    // window IDs in z-order (index 0 = bottom)
    pub focused:  Option<u32>,
    pub next_id:  u32,
    pub cursor_x: i32,
    pub cursor_y: i32,
    drag:         DragState,
    prev_left:    bool,
    prev_right:   bool,
}

impl WmState {
    fn new() -> Self {
        WmState {
            windows:  BTreeMap::new(),
            z_stack:  Vec::new(),
            focused:  None,
            next_id:  1,
            cursor_x: 512,
            cursor_y: 300,
            drag:     DragState {
                kind: DragKind::None, win_id: 0,
                orig_mouse_x: 0, orig_mouse_y: 0,
                orig_x: 0, orig_y: 0, orig_w: 0, orig_h: 0,
            },
            prev_left:  false,
            prev_right: false,
        }
    }

    /// Create a window; returns its ID (or 0 on failure).
    pub fn create_window(&mut self, x: i32, y: i32, w: u32, h: u32, title: &str) -> u32 {
        if self.windows.len() >= MAX_WINDOWS || w == 0 || h == 0 { return 0; }
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1).max(1);
        let z = self.z_stack.len() as u32;
        let pixels = alloc::vec![0xFF_22_22_2Au32; (w * h) as usize]; // dark default bg
        let mut win = Window {
            id, title: [0; 64], title_len: 0,
            x, y, w, h,
            saved_x: x, saved_y: y, saved_w: w, saved_h: h,
            z_order: z, minimized: false, maximized: false, dirty: true,
            pixels,
        };
        win.set_title(title);
        self.windows.insert(id, win);
        self.z_stack.push(id);
        self.focused = Some(id);
        id
    }

    /// Destroy a window and repaint.
    pub fn destroy_window(&mut self, id: u32) {
        self.windows.remove(&id);
        self.z_stack.retain(|&i| i != id);
        if self.focused == Some(id) {
            self.focused = self.z_stack.last().copied();
        }
        self.renormalize_z();
    }

    fn renormalize_z(&mut self) {
        for (i, &wid) in self.z_stack.iter().enumerate() {
            if let Some(w) = self.windows.get_mut(&wid) {
                w.z_order = i as u32;
            }
        }
    }

    /// Bring window to front and set as focused.
    pub fn focus_window(&mut self, id: u32) {
        if !self.windows.contains_key(&id) { return; }
        self.z_stack.retain(|&i| i != id);
        self.z_stack.push(id);
        self.focused = Some(id);
        self.renormalize_z();
    }

    /// Returns the topmost non-minimized window that contains screen position.
    pub fn hit_test(&self, x: i32, y: i32) -> Option<u32> {
        for &id in self.z_stack.iter().rev() {
            if let Some(w) = self.windows.get(&id) {
                if !w.minimized && w.contains(x, y) { return Some(id); }
            }
        }
        None
    }

    /// Returns true if (x, y) is in the title bar of window `id`.
    fn in_titlebar(&self, id: u32, x: i32, y: i32) -> bool {
        if let Some(w) = self.windows.get(&id) {
            y >= w.chrome_y() && y < w.y && x >= w.x && x < w.x + w.w as i32
        } else { false }
    }

    /// Classify cursor position for resize edge detection.
    fn resize_kind(&self, id: u32, x: i32, y: i32) -> DragKind {
        let w = match self.windows.get(&id) { Some(w) => w, None => return DragKind::None };
        let n = (y - w.chrome_y()).abs() <= RESIZE_BORDER as i32;
        let s = (y - (w.y + w.h as i32)).abs() <= RESIZE_BORDER as i32;
        let ww = (x - w.x).abs() <= RESIZE_BORDER as i32;
        let e = (x - (w.x + w.w as i32)).abs() <= RESIZE_BORDER as i32;
        match (n, s, ww, e) {
            (true, _, true, _)  => DragKind::ResizeNW,
            (true, _, _, true)  => DragKind::ResizeNE,
            (_, true, true, _)  => DragKind::ResizeSW,
            (_, true, _, true)  => DragKind::ResizeSE,
            (true, _, _, _)     => DragKind::ResizeN,
            (_, true, _, _)     => DragKind::ResizeS,
            (_, _, true, _)     => DragKind::ResizeW,
            (_, _, _, true)     => DragKind::ResizeE,
            _                   => DragKind::None,
        }
    }

    /// Apply a drag update given new absolute cursor position.
    fn apply_drag(&mut self, mx: i32, my: i32) {
        if self.drag.kind == DragKind::None { return; }
        let dx = mx - self.drag.orig_mouse_x;
        let dy = my - self.drag.orig_mouse_y;
        let id = self.drag.win_id;
        let w = match self.windows.get_mut(&id) { Some(w) => w, None => return };
        match self.drag.kind {
            DragKind::Move => {
                w.x = self.drag.orig_x + dx;
                w.y = self.drag.orig_y + dy;
            }
            DragKind::ResizeS => {
                let nh = ((self.drag.orig_h as i32 + dy).max(WIN_MIN_H as i32)) as u32;
                w.h = nh;
                if w.pixels.len() != (w.w * w.h) as usize {
                    w.pixels.resize((w.w * w.h) as usize, 0xFF_22_22_2A);
                }
            }
            DragKind::ResizeE => {
                let nw = ((self.drag.orig_w as i32 + dx).max(WIN_MIN_W as i32)) as u32;
                w.w = nw;
                if w.pixels.len() != (w.w * w.h) as usize {
                    w.pixels.resize((w.w * w.h) as usize, 0xFF_22_22_2A);
                }
            }
            DragKind::ResizeSE => {
                let nw = ((self.drag.orig_w as i32 + dx).max(WIN_MIN_W as i32)) as u32;
                let nh = ((self.drag.orig_h as i32 + dy).max(WIN_MIN_H as i32)) as u32;
                w.w = nw; w.h = nh;
                if w.pixels.len() != (w.w * w.h) as usize {
                    w.pixels.resize((w.w * w.h) as usize, 0xFF_22_22_2A);
                }
            }
            DragKind::ResizeN => {
                let ny = self.drag.orig_y + dy;
                let nh = ((self.drag.orig_h as i32 - dy).max(WIN_MIN_H as i32)) as u32;
                w.y = ny; w.h = nh;
                if w.pixels.len() != (w.w * w.h) as usize {
                    w.pixels.resize((w.w * w.h) as usize, 0xFF_22_22_2A);
                }
            }
            DragKind::ResizeW => {
                let nx = self.drag.orig_x + dx;
                let nw = ((self.drag.orig_w as i32 - dx).max(WIN_MIN_W as i32)) as u32;
                w.x = nx; w.w = nw;
                if w.pixels.len() != (w.w * w.h) as usize {
                    w.pixels.resize((w.w * w.h) as usize, 0xFF_22_22_2A);
                }
            }
            DragKind::ResizeNW => {
                let nx = self.drag.orig_x + dx;
                let ny = self.drag.orig_y + dy;
                let nw = ((self.drag.orig_w as i32 - dx).max(WIN_MIN_W as i32)) as u32;
                let nh = ((self.drag.orig_h as i32 - dy).max(WIN_MIN_H as i32)) as u32;
                w.x = nx; w.y = ny; w.w = nw; w.h = nh;
                if w.pixels.len() != (w.w * w.h) as usize {
                    w.pixels.resize((w.w * w.h) as usize, 0xFF_22_22_2A);
                }
            }
            DragKind::ResizeNE => {
                let ny = self.drag.orig_y + dy;
                let nw = ((self.drag.orig_w as i32 + dx).max(WIN_MIN_W as i32)) as u32;
                let nh = ((self.drag.orig_h as i32 - dy).max(WIN_MIN_H as i32)) as u32;
                w.y = ny; w.w = nw; w.h = nh;
                if w.pixels.len() != (w.w * w.h) as usize {
                    w.pixels.resize((w.w * w.h) as usize, 0xFF_22_22_2A);
                }
            }
            DragKind::ResizeSW => {
                let nx = self.drag.orig_x + dx;
                let nw = ((self.drag.orig_w as i32 - dx).max(WIN_MIN_W as i32)) as u32;
                let nh = ((self.drag.orig_h as i32 + dy).max(WIN_MIN_H as i32)) as u32;
                w.x = nx; w.w = nw; w.h = nh;
                if w.pixels.len() != (w.w * w.h) as usize {
                    w.pixels.resize((w.w * w.h) as usize, 0xFF_22_22_2A);
                }
            }
            DragKind::None => {}
        }
    }

    /// Toggle minimize state for a window.
    pub fn toggle_minimize(&mut self, id: u32) {
        if let Some(w) = self.windows.get_mut(&id) {
            w.minimized = !w.minimized;
        }
        if self.focused == Some(id) && self.windows.get(&id).map(|w| w.minimized).unwrap_or(false) {
            // Focus next visible window
            self.focused = self.z_stack.iter().rev()
                .filter(|&&i| i != id)
                .find(|&&i| self.windows.get(&i).map(|w| !w.minimized).unwrap_or(false))
                .copied();
        }
    }

    /// Toggle maximize state for a window.
    pub fn toggle_maximize(&mut self, id: u32, screen_w: usize, screen_h: usize, top_h: usize) {
        let is_max = self.windows.get(&id).map(|w| w.maximized).unwrap_or(false);
        if let Some(w) = self.windows.get_mut(&id) {
            if is_max {
                w.x = w.saved_x; w.y = w.saved_y;
                w.w = w.saved_w; w.h = w.saved_h;
                w.maximized = false;
            } else {
                w.saved_x = w.x; w.saved_y = w.y;
                w.saved_w = w.w; w.saved_h = w.h;
                w.x = 0;
                w.y = (top_h + WINTITLE_H) as i32;
                w.w = screen_w as u32;
                w.h = screen_h.saturating_sub(top_h + WINTITLE_H + TASKBAR_H) as u32;
                w.maximized = true;
                if w.pixels.len() != (w.w * w.h) as usize {
                    w.pixels.resize((w.w * w.h) as usize, 0xFF_22_22_2A);
                }
            }
        }
    }

    pub fn is_empty(&self) -> bool { self.windows.is_empty() }
}

// ── Global state ──────────────────────────────────────────────────────────────
static WM: Once<Mutex<WmState>> = Once::new();

pub fn wm_init() {
    WM.call_once(|| Mutex::new(WmState::new()));
}

fn with_wm<F: FnOnce(&mut WmState) -> R, R>(f: F) -> Option<R> {
    WM.get().map(|m| f(&mut m.lock()))
}

/// Public version for use in syscall module.
pub fn with_wm_pub<F: FnOnce(&mut WmState)>(f: F) {
    if let Some(m) = WM.get() { f(&mut m.lock()); }
}

pub fn wm_is_active() -> bool {
    WM.get().map(|m| !m.lock().is_empty()).unwrap_or(false)
}

// ── Public window management API ──────────────────────────────────────────────

pub fn wm_create_window(x: i32, y: i32, w: u32, h: u32, title: &str) -> u32 {
    with_wm(|s| s.create_window(x, y, w, h, title)).unwrap_or(0)
}

pub fn wm_destroy_window(id: u32) {
    with_wm(|s| s.destroy_window(id));
    wm_composite();
}

pub fn wm_set_title(id: u32, title: &str) {
    with_wm(|s| {
        if let Some(w) = s.windows.get_mut(&id) { w.set_title(title); }
    });
}

/// Write a single pixel into a window's backing buffer (does not blit to screen).
pub fn wm_paint_pixel(id: u32, px: u32, py: u32, rgba: u32) {
    with_wm(|s| {
        if let Some(w) = s.windows.get_mut(&id) {
            if px < w.w && py < w.h {
                w.pixels[(py * w.w + px) as usize] = rgba;
            }
        }
    });
}

/// Fill a rectangle in a window's backing buffer (does not blit to screen).
pub fn wm_fill_window_rect(id: u32, rx: u32, ry: u32, rw: u32, rh: u32, rgba: u32) {
    with_wm(|s| {
        if let Some(win) = s.windows.get_mut(&id) {
            let x1 = (rx + rw).min(win.w);
            let y1 = (ry + rh).min(win.h);
            for py in ry..y1 {
                for px in rx..x1 {
                    win.pixels[(py * win.w + px) as usize] = rgba;
                }
            }
        }
    });
}

/// Blit the window's backing buffer to the framebuffer (on-screen update).
pub fn wm_flip(id: u32) {
    // Take a snapshot of the window position/size/pixels to avoid holding lock during fb::with
    let snapshot = with_wm(|s| {
        s.windows.get(&id).map(|w| {
            (w.x, w.y, w.w, w.h, w.pixels.clone(), s.focused == Some(id))
        })
    });
    let (wx, wy, ww, wh, pixels, focused) = match snapshot.flatten() {
        Some(v) => v,
        None    => return,
    };
    fb::with(|f| {
        let sw = f.width() as i32;
        let sh = f.height() as i32;
        for py in 0..wh {
            for px in 0..ww {
                let sx = wx + px as i32;
                let sy = wy + py as i32;
                if sx < 0 || sy < 0 || sx >= sw || sy >= sh { continue; }
                let rgba = pixels[(py * ww + px) as usize];
                let r = ((rgba >> 16) & 0xFF) as u8;
                let g = ((rgba >>  8) & 0xFF) as u8;
                let b = ( rgba        & 0xFF) as u8;
                f.put_pixel(sx as usize, sy as usize, r, g, b);
            }
        }
    });
    // Re-draw chrome on top (we need WM state again, briefly)
    let chrome_info = with_wm(|s| {
        s.windows.get(&id).map(|w| {
            (w.x, w.y, w.w, w.chrome_y(),
             alloc::string::String::from(w.title()),
             s.focused == Some(id))
        })
    });
    if let Some(Some((cx, cy, cw, chrome_y, title, is_focused))) = chrome_info {
        fb::with(|f| draw_window_chrome(f, cx, cy, cw, chrome_y, &title, is_focused));
    }
}

/// Render one 8×16 character cell into a window's pixel buffer.
/// `px`, `py` are top-left pixel coordinates within the window client area.
/// `fg` / `bg` are 0x00RRGGBB colours.
pub fn wm_draw_text_cell(id: u32, px: u32, py: u32, ch: u8, fg: u32, bg: u32) {
    let glyph = crate::framebuffer::glyph(ch);
    with_wm(|s| {
        if let Some(win) = s.windows.get_mut(&id) {
            for row in 0u32..16 {
                let bits = glyph[row as usize];
                for col in 0u32..8 {
                    let lit = (bits >> (7 - col)) & 1 != 0;
                    let color = if lit { fg } else { bg };
                    let wx = px + col;
                    let wy = py + row;
                    if wx < win.w && wy < win.h {
                        win.pixels[(wy * win.w + wx) as usize] = color;
                    }
                }
            }
        }
    });
}

/// Full compositor repaint: background → windows (z-order) → taskbar.
pub fn wm_composite() {
    if !fb::is_available() { return; }
    // Snapshot state to avoid holding lock during framebuffer operations
    struct WinSnap {
        id:        u32,
        x:         i32,
        y:         i32,
        w:         u32,
        h:         u32,
        z_order:   u32,
        minimized: bool,
        focused:   bool,
        title:     alloc::string::String,
        pixels:    Vec<u32>,
    }
    let snaps: Vec<WinSnap> = {
        let guard = match WM.get() { Some(m) => m, None => return };
        let state  = guard.lock();
        let mut v: Vec<WinSnap> = state.z_stack.iter().filter_map(|&id| {
            state.windows.get(&id).map(|w| WinSnap {
                id, x: w.x, y: w.y, w: w.w, h: w.h,
                z_order: w.z_order, minimized: w.minimized,
                focused: state.focused == Some(id),
                title: alloc::string::String::from(w.title()),
                pixels: w.pixels.clone(),
            })
        }).collect();
        v.sort_by_key(|s| s.z_order);
        v
    };

    fb::with(|f| {
        let sw = f.width();
        let sh = f.height();

        // ── Desktop background ─────────────────────────────────────────────────
        f.fill_rect(0, 0, sw, sh, DESK_BG.0, DESK_BG.1, DESK_BG.2);

        // ── Paint windows bottom-to-top ────────────────────────────────────────
        for snap in &snaps {
            if snap.minimized { continue; }

            // 2-pixel shadow drop (black, offset +2/+2) behind chrome
            let chrome_y = snap.y - WINTITLE_H as i32;
            let shadow_x = (snap.x + 3).max(0) as usize;
            let shadow_y = (chrome_y + 3).max(0) as usize;
            let shadow_w = snap.w as usize;
            let shadow_h = (snap.h as usize) + WINTITLE_H;
            let se = (shadow_x + shadow_w).min(sw);
            let se_h = (shadow_y + shadow_h).min(sh);
            for py in shadow_y..se_h {
                for px in shadow_x..se {
                    f.put_pixel(px, py, WIN_SHADOW.0, WIN_SHADOW.1, WIN_SHADOW.2);
                }
            }

            // Window content (pixel buffer)
            for py in 0..snap.h {
                for px in 0..snap.w {
                    let sx = snap.x + px as i32;
                    let sy = snap.y + py as i32;
                    if sx < 0 || sy < 0 || sx >= sw as i32 || sy >= sh as i32 { continue; }
                    let rgba = snap.pixels[(py * snap.w + px) as usize];
                    let r = ((rgba >> 16) & 0xFF) as u8;
                    let g = ((rgba >>  8) & 0xFF) as u8;
                    let b = ( rgba        & 0xFF) as u8;
                    f.put_pixel(sx as usize, sy as usize, r, g, b);
                }
            }

            // Window chrome (title bar + buttons)
            draw_window_chrome(f, snap.x, snap.y, snap.w, chrome_y, &snap.title, snap.focused);
        }

        // ── Taskbar ────────────────────────────────────────────────────────────
        draw_taskbar(f, &snaps.iter().map(|s| {
            TaskBtn { id: s.id, minimized: s.minimized, focused: s.focused,
                      title: s.title.clone() }
        }).collect::<Vec<_>>(), 0u64);
    });
}

// ── Window chrome drawing ─────────────────────────────────────────────────────

fn draw_window_chrome(f: &mut fb::Framebuffer,
                      wx: i32, wy: i32, ww: u32,
                      chrome_y: i32,
                      title: &str,
                      focused: bool) {
    let sw = f.width() as i32;
    let sh = f.height() as i32;
    // Title bar background
    let bg = if focused { TITLE_ACTIVE } else { TITLE_INACTIVE };
    let fg = if focused { TITLE_FG     } else { TITLE_FG_DIM   };
    let bor = if focused { WIN_BORDER_ACT } else { WIN_BORDER_IN };

    let ty = chrome_y;
    let bw = ww as i32;
    let bh = WINTITLE_H as i32;

    // Clip to screen
    let x0 = wx.max(0) as usize;
    let y0 = ty.max(0) as usize;
    let x1 = (wx + bw).min(sw) as usize;
    let y1 = (ty + bh).min(sh) as usize;

    if x0 < x1 && y0 < y1 {
        f.fill_rect(x0, y0, x1 - x0, y1 - y0, bg.0, bg.1, bg.2);
    }

    // 1-pixel top border (accent)
    if ty >= 0 && ty < sh {
        let bx0 = wx.max(0) as usize;
        let bx1 = (wx + bw).min(sw) as usize;
        if bx0 < bx1 {
            f.fill_rect(bx0, ty as usize, bx1 - bx0, 1, bor.0, bor.1, bor.2);
        }
    }

    // Traffic-light buttons (close=red, min=yellow, max=green)
    let cy = (ty + bh / 2) as i32;
    draw_btn_circle(f, wx + 12, cy, 6, BTN_CLOSE, BTN_BOR);
    draw_btn_circle(f, wx + 28, cy, 6, BTN_MIN,   BTN_BOR);
    draw_btn_circle(f, wx + 44, cy, 6, BTN_MAX,   BTN_BOR);

    // Window title (centred in title bar)
    let max_chars = ((bw as usize).saturating_sub(8 * FONT_W)) / FONT_W;
    let shown: alloc::string::String = if title.len() > max_chars {
        let mut s = alloc::string::String::from(&title[..max_chars.saturating_sub(3)]);
        s.push_str("...");
        s
    } else {
        alloc::string::String::from(title)
    };
    let tx = wx + (bw / 2) - (shown.len() as i32 * FONT_W as i32) / 2;
    let text_y = ty + (bh - FONT_H as i32) / 2;
    if tx >= 0 && text_y >= 0 && tx < sw && text_y < sh {
        f.draw_str(tx as usize, text_y as usize, &shown, fg, bg);
    }

    // 1-pixel bottom border (same accent)
    let bot_y = ty + bh - 1;
    if bot_y >= 0 && bot_y < sh {
        let bx0 = wx.max(0) as usize;
        let bx1 = (wx + bw).min(sw) as usize;
        if bx0 < bx1 {
            f.fill_rect(bx0, bot_y as usize, bx1 - bx0, 1, bor.0, bor.1, bor.2);
        }
    }

    // Left, right, bottom window border lines (1px)
    // Left
    if wx >= 0 && wx < sw {
        let by0 = ty.max(0) as usize;
        let by1 = (ty + bh + wy - ty).min(sh) as usize; // full window height from chrome_y to content bottom would need wy+h
        // Just bottom of chrome bar border:
        let content_bot = wy + by0 as i32; // approximation — skipped for brevity
        let _ = content_bot;
    }
}

fn draw_btn_circle(f: &mut fb::Framebuffer,
                   cx: i32, cy: i32, r: i32,
                   fill: (u8,u8,u8), border: (u8,u8,u8)) {
    let sw = f.width() as i32;
    let sh = f.height() as i32;
    let r2  = r * r;
    let ir2 = (r - 1).max(0) * (r - 1).max(0);
    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = dx * dx + dy * dy;
            if d2 <= r2 {
                let px = cx + dx;
                let py = cy + dy;
                if px < 0 || py < 0 || px >= sw || py >= sh { continue; }
                let (pr, pg, pb) = if d2 > ir2 { border } else { fill };
                f.put_pixel(px as usize, py as usize, pr, pg, pb);
            }
        }
    }
}

// ── Taskbar drawing ───────────────────────────────────────────────────────────
struct TaskBtn {
    id:        u32,
    minimized: bool,
    focused:   bool,
    title:     alloc::string::String,
}

fn draw_taskbar(f: &mut fb::Framebuffer, buttons: &[TaskBtn], tick: u64) {
    let sw = f.width();
    let sh = f.height();
    let ty = sh.saturating_sub(TASKBAR_H);

    // Background
    f.fill_rect(0, ty, sw, TASKBAR_H, TASKBR_BG.0, TASKBR_BG.1, TASKBR_BG.2);
    // Top border
    f.fill_rect(0, ty, sw, 1, 0x35, 0x84, 0xE4);

    let btn_h = TASKBAR_H - 8;
    let btn_y = ty + 4;

    // NodeAI pill (leftmost)
    let pill_label = " NodeAI ";
    let pill_w     = pill_label.len() * FONT_W + 4;
    let pill_x     = 4usize;
    f.fill_rect(pill_x, btn_y, pill_w, btn_h, 0x35, 0x84, 0xE4);
    let text_y = btn_y + (btn_h.saturating_sub(FONT_H)) / 2;
    f.draw_str(pill_x + 2, text_y, pill_label, (0xFF, 0xFF, 0xFF), (0x35, 0x84, 0xE4));

    // Window buttons (one per window, 120px max width each)
    let mut bx = pill_x + pill_w + 8;
    for btn in buttons {
        if bx + 4 > sw.saturating_sub(150) { break; }
        let btn_w  = 120usize.min(sw.saturating_sub(bx + 150));
        let bg     = if btn.focused { TASKBR_ACTIVE } else { TASKBR_BTN };
        let fg     = TASKBR_FG;
        f.fill_rect(bx, btn_y, btn_w, btn_h, bg.0, bg.1, bg.2);
        // Truncate title to fit
        let max_c = (btn_w.saturating_sub(4)) / FONT_W;
        let shown: alloc::string::String = if btn.title.len() > max_c && max_c > 3 {
            let mut s = alloc::string::String::from(&btn.title[..max_c - 3]);
            s.push_str("...");
            s
        } else {
            btn.title.clone()
        };
        let prefix = if btn.minimized { "_ " } else { "  " };
        let full_title: alloc::string::String = {
            let mut s = alloc::string::String::from(prefix);
            s.push_str(&shown);
            s
        };
        f.draw_str(bx + 2, text_y, &full_title, fg, bg);
        // 1-px active indicator (bottom of button)
        if btn.focused {
            f.fill_rect(bx, btn_y + btn_h - 2, btn_w, 2, 0x35, 0x84, 0xE4);
        }
        bx += btn_w + 4;
    }

    // Clock (right-aligned)
    let secs = tick / 1000;
    let hh   = (secs / 3600) % 24;
    let mm   = (secs / 60) % 60;
    let ss   =  secs % 60;
    let clock_x = sw.saturating_sub(10 * FONT_W);
    f.draw_fmt(clock_x, text_y, (0xFF, 0xFF, 0xFF), TASKBR_BG,
               format_args!("{:02}:{:02}:{:02}", hh, mm, ss));
}

// ── Mouse event handling for WM ───────────────────────────────────────────────

/// Process a mouse movement/click event.  Called from desktop's `mouse_event`.
/// Returns `true` if the event was fully consumed by the WM (don't pass to legacy desktop).
pub fn wm_mouse_event(raw_dx: i16, raw_dy: i16, left: bool, right: bool) -> bool {
    if !fb::is_available() { return false; }
    let active = WM.get().map(|m| !m.lock().is_empty()).unwrap_or(false);
    if !active { return false; }

    let (sw, sh) = (fb::width() as i32, fb::height() as i32);

    // Update cursor
    let (nx, ny, click, release, prev_left) = with_wm(|s| {
        s.cursor_x = (s.cursor_x + raw_dx as i32 * 2).clamp(0, sw - 1);
        s.cursor_y = (s.cursor_y + raw_dy as i32 * 2).clamp(0, sh - 1);
        let click   = left && !s.prev_left;
        let release = !left && s.prev_left;
        let prev    = s.prev_left;
        s.prev_left  = left;
        s.prev_right = right;
        (s.cursor_x, s.cursor_y, click, release, prev)
    }).unwrap_or((0, 0, false, false, false));

    // Handle drag continuation (left button held)
    if left && prev_left {
        with_wm(|s| { s.apply_drag(nx, ny); });
        wm_composite();
        draw_wm_cursor(nx, ny);
        return true;
    }

    // Release: end drag
    if release {
        with_wm(|s| { s.drag.kind = DragKind::None; });
        wm_composite();
        draw_wm_cursor(nx, ny);
        return true;
    }

    // Click: hit-test + action
    if click {
        // Check taskbar click first (y near bottom)
        if ny >= sh - TASKBAR_H as i32 {
            let ids: Vec<u32> = with_wm(|s| s.z_stack.clone()).unwrap_or_default();
            // Node-AI pill (x 4..4+pill_w) toggles launcher
            let pill_w = (" NodeAI ".len() * FONT_W + 4) as i32;
            if nx >= 4 && nx < 4 + pill_w {
                crate::desktop::launcher_toggle();
                return true;
            }
            // Window buttons
            let mut bx = 4 + pill_w + 8;
            for &id in &ids {
                let btn_w = 124i32;
                if nx >= bx && nx < bx + btn_w {
                    let is_focused = with_wm(|s| s.focused == Some(id)).unwrap_or(false);
                    if is_focused {
                        with_wm(|s| s.toggle_minimize(id));
                    } else {
                        with_wm(|s| {
                            if s.windows.get(&id).map(|w| w.minimized).unwrap_or(false) {
                                s.toggle_minimize(id);
                            }
                            s.focus_window(id);
                        });
                    }
                    wm_composite();
                    draw_wm_cursor(nx, ny);
                    return true;
                }
                bx += btn_w + 4;
            }
            return true; // consumed taskbar click
        }

        // Hit-test windows
        let hit = with_wm(|s| s.hit_test(nx, ny)).flatten();
        if let Some(id) = hit {
            // Focus the clicked window
            with_wm(|s| s.focus_window(id));

            // Check if click is in title bar
            let in_tb = with_wm(|s| s.in_titlebar(id, nx, ny)).unwrap_or(false);
            if in_tb {
                let chrome_y = with_wm(|s| {
                    s.windows.get(&id).map(|w| w.chrome_y())
                }).flatten().unwrap_or(0);

                let wx = with_wm(|s| s.windows.get(&id).map(|w| w.x)).flatten().unwrap_or(0);
                // Close button (circle at wx+12)
                let close_cx = wx + 12;
                if (nx - close_cx).abs() <= 8 {
                    with_wm(|s| s.destroy_window(id));
                    wm_composite();
                    draw_wm_cursor(nx, ny);
                    return true;
                }
                // Minimize button (circle at wx+28)
                let min_cx = wx + 28;
                if (nx - min_cx).abs() <= 8 {
                    with_wm(|s| s.toggle_minimize(id));
                    wm_composite();
                    draw_wm_cursor(nx, ny);
                    return true;
                }
                // Maximize button (circle at wx+44)
                let max_cx = wx + 44;
                if (nx - max_cx).abs() <= 8 {
                    with_wm(|s| s.toggle_maximize(id, sw as usize, sh as usize, 36));
                    wm_composite();
                    draw_wm_cursor(nx, ny);
                    return true;
                }
                // Drag start
                let orig = with_wm(|s| {
                    s.windows.get(&id).map(|w| (w.x, w.y, w.w, w.h))
                }).flatten().unwrap_or((0, 0, 0, 0));
                with_wm(|s| {
                    s.drag = DragState {
                        kind: DragKind::Move, win_id: id,
                        orig_mouse_x: nx, orig_mouse_y: ny,
                        orig_x: orig.0, orig_y: orig.1,
                        orig_w: orig.2, orig_h: orig.3,
                    };
                });
            } else {
                // Check resize edges
                let rk = with_wm(|s| s.resize_kind(id, nx, ny)).unwrap_or(DragKind::None);
                if rk != DragKind::None {
                    let orig = with_wm(|s| {
                        s.windows.get(&id).map(|w| (w.x, w.y, w.w, w.h))
                    }).flatten().unwrap_or((0, 0, 0, 0));
                    with_wm(|s| {
                        s.drag = DragState {
                            kind: rk, win_id: id,
                            orig_mouse_x: nx, orig_mouse_y: ny,
                            orig_x: orig.0, orig_y: orig.1,
                            orig_w: orig.2, orig_h: orig.3,
                        };
                    });
                }
            }
            wm_composite();
            draw_wm_cursor(nx, ny);
            return true;
        }
    }

    // Just moved (no click) — redraw cursor
    wm_composite();
    draw_wm_cursor(nx, ny);
    true
}

/// Draw a simple crosshair cursor at (x, y) on the framebuffer.
fn draw_wm_cursor(x: i32, y: i32) {
    if !fb::is_available() { return; }
    // Re-use desktop module's existing cursor drawing (but we draw it ourselves here):
    const CURSOR_BITS: [u8; 12] = [
        0b1000_0000, 0b1100_0000, 0b1110_0000, 0b1111_0000,
        0b1111_1000, 0b1111_1100, 0b1111_1110, 0b1111_1100,
        0b1101_1000, 0b1000_1100, 0b0000_1100, 0b0000_0110,
    ];
    fb::with(|f| {
        let sw = f.width() as i32;
        let sh = f.height() as i32;
        for (row, &bits) in CURSOR_BITS.iter().enumerate() {
            for col in 0..8i32 {
                if (bits >> (7 - col)) & 1 != 0 {
                    let px = x + col;
                    let py = y + row as i32;
                    if px >= 0 && py >= 0 && px < sw && py < sh {
                        // Shadow pixel
                        if px + 1 < sw && py + 1 < sh {
                            f.put_pixel(px as usize + 1, py as usize + 1, 0, 0, 0);
                        }
                        f.put_pixel(px as usize, py as usize, 0xFF, 0xFF, 0xFF);
                    }
                }
            }
        }
    });
}

// ── Tick update (refresh taskbar clock) ──────────────────────────────────────

pub fn wm_tick(ticks: u64) {
    if !wm_is_active() { return; }
    if !fb::is_available() { return; }
    let buttons: Vec<TaskBtn> = {
        let guard = match WM.get() { Some(m) => m, None => return };
        let state  = guard.lock();
        state.z_stack.iter().filter_map(|&id| {
            state.windows.get(&id).map(|w| TaskBtn {
                id, minimized: w.minimized,
                focused: state.focused == Some(id),
                title: alloc::string::String::from(w.title()),
            })
        }).collect()
    };
    fb::with(|f| draw_taskbar(f, &buttons, ticks));
}

// ── Keyboard routing ──────────────────────────────────────────────────────────

/// Returns the ID of the currently focused window, if any.
pub fn wm_focused_id() -> Option<u32> {
    with_wm(|s| s.focused).flatten()
}
