//! NodeAI Desktop Compositor — GNOME-inspired kernel UI.
//!
//! Layout (1024×768 reference):
//!
//!   ┌──────────────────────────────────────────────────────────────────┐
//!   │ [NodeAI]  Terminal          11:23:45       256M · tasks:4 · root │ TOP_H=36px
//!   ├──────────────────────────────────────────────────────────────────┤
//!   │  ● ● ●   Terminal — root@nodeai                            [...]  │ TITLEBAR_H=28px
//!   ├──────────────────────────────────────────────────────────────────┤
//!   │                                                                  │
//!   │   Full-width terminal (dark background, ANSI colours)            │
//!   │                                                                  │
//!   └──────────────────────────────────────────────────────────────────┘
//!
//! Public API (same as before, extended with `set_title_user`):
//!   - `init()`                      — draw the full desktop.
//!   - `tick(ticks)`                 — refresh clock + system-tray stats.
//!   - `terminal_input(byte)`        — push one byte (ANSI aware).
//!   - `terminal_put_char(byte)`     — write char at cursor, advance col.
//!   - `terminal_clear_to_eol()`     — clear from cursor to end of line.
//!   - `terminal_redraw_line(row)`   — repaint one terminal row.
//!   - `terminal_col/row()`          — current cursor position.
//!   - `terminal_set_col(col)`       — reposition column.
//!   - `clear_terminal()`            — blank the terminal area.
//!   - `set_title_user(name)`        — update username in titlebar/tray.

pub mod compositor;
pub use compositor::{
    wm_init, wm_is_active, wm_create_window, wm_destroy_window,
    wm_set_title, wm_paint_pixel, wm_fill_window_rect, wm_flip,
    wm_composite, wm_mouse_event, wm_tick, wm_focused_id,
    wm_draw_text_cell,
    COMPOSER_CREATE_WINDOW, COMPOSER_DESTROY_WINDOW, COMPOSER_FLIP,
    COMPOSER_MOVE, COMPOSER_RESIZE, COMPOSER_SET_TITLE,
};

pub mod browser;
pub use browser::{
    browser_init, browser_is_open, browser_navigate,
};

pub mod term_window;
pub use term_window::{
    term_window_init, term_window_is_open, term_window_write, term_window_key, tty_read,
};

pub mod notepad_pro;
pub use notepad_pro::{notepad_open, notepad_pro_is_open, notepad_pro_key};

pub mod filemanager_pro;
pub use filemanager_pro::{fm_pro_open, fm_pro_is_open, fm_pro_key};

pub mod terminal_app;
pub use terminal_app::{
    terminal_app_open, terminal_app_is_open, terminal_app_write, terminal_app_key,
};

pub mod image_viewer;
pub use image_viewer::{imgview_open, imgview_is_open, imgview_key};

pub mod ai_chat;
pub use ai_chat::{ai_chat_open, ai_chat_is_open, ai_chat_key};

pub mod sysmon;
pub use sysmon::{sysmon_open, sysmon_is_open, sysmon_tick};

pub mod settings;
pub use settings::{settings_open, settings_is_open, settings_key};

pub mod appstore;
pub use appstore::{appstore_open, appstore_is_open, appstore_key};

use crate::framebuffer::{self as fb, colour};
use alloc::borrow::ToOwned;
use alloc::string::String;
use alloc::vec::Vec;

// ── Layout ────────────────────────────────────────────────────────────────────
/// Height of the GNOME-style top panel.
const TOP_H:        usize = 36;
/// Height of the terminal window title bar.
const TITLEBAR_H:   usize = 28;
/// Y-coordinate where terminal text content starts.
const TERM_Y:       usize = TOP_H + TITLEBAR_H;   // 64
const FONT_W:       usize = 8;
const FONT_H:       usize = 16;

// ── Terminal ring buffer ──────────────────────────────────────────────────────
const TERM_COLS_MAX: usize = 128;
const TERM_ROWS_MAX: usize = 64;

static mut TERM_BUF:   [[u8; TERM_COLS_MAX]; TERM_ROWS_MAX]
    = [[0u8; TERM_COLS_MAX]; TERM_ROWS_MAX];
static mut TERM_COLOR: [[u8; TERM_COLS_MAX]; TERM_ROWS_MAX]
    = [[0u8; TERM_COLS_MAX]; TERM_ROWS_MAX];
static mut TERM_ROW: usize = 0;
static mut TERM_COL: usize = 0;
static mut TICK:     u64   = 0;

// Dynamic username displayed in title-bar and system tray (updated by set_title_user).
static mut TITLE_USER:     [u8; 32] = *b"root\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
static mut TITLE_USER_LEN: usize    = 4;

// ── Mouse cursor state ────────────────────────────────────────────────────────
/// Width and height of the software cursor sprite in pixels.
const CURSOR_W: usize = 8;
const CURSOR_H: usize = 12;
/// Arrow-pointer bitmap: each byte is one row, bit7 = leftmost pixel.
const CURSOR_BITS: [u8; CURSOR_H] = [
    0b1000_0000, // *
    0b1100_0000, // **
    0b1110_0000, // ***
    0b1111_0000, // ****
    0b1111_1000, // *****
    0b1111_1100, // ******
    0b1111_1110, // *******
    0b1111_1100, // ******
    0b1101_1000, // ** **
    0b1000_1100, // *   **
    0b0000_1100, //     **
    0b0000_0110, //      *
];
static mut CURSOR_X:     usize = 512;
static mut CURSOR_Y:     usize = 400;
/// True when cursor pixels are currently painted on the framebuffer.
static mut CURSOR_DRAWN: bool  = false;
/// Previous left-button state — used to detect rising edge (click).
static mut PREV_LEFT:    bool  = false;
/// Save area for pixels under the cursor sprite (+1 shadow = 9×13 pixels max).
const CURSOR_SAVE_W: usize = CURSOR_W + 1;
const CURSOR_SAVE_H: usize = CURSOR_H + 1;
static mut CURSOR_SAVE: [(u8, u8, u8); CURSOR_SAVE_W * CURSOR_SAVE_H]
    = [(0, 0, 0); CURSOR_SAVE_W * CURSOR_SAVE_H];
static mut CURSOR_SAVE_X: usize = 0;
static mut CURSOR_SAVE_Y: usize = 0;

// ── App launcher ──────────────────────────────────────────────────────────────
struct AppDesc { icon: &'static str, name: &'static str, cmd: &'static str }
const APPS: [AppDesc; 7] = [
    AppDesc { icon: ">_", name: "Terminal",     cmd: ""        },
    AppDesc { icon: "FM", name: "File Manager", cmd: "fm"      },
    AppDesc { icon: "NT", name: "Note Pad",     cmd: "note"    },
    AppDesc { icon: "CA", name: "Calculator",   cmd: "calc"    },
    AppDesc { icon: "SI", name: "System Info",  cmd: "sysinfo" },
    AppDesc { icon: "IB", name: "Browser",      cmd: "browser" },
    AppDesc { icon: "NW", name: "Network",      cmd: "netmgr"  },
];
const LAUNCHER_COLS:  usize = 3;
const TILE_W:         usize = 160;
const TILE_H:         usize = 80;
const TILE_GAP:       usize = 20;
const SEARCHBAR_H:    usize = 28;
// Launcher overlay colour scheme
const LAUNCH_BG:  (u8,u8,u8) = (0x0E, 0x0E, 0x16); // dark translucent overlay
const LAUNCH_TBG: (u8,u8,u8) = (0x28, 0x28, 0x38); // tile background
const LAUNCH_DIM: (u8,u8,u8) = (0x18, 0x18, 0x22); // dimmed (non-matching) tile
const LAUNCH_SBG: (u8,u8,u8) = (0x1C, 0x1C, 0x28); // search bar background
const LAUNCH_SFG: (u8,u8,u8) = (0xEE, 0xEE, 0xEE); // search text
const LAUNCH_IFG: (u8,u8,u8) = (0xFF, 0xFF, 0xFF); // icon text
const LAUNCH_NFG: (u8,u8,u8) = (0xCC, 0xCC, 0xCC); // app name text

static mut LAUNCHER_OPEN:       bool     = false;
static mut LAUNCHER_SEARCH:     [u8; 32] = [0u8; 32];
static mut LAUNCHER_SEARCH_LEN: usize    = 0;

// ── GUI Application Windows ───────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq, Eq)]
enum ActiveApp { Terminal, Notepad, FileManager, Browser, Network }
static mut ACTIVE_APP: ActiveApp = ActiveApp::Terminal;
/// Y where app content starts (below the window titlebar).
const APP_STATUS_H: usize = 20;

// ── Notepad state ─────────────────────────────────────────────────────────────
const NP_ROWS:     usize = 200;
const NP_COLS:     usize = 120;
const NP_GUTTER_W: usize = 40;
static mut NP_BUF:       [[u8; NP_COLS]; NP_ROWS] = [[0u8; NP_COLS]; NP_ROWS];
static mut NP_LEN:       [usize; NP_ROWS]          = [0usize; NP_ROWS];
static mut NP_ROWS_USED: usize = 1;
static mut NP_EDIT_ROW:  usize = 0;
static mut NP_EDIT_COL:  usize = 0;
static mut NP_SCROLL:    usize = 0;
static mut NP_FNAME:     [u8; 64] = [0u8; 64];
static mut NP_FNAME_LEN: usize = 0;
static mut NP_DIRTY:     bool = false;
const NP_BG:  (u8,u8,u8) = (0xF8, 0xF8, 0xF8);
const NP_FG:  (u8,u8,u8) = (0x18, 0x18, 0x18);
const NP_LN:  (u8,u8,u8) = (0x88, 0x88, 0x9A);
const NP_GUT: (u8,u8,u8) = (0xE8, 0xE8, 0xEC);
const NP_CUR: (u8,u8,u8) = (0xC8, 0xDC, 0xFF);
const NP_SBG: (u8,u8,u8) = (0x24, 0x24, 0x30);
const NP_SFG: (u8,u8,u8) = (0xAA, 0xAA, 0xCC);

// ── File-Manager state ────────────────────────────────────────────────────────
const FM_MAX: usize = 128;
static mut FM_PATH:     [u8; 256]          = [0u8; 256];
static mut FM_PATH_LEN: usize              = 0;
static mut FM_NAMES:    [[u8; 64]; FM_MAX] = [[0u8; 64]; FM_MAX];
static mut FM_NLENS:    [usize; FM_MAX]    = [0usize; FM_MAX];
static mut FM_IS_DIR:   [bool; FM_MAX]     = [false; FM_MAX];
static mut FM_SIZES:    [u64; FM_MAX]      = [0u64; FM_MAX];
static mut FM_COUNT:    usize              = 0;
static mut FM_SEL:      usize              = 0;
static mut FM_SCROLL:   usize              = 0;
const FM_BG:  (u8,u8,u8) = (0x14, 0x14, 0x20);
const FM_HDR: (u8,u8,u8) = (0x20, 0x20, 0x34);
const FM_SL:  (u8,u8,u8) = (0x28, 0x50, 0x88);
const FM_DIR_C: (u8,u8,u8) = (0x55, 0xAA, 0xFF);
const FM_FIL: (u8,u8,u8) = (0xCC, 0xCC, 0xCC);
const FM_FG:  (u8,u8,u8) = (0xE0, 0xE0, 0xE0);

// ── Browser state ─────────────────────────────────────────────────────────────
const BR_CROWS: usize = 60;
const BR_CCOLS: usize = 120;
const BR_URLBAR_H: usize = 28;
static mut BR_URL:      [u8; 256]              = [0u8; 256];
static mut BR_URL_LEN:  usize                  = 0;
static mut BR_CONT:     [[u8; BR_CCOLS]; BR_CROWS] = [[0u8; BR_CCOLS]; BR_CROWS];
static mut BR_CLENS:    [usize; BR_CROWS]      = [0usize; BR_CROWS];
static mut BR_CROWS_USED: usize                = 0;
static mut BR_SCROLL:   usize                  = 0;
static mut BR_STATUS:   [u8; 80]               = [0u8; 80];
static mut BR_STAT_LEN: usize                  = 0;
/// Pending URL to fetch — set from keyboard IRQ, consumed in idle loop.
static BR_FETCH_PENDING: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);
static mut BR_FETCH_URL: [u8; 256] = [0u8; 256];
static mut BR_FETCH_URL_LEN: usize = 0;
const BR_BG:   (u8,u8,u8) = (0x16, 0x16, 0x20);
const BR_UBGN: (u8,u8,u8) = (0x28, 0x28, 0x3C);
const BR_UBGF: (u8,u8,u8) = (0x18, 0x38, 0x60);
const BR_FG:   (u8,u8,u8) = (0xE0, 0xE0, 0xE0);
const BR_SBG:  (u8,u8,u8) = (0x10, 0x10, 0x1C);

// ── ANSI escape sequence parser state ────────────────────────────────────────
#[derive(Clone, Copy, PartialEq)]
enum AnsiState { Normal, Esc, Csi }
static mut ANSI_STATE:    AnsiState = AnsiState::Normal;
static mut ANSI_BUF:      [u8; 32]  = [0u8; 32];
static mut ANSI_LEN:      usize     = 0;
static mut CUR_COLOR_IDX: u8        = 0;

// ── Color palette: GNOME Adwaita Dark + NodeAI branding ──────────────────────
/// Desktop background — deep dark blue-grey.
const DESK_BG:   (u8,u8,u8) = (0x22, 0x22, 0x2A);
/// Top panel background.
const TOP_BG:    (u8,u8,u8) = (0x18, 0x18, 0x20);
/// Top panel primary text.
const TOP_FG:    (u8,u8,u8) = (0xEE, 0xEE, 0xEE);
/// Top panel dimmed/secondary text.
const TOP_DIM:   (u8,u8,u8) = (0x88, 0x88, 0x99);
/// Subtle 1-pixel separator lines in the panel.
const TOP_SEP:   (u8,u8,u8) = (0x40, 0x40, 0x58);
/// NodeAI accent blue (GNOME/Activities button).
const ACCENT:    (u8,u8,u8) = (0x35, 0x84, 0xE4);
/// Text on accent-coloured background.
const ACCENT_FG: (u8,u8,u8) = (0xFF, 0xFF, 0xFF);
/// Bright white clock digits.
const CLOCK_FG:  (u8,u8,u8) = (0xFF, 0xFF, 0xFF);
/// System tray: memory indicator colour (purple).
const S_MEM:     (u8,u8,u8) = (0xB4, 0x88, 0xFF);
/// System tray: task-count colour (green).
const S_TASK:    (u8,u8,u8) = (0x61, 0xC5, 0x54);

/// Terminal window title bar.
const WIN_BG:    (u8,u8,u8) = (0x2C, 0x2C, 0x36);
/// Window title-bar border / separator line.
const WIN_BOR:   (u8,u8,u8) = (0x44, 0x44, 0x58);
/// Window title text.
const WIN_FG:    (u8,u8,u8) = (0xC0, 0xC0, 0xC8);

/// Traffic-light close button (red).
const BTN_R:     (u8,u8,u8) = (0xEC, 0x6A, 0x5E);
/// Traffic-light minimise button (yellow).
const BTN_Y:     (u8,u8,u8) = (0xF4, 0xBF, 0x4F);
/// Traffic-light maximise button (green).
const BTN_G:     (u8,u8,u8) = (0x61, 0xC5, 0x54);
/// Button border ring — very dark.
const BTN_BOR:   (u8,u8,u8) = (0x18, 0x18, 0x18);

/// Terminal content background.
const TERM_BG_C: (u8,u8,u8) = (0x14, 0x14, 0x1C);
/// Terminal default foreground text.
const TERM_FG:   (u8,u8,u8) = (0xCC, 0xCC, 0xCC);

// ANSI 16-colour palette
const ANSI_COLORS: [(u8,u8,u8); 16] = [
    (0x00, 0x00, 0x00), // 0: Black
    (0xCC, 0x00, 0x00), // 1: Red
    (0x00, 0xCC, 0x00), // 2: Green
    (0xCC, 0xCC, 0x00), // 3: Yellow
    (0x00, 0x00, 0xCC), // 4: Blue
    (0xCC, 0x00, 0xCC), // 5: Magenta
    (0x00, 0xCC, 0xCC), // 6: Cyan
    (0xCC, 0xCC, 0xCC), // 7: White (light grey)
    (0x66, 0x66, 0x66), // 8: Bright Black (dark grey)
    (0xFF, 0x00, 0x00), // 9: Bright Red
    (0x00, 0xFF, 0x00), // 10: Bright Green
    (0xFF, 0xFF, 0x00), // 11: Bright Yellow
    (0x55, 0x55, 0xFF), // 12: Bright Blue
    (0xFF, 0x55, 0xFF), // 13: Bright Magenta
    (0x55, 0xFF, 0xFF), // 14: Bright Cyan
    (0xFF, 0xFF, 0xFF), // 15: Bright White
];

fn color_from_idx(idx: u8) -> (u8,u8,u8) {
    if idx == 0 { TERM_FG }
    else if (idx as usize) <= ANSI_COLORS.len() { ANSI_COLORS[(idx - 1) as usize] }
    else { TERM_FG }
}

// ── Drawing helpers ───────────────────────────────────────────────────────────

/// Draw a filled circle (traffic-light style) using distance formula.
/// `r` is the outer radius; the inner `r-1` area is filled with `fill`,
/// the outer ring pixel is filled with `border` for a subtle outline.
fn draw_circle(f: &mut fb::Framebuffer,
               cx: i32, cy: i32, r: i32,
               fill: (u8,u8,u8), border: (u8,u8,u8)) {
    let r2  = r * r;
    let ir2 = (r - 1).max(0) * (r - 1).max(0);
    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = dx * dx + dy * dy;
            if d2 <= r2 {
                let px = (cx + dx) as usize;
                let py = (cy + dy) as usize;
                let (pr, pg, pb) = if d2 > ir2 { border } else { fill };
                f.put_pixel(px, py, pr, pg, pb);
            }
        }
    }
}

/// Draw a rounded-corner pill button (2-pixel corner clipping).
/// Returns the x-coordinate immediately after the button.
fn draw_pill(f: &mut fb::Framebuffer,
             x: usize, y: usize, w: usize, h: usize,
             bg: (u8,u8,u8), fg: (u8,u8,u8),
             label: &str,
             panel_bg: (u8,u8,u8)) -> usize {
    f.fill_rect(x, y, w, h, bg.0, bg.1, bg.2);
    // Clip 2×2 pixel corners to simulate rounded look
    for cy in 0..2usize {
        for cx in 0..2usize {
            f.put_pixel(x + cx,         y + cy,         panel_bg.0, panel_bg.1, panel_bg.2);
            f.put_pixel(x + w - 1 - cx, y + cy,         panel_bg.0, panel_bg.1, panel_bg.2);
            f.put_pixel(x + cx,         y + h - 1 - cy, panel_bg.0, panel_bg.1, panel_bg.2);
            f.put_pixel(x + w - 1 - cx, y + h - 1 - cy, panel_bg.0, panel_bg.1, panel_bg.2);
        }
    }
    let tx = x + (w.saturating_sub(label.len() * FONT_W)) / 2;
    let ty = y + (h.saturating_sub(FONT_H)) / 2;
    f.draw_str(tx, ty, label, fg, bg);
    x + w
}

// ── Public: update username displayed in the titlebar and system tray ─────────

/// Call this from the login/`su` flow whenever the active user changes.
pub fn set_title_user(name: &str) {
    unsafe {
        let bytes = name.as_bytes();
        let len   = bytes.len().min(31);
        TITLE_USER[..len].copy_from_slice(&bytes[..len]);
        TITLE_USER_LEN = len;
    }
    // Trigger a visual refresh if the framebuffer is ready
    if fb::is_available() {
        fb::with(|f| {
            draw_win_titlebar(f);
            refresh_top_panel_right(f, unsafe { TICK });
        });
    }
}

// ── Mouse input ───────────────────────────────────────────────────────────────

/// Called from the PS/2 mouse IRQ handler every time a complete 3-byte packet
/// has been decoded.  Moves the cursor and fires click actions.
pub fn mouse_event(dx: i16, dy: i16, left: bool, right: bool) {
    if !fb::is_available() { return; }
    // Delegate to multi-window compositor when any WM windows are open
    if wm_is_active() {
        wm_mouse_event(dx, dy, left, right);
        return;
    }
    unsafe {
        erase_cursor();

        let w  = fb::width();
        let h  = fb::height();
        // Scale raw PS/2 deltas ×2 for comfortable pointer speed
        let nx = (CURSOR_X as i32 + dx as i32 * 2).clamp(0, w as i32 - 1) as usize;
        let ny = (CURSOR_Y as i32 + dy as i32 * 2).clamp(0, h as i32 - 1) as usize;
        CURSOR_X = nx;
        CURSOR_Y = ny;

        // Detect left-click rising edge (button pressed this event, not last)
        let click = left && !PREV_LEFT;
        PREV_LEFT = left;

        draw_cursor();

        // Handle click *after* cursor is redrawn
        if click { check_click(nx, ny); }
    }
}

/// Hit-test a left-click against interactive desktop regions.
fn check_click(x: usize, y: usize) {
    let titlebar_cy = (TOP_H + TITLEBAR_H / 2) as i32;
    let ix = x as i32;
    let iy = y as i32;

    // ── Top panel (0 .. TOP_H) ────────────────────────────────────────────────
    if iy < TOP_H as i32 {
        // "NodeAI" pill button (x ≈ 4 .. 76) → toggle launcher
        if ix >= 4 && ix < 76 {
            launcher_toggle();
            return;
        }
        // "Terminal" label (x ≈ 86 .. 86 + 8*8 = 150) → return to terminal.
        // Closes the launcher if open, or closes any active app window.
        if ix >= 80 && ix < 160 {
            if unsafe { LAUNCHER_OPEN } {
                launcher_toggle(); // close launcher
            } else if unsafe { ACTIVE_APP != ActiveApp::Terminal } {
                close_app_window();
            }
        }
        return;
    }

    // ── Launcher overlay (open): all clicks below TOP_H go to it ─────────────
    if unsafe { LAUNCHER_OPEN } {
        if let Some(idx) = launcher_tile_at(x, y) {
            launcher_toggle(); // close the launcher first
            open_app_window(idx);
        }
        return; // all launcher-area clicks handled or ignored
    }

    // ── Active app window: route titlebar buttons + content clicks ──────────
    if unsafe { ACTIVE_APP != ActiveApp::Terminal } {
        let titlebar_cy = (TOP_H + TITLEBAR_H / 2) as i32;
        if iy >= TOP_H as i32 && iy < TERM_Y as i32 {
            // Close button (red ●) closes the app
            if (ix - 18) * (ix - 18) + (iy - titlebar_cy) * (iy - titlebar_cy) <= 49 {
                close_app_window();
            }
        } else if iy >= TERM_Y as i32 {
            unsafe {
                match ACTIVE_APP {
                    ActiveApp::FileManager => fm_click(x, y),
                    _ => {}
                }
            }
        }
        return;
    }

    // ── Titlebar traffic-light buttons (TOP_H .. TOP_H+TITLEBAR_H) ───────────
    if iy >= (TOP_H + TITLEBAR_H) as i32 { return; }

    // Close button (red ● at x=18) — clear terminal + reprint prompt
    if (ix - 18) * (ix - 18) + (iy - titlebar_cy) * (iy - titlebar_cy) <= 49 {
        clear_terminal();
        crate::shell::reprint_prompt();
        return;
    }
    // Minimize button (yellow ● at x=36) — stub
    if (ix - 36) * (ix - 36) + (iy - titlebar_cy) * (iy - titlebar_cy) <= 49 {
        return;
    }
    // Maximize button (green ● at x=54) — stub (already full-screen)
    if (ix - 54) * (ix - 54) + (iy - titlebar_cy) * (iy - titlebar_cy) <= 49 {
        return;
    }
    // "..." dots menu button — right side of titlebar (x ≈ w-30 .. w-6)
    let w = fb::width() as i32;
    if ix >= w - 30 && ix < w - 6 {
        crate::shell::print_sysinfo_banner();
        return;
    }
}

/// Erase the cursor from the framebuffer by redrawing every zone the sprite touched.
/// Handles both normal desktop and the launcher overlay.
/// Must be called inside an `unsafe` block (accesses mutable statics).
unsafe fn erase_cursor() {
    if !CURSOR_DRAWN { return; }
    CURSOR_DRAWN = false;

    let cy     = CURSOR_Y;
    let cy_bot = cy.saturating_add(CURSOR_H);

    // Zone 1: top panel — same whether launcher is open or not.
    if cy < TOP_H {
        fb::with(|f| draw_top_panel(f, TICK));
    }

    if LAUNCHER_OPEN {
        // Launcher covers everything below the top panel.
        if cy_bot > TOP_H {
            fb::with(|f| draw_launcher_overlay(f));
        }
        return;
    }

    if ACTIVE_APP != ActiveApp::Terminal {
        // Restore the pixels saved before the cursor was drawn.
        // This avoids repainting the entire app window on every mouse move.
        fb::with(|f| {
            let w = f.width();
            let h = f.height();
            for sy in 0..CURSOR_SAVE_H {
                for sx in 0..CURSOR_SAVE_W {
                    let px = CURSOR_SAVE_X + sx;
                    let py = CURSOR_SAVE_Y + sy;
                    if px < w && py < h {
                        let (r, g, b) = CURSOR_SAVE[sy * CURSOR_SAVE_W + sx];
                        f.put_pixel(px, py, r, g, b);
                    }
                }
            }
        });
        return;
    }

    // Zone 2: titlebar
    if cy_bot > TOP_H && cy < TOP_H + TITLEBAR_H {
        fb::with(|f| draw_win_titlebar(f));
    }
    // Zone 3: terminal content
    // redraw_terminal_line calls fb::with() itself — must NOT nest inside fb::with().
    if cy_bot > TERM_Y {
        let w  = fb::width();
        let h  = fb::height();
        let px_y0 = cy.max(TERM_Y);
        let px_y1 = cy_bot.min(h);
        if px_y1 > px_y0 {
            fb::with(|f| f.fill_rect(0, px_y0, w, px_y1 - px_y0,
                                     TERM_BG_C.0, TERM_BG_C.1, TERM_BG_C.2));
        }
        let start   = cy.max(TERM_Y).saturating_sub(TERM_Y);
        let row0    = start / FONT_H;
        let row_end = (start + CURSOR_H) / FONT_H + 1;
        let max_r   = term_rows();
        for r in row0..row_end.min(max_r) { redraw_terminal_line(r); }
    }
}

/// Paint the cursor sprite at (CURSOR_X, CURSOR_Y).
/// Saves the pixels it will overwrite so erase_cursor can restore them exactly,
/// eliminating the need to repaint the entire app window on each mouse move.
/// Must be called inside an `unsafe` block.
unsafe fn draw_cursor() {
    let x0 = CURSOR_X;
    let y0 = CURSOR_Y;
    CURSOR_SAVE_X = x0;
    CURSOR_SAVE_Y = y0;
    fb::with(|f| {
        let w = f.width();
        let h = f.height();
        // Save the bounding rectangle (CURSOR_W+1) × (CURSOR_H+1) pixels.
        for sy in 0..CURSOR_SAVE_H {
            for sx in 0..CURSOR_SAVE_W {
                let px = x0 + sx;
                let py = y0 + sy;
                CURSOR_SAVE[sy * CURSOR_SAVE_W + sx] =
                    if px < w && py < h { f.get_pixel(px, py) } else { (0, 0, 0) };
            }
        }
        // Draw cursor sprite.
        for row in 0..CURSOR_H {
            let bits = CURSOR_BITS[row];
            for col in 0..CURSOR_W {
                if (bits >> (7 - col)) & 1 != 0 {
                    let px = x0 + col;
                    let py = y0 + row;
                    if px < w && py < h {
                        if px + 1 < w && py + 1 < h {
                            f.put_pixel(px + 1, py + 1, 0x00, 0x00, 0x00);
                        }
                        f.put_pixel(px, py, 0xFF, 0xFF, 0xFF);
                    }
                }
            }
        }
    });
    CURSOR_DRAWN = true;
}

// ── App launcher helpers ──────────────────────────────────────────────────────

/// Returns true if `name` contains `search` (case-insensitive ASCII).
fn app_name_matches(name: &str, search: &[u8]) -> bool {
    if search.is_empty() { return true; }
    let nb = name.as_bytes();
    if nb.len() < search.len() { return false; }
    'outer: for start in 0..=(nb.len() - search.len()) {
        for i in 0..search.len() {
            let a  = if nb[start+i] >= b'A' && nb[start+i] <= b'Z' { nb[start+i] + 32 } else { nb[start+i] };
            let b2 = if search[i]   >= b'A' && search[i]   <= b'Z' { search[i]   + 32 } else { search[i]   };
            if a != b2 { continue 'outer; }
        }
        return true;
    }
    false
}

/// Returns the grid start x/y origin based on current framebuffer width.
/// Returns the tile grid origin given the screen width (pass f.width() when inside fb::with).
fn launcher_grid_origin(screen_w: usize) -> (usize, usize) {
    let total_w = LAUNCHER_COLS * TILE_W + (LAUNCHER_COLS - 1) * TILE_GAP;
    let gx = screen_w.saturating_sub(total_w) / 2;
    let gy = TOP_H + 32 + SEARCHBAR_H + 22; // below search bar
    (gx, gy)
}

/// Returns the APPS index of the tile at screen position (x, y), or None.
fn launcher_tile_at(x: usize, y: usize) -> Option<usize> {
    // When called outside fb::with(), use the free-standing fb::width() — no deadlock here.
    let (gx, gy) = launcher_grid_origin(fb::width());
    if x < gx || y < gy { return None; }
    let rx = x - gx;
    let ry = y - gy;
    let col   = rx / (TILE_W + TILE_GAP);
    let row   = ry / (TILE_H + TILE_GAP);
    let c_off = rx % (TILE_W + TILE_GAP);
    let r_off = ry % (TILE_H + TILE_GAP);
    if c_off >= TILE_W || r_off >= TILE_H || col >= LAUNCHER_COLS { return None; }
    let idx = row * LAUNCHER_COLS + col;
    if idx >= APPS.len() { return None; }
    let search = unsafe { &LAUNCHER_SEARCH[..LAUNCHER_SEARCH_LEN] };
    if app_name_matches(APPS[idx].name, search) { Some(idx) } else { None }
}

/// Draw the full launcher overlay (covers titlebar + terminal area).
fn draw_launcher_overlay(f: &mut fb::Framebuffer) {
    let w = f.width();
    let h = f.height();

    // Background covering everything below the top panel
    f.fill_rect(0, TOP_H, w, h.saturating_sub(TOP_H), LAUNCH_BG.0, LAUNCH_BG.1, LAUNCH_BG.2);

    // Title
    let title = "Applications";
    let title_x = (w.saturating_sub(title.len() * FONT_W)) / 2;
    f.draw_str(title_x, TOP_H + 8, title, TOP_FG, LAUNCH_BG);

    // Search bar
    let search_w: usize = 400.min(w.saturating_sub(20));
    let search_x = (w.saturating_sub(search_w)) / 2;
    let search_y = TOP_H + 32;
    f.fill_rect(search_x, search_y, search_w, SEARCHBAR_H, LAUNCH_SBG.0, LAUNCH_SBG.1, LAUNCH_SBG.2);
    // border
    f.fill_rect(search_x, search_y, search_w, 1, ACCENT.0, ACCENT.1, ACCENT.2);
    f.fill_rect(search_x, search_y + SEARCHBAR_H - 1, search_w, 1, ACCENT.0, ACCENT.1, ACCENT.2);
    f.fill_rect(search_x, search_y, 1, SEARCHBAR_H, ACCENT.0, ACCENT.1, ACCENT.2);
    f.fill_rect(search_x + search_w - 1, search_y, 1, SEARCHBAR_H, ACCENT.0, ACCENT.1, ACCENT.2);
    // hint text or typed search
    let text_y = search_y + (SEARCHBAR_H.saturating_sub(FONT_H)) / 2;
    let search_str = unsafe { core::str::from_utf8(&LAUNCHER_SEARCH[..LAUNCHER_SEARCH_LEN]).unwrap_or("") };
    if search_str.is_empty() {
        f.draw_str(search_x + 8, text_y, "Search apps...", TOP_DIM, LAUNCH_SBG);
    } else {
        f.draw_str(search_x + 8, text_y, search_str, LAUNCH_SFG, LAUNCH_SBG);
        // blinking cursor indicator
        let cur_x = (search_x + 8 + search_str.len() * FONT_W).min(search_x + search_w - 4);
        f.fill_rect(cur_x, text_y, 2, FONT_H, ACCENT.0, ACCENT.1, ACCENT.2);
    }

    // App tiles — fixed 3×2 grid, non-matching tiles are dimmed
    // Use f.width() here — we are already inside fb::with(), so fb::width() would deadlock.
    let (gx, gy) = launcher_grid_origin(w);
    let search_bytes = unsafe { &LAUNCHER_SEARCH[..LAUNCHER_SEARCH_LEN] };
    for (idx, app) in APPS.iter().enumerate() {
        let col = idx % LAUNCHER_COLS;
        let row = idx / LAUNCHER_COLS;
        let tx  = gx + col * (TILE_W + TILE_GAP);
        let ty  = gy + row * (TILE_H + TILE_GAP);
        let visible = app_name_matches(app.name, search_bytes);
        if !visible {
            f.fill_rect(tx, ty, TILE_W, TILE_H, LAUNCH_DIM.0, LAUNCH_DIM.1, LAUNCH_DIM.2);
        } else {
            // Tile body
            f.fill_rect(tx, ty, TILE_W, TILE_H, LAUNCH_TBG.0, LAUNCH_TBG.1, LAUNCH_TBG.2);
            // Accent top strip
            f.fill_rect(tx, ty, TILE_W, 3, ACCENT.0, ACCENT.1, ACCENT.2);
            // 2-char icon, vertically centred in top half
            let icon_x = tx + (TILE_W.saturating_sub(app.icon.len() * FONT_W)) / 2;
            let icon_y = ty + (TILE_H / 2).saturating_sub(FONT_H + 2);
            f.draw_str(icon_x, icon_y, app.icon, LAUNCH_IFG, LAUNCH_TBG);
            // App name centred in bottom half
            let name_w = app.name.len() * FONT_W;
            let name_x = if name_w >= TILE_W { tx + 2 } else { tx + (TILE_W - name_w) / 2 };
            let name_y = ty + TILE_H - FONT_H - 8;
            f.draw_str(name_x, name_y, app.name, LAUNCH_NFG, LAUNCH_TBG);
        }
    }
}

// ── Launcher public API ───────────────────────────────────────────────────────

/// Toggle the application launcher overlay open/closed.
pub fn launcher_toggle() {
    unsafe {
        LAUNCHER_OPEN = !LAUNCHER_OPEN;
        if LAUNCHER_OPEN {
            fb::with(|f| draw_launcher_overlay(f));
        } else {
            // Reset search and restore the desktop below the top panel
            LAUNCHER_SEARCH_LEN = 0;
            LAUNCHER_SEARCH = [0u8; 32];
            fb::with(|f| {
                let w = f.width();
                let h = f.height();
                draw_win_titlebar(f);
                f.fill_rect(0, TERM_Y, w, h.saturating_sub(TERM_Y),
                            TERM_BG_C.0, TERM_BG_C.1, TERM_BG_C.2);
            });
            let rows = term_rows();
            for r in 0..rows { redraw_terminal_line(r); }
        }
        draw_cursor();
    }
}

/// Returns true when the launcher overlay is visible.
#[inline]
pub fn launcher_is_open() -> bool { unsafe { LAUNCHER_OPEN } }

/// Handle a raw keypress while the launcher is open.
/// 0x08=Backspace  0x1B=ESC(close)  b'\n'=launch-first-match  else=search char
pub fn launcher_key(b: u8) {
    unsafe {
        match b {
            0x1B => {
                launcher_toggle(); // ESC closes the launcher
            }
            0x08 => {
                if LAUNCHER_SEARCH_LEN > 0 {
                    LAUNCHER_SEARCH_LEN -= 1;
                    LAUNCHER_SEARCH[LAUNCHER_SEARCH_LEN] = 0;
                    fb::with(|f| draw_launcher_overlay(f));
                    draw_cursor();
                }
            }
            0x0A | 0x0D => {   // Enter: launch first matching app
                let search = &LAUNCHER_SEARCH[..LAUNCHER_SEARCH_LEN];
                let mut first_cmd: Option<&'static str> = None;
                for app in APPS.iter() {
                    if app_name_matches(app.name, search) {
                        first_cmd = Some(app.cmd);
                        break;
                    }
                }
                launcher_toggle(); // close first
                if let Some(cmd) = first_cmd {
                    if !cmd.is_empty() { crate::shell::launch_app(cmd); }
                }
            }
            0x20..=0x7E => {
                if LAUNCHER_SEARCH_LEN < 32 {
                    LAUNCHER_SEARCH[LAUNCHER_SEARCH_LEN] = b;
                    LAUNCHER_SEARCH_LEN += 1;
                    fb::with(|f| draw_launcher_overlay(f));
                    draw_cursor();
                }
            }
            _ => {}
        }
    }
}

// ── Initialise ────────────────────────────────────────────────────────────────

/// Initialise and paint the full desktop.  Must be called after `framebuffer::init()`.
pub fn init() {
    if !fb::is_available() { return; }
    // Initialise the multi-window compositor
    wm_init();
    unsafe {
        CURSOR_X = fb::width() / 2;
        CURSOR_Y = fb::height() / 2;
    }
    fb::with(|f| {
        let w = f.width();
        let h = f.height();
        // Fill everything with desktop background first
        f.fill_rect(0, 0, w, h, DESK_BG.0, DESK_BG.1, DESK_BG.2);
        draw_top_panel(f, 0);
        draw_win_titlebar(f);
        // Terminal content area
        f.fill_rect(0, TERM_Y, w, h.saturating_sub(TERM_Y), TERM_BG_C.0, TERM_BG_C.1, TERM_BG_C.2);
    });
    // Draw initial cursor
    unsafe { draw_cursor(); }
}

/// Called from the idle loop to perform any pending browser fetch outside of
/// interrupt context.  This prevents br_fetch()'s blocking spin-loops from
/// locking up the screen / keyboard inside the keyboard IRQ handler.
pub fn browser_fetch_tick() {
    use core::sync::atomic::Ordering;
    if !BR_FETCH_PENDING.load(Ordering::Acquire) { return; }
    BR_FETCH_PENDING.store(false, Ordering::Release);
    let url: alloc::string::String = unsafe {
        let len = BR_FETCH_URL_LEN.min(BR_FETCH_URL.len());
        core::str::from_utf8(&BR_FETCH_URL[..len]).unwrap_or("").into()
    };
    if url.is_empty() {
        unsafe { br_set_status(b"Empty URL"); }
        fb::with(|f| unsafe { draw_browser_overlay(f); });
        return;
    }
    unsafe {
        let rows = br_fetch(&url);
        if rows == 0 { /* error already set inside br_fetch */ }
        fb::with(|f| { draw_win_titlebar(f); draw_browser_overlay(f); });
    }
}

/// Called every kernel timer tick to refresh dynamic UI elements.
pub fn tick(ticks_since_boot: u64) {
    if !fb::is_available() { return; }
    unsafe { TICK = ticks_since_boot; }
    // Update WM taskbar clock if any windows are open
    if wm_is_active() {
        wm_tick(ticks_since_boot);
    } else {
        fb::with(|f| { refresh_top_panel_right(f, ticks_since_boot); });
    }
    // Tick system monitor if running
    sysmon_tick();
}

/// Redraw a given terminal line by its row index (for shell line-editing).
pub fn terminal_redraw_line(row: usize) {
    if !fb::is_available() { return; }
    unsafe { redraw_terminal_line(row); }
}

/// Get current terminal column.
pub fn terminal_col() -> usize { unsafe { TERM_COL } }

/// Get current terminal row.
pub fn terminal_row() -> usize { unsafe { TERM_ROW } }


/// Set terminal column directly (for cursor repositioning).
pub fn terminal_set_col(col: usize) {
    unsafe { TERM_COL = col; }
}

/// Write a character at the current cursor position with the current color,
/// then advance the cursor. Does NOT trigger a line redraw.
pub fn terminal_put_char(byte: u8) {
    unsafe {
        if TERM_COL < term_cols() {
            TERM_BUF[TERM_ROW][TERM_COL] = byte;
            TERM_COLOR[TERM_ROW][TERM_COL] = CUR_COLOR_IDX;
            TERM_COL += 1;
        }
    }
}

/// Clear from current column to end of current line in the buffer.
pub fn terminal_clear_to_eol() {
    unsafe {
        let cols = term_cols();
        for c in TERM_COL..cols.min(TERM_COLS_MAX) {
            TERM_BUF[TERM_ROW][c] = 0;
            TERM_COLOR[TERM_ROW][c] = 0;
        }
    }
}

/// Push one ASCII byte into the terminal region, with ANSI escape sequence support.
pub fn terminal_input(byte: u8) {
    if !fb::is_available() { return; }
    unsafe {
        match ANSI_STATE {
            AnsiState::Normal => {
                if byte == 0x1B {
                    // ESC — start escape sequence
                    ANSI_STATE = AnsiState::Esc;
                    ANSI_LEN = 0;
                    return;
                }
                emit_byte(byte);
            }
            AnsiState::Esc => {
                if byte == b'[' {
                    ANSI_STATE = AnsiState::Csi;
                } else {
                    // Unknown escape, ignore and reset
                    ANSI_STATE = AnsiState::Normal;
                }
            }
            AnsiState::Csi => {
                if byte >= b'0' && byte <= b'9' || byte == b';' {
                    // Parameter byte — accumulate
                    if ANSI_LEN < 32 {
                        ANSI_BUF[ANSI_LEN] = byte;
                        ANSI_LEN += 1;
                    }
                } else {
                    // Final byte — dispatch
                    process_csi(byte);
                    ANSI_STATE = AnsiState::Normal;
                }
            }
        }
    }
}

/// Emit a raw byte to the terminal (no escape processing).
unsafe fn emit_byte(byte: u8) {
    match byte {
        b'\n' | b'\r' => {
            TERM_COL = 0;
            TERM_ROW += 1;
            if TERM_ROW >= term_rows() { scroll_terminal(); }
        }
        0x08 | 0x7F => {
            // Backspace: erase the previous character
            if TERM_COL > 0 {
                TERM_COL -= 1;
                TERM_BUF[TERM_ROW][TERM_COL] = 0;
                TERM_COLOR[TERM_ROW][TERM_COL] = 0;
            }
        }
        _ => {
            if TERM_COL < term_cols() {
                TERM_BUF[TERM_ROW][TERM_COL] = byte;
                TERM_COLOR[TERM_ROW][TERM_COL] = CUR_COLOR_IDX;
                TERM_COL += 1;
            }
        }
    }
    redraw_terminal_line(TERM_ROW.saturating_sub(if byte == b'\n' { 1 } else { 0 }));
}

/// Process a CSI (Control Sequence Introducer) escape sequence.
unsafe fn process_csi(final_byte: u8) {
    let params_str = core::str::from_utf8(&ANSI_BUF[..ANSI_LEN]).unwrap_or("");
    match final_byte {
        b'm' => {
            // SGR — Select Graphic Rendition
            if params_str.is_empty() || params_str == "0" {
                CUR_COLOR_IDX = 0; // reset
            } else {
                for part in params_str.split(';') {
                    if let Ok(n) = part.parse::<u32>() {
                        match n {
                            0 => CUR_COLOR_IDX = 0,
                            1 => {} // bold — ignore for now
                            30..=37 => CUR_COLOR_IDX = (n - 30 + 1) as u8,
                            90..=97 => CUR_COLOR_IDX = (n - 90 + 9) as u8,
                            _ => {} // ignore BG, 256-color, etc. for now
                        }
                    }
                }
            }
        }
        b'J' => {
            // Erase in display: CSI 2J = clear screen
            let n = params_str.parse::<u32>().unwrap_or(0);
            if n == 2 {
                for row in &mut TERM_BUF { *row = [0u8; TERM_COLS_MAX]; }
                for row in &mut TERM_COLOR { *row = [0u8; TERM_COLS_MAX]; }
                TERM_ROW = 0;
                TERM_COL = 0;
                fb::with(|f| {
                    let w = f.width();
                    let h = f.height();
                    f.fill_rect(0, TERM_Y, w, h.saturating_sub(TERM_Y),
                                TERM_BG_C.0, TERM_BG_C.1, TERM_BG_C.2);
                });
            }
        }
        b'K' => {
            // Erase in line: CSI 0K = clear to end of line
            let n = params_str.parse::<u32>().unwrap_or(0);
            if n == 0 {
                let cols = term_cols();
                for c in TERM_COL..cols.min(TERM_COLS_MAX) {
                    TERM_BUF[TERM_ROW][c] = 0;
                    TERM_COLOR[TERM_ROW][c] = 0;
                }
                redraw_terminal_line(TERM_ROW);
            }
        }
        _ => {} // ignore unknown CSI sequences
    }
}

/// Clear the terminal region and reset cursor to the top-left.
pub fn clear_terminal() {
    if !fb::is_available() { return; }
    unsafe {
        for row in &mut TERM_BUF   { *row = [0u8; TERM_COLS_MAX]; }
        for row in &mut TERM_COLOR { *row = [0u8; TERM_COLS_MAX]; }
        TERM_ROW = 0;
        TERM_COL = 0;
        CUR_COLOR_IDX = 0;
    }
    fb::with(|f| {
        let w = f.width();
        let h = f.height();
        f.fill_rect(0, TERM_Y, w, h.saturating_sub(TERM_Y),
                    TERM_BG_C.0, TERM_BG_C.1, TERM_BG_C.2);
    });
}

// ── Top panel (GNOME-style) ───────────────────────────────────────────────────

/// Draw the full top panel (static layout + dynamic clock/stats).
/// Pass `tick` from `TICK` to show the current time; pass 0 on first paint.
fn draw_top_panel(f: &mut fb::Framebuffer, tick: u64) {
    let w = f.width();
    // Solid near-black panel background
    f.fill_rect(0, 0, w, TOP_H, TOP_BG.0, TOP_BG.1, TOP_BG.2);
    // 1-pixel bottom border/shadow
    f.fill_rect(0, TOP_H - 1, w, 1, TOP_SEP.0, TOP_SEP.1, TOP_SEP.2);

    // LEFT: Activities pill button ("NodeAI")
    let label    = " NodeAI ";
    let pill_w   = label.len() * FONT_W + 8;
    let pill_y   = (TOP_H - 26) / 2;
    let after_x  = draw_pill(f, 4, pill_y, pill_w, 26, ACCENT, ACCENT_FG, label, TOP_BG);

    // Current app name (dimmed, immediately after pill)
    let app_x = after_x + 10;
    let text_y = (TOP_H - FONT_H) / 2;
    f.draw_str(app_x, text_y, "Terminal", TOP_DIM, TOP_BG);
    // Separator after app name
    let sep_x = app_x + 8 * FONT_W + 6;
    f.fill_rect(sep_x, 6, 1, TOP_H - 12, TOP_SEP.0, TOP_SEP.1, TOP_SEP.2);

    // Dynamic right zone (clock, memory, username)
    refresh_top_panel_right(f, tick);
}

/// Refresh the dynamic right zone of the top panel every tick:
/// clock (centred) + memory / task-count / username (right-aligned).
fn refresh_top_panel_right(f: &mut fb::Framebuffer, tick: u64) {
    let w      = f.width();
    let text_y = (TOP_H - FONT_H) / 2;

    let secs    = tick / 1000;
    let hh      = (secs / 3600) % 24;
    let mm      = (secs / 60) % 60;
    let ss      =  secs % 60;
    let free_mb = crate::memory::free_mb();
    let tasks   = crate::scheduler::task_count();

    // Clear everything from the midpoint rightward so redraw is clean
    let clear_x = w / 2 - 60;
    f.fill_rect(clear_x, 0, w - clear_x, TOP_H - 1, TOP_BG.0, TOP_BG.1, TOP_BG.2);

    // ── CENTRE: clock ──────────────────────────────────────────────────────────
    // "HH:MM:SS" = 8 chars = 64px, centred on screen
    let clock_x = (w - 8 * FONT_W) / 2;
    f.draw_fmt(clock_x, text_y, CLOCK_FG, TOP_BG,
               format_args!("{:02}:{:02}:{:02}", hh, mm, ss));

    // ── RIGHT: username ────────────────────────────────────────────────────────
    let username = unsafe {
        core::str::from_utf8(&TITLE_USER[..TITLE_USER_LEN]).unwrap_or("root")
    };
    let ux = w.saturating_sub(username.len() * FONT_W + 10);
    f.draw_str(ux, text_y, username, TOP_FG, TOP_BG);

    // Separator before username
    let s1 = ux.saturating_sub(8);
    f.fill_rect(s1, 6, 1, TOP_H - 12, TOP_SEP.0, TOP_SEP.1, TOP_SEP.2);

    // Memory "256M"
    let mx = s1.saturating_sub(5 * FONT_W + 10);
    f.draw_fmt(mx, text_y, S_MEM, TOP_BG, format_args!("{}M", free_mb));

    // Separator
    let s2 = mx.saturating_sub(8);
    f.fill_rect(s2, 6, 1, TOP_H - 12, TOP_SEP.0, TOP_SEP.1, TOP_SEP.2);

    // Task count "tasks:N"
    let tx = s2.saturating_sub(8 * FONT_W + 10);
    // Only draw if it fits after the clock rightward
    if tx > clock_x + 8 * FONT_W + 16 {
        f.draw_fmt(tx, text_y, S_TASK, TOP_BG, format_args!("tasks:{}", tasks));
    }
}

// ── Terminal window title bar ─────────────────────────────────────────────────

fn draw_win_titlebar(f: &mut fb::Framebuffer) {
    let w  = f.width();
    let ty = TOP_H;  // y start for titlebar

    // Background
    f.fill_rect(0, ty, w, TITLEBAR_H, WIN_BG.0, WIN_BG.1, WIN_BG.2);
    // Bottom border
    f.fill_rect(0, ty + TITLEBAR_H - 1, w, 1, WIN_BOR.0, WIN_BOR.1, WIN_BOR.2);

    // ── Traffic-light buttons ──────────────────────────────────────────────────
    let cy = (ty + TITLEBAR_H / 2) as i32;
    draw_circle(f, 18, cy, 7, BTN_R, BTN_BOR);
    draw_circle(f, 36, cy, 7, BTN_Y, BTN_BOR);
    draw_circle(f, 54, cy, 7, BTN_G, BTN_BOR);

    // ── Centred title — depends on active app ─────────────────────────────────
    let title_y = ty + (TITLEBAR_H - FONT_H) / 2;
    match unsafe { ACTIVE_APP } {
        ActiveApp::Terminal => {
            let username = unsafe {
                core::str::from_utf8(&TITLE_USER[..TITLE_USER_LEN]).unwrap_or("root")
            };
            let title_chars = 11 + username.len() + 7;
            let title_x = (w.saturating_sub(title_chars * FONT_W)) / 2;
            let mut x = f.draw_str(title_x, title_y, "Terminal - ", WIN_FG, WIN_BG);
            x = f.draw_str(x, title_y, username, colour::YELLOW, WIN_BG);
            f.draw_str(x, title_y, "@nodeai", WIN_FG, WIN_BG);
        }
        ActiveApp::Notepad => {
            let fname = unsafe {
                if NP_FNAME_LEN > 0 {
                    core::str::from_utf8(&NP_FNAME[..NP_FNAME_LEN]).unwrap_or("untitled")
                } else { "untitled" }
            };
            let dirty = unsafe { NP_DIRTY };
            let prefix = "Notepad — ";
            let suffix = if dirty { " *" } else { "" };
            let chars = prefix.len() + fname.len() + suffix.len();
            let tx = (w.saturating_sub(chars * FONT_W)) / 2;
            let mut x = f.draw_str(tx, title_y, prefix, WIN_FG, WIN_BG);
            x = f.draw_str(x, title_y, fname, colour::YELLOW, WIN_BG);
            if dirty { f.draw_str(x, title_y, " *", (0xFF,0x88,0x44), WIN_BG); }
        }
        ActiveApp::FileManager => {
            let path = unsafe {
                core::str::from_utf8(&FM_PATH[..FM_PATH_LEN]).unwrap_or("/")
            };
            let prefix = "File Manager — ";
            let chars = prefix.len() + path.len();
            let tx = (w.saturating_sub(chars * FONT_W)) / 2;
            let mut x = f.draw_str(tx, title_y, prefix, WIN_FG, WIN_BG);
            f.draw_str(x, title_y, path, FM_DIR_C, WIN_BG);
        }
        ActiveApp::Browser => {
            let title = "Intelli Browser";
            let tx = (w.saturating_sub(title.len() * FONT_W)) / 2;
            f.draw_str(tx, title_y, title, (0x44, 0xAA, 0xFF), WIN_BG);
        }
        ActiveApp::Network => {
            let title = "Network Manager";
            let tx = (w.saturating_sub(title.len() * FONT_W)) / 2;
            f.draw_str(tx, title_y, title, (0x44, 0xFF, 0xAA), WIN_BG);
        }
    }

    // ── Right: three-dot menu indicator ───────────────────────────────────────
    f.draw_str(w.saturating_sub(30), title_y, "...", TOP_DIM, WIN_BG);
}

// ── Terminal geometry & rendering ─────────────────────────────────────────────

fn term_cols() -> usize {
    (fb::width().saturating_sub(8)) / FONT_W
}

fn term_rows() -> usize {
    (fb::height().saturating_sub(TERM_Y + 4)) / FONT_H
}

unsafe fn redraw_terminal_line(row: usize) {
    if !fb::is_available() { return; }
    let cols = term_cols();
    let w    = fb::width();
    let y    = TERM_Y + 4 + row * FONT_H;
    fb::with(|f| {
        // Clear the full-width line background
        f.fill_rect(0, y, w, FONT_H, TERM_BG_C.0, TERM_BG_C.1, TERM_BG_C.2);
        let mut x = 4usize;
        for col in 0..cols.min(TERM_COLS_MAX) {
            let ch = TERM_BUF[row][col];
            if ch == 0 { break; }
            let fg = color_from_idx(TERM_COLOR[row][col]);
            x = f.draw_char(x, y, ch, fg, TERM_BG_C);
        }
    });
}

unsafe fn scroll_terminal() {
    // Shift buffer rows up by one
    let rows = TERM_ROWS_MAX - 1;
    for r in 0..rows {
        TERM_BUF[r]   = TERM_BUF[r + 1];
        TERM_COLOR[r] = TERM_COLOR[r + 1];
    }
    TERM_BUF[rows]   = [0u8; TERM_COLS_MAX];
    TERM_COLOR[rows] = [0u8; TERM_COLS_MAX];
    TERM_ROW = rows;
    TERM_COL = 0;

    // Repaint all terminal lines in ONE fb::with — redraw_terminal_line also calls
    // fb::with internally, so we must NOT nest it inside another fb::with.
    let cols = term_cols();
    let w    = fb::width();
    fb::with(|f| {
        for r in 0..TERM_ROWS_MAX {
            let y = TERM_Y + 4 + r * FONT_H;
            f.fill_rect(0, y, w, FONT_H, TERM_BG_C.0, TERM_BG_C.1, TERM_BG_C.2);
            let mut x = 4usize;
            for col in 0..cols.min(TERM_COLS_MAX) {
                let ch = TERM_BUF[r][col];
                if ch == 0 { break; }
                let fg = color_from_idx(TERM_COLOR[r][col]);
                x = f.draw_char(x, y, ch, fg, TERM_BG_C);
            }
        }
    });
}

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  GUI Application Windows — Notepad · File Manager · Browser                ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Write `n` as decimal into `buf`; returns the ASCII slice.
fn fmt_usize(n: usize, buf: &mut [u8; 20]) -> &str {
    if n == 0 {
        buf[0] = b'0';
        return core::str::from_utf8(&buf[0..1]).unwrap_or("0");
    }
    let mut i = 20usize;
    let mut v = n;
    while v > 0 && i > 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    core::str::from_utf8(&buf[i..]).unwrap_or("?")
}

/// Format a u64 byte-count into a human-readable "N KB" / "N MB" / "N B" string.
fn fmt_size(bytes: u64, buf: &mut [u8; 24]) -> &str {
    let (val, unit) = if bytes >= 1024 * 1024 {
        (bytes / (1024 * 1024), b"MB" as &[u8])
    } else if bytes >= 1024 {
        (bytes / 1024, b"KB" as &[u8])
    } else {
        (bytes,         b"B " as &[u8])
    };
    let mut nb = [0u8; 20];
    let s  = fmt_usize(val as usize, &mut nb).as_bytes();
    let sl = s.len().min(20);
    buf[..sl].copy_from_slice(&s[..sl]);
    buf[sl]     = b' ';
    buf[sl + 1] = unit[0];
    buf[sl + 2] = unit[1];
    core::str::from_utf8(&buf[..sl + 3]).unwrap_or("?")
}

/// Dispatch to the correct draw function for the current active app.
unsafe fn draw_active_app_overlay(f: &mut fb::Framebuffer) {
    match ACTIVE_APP {
        ActiveApp::Terminal    => {}
        ActiveApp::Notepad     => draw_notepad_overlay(f),
        ActiveApp::FileManager => draw_fm_overlay(f),
        ActiveApp::Browser     => draw_browser_overlay(f),
        ActiveApp::Network     => draw_netmgr_overlay(f),
    }
}

/// Close the current app window and restore the terminal.
fn close_app_window() {
    unsafe { ACTIVE_APP = ActiveApp::Terminal; }
    fb::with(|f| {
        draw_win_titlebar(f);
        let w = f.width();
        let h = f.height();
        f.fill_rect(0, TERM_Y, w, h.saturating_sub(TERM_Y),
                    TERM_BG_C.0, TERM_BG_C.1, TERM_BG_C.2);
    });
    let rows = unsafe { term_rows().min(TERM_ROWS_MAX) };
    for r in 0..rows { unsafe { redraw_terminal_line(r); } }
}

/// Open the correct GUI window for the given APPS[] index.
fn open_app_window(app_idx: usize) {
    if app_idx >= APPS.len() { return; }
    match APPS[app_idx].name {
        "File Manager" => {
            let path = crate::users::cwd();
            fm_open_gui(&path);
        }
        "Notepad" => notepad_open_gui(""),
        "Browser" => browser_open_gui(),
        "Network" => netmgr_open(),
        _ => {
            let cmd = APPS[app_idx].cmd;
            if !cmd.is_empty() { crate::shell::launch_app(cmd); }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Notepad
// ══════════════════════════════════════════════════════════════════════════════

fn notepad_open_gui(path: &str) {
    np_load(path);
    unsafe { ACTIVE_APP = ActiveApp::Notepad; }
    fb::with(|f| unsafe {
        draw_win_titlebar(f);
        draw_notepad_overlay(f);
    });
}

fn np_load(path: &str) {
    unsafe {
        NP_ROWS_USED = 1;
        NP_EDIT_ROW  = 0;
        NP_EDIT_COL  = 0;
        NP_SCROLL    = 0;
        NP_DIRTY     = false;
        for r in 0..NP_ROWS { NP_LEN[r] = 0; }
        let pb  = path.as_bytes();
        let len = pb.len().min(63);
        NP_FNAME[..len].copy_from_slice(&pb[..len]);
        NP_FNAME_LEN = len;
    }
    if path.is_empty() { return; }
    if let Ok(node) = crate::vfs::lookup(path) {
        if let Ok(mut fh) = node.open() {
            let mut buf  = [0u8; 512];
            let mut row  = 0usize;
            let mut col  = 0usize;
            'rd: loop {
                match fh.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        for &b in &buf[..n] {
                            if b == b'\n' {
                                row += 1;
                                col  = 0;
                                if row >= NP_ROWS { break 'rd; }
                            } else if b != b'\r' && col < NP_COLS {
                                unsafe { NP_BUF[row][col] = b; NP_LEN[row] = col + 1; }
                                col += 1;
                            }
                        }
                    }
                }
            }
            unsafe { NP_ROWS_USED = (row + 1).max(1); }
        }
    }
}

fn np_save() {
    let path = unsafe {
        core::str::from_utf8(&NP_FNAME[..NP_FNAME_LEN]).unwrap_or("")
    };
    if path.is_empty() { return; }
    if let Ok(node) = crate::vfs::lookup(path) {
        if let Ok(mut fh) = node.open() {
            let _ = fh.truncate(0);
            let _ = fh.seek(0);
            let rows = unsafe { NP_ROWS_USED };
            for r in 0..rows {
                let len = unsafe { NP_LEN[r] };
                let _ = fh.write(unsafe { &NP_BUF[r][..len] });
                let _ = fh.write(b"\n");
            }
            unsafe { NP_DIRTY = false; }
        }
    }
}

/// Insert a printable byte at the current cursor position.
unsafe fn np_insert(b: u8) {
    if NP_EDIT_ROW >= NP_ROWS { return; }
    let row = NP_EDIT_ROW;
    let col = NP_EDIT_COL;
    if col < NP_COLS {
        // Shift chars right from col onwards (insert mode)
        let len = NP_LEN[row].min(NP_COLS - 1);
        let mut i = len;
        while i > col { NP_BUF[row][i] = NP_BUF[row][i - 1]; i -= 1; }
        NP_BUF[row][col] = b;
        NP_LEN[row] = (NP_LEN[row] + 1).min(NP_COLS);
        NP_EDIT_COL += 1;
        NP_DIRTY = true;
    }
}

/// Handle Enter key: split current line at cursor into two lines.
unsafe fn np_newline() {
    if NP_ROWS_USED >= NP_ROWS { return; }
    let row = NP_EDIT_ROW;
    let col = NP_EDIT_COL;
    // Shift all rows below down by one
    let last = NP_ROWS_USED.min(NP_ROWS - 1);
    let mut r = last;
    while r > row + 1 {
        NP_BUF[r]  = NP_BUF[r - 1];
        NP_LEN[r]  = NP_LEN[r - 1];
        r -= 1;
    }
    // New row gets chars from col onwards
    let old_len = NP_LEN[row];
    let tail_len = old_len.saturating_sub(col);
    NP_BUF[row + 1][..tail_len].copy_from_slice(&NP_BUF[row][col..col + tail_len]);
    NP_LEN[row + 1] = tail_len;
    // Truncate current row at col
    NP_LEN[row] = col;
    for c in col..NP_COLS { NP_BUF[row][c] = 0; }
    NP_ROWS_USED = (NP_ROWS_USED + 1).min(NP_ROWS);
    NP_EDIT_ROW += 1;
    NP_EDIT_COL  = 0;
    NP_DIRTY = true;
    np_ensure_scroll_visible();
}

/// Handle Backspace.
unsafe fn np_backspace() {
    let row = NP_EDIT_ROW;
    let col = NP_EDIT_COL;
    if col > 0 {
        // Delete char before cursor
        let len = NP_LEN[row];
        let mut i = col - 1;
        while i + 1 < len { NP_BUF[row][i] = NP_BUF[row][i + 1]; i += 1; }
        if len > 0 { NP_LEN[row] -= 1; NP_BUF[row][NP_LEN[row]] = 0; }
        NP_EDIT_COL -= 1;
        NP_DIRTY = true;
    } else if row > 0 {
        // Merge this line onto previous
        let prev_len  = NP_LEN[row - 1];
        let cur_len   = NP_LEN[row];
        let merged    = prev_len + cur_len;
        if merged <= NP_COLS {
            NP_BUF[row - 1][prev_len..merged].copy_from_slice(&NP_BUF[row][..cur_len]);
            NP_LEN[row - 1] = merged;
            // Shift rows above current up
            let mut r = row;
            while r + 1 < NP_ROWS_USED {
                NP_BUF[r]  = NP_BUF[r + 1];
                NP_LEN[r]  = NP_LEN[r + 1];
                r += 1;
            }
            NP_LEN[NP_ROWS_USED - 1] = 0;
            NP_ROWS_USED = NP_ROWS_USED.saturating_sub(1).max(1);
            NP_EDIT_ROW -= 1;
            NP_EDIT_COL  = prev_len;
            NP_DIRTY = true;
        }
    }
}

unsafe fn np_ensure_scroll_visible() {
    let h = fb::height();
    let content_h  = h.saturating_sub(TERM_Y + APP_STATUS_H);
    let visible    = content_h / FONT_H;
    if NP_EDIT_ROW < NP_SCROLL {
        NP_SCROLL = NP_EDIT_ROW;
    } else if NP_EDIT_ROW >= NP_SCROLL + visible {
        NP_SCROLL = NP_EDIT_ROW.saturating_sub(visible - 1);
    }
}

unsafe fn draw_notepad_overlay(f: &mut fb::Framebuffer) {
    let w = f.width();
    let h = f.height();
    let content_h = h.saturating_sub(TERM_Y + APP_STATUS_H);
    let visible   = content_h / FONT_H;

    // Background + gutter
    f.fill_rect(0, TERM_Y, w, content_h, NP_BG.0, NP_BG.1, NP_BG.2);
    f.fill_rect(0, TERM_Y, NP_GUTTER_W, content_h, NP_GUT.0, NP_GUT.1, NP_GUT.2);
    f.fill_rect(NP_GUTTER_W, TERM_Y, 1, content_h, 0xCC, 0xCC, 0xD8);

    for vr in 0..visible {
        let dr = NP_SCROLL + vr;
        if dr >= NP_ROWS_USED { break; }
        let y = TERM_Y + vr * FONT_H;
        let is_cur = dr == NP_EDIT_ROW;
        // Current-line highlight
        if is_cur {
            f.fill_rect(0,             y, NP_GUTTER_W, FONT_H, NP_CUR.0, NP_CUR.1, NP_CUR.2);
            f.fill_rect(NP_GUTTER_W+1, y, w.saturating_sub(NP_GUTTER_W+1), FONT_H,
                        NP_CUR.0, NP_CUR.1, NP_CUR.2);
        }
        let gut_bg  = if is_cur { NP_CUR } else { NP_GUT };
        let text_bg = if is_cur { NP_CUR } else { NP_BG };
        // Line number
        let mut nb  = [0u8; 20];
        let ln_str  = fmt_usize(dr + 1, &mut nb);
        let ln_x    = NP_GUTTER_W.saturating_sub(ln_str.len() * FONT_W + 4).max(2);
        f.draw_str(ln_x, y, ln_str, NP_LN, gut_bg);
        // Text content
        let len = NP_LEN[dr].min(NP_COLS);
        if len > 0 {
            let text = core::str::from_utf8(&NP_BUF[dr][..len]).unwrap_or("");
            f.draw_str(NP_GUTTER_W + 4, y, text, NP_FG, text_bg);
        }
        // Cursor caret
        if is_cur {
            let cx = NP_GUTTER_W + 4 + NP_EDIT_COL * FONT_W;
            if cx + 2 <= w {
                f.fill_rect(cx, y, 2, FONT_H, 0x20, 0x80, 0xFF);
            }
        }
    }

    // Status bar
    let sb_y = h - APP_STATUS_H;
    f.fill_rect(0, sb_y, w, APP_STATUS_H, NP_SBG.0, NP_SBG.1, NP_SBG.2);
    let sy = sb_y + (APP_STATUS_H - FONT_H) / 2;
    let fname  = if NP_FNAME_LEN > 0 {
        core::str::from_utf8(&NP_FNAME[..NP_FNAME_LEN]).unwrap_or("???")
    } else { "untitled" };
    let mut x = f.draw_str(8, sy, fname, NP_SFG, NP_SBG);
    if NP_DIRTY { x = f.draw_str(x, sy, " *modified*", colour::YELLOW, NP_SBG); }
    let mut rb  = [0u8; 20];
    let mut cb  = [0u8; 20];
    x = f.draw_str(x, sy, "  Ln:", NP_SFG, NP_SBG);
    x = f.draw_str(x, sy, fmt_usize(NP_EDIT_ROW + 1, &mut rb), colour::WHITE, NP_SBG);
    x = f.draw_str(x, sy, " Col:", NP_SFG, NP_SBG);
    let _ = f.draw_str(x, sy, fmt_usize(NP_EDIT_COL + 1, &mut cb), colour::WHITE, NP_SBG);
    f.draw_str(w.saturating_sub(23 * FONT_W), sy, "F2:Save  F3:New  ESC:X", NP_SFG, NP_SBG);
}

/// Handle a printable or control byte in the Notepad.
pub fn notepad_key(b: u8) {
    unsafe {
        match b {
            0x1B => { close_app_window(); return; } // ESC
            0x08 => np_backspace(),
            b'\n' | 0x0D => np_newline(),
            0x20..=0x7E => np_insert(b),
            _ => {}
        }
        np_ensure_scroll_visible();
        fb::with(|f| unsafe { draw_win_titlebar(f); draw_notepad_overlay(f); });
    }
}

/// Handle a special (arrow/F-key) keystroke in the Notepad.
pub fn notepad_special(key: drivers::input::SpecialKey) {
    use drivers::input::SpecialKey::*;
    unsafe {
        match key {
            Up    => { if NP_EDIT_ROW > 0 { NP_EDIT_ROW -= 1; NP_EDIT_COL = NP_EDIT_COL.min(NP_LEN[NP_EDIT_ROW]); } }
            Down  => { if NP_EDIT_ROW + 1 < NP_ROWS_USED { NP_EDIT_ROW += 1; NP_EDIT_COL = NP_EDIT_COL.min(NP_LEN[NP_EDIT_ROW]); } }
            Left  => { if NP_EDIT_COL > 0 { NP_EDIT_COL -= 1; } else if NP_EDIT_ROW > 0 { NP_EDIT_ROW -= 1; NP_EDIT_COL = NP_LEN[NP_EDIT_ROW]; } }
            Right => { if NP_EDIT_COL < NP_LEN[NP_EDIT_ROW] { NP_EDIT_COL += 1; } else if NP_EDIT_ROW + 1 < NP_ROWS_USED { NP_EDIT_ROW += 1; NP_EDIT_COL = 0; } }
            Home  => { NP_EDIT_COL = 0; }
            End   => { NP_EDIT_COL = NP_LEN[NP_EDIT_ROW]; }
            PageUp   => { NP_SCROLL = NP_SCROLL.saturating_sub(10); NP_EDIT_ROW = NP_EDIT_ROW.saturating_sub(10); NP_EDIT_COL = NP_EDIT_COL.min(NP_LEN[NP_EDIT_ROW]); }
            PageDown => {
                let h = fb::height();
                let vis = h.saturating_sub(TERM_Y + APP_STATUS_H) / FONT_H;
                let max_scroll = NP_ROWS_USED.saturating_sub(1);
                NP_SCROLL = (NP_SCROLL + vis).min(max_scroll);
                NP_EDIT_ROW = (NP_EDIT_ROW + vis).min(NP_ROWS_USED.saturating_sub(1));
                NP_EDIT_COL = NP_EDIT_COL.min(NP_LEN[NP_EDIT_ROW]);
            }
            F2 => { np_save(); }
            F3 => { // New file
                for r in 0..NP_ROWS { NP_LEN[r] = 0; }
                NP_ROWS_USED = 1; NP_EDIT_ROW = 0; NP_EDIT_COL = 0;
                NP_SCROLL = 0; NP_FNAME_LEN = 0; NP_DIRTY = false;
            }
            _ => { return; }
        }
        np_ensure_scroll_visible();
        fb::with(|f| unsafe { draw_win_titlebar(f); draw_notepad_overlay(f); });
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// File Manager
// ══════════════════════════════════════════════════════════════════════════════

fn fm_open_gui(path: &str) {
    fm_load_dir(path);
    unsafe { ACTIVE_APP = ActiveApp::FileManager; }
    fb::with(|f| unsafe {
        draw_win_titlebar(f);
        draw_fm_overlay(f);
    });
}

fn fm_load_dir(path: &str) {
    unsafe {
        FM_COUNT = 0; FM_SEL = 0; FM_SCROLL = 0;
        let pb  = path.as_bytes();
        let len = pb.len().min(255);
        FM_PATH[..len].copy_from_slice(&pb[..len]);
        FM_PATH_LEN = len;
    }
    if let Ok(node) = crate::vfs::lookup(path) {
        if let Ok(entries) = node.readdir() {
            for e in entries.iter().take(FM_MAX) {
                let idx  = unsafe { FM_COUNT };
                let nb   = e.name.as_bytes();
                let nlen = nb.len().min(63);
                unsafe {
                    FM_NAMES[idx][..nlen].copy_from_slice(&nb[..nlen]);
                    FM_NLENS[idx] = nlen;
                    FM_IS_DIR[idx] = e.is_dir;
                    FM_SIZES[idx] = if !e.is_dir {
                        // Build full path to stat
                        let full = if path == "/" {
                            alloc::format!("/{}", e.name)
                        } else {
                            alloc::format!("{}/{}", path, e.name)
                        };
                        crate::vfs::lookup(&full)
                            .and_then(|n| n.stat())
                            .map(|s| s.size)
                            .unwrap_or(0)
                    } else { 0 };
                    FM_COUNT += 1;
                }
            }
        }
    }
}

/// Navigate to parent of FM_PATH.
fn fm_go_parent() {
    let cur_path = unsafe {
        core::str::from_utf8(&FM_PATH[..FM_PATH_LEN]).unwrap_or("/").to_owned()
    };
    let parent = match cur_path.rfind('/') {
        Some(0) | None => "/".to_owned(),
        Some(i)        => cur_path[..i].to_owned(),
    };
    fm_load_dir(&parent);
    fb::with(|f| unsafe { draw_win_titlebar(f); draw_fm_overlay(f); });
}

/// Handle a mouse click in the File Manager content area.
fn fm_click(x: usize, y: usize) {
    let path_bar_h = 24usize;
    let hdr_h      = FONT_H + 4;
    let row_h      = FONT_H + 2;
    let list_y     = TERM_Y + path_bar_h + hdr_h;
    if y < list_y { return; }
    let rel = (y - list_y) / row_h;
    let idx = unsafe { FM_SCROLL } + rel;
    if idx >= unsafe { FM_COUNT } { return; }
    unsafe { FM_SEL = idx; }
    fb::with(|f| unsafe { draw_fm_overlay(f) });
}

unsafe fn draw_fm_overlay(f: &mut fb::Framebuffer) {
    let w          = f.width();
    let h          = f.height();
    let path_bar_h = 24usize;
    let hdr_h      = FONT_H + 4;
    let row_h      = FONT_H + 2;
    let list_y     = TERM_Y + path_bar_h + hdr_h;
    let list_h     = h.saturating_sub(list_y + APP_STATUS_H);
    let visible    = list_h / row_h;

    // Full background
    f.fill_rect(0, TERM_Y, w, h.saturating_sub(TERM_Y), FM_BG.0, FM_BG.1, FM_BG.2);

    // Path bar
    f.fill_rect(0, TERM_Y, w, path_bar_h, FM_HDR.0, FM_HDR.1, FM_HDR.2);
    let path = core::str::from_utf8(&FM_PATH[..FM_PATH_LEN]).unwrap_or("/");
    let py   = TERM_Y + (path_bar_h - FONT_H) / 2;
    let mut px = f.draw_str(8, py, ">> ", FM_DIR_C, FM_HDR);
    f.draw_str(px, py, path, FM_FG, FM_HDR);

    // Column header
    let hdr_y = TERM_Y + path_bar_h;
    f.fill_rect(0, hdr_y, w, hdr_h, FM_HDR.0, FM_HDR.1, FM_HDR.2);
    f.draw_str(8,  hdr_y + 2, "TYPE", (0x88,0x88,0xAA), FM_HDR);
    f.draw_str(64, hdr_y + 2, "NAME", (0x88,0x88,0xAA), FM_HDR);
    f.draw_str(w.saturating_sub(68), hdr_y + 2, "SIZE", (0x88,0x88,0xAA), FM_HDR);
    f.fill_rect(0, hdr_y + hdr_h - 1, w, 1, 0x30, 0x30, 0x48);

    // Entry rows
    for vr in 0..visible {
        let idx = FM_SCROLL + vr;
        if idx >= FM_COUNT { break; }
        let ry       = list_y + vr * row_h;
        let selected = idx == FM_SEL;
        let row_bg   = if selected { FM_SL } else { FM_BG };
        let lbl_col  = if selected { FM_FG } else if FM_IS_DIR[idx] { FM_DIR_C } else { FM_FIL };
        f.fill_rect(0, ry, w, row_h, row_bg.0, row_bg.1, row_bg.2);
        let type_str = if FM_IS_DIR[idx] { "DIR " } else { "FILE" };
        let tc = if selected { FM_FG } else if FM_IS_DIR[idx] { FM_DIR_C } else { (0x77,0x77,0x88) };
        f.draw_str(8,  ry + 1, type_str, tc,      row_bg);
        let nlen = FM_NLENS[idx].min(63);
        let name = core::str::from_utf8(&FM_NAMES[idx][..nlen]).unwrap_or("?");
        f.draw_str(64, ry + 1, name,     lbl_col, row_bg);
        if !FM_IS_DIR[idx] {
            let mut sb = [0u8; 24];
            let ss   = fmt_size(FM_SIZES[idx], &mut sb);
            let sx   = w.saturating_sub(ss.len() * FONT_W + 8);
            let scol = if selected { FM_FG } else { (0x77,0x77,0x88) };
            f.draw_str(sx, ry + 1, ss, scol, row_bg);
        }
    }

    // Status bar
    let sb_y = h - APP_STATUS_H;
    f.fill_rect(0, sb_y, w, APP_STATUS_H, FM_HDR.0, FM_HDR.1, FM_HDR.2);
    let sy = sb_y + (APP_STATUS_H - FONT_H) / 2;
    f.draw_str(8, sy, "^v:Nav  Enter:Open  BS:Up  ESC:Close", FM_FG, FM_HDR);
    let mut cb = [0u8; 20];
    let cs = fmt_usize(FM_COUNT, &mut cb);
    let cx = w.saturating_sub((cs.len() + 6) * FONT_W);
    f.draw_str(cx, sy, cs, colour::YELLOW, FM_HDR);
    f.draw_str(cx + cs.len() * FONT_W, sy, " items", FM_FG, FM_HDR);
}

/// Handle a char/control byte in the File Manager (only ESC used; Enter/arrow via special).
pub fn fm_key(b: u8) {
    if b == 0x1B { close_app_window(); }
}

/// Handle a special key in the File Manager.
pub fn fm_special(key: drivers::input::SpecialKey) {
    use drivers::input::SpecialKey::*;
    unsafe {
        match key {
            Up => {
                if FM_SEL > 0 { FM_SEL -= 1; }
                if FM_SEL < FM_SCROLL { FM_SCROLL = FM_SEL; }
            }
            Down => {
                if FM_SEL + 1 < FM_COUNT { FM_SEL += 1; }
                let h = fb::height();
                let visible = (h.saturating_sub(TERM_Y + 24 + FONT_H + 4 + APP_STATUS_H)) / (FONT_H + 2);
                if FM_SEL >= FM_SCROLL + visible { FM_SCROLL += 1; }
            }
            PageUp => {
                let h = fb::height();
                let vis = (h.saturating_sub(TERM_Y + 24 + FONT_H + 4 + APP_STATUS_H)) / (FONT_H + 2);
                FM_SCROLL = FM_SCROLL.saturating_sub(vis);
                FM_SEL    = FM_SEL.saturating_sub(vis);
            }
            PageDown => {
                let h = fb::height();
                let vis = (h.saturating_sub(TERM_Y + 24 + FONT_H + 4 + APP_STATUS_H)) / (FONT_H + 2);
                let max = FM_COUNT.saturating_sub(1);
                FM_SEL    = (FM_SEL + vis).min(max);
                FM_SCROLL = (FM_SCROLL + vis).min(max.saturating_sub(vis));
            }
            F5 => {
                // Refresh current directory
                let path = core::str::from_utf8(&FM_PATH[..FM_PATH_LEN])
                    .unwrap_or("/").to_owned();
                fm_load_dir(&path);
            }
            _ => { return; }
        }
    }
    fb::with(|f| unsafe { draw_win_titlebar(f); draw_fm_overlay(f); });
}

/// Called when Enter is pressed in FM — open selected item.
fn fm_enter() {
    let (is_dir, mut path_buf) = unsafe {
        let idx = FM_SEL;
        if idx >= FM_COUNT { return; }
        let is_dir  = FM_IS_DIR[idx];
        let nlen    = FM_NLENS[idx].min(63);
        let name_b  = &FM_NAMES[idx][..nlen];
        let cur     = core::str::from_utf8(&FM_PATH[..FM_PATH_LEN]).unwrap_or("/");
        let full    = if cur == "/" {
            alloc::format!("/{}", core::str::from_utf8(name_b).unwrap_or("?"))
        } else {
            alloc::format!("{}/{}", cur, core::str::from_utf8(name_b).unwrap_or("?"))
        };
        (is_dir, full)
    };
    if is_dir {
        fm_load_dir(&path_buf);
        fb::with(|f| unsafe { draw_win_titlebar(f); draw_fm_overlay(f); });
    } else {
        // Open in Notepad
        notepad_open_gui(&path_buf);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Browser
// ══════════════════════════════════════════════════════════════════════════════

fn browser_open_gui() {
    unsafe {
        ACTIVE_APP     = ActiveApp::Browser;
        BR_URL_LEN     = 0;
        BR_CROWS_USED  = 0;
        BR_SCROLL      = 0;
        BR_STAT_LEN    = 0;
        let msg = b"Welcome to Intelli Browser! Type a URL and press Enter.";
        let l = msg.len().min(BR_CCOLS);
        BR_CONT[0][..l].copy_from_slice(&msg[..l]);
        BR_CLENS[0]    = l;
        BR_CROWS_USED  = 1;
    }
    fb::with(|f| unsafe {
        draw_win_titlebar(f);
        draw_browser_overlay(f);
    });
}

unsafe fn br_set_status(msg: &[u8]) {
    let l = msg.len().min(79);
    BR_STATUS[..l].copy_from_slice(&msg[..l]);
    BR_STAT_LEN = l;
}

/// Very naive HTML → plain-text: strips `<...>` tags, decodes a few entities.
fn br_strip_html(src: &[u8], dest: &mut [[u8; BR_CCOLS]; BR_CROWS], lens: &mut [usize; BR_CROWS]) -> usize {
    let mut row  = 0usize;
    let mut col  = 0usize;
    let mut tag  = false;
    for &b in src {
        if row >= BR_CROWS { break; }
        match b {
            b'<'  => { tag = true; }
            b'>'  => { tag = false; }
            _ if tag => {}
            b'\n' => {
                row += 1; col = 0;
                lens[row.saturating_sub(1)] = col;  // already set via col writes below actually
            }
            b'\r' => {}
            _ => {
                if col < BR_CCOLS {
                    dest[row][col] = b;
                    col += 1;
                    lens[row] = col;
                }
            }
        }
    }
    (row + 1).min(BR_CROWS)
}

/// Perform a simple blocking HTTP GET for `url` (e.g. "http://host/path").
/// Uses the kernel TCP stack — mirrors the approach used by shell's `wget` command.
fn br_fetch(url: &str) -> usize {
    let rest = url.trim_start_matches("http://").trim_start_matches("https://");
    let (hostport, path_part) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None    => (rest, "/"),
    };
    let (host, port) = match hostport.rfind(':') {
        Some(i) => (&hostport[..i], hostport[i+1..].parse::<u16>().unwrap_or(80)),
        None    => (hostport, 80u16),
    };
    if host.is_empty() { return 0; }

    let dst_ip = match crate::net::resolve(host) {
        Some(a) => a,
        None    => { unsafe { br_set_status(b"DNS failed"); } return 0; }
    };
    let our_ip  = unsafe { crate::net::OUR_IP };
    let our_mac = unsafe { crate::net::OUR_MAC };
    let gw: [u8; 4] = [10, 0, 2, 2];

    let dst_mac = if dst_ip[..3] == our_ip[..3] {
        crate::net::arp_cache_lookup(&dst_ip).unwrap_or_else(|| {
            crate::net::arp_request(dst_ip);
            for _ in 0..5000 { crate::net::poll(); core::hint::spin_loop(); }
            crate::net::arp_cache_lookup(&dst_ip).unwrap_or([0xFF; 6])
        })
    } else {
        crate::net::arp_cache_lookup(&gw).unwrap_or_else(|| {
            crate::net::arp_request(gw);
            for _ in 0..5000 { crate::net::poll(); core::hint::spin_loop(); }
            crate::net::arp_cache_lookup(&gw).unwrap_or([0xFF; 6])
        })
    };

    let local_port = 49152u16.wrapping_add((crate::scheduler::uptime_ms() & 0x3FFF) as u16);
    let isn: u32   = (crate::scheduler::uptime_ms() & 0xFFFF_FFFF) as u32;
    let syn = crate::net::tcp::TcpHeader::build(
        local_port, port, isn, 0, crate::net::tcp::SYN, 65535, our_ip, dst_ip, &[],
    );
    let ip_hdr = crate::net::Ipv4Header::build(crate::net::IP_PROTO_TCP, our_ip, dst_ip, syn.len());
    let mut pkt = ip_hdr;
    pkt.extend_from_slice(&syn);
    let frame = crate::net::EthFrame::build(dst_mac, our_mac, crate::net::ETHERTYPE_IPV4, &pkt);
    crate::net::transmit(&frame);

    let key = crate::net::tcp::TcpSocketKey { local_port, remote_ip: dst_ip, remote_port: port };
    {
        let mut sockets = crate::net::tcp::SOCKETS.lock();
        sockets.insert(key.clone(), crate::net::tcp::TcpSocket {
            state:   crate::net::tcp::TcpState::SynSent,
            snd_nxt: isn.wrapping_add(1),
            snd_una: isn,
            rcv_nxt: 0,
            snd_wnd: 65535,
            rcv_buf: Vec::new(),
            cwnd: 1460, ssthresh: 65535,
            last_send_ms: 0, rto_ms: 1000, retransmit_buf: Vec::new(),
        });
    }

    // Wait for SYN-ACK
    let deadline = crate::scheduler::uptime_ms() + 5000;
    let mut established = false;
    while crate::scheduler::uptime_ms() < deadline {
        crate::net::poll();
        let sockets = crate::net::tcp::SOCKETS.lock();
        if let Some(s) = sockets.get(&key) {
            if s.state == crate::net::tcp::TcpState::Established { established = true; break; }
        }
        drop(sockets);
        core::hint::spin_loop();
    }
    if !established {
        crate::net::tcp::SOCKETS.lock().remove(&key);
        unsafe { br_set_status(b"Connection failed"); }
        return 0;
    }

    let req = alloc::format!(
        "GET {} HTTP/1.0\r\nHost: {}\r\nUser-Agent: IntelliKernel/0.1\r\nConnection: close\r\n\r\n",
        path_part, host
    );
    crate::net::tcp::send(local_port, dst_ip, port, req.as_bytes());

    let deadline = crate::scheduler::uptime_ms() + 10000;
    while crate::scheduler::uptime_ms() < deadline {
        crate::net::poll();
        let sockets = crate::net::tcp::SOCKETS.lock();
        if let Some(s) = sockets.get(&key) {
            if s.state == crate::net::tcp::TcpState::CloseWait { break; }
            if !s.rcv_buf.is_empty() { break; }  // data arrived — don't wait for FIN
        }
        drop(sockets);
        core::hint::spin_loop();
    }

    let raw = {
        let mut sockets = crate::net::tcp::SOCKETS.lock();
        sockets.get_mut(&key).map(|s| core::mem::take(&mut s.rcv_buf)).unwrap_or_default()
    };
    crate::net::tcp::close(local_port, dst_ip, port);

    if raw.is_empty() { unsafe { br_set_status(b"No data received"); } return 0; }

    let body = raw.windows(4).position(|w| w == b"\r\n\r\n")
        .map(|i| &raw[i+4..])
        .unwrap_or(&raw);
    unsafe {
        for r in 0..BR_CROWS { BR_CLENS[r] = 0; }
        let rows = br_strip_html(body, &mut BR_CONT, &mut BR_CLENS);
        BR_CROWS_USED = rows;
        BR_SCROLL     = 0;
        br_set_status(b"Done");
        rows
    }
}

/// Public: perform a real HTTP GET and return the raw response body bytes.
/// Called by browser.rs Tab::fetch_url() to get genuine HTML from the network.
pub fn br_fetch_raw(url: &str) -> Vec<u8> {
    let rest = url.trim_start_matches("http://").trim_start_matches("https://");
    let (hostport, path_part) = if let Some(p) = rest.find('/') {
        (&rest[..p], &rest[p..])
    } else {
        (rest, "/")
    };
    let (host, port) = if let Some(i) = hostport.rfind(':') {
        (&hostport[..i], hostport[i+1..].parse::<u16>().unwrap_or(80))
    } else {
        (hostport, 80u16)
    };
    if host.is_empty() { return Vec::new(); }

    let dst_ip = match crate::net::resolve(host) {
        Some(a) => a,
        None    => return Vec::new(),
    };
    let our_ip  = unsafe { crate::net::OUR_IP };
    let our_mac = unsafe { crate::net::OUR_MAC };
    let gw: [u8; 4] = [10, 0, 2, 2];
    let dst_mac = if dst_ip[..3] == our_ip[..3] {
        crate::net::arp_cache_lookup(&dst_ip).unwrap_or_else(|| {
            crate::net::arp_request(dst_ip);
            for _ in 0..5000 { crate::net::poll(); core::hint::spin_loop(); }
            crate::net::arp_cache_lookup(&dst_ip).unwrap_or([0xFF; 6])
        })
    } else {
        crate::net::arp_cache_lookup(&gw).unwrap_or_else(|| {
            crate::net::arp_request(gw);
            for _ in 0..5000 { crate::net::poll(); core::hint::spin_loop(); }
            crate::net::arp_cache_lookup(&gw).unwrap_or([0xFF; 6])
        })
    };

    let local_port = 49200u16.wrapping_add((crate::scheduler::uptime_ms() & 0x3FFF) as u16);
    let isn: u32   = (crate::scheduler::uptime_ms() & 0xFFFF_FFFF) as u32;
    let syn = crate::net::tcp::TcpHeader::build(
        local_port, port, isn, 0, crate::net::tcp::SYN, 65535, our_ip, dst_ip, &[],
    );
    let ip_hdr = crate::net::Ipv4Header::build(crate::net::IP_PROTO_TCP, our_ip, dst_ip, syn.len());
    let mut pkt = ip_hdr; pkt.extend_from_slice(&syn);
    let frame = crate::net::EthFrame::build(dst_mac, our_mac, crate::net::ETHERTYPE_IPV4, &pkt);
    crate::net::transmit(&frame);

    let key = crate::net::tcp::TcpSocketKey { local_port, remote_ip: dst_ip, remote_port: port };
    crate::net::tcp::SOCKETS.lock().insert(key.clone(), crate::net::tcp::TcpSocket {
        state: crate::net::tcp::TcpState::SynSent,
        snd_nxt: isn.wrapping_add(1), snd_una: isn,
        rcv_nxt: 0, snd_wnd: 65535, rcv_buf: Vec::new(),
        cwnd: 1460, ssthresh: 65535,
        last_send_ms: 0, rto_ms: 1000, retransmit_buf: Vec::new(),
    });

    let deadline = crate::scheduler::uptime_ms() + 5000;
    while crate::scheduler::uptime_ms() < deadline {
        crate::net::poll();
        if crate::net::tcp::SOCKETS.lock().get(&key)
            .map(|s| s.state == crate::net::tcp::TcpState::Established).unwrap_or(false) { break; }
        core::hint::spin_loop();
    }
    if !crate::net::tcp::SOCKETS.lock().get(&key)
        .map(|s| s.state == crate::net::tcp::TcpState::Established).unwrap_or(false) {
        crate::net::tcp::SOCKETS.lock().remove(&key);
        return Vec::new();
    }

    let req = alloc::format!(
        "GET {} HTTP/1.0\r\nHost: {}\r\nUser-Agent: NodeAI/0.1\r\nConnection: close\r\n\r\n",
        path_part, host
    );
    crate::net::tcp::send(local_port, dst_ip, port, req.as_bytes());

    let deadline = crate::scheduler::uptime_ms() + 10000;
    while crate::scheduler::uptime_ms() < deadline {
        crate::net::poll();
        let done = crate::net::tcp::SOCKETS.lock().get(&key)
            .map(|s| s.state == crate::net::tcp::TcpState::CloseWait || !s.rcv_buf.is_empty())
            .unwrap_or(false);
        if done { break; }
        core::hint::spin_loop();
    }

    let raw = crate::net::tcp::SOCKETS.lock().get_mut(&key)
        .map(|s| core::mem::take(&mut s.rcv_buf)).unwrap_or_default();
    crate::net::tcp::close(local_port, dst_ip, port);

    // Strip HTTP headers — body starts after \r\n\r\n
    raw.windows(4).position(|w| w == b"\r\n\r\n")
        .map(|i| raw[i+4..].to_vec())
        .unwrap_or(raw)
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── Network Manager ───────────────────────────────────────────────────────────
// Real data: IP/MAC/gateway from net globals, ARP cache, TCP sockets, counters.
// ═══════════════════════════════════════════════════════════════════════════════

fn netmgr_open() {
    unsafe { ACTIVE_APP = ActiveApp::Network; }
    fb::with(|f| unsafe {
        draw_win_titlebar(f);
        draw_netmgr_overlay(f);
    });
}

unsafe fn draw_netmgr_overlay(f: &mut fb::Framebuffer) {
    let w   = f.width();
    let h   = f.height();
    let bg  = (0x0A, 0x10, 0x18);
    let hdr = (0x44, 0xFF, 0xAA);
    let fg  = (0xCC, 0xCC, 0xCC);
    let dim = (0x66, 0x88, 0x66);
    let val = (0xFF, 0xFF, 0x66);
    let sep = (0x22, 0x33, 0x22);

    f.fill_rect(0, TERM_Y, w, h.saturating_sub(TERM_Y), bg.0, bg.1, bg.2);

    let mut y = TERM_Y + 8;
    let x1    = 16usize;
    let x2    = 200usize;

    // ── Section: Interface ────────────────────────────────────────────────────
    f.draw_str(x1, y, "INTERFACE", hdr, bg); y += FONT_H + 4;
    f.fill_rect(x1, y, w - x1*2, 1, sep.0, sep.1, sep.2); y += 6;

    let ip  = crate::net::OUR_IP;
    let mac = crate::net::OUR_MAC;
    f.draw_str(x1, y, "IP Address :", dim, bg);
    f.draw_fmt(x2, y, val, bg, format_args!("{}.{}.{}.{}", ip[0], ip[1], ip[2], ip[3]));
    y += FONT_H + 2;

    f.draw_str(x1, y, "MAC Address:", dim, bg);
    f.draw_fmt(x2, y, val, bg, format_args!("{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
        mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]));
    y += FONT_H + 2;

    let gw = crate::net::route_entries().into_iter()
        .find(|r| r.destination == [0,0,0,0])
        .map(|r| r.gateway)
        .unwrap_or([10,0,2,2]);
    f.draw_str(x1, y, "Gateway    :", dim, bg);
    f.draw_fmt(x2, y, val, bg, format_args!("{}.{}.{}.{}", gw[0], gw[1], gw[2], gw[3]));
    y += FONT_H + 2;

    f.draw_str(x1, y, "DNS Server :", dim, bg);
    f.draw_str(x2, y, "10.0.2.3  (QEMU virtual)", val, bg);
    y += FONT_H + 8;

    // ── Section: Traffic ─────────────────────────────────────────────────────
    f.draw_str(x1, y, "TRAFFIC", hdr, bg); y += FONT_H + 4;
    f.fill_rect(x1, y, w - x1*2, 1, sep.0, sep.1, sep.2); y += 6;

    let rx = crate::telemetry::NET_RX_BYTES.load(core::sync::atomic::Ordering::Relaxed);
    let tx = crate::telemetry::NET_TX_BYTES.load(core::sync::atomic::Ordering::Relaxed);
    f.draw_str(x1, y, "RX bytes   :", dim, bg);
    f.draw_fmt(x2, y, val, bg, format_args!("{}", rx));
    y += FONT_H + 2;
    f.draw_str(x1, y, "TX bytes   :", dim, bg);
    f.draw_fmt(x2, y, val, bg, format_args!("{}", tx));
    y += FONT_H + 8;

    // ── Section: ARP Cache ───────────────────────────────────────────────────
    f.draw_str(x1, y, "ARP CACHE", hdr, bg); y += FONT_H + 4;
    f.fill_rect(x1, y, w - x1*2, 1, sep.0, sep.1, sep.2); y += 6;

    let arp = crate::net::arp_cache_entries();
    if arp.is_empty() {
        f.draw_str(x1, y, "  (empty)", dim, bg); y += FONT_H + 2;
    } else {
        f.draw_str(x1, y, "  IP Address        MAC Address        Age(ms)", dim, bg);
        y += FONT_H + 2;
        let now = crate::scheduler::uptime_ms();
        for (ip4, mac4, ts) in arp.iter().take(6) {
            if y + FONT_H > h { break; }
            let age = now.saturating_sub(*ts);
            f.draw_fmt(x1 + 8, y, fg, bg, format_args!(
                "{:>3}.{:>3}.{:>3}.{:>3}  {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}  {}ms",
                ip4[0], ip4[1], ip4[2], ip4[3],
                mac4[0], mac4[1], mac4[2], mac4[3], mac4[4], mac4[5],
                age));
            y += FONT_H + 2;
        }
    }
    y += 6;

    // ── Section: Active TCP connections ──────────────────────────────────────
    if y + FONT_H * 4 < h {
        f.draw_str(x1, y, "TCP CONNECTIONS", hdr, bg); y += FONT_H + 4;
        f.fill_rect(x1, y, w - x1*2, 1, sep.0, sep.1, sep.2); y += 6;

        let sockets = crate::net::tcp::SOCKETS.lock();
        let active: alloc::vec::Vec<_> = sockets.iter().collect();
        if active.is_empty() {
            f.draw_str(x1, y, "  (no active connections)", dim, bg);
            y += FONT_H + 2;
        } else {
            f.draw_str(x1, y, "  Local Port  Remote IP           Remote Port  State", dim, bg);
            y += FONT_H + 2;
            for (key, sock) in active.iter().take(5) {
                if y + FONT_H > h { break; }
                let state = match sock.state {
                    crate::net::tcp::TcpState::Established => "ESTABLISHED",
                    crate::net::tcp::TcpState::SynSent     => "SYN_SENT",
                    crate::net::tcp::TcpState::CloseWait   => "CLOSE_WAIT",
                    _                                       => "OTHER",
                };
                f.draw_fmt(x1 + 8, y, fg, bg, format_args!(
                    "{:<13} {}.{}.{}.{:<16} {:<12} {}",
                    key.local_port,
                    key.remote_ip[0], key.remote_ip[1], key.remote_ip[2], key.remote_ip[3],
                    key.remote_port, state));
                y += FONT_H + 2;
            }
        }
        drop(sockets);
        y += 6;
    }

    // ── Section: WiFi ────────────────────────────────────────────────────────
    if y + FONT_H * 3 < h {
        y += 4;
        f.draw_str(x1, y, "WIFI", hdr, bg); y += FONT_H + 4;
        f.fill_rect(x1, y, w - x1*2, 1, sep.0, sep.1, sep.2); y += 6;

        let wifi_avail = crate::wifi::is_available();
        let wifi_conn  = crate::wifi::is_connected();
        let wifi_ip    = crate::wifi::get_ip();

        if !wifi_avail {
            f.draw_str(x1, y, "  No AR9271 adapter detected", dim, bg); y += FONT_H + 2;
        } else if !wifi_conn {
            f.draw_str(x1, y, "Status     :", dim, bg);
            f.draw_str(x2, y, "Not connected", (0xFF, 0x88, 0x44), bg); y += FONT_H + 2;
            // Show scan results if any
            let aps = crate::wifi::scan_cache();
            if aps.is_empty() {
                f.draw_str(x1, y, "  Press W to scan for networks", dim, bg); y += FONT_H + 2;
            } else {
                f.draw_str(x1, y, "  SSID                           RSSI  CH  SEC", dim, bg);
                y += FONT_H + 2;
                for (i, ap) in aps.iter().take(6).enumerate() {
                    if y + FONT_H > h { break; }
                    let sec = if ap.secured { "WPA2" } else { "open" };
                    f.draw_fmt(x1 + 8, y, fg, bg, format_args!(
                        "[{}] {:<32} {:>4}  {:>2}  {}",
                        i + 1, ap.ssid, ap.rssi, ap.channel, sec));
                    y += FONT_H + 2;
                }
                f.draw_str(x1, y, "  Press C+<n> to connect to network #n", dim, bg);
                y += FONT_H + 2;
            }
        } else {
            let ssid = crate::wifi::ssid().unwrap_or_default();
            f.draw_str(x1, y, "Status     :", dim, bg);
            f.draw_str(x2, y, "Connected", (0x44, 0xFF, 0x88), bg); y += FONT_H + 2;
            f.draw_str(x1, y, "Network    :", dim, bg);
            f.draw_str(x2, y, &ssid, val, bg); y += FONT_H + 2;
            f.draw_str(x1, y, "WiFi IP    :", dim, bg);
            f.draw_fmt(x2, y, val, bg, format_args!("{}.{}.{}.{}",
                wifi_ip[0], wifi_ip[1], wifi_ip[2], wifi_ip[3])); y += FONT_H + 2;
            let wm = crate::wifi::wifi_mac();
            f.draw_str(x1, y, "WiFi MAC   :", dim, bg);
            f.draw_fmt(x2, y, val, bg, format_args!("{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                wm[0], wm[1], wm[2], wm[3], wm[4], wm[5])); y += FONT_H + 2;
        }
        y += 4;
    }

    // ── Ping button hint ──────────────────────────────────────────────────────
    if y + FONT_H < h {
        f.draw_str(x1, y,
            "P=ping gw  R=refresh  W=wifi scan  D=wifi disconnect", dim, bg);
    }
}

fn netmgr_key(b: u8) {
    match b | 0x20 { // lowercase
        b'p' => {
            // Ping the gateway and show result in terminal on close
            let gw = crate::net::route_entries().into_iter()
                .find(|r| r.destination == [0,0,0,0])
                .map(|r| r.gateway)
                .unwrap_or([10,0,2,2]);
            let rtt = crate::net::ping(gw, 1, 1, 2000);
            let msg: alloc::string::String = match rtt {
                Some(ms) => alloc::format!("Ping {}.{}.{}.{}: {}ms\n",
                    gw[0], gw[1], gw[2], gw[3], ms),
                None     => alloc::format!("Ping {}.{}.{}.{}: timeout\n",
                    gw[0], gw[1], gw[2], gw[3]),
            };
            // Redraw with ping result shown briefly in status area
            fb::with(|f| unsafe {
                let y = f.height().saturating_sub(FONT_H * 2);
                let bg = (0x0A, 0x10, 0x18);
                let ok = if rtt.is_some() { (0x44, 0xFF, 0x44) } else { (0xFF, 0x44, 0x44) };
                f.fill_rect(16, y, f.width() - 32, FONT_H, bg.0, bg.1, bg.2);
                f.draw_str(16, y, &msg, ok, bg);
            });
        }
        b'r' => {
            fb::with(|f| unsafe { draw_netmgr_overlay(f); });
        }
        b'w' => {
            // WiFi scan
            fb::with(|f| {
                let bg = (0x0A, 0x10, 0x18);
                let y  = f.height().saturating_sub(FONT_H * 2);
                f.fill_rect(16, y, f.width() - 32, FONT_H, bg.0, bg.1, bg.2);
                f.draw_str(16, y, "Scanning for WiFi networks...", (0xFF, 0xFF, 0x44), bg);
            });
            let count = crate::wifi::scan().len();
            fb::with(|f| unsafe {
                let bg = (0x0A, 0x10, 0x18);
                let ok = (0x44, 0xFF, 0x88);
                let y  = f.height().saturating_sub(FONT_H * 2);
                f.fill_rect(16, y, f.width() - 32, FONT_H, bg.0, bg.1, bg.2);
                let msg = alloc::format!("Found {} network(s) — press R to refresh list", count);
                f.draw_str(16, y, &msg, ok, bg);
                draw_netmgr_overlay(f);
            });
        }
        b'd' => {
            // WiFi disconnect
            crate::wifi::disconnect();
            fb::with(|f| unsafe { draw_netmgr_overlay(f); });
        }
        _ => {}
    }
}

unsafe fn draw_browser_overlay(f: &mut fb::Framebuffer) {
    let w = f.width();
    let h = f.height();
    let url_y  = TERM_Y;
    let cont_y = url_y + BR_URLBAR_H + 1;
    let cont_h = h.saturating_sub(cont_y + APP_STATUS_H);
    let visible = cont_h / FONT_H;

    // URL bar
    f.fill_rect(0, url_y, w, BR_URLBAR_H, BR_UBGF.0, BR_UBGF.1, BR_UBGF.2);
    let uy = url_y + (BR_URLBAR_H - FONT_H) / 2;
    let ux = f.draw_str(8, uy, "URL: ", (0x66,0xAA,0xFF), BR_UBGF);
    let url_str = core::str::from_utf8(&BR_URL[..BR_URL_LEN]).unwrap_or("");
    let ux2 = f.draw_str(ux, uy, url_str, BR_FG, BR_UBGF);
    // Cursor
    if ux2 + 2 <= w { f.fill_rect(ux2, uy, 2, FONT_H, 0x44, 0xBB, 0xFF); }
    // Separator
    f.fill_rect(0, url_y + BR_URLBAR_H, w, 1, 0x28, 0x48, 0x88);

    // Content
    f.fill_rect(0, cont_y, w, cont_h, BR_BG.0, BR_BG.1, BR_BG.2);
    for vr in 0..visible {
        let dr = BR_SCROLL + vr;
        if dr >= BR_CROWS_USED { break; }
        let y   = cont_y + vr * FONT_H;
        let len = BR_CLENS[dr].min(BR_CCOLS);
        if len > 0 {
            let s = core::str::from_utf8(&BR_CONT[dr][..len]).unwrap_or("");
            f.draw_str(8, y, s, BR_FG, BR_BG);
        }
    }

    // Status bar
    let sb_y = h - APP_STATUS_H;
    f.fill_rect(0, sb_y, w, APP_STATUS_H, BR_SBG.0, BR_SBG.1, BR_SBG.2);
    let sy = sb_y + (APP_STATUS_H - FONT_H) / 2;
    let stat = if BR_STAT_LEN > 0 {
        core::str::from_utf8(&BR_STATUS[..BR_STAT_LEN]).unwrap_or("")
    } else { "Enter URL and press Enter to navigate" };
    f.draw_str(8, sy, stat, BR_FG, BR_SBG);
    f.draw_str(w.saturating_sub(10 * FONT_W), sy, "ESC:Close", (0x88,0x88,0xAA), BR_SBG);
}

pub fn browser_key(b: u8) {
    unsafe {
        match b {
            0x1B => { close_app_window(); return; }
            0x08 => { if BR_URL_LEN > 0 { BR_URL_LEN -= 1; } }
            b'\n' | 0x0D => {
                let url = core::str::from_utf8(&BR_URL[..BR_URL_LEN])
                    .unwrap_or("").to_owned();
                // Store the URL for deferred fetch (processed in idle loop, not IRQ)
                let len = url.len().min(BR_FETCH_URL.len());
                BR_FETCH_URL[..len].copy_from_slice(&url.as_bytes()[..len]);
                BR_FETCH_URL_LEN = len;
                BR_FETCH_PENDING.store(true, core::sync::atomic::Ordering::Release);
                br_set_status(b"Fetching...");
                fb::with(|f| unsafe { draw_browser_overlay(f); });
                return;
            }
            0x20..=0x7E => {
                if BR_URL_LEN < 255 { BR_URL[BR_URL_LEN] = b; BR_URL_LEN += 1; }
            }
            _ => {}
        }
        fb::with(|f| unsafe { draw_browser_overlay(f); });
    }
}

pub fn browser_special(key: drivers::input::SpecialKey) {
    use drivers::input::SpecialKey::*;
    unsafe {
        match key {
            Up       => { if BR_SCROLL > 0 { BR_SCROLL -= 1; } }
            Down     => { if BR_SCROLL + 1 < BR_CROWS_USED { BR_SCROLL += 1; } }
            PageUp   => { BR_SCROLL = BR_SCROLL.saturating_sub(10); }
            PageDown => { BR_SCROLL = (BR_SCROLL + 10).min(BR_CROWS_USED.saturating_sub(1)); }
            _ => { return; }
        }
        fb::with(|f| unsafe { draw_browser_overlay(f); });
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Public routing API called from interrupts/mod.rs
// ══════════════════════════════════════════════════════════════════════════════

/// Returns true when any GUI app window is currently open.
pub fn app_is_open() -> bool {
    unsafe { ACTIVE_APP != ActiveApp::Terminal }
}

/// Route a printable/control byte to the currently active app.
pub fn app_char_key(b: u8) {
    unsafe {
        match ACTIVE_APP {
            ActiveApp::Terminal    => {}
            ActiveApp::Notepad     => notepad_key(b),
            ActiveApp::FileManager => {
                match b {
                    0x1B        => fm_key(b),
                    b'\n'|0x0D  => fm_enter(),
                    0x08        => fm_go_parent(),
                    _           => {}
                }
            }
            ActiveApp::Browser     => browser_key(b),
            ActiveApp::Network     => netmgr_key(b),
        }
    }
}

/// Route a special (arrow / F-key) keystroke to the currently active app.
pub fn app_special_key(key: drivers::input::SpecialKey) {
    unsafe {
        match ACTIVE_APP {
            ActiveApp::Terminal    => {}
            ActiveApp::Notepad     => notepad_special(key),
            ActiveApp::FileManager => fm_special(key),
            ActiveApp::Browser     => browser_special(key),
            ActiveApp::Network     => {}
        }
    }
}

/// Poll hardware input queues and route events to the desktop or shell.
/// Called safely from the main idle loop (out of interrupt context).
pub fn process_input_events() {
    // Process Mouse
    while let Some(ev) = drivers::input::poll_mouse_event() {
        mouse_event(ev.dx, ev.dy, ev.left, ev.right);
    }

    // Process Keyboard
    while let Some(ev) = drivers::input::poll_event() {
        if ev.pressed {
            if launcher_is_open() {
                // Inside launcher
                match ev.scancode {
                    0x01 => launcher_key(0x1B), // ESC
                    0x0E => launcher_key(0x08), // Backspace
                    0x1C => launcher_key(b'\n'), // Enter
                    _ => {
                        if let Some(ch) = ev.ascii {
                            let b = ch as u8;
                            if b >= 0x20 && b < 0x7F {
                                launcher_key(b);
                            }
                        }
                    }
                }
            } else if app_is_open() {
                // GUI app window: route to app key handlers
                if let Some(special) = ev.special {
                    app_special_key(special);
                } else {
                    match ev.scancode {
                        0x01 => app_char_key(0x1B), // ESC
                        0x0E => app_char_key(0x08), // Backspace
                        0x1C => app_char_key(b'\n'), // Enter
                        _ => {
                            if let Some(ch) = ev.ascii {
                                let b = ch as u8;
                                if b >= 0x20 && b < 0x7F {
                                    app_char_key(b);
                                }
                            }
                        }
                    }
                }
            } else {
                // Normal shell routing
                if ev.ctrl {
                    // Ctrl+key: generate control code (0x01..0x1A for a..z)
                    if let Some(ch) = ev.ascii {
                        let b = ch as u8;
                        let ctrl_byte = if b >= b'a' && b <= b'z' { b - b'a' + 1 }
                                        else if b >= b'A' && b <= b'Z' { b - b'A' + 1 }
                                        else { 0 };
                        if ctrl_byte > 0 { crate::shell::on_char(ctrl_byte); }
                    }
                } else if let Some(special) = ev.special {
                    crate::shell::on_special_key(special);
                } else {
                    match ev.scancode {
                        0x0E => crate::shell::on_char(0x08),
                        0x1C => crate::shell::on_char(b'\n'),
                        0x0F => crate::shell::on_char(b'\t'),
                        _ => {
                            if let Some(ch) = ev.ascii {
                                let b = ch as u8;
                                if b >= 0x20 && b < 0x7F {
                                    crate::shell::on_char(b);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
