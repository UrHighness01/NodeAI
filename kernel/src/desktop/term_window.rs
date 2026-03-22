//! VT100/ANSI terminal emulator window — Phase 24.
//!
//! Creates a WM window with an 80×24 character grid, full ANSI colour and
//! cursor-movement support, and a keyboard-input ring buffer for /dev/tty reads.

use spin::{Mutex, Once};

// ── Dimensions ────────────────────────────────────────────────────────────────
pub const TERM_COLS: usize = 80;
pub const TERM_ROWS: usize = 24;
const CELL_W: u32 = 8;
const CELL_H: u32 = 16;
const PAD_X:  u32 = 10;
const PAD_Y:  u32 = 10;
const WIN_W:  u32 = TERM_COLS as u32 * CELL_W + PAD_X * 2;   // 660
const WIN_H:  u32 = TERM_ROWS as u32 * CELL_H + PAD_Y * 2;   // 404

// ── 16-colour ANSI palette (0x00RRGGBB) ─────────────────────────────────────
const PALETTE: [u32; 16] = [
    0x000000, // 0  black
    0xAA0000, // 1  red
    0x00AA00, // 2  green
    0xAAAA00, // 3  yellow
    0x0000AA, // 4  blue
    0xAA00AA, // 5  magenta
    0x00AAAA, // 6  cyan
    0xAAAAAA, // 7  white
    0x555555, // 8  bright black
    0xFF5555, // 9  bright red
    0x55FF55, // 10 bright green
    0xFFFF55, // 11 bright yellow
    0x5555FF, // 12 bright blue
    0xFF55FF, // 13 bright magenta
    0x55FFFF, // 14 bright cyan
    0xFFFFFF, // 15 bright white
];

const DEFAULT_FG: u32 = 0xAAAAAA;   // ANSI white
const DEFAULT_BG: u32 = 0x1E1E2A;   // dark background

// ── Cell ─────────────────────────────────────────────────────────────────────
#[derive(Clone, Copy)]
struct Cell {
    ch: u8,
    fg: u32,
    bg: u32,
}

impl Cell {
    const fn blank() -> Self {
        Self { ch: b' ', fg: DEFAULT_FG, bg: DEFAULT_BG }
    }
}

// ── VT100 parser state ────────────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq)]
enum VtState {
    Normal,
    Escape,
    Csi,
}

// ── Simple power-of-2 ring buffer ─────────────────────────────────────────────
const RING_SIZE: usize = 4096;

struct RingBuf {
    buf: [u8; RING_SIZE],
    rd:  usize,
    wr:  usize,
}

impl RingBuf {
    const fn new() -> Self {
        Self { buf: [0u8; RING_SIZE], rd: 0, wr: 0 }
    }
    fn push(&mut self, b: u8) {
        let next = (self.wr + 1) & (RING_SIZE - 1);
        if next != self.rd {
            self.buf[self.wr] = b;
            self.wr = next;
        }
    }
    fn pop(&mut self) -> Option<u8> {
        if self.rd == self.wr { return None; }
        let b = self.buf[self.rd];
        self.rd = (self.rd + 1) & (RING_SIZE - 1);
        Some(b)
    }
}

// ── TermWindow ────────────────────────────────────────────────────────────────
struct TermWindow {
    win_id:      u32,
    cells:       [Cell; TERM_COLS * TERM_ROWS],
    cursor_row:  usize,
    cursor_col:  usize,
    fg:          u32,
    bg:          u32,
    state:       VtState,
    csi_params:  [u32; 8],
    csi_nparams: usize,
    csi_cur:     u32,
    input_ring:  RingBuf,
}

impl TermWindow {
    fn new(win_id: u32) -> Self {
        Self {
            win_id,
            cells:       [Cell::blank(); TERM_COLS * TERM_ROWS],
            cursor_row:  0,
            cursor_col:  0,
            fg:          DEFAULT_FG,
            bg:          DEFAULT_BG,
            state:       VtState::Normal,
            csi_params:  [0u32; 8],
            csi_nparams: 0,
            csi_cur:     0,
            input_ring:  RingBuf::new(),
        }
    }

    #[inline]
    fn cell_at(&self, row: usize, col: usize) -> Cell {
        self.cells[row * TERM_COLS + col]
    }
    #[inline]
    fn set_cell(&mut self, row: usize, col: usize, cell: Cell) {
        self.cells[row * TERM_COLS + col] = cell;
    }

    // ── Render one cell into the WM window buffer ──────────────────────────

    fn paint_cell(&self, row: usize, col: usize) {
        let cell = self.cell_at(row, col);
        let px = PAD_X + col as u32 * CELL_W;
        let py = PAD_Y + row as u32 * CELL_H;
        crate::desktop::wm_draw_text_cell(self.win_id, px, py, cell.ch, cell.fg, cell.bg);
    }

    fn paint_cursor(&self) {
        let row = self.cursor_row.min(TERM_ROWS - 1);
        let col = self.cursor_col.min(TERM_COLS - 1);
        let cell = self.cell_at(row, col);
        // Invert fg/bg to show cursor block
        crate::desktop::wm_draw_text_cell(
            self.win_id,
            PAD_X + col as u32 * CELL_W,
            PAD_Y + row as u32 * CELL_H,
            cell.ch, cell.bg, cell.fg,
        );
    }

    fn redraw(&self) {
        crate::desktop::wm_fill_window_rect(self.win_id, 0, 0, WIN_W, WIN_H, DEFAULT_BG);
        for r in 0..TERM_ROWS {
            for c in 0..TERM_COLS {
                self.paint_cell(r, c);
            }
        }
        self.paint_cursor();
        crate::desktop::wm_flip(self.win_id);
    }

    // ── Scrolling ─────────────────────────────────────────────────────────

    fn scroll_up_one(&mut self) {
        for r in 1..TERM_ROWS {
            let src = r * TERM_COLS;
            let dst = (r - 1) * TERM_COLS;
            for c in 0..TERM_COLS {
                self.cells[dst + c] = self.cells[src + c];
            }
        }
        let last = (TERM_ROWS - 1) * TERM_COLS;
        for c in 0..TERM_COLS {
            self.cells[last + c] = Cell::blank();
        }
    }

    fn newline(&mut self) {
        self.cursor_row += 1;
        if self.cursor_row >= TERM_ROWS {
            self.scroll_up_one();
            self.cursor_row = TERM_ROWS - 1;
        }
    }

    fn cursor_advance(&mut self) {
        self.cursor_col += 1;
        if self.cursor_col >= TERM_COLS {
            self.cursor_col = 0;
            self.newline();
        }
    }

    // ── VT100 state machine ───────────────────────────────────────────────

    fn write_byte(&mut self, b: u8) {
        match self.state {
            VtState::Normal => self.process_normal(b),
            VtState::Escape => self.process_escape(b),
            VtState::Csi    => self.process_csi(b),
        }
    }

    fn process_normal(&mut self, b: u8) {
        match b {
            0x07 => {}   // BEL — ignore
            0x08 => {    // BS
                if self.cursor_col > 0 { self.cursor_col -= 1; }
            }
            0x09 => {    // HT — tab to next 8-col boundary
                self.cursor_col = ((self.cursor_col | 7) + 1).min(TERM_COLS - 1);
            }
            0x0A | 0x0B | 0x0C => { // LF / VT / FF
                self.newline();
            }
            0x0D => {    // CR
                self.cursor_col = 0;
            }
            0x1B => {    // ESC
                self.state = VtState::Escape;
            }
            0x7F => {}   // DEL — ignore
            b if b >= 0x20 => {
                let row = self.cursor_row.min(TERM_ROWS - 1);
                let col = self.cursor_col.min(TERM_COLS - 1);
                self.set_cell(row, col, Cell { ch: b, fg: self.fg, bg: self.bg });
                self.cursor_advance();
            }
            _ => {}
        }
    }

    fn process_escape(&mut self, b: u8) {
        match b {
            b'[' => {
                self.csi_params  = [0u32; 8];
                self.csi_nparams = 0;
                self.csi_cur     = 0;
                self.state = VtState::Csi;
            }
            b'c' => {    // RIS — full reset
                self.cells      = [Cell::blank(); TERM_COLS * TERM_ROWS];
                self.cursor_row = 0;
                self.cursor_col = 0;
                self.fg = DEFAULT_FG;
                self.bg = DEFAULT_BG;
                self.state = VtState::Normal;
            }
            b'D' => { self.newline(); self.state = VtState::Normal; }
            b'M' => {    // RI — reverse index
                if self.cursor_row > 0 { self.cursor_row -= 1; }
                self.state = VtState::Normal;
            }
            _ => { self.state = VtState::Normal; }
        }
    }

    fn process_csi(&mut self, b: u8) {
        if b >= b'0' && b <= b'9' {
            self.csi_cur = self.csi_cur.saturating_mul(10)
                                       .saturating_add((b - b'0') as u32);
        } else if b == b';' {
            if self.csi_nparams < 7 {
                self.csi_params[self.csi_nparams] = self.csi_cur;
                self.csi_nparams += 1;
                self.csi_cur = 0;
            }
        } else if b == b'?' || b == b'>' {
            // private mode prefix — ignore, keep accumulating
        } else if b >= 0x40 && b <= 0x7E {
            // Flush last param
            if self.csi_nparams < 8 {
                self.csi_params[self.csi_nparams] = self.csi_cur;
            }
            let nparams = (self.csi_nparams + 1).min(8);
            self.dispatch_csi(b, nparams);
            self.state = VtState::Normal;
        }
        // Intermediate bytes (0x20–0x2F) — skip
    }

    fn dispatch_csi(&mut self, final_byte: u8, nparams: usize) {
        let p0 = self.csi_params[0];
        let p1 = self.csi_params[1];

        match final_byte {
            b'A' => { // cursor up
                let n = p0.max(1) as usize;
                self.cursor_row = self.cursor_row.saturating_sub(n);
            }
            b'B' => { // cursor down
                let n = p0.max(1) as usize;
                self.cursor_row = (self.cursor_row + n).min(TERM_ROWS - 1);
            }
            b'C' => { // cursor forward
                let n = p0.max(1) as usize;
                self.cursor_col = (self.cursor_col + n).min(TERM_COLS - 1);
            }
            b'D' => { // cursor back
                let n = p0.max(1) as usize;
                self.cursor_col = self.cursor_col.saturating_sub(n);
            }
            b'E' => { // cursor next line
                let n = p0.max(1) as usize;
                self.cursor_row = (self.cursor_row + n).min(TERM_ROWS - 1);
                self.cursor_col = 0;
            }
            b'F' => { // cursor previous line
                let n = p0.max(1) as usize;
                self.cursor_row = self.cursor_row.saturating_sub(n);
                self.cursor_col = 0;
            }
            b'G' => { // cursor horizontal absolute
                self.cursor_col = if p0 == 0 { 0 } else { (p0 as usize - 1).min(TERM_COLS - 1) };
            }
            b'H' | b'f' => { // cursor position (1-based row;col)
                self.cursor_row = if p0 == 0 { 0 } else { (p0 as usize - 1).min(TERM_ROWS - 1) };
                self.cursor_col = if p1 == 0 { 0 } else { (p1 as usize - 1).min(TERM_COLS - 1) };
            }
            b'J' => { // erase in display
                let blank = Cell { ch: b' ', fg: self.fg, bg: self.bg };
                match p0 {
                    0 => {
                        for c in self.cursor_col..TERM_COLS {
                            self.set_cell(self.cursor_row, c, blank);
                        }
                        for r in (self.cursor_row + 1)..TERM_ROWS {
                            for c in 0..TERM_COLS { self.set_cell(r, c, blank); }
                        }
                    }
                    1 => {
                        for r in 0..self.cursor_row {
                            for c in 0..TERM_COLS { self.set_cell(r, c, blank); }
                        }
                        for c in 0..=self.cursor_col {
                            self.set_cell(self.cursor_row, c, blank);
                        }
                    }
                    _ => { // 2 / 3 — whole screen
                        for r in 0..TERM_ROWS {
                            for c in 0..TERM_COLS { self.set_cell(r, c, blank); }
                        }
                        self.cursor_row = 0;
                        self.cursor_col = 0;
                    }
                }
            }
            b'K' => { // erase in line
                let blank = Cell { ch: b' ', fg: self.fg, bg: self.bg };
                match p0 {
                    0 => { for c in self.cursor_col..TERM_COLS  { self.set_cell(self.cursor_row, c, blank); } }
                    1 => { for c in 0..=self.cursor_col          { self.set_cell(self.cursor_row, c, blank); } }
                    _ => { for c in 0..TERM_COLS                 { self.set_cell(self.cursor_row, c, blank); } }
                }
            }
            b'L' => { // insert line(s)
                let n = p0.max(1) as usize;
                let row = self.cursor_row;
                for _ in 0..n {
                    let last = TERM_ROWS - 1;
                    for r in (row..last).rev() {
                        let src = r * TERM_COLS;
                        let dst = (r + 1) * TERM_COLS;
                        for c in 0..TERM_COLS { let v = self.cells[src+c]; self.cells[dst+c] = v; }
                    }
                    for c in 0..TERM_COLS { self.set_cell(row, c, Cell::blank()); }
                }
            }
            b'M' => { // delete line(s)
                let n = p0.max(1) as usize;
                let row = self.cursor_row;
                for _ in 0..n {
                    for r in row..TERM_ROWS - 1 {
                        let src = (r + 1) * TERM_COLS;
                        let dst = r * TERM_COLS;
                        for c in 0..TERM_COLS { let v = self.cells[src+c]; self.cells[dst+c] = v; }
                    }
                    for c in 0..TERM_COLS { self.set_cell(TERM_ROWS - 1, c, Cell::blank()); }
                }
            }
            b'P' => { // delete char(s)
                let n = p0.max(1) as usize;
                let row = self.cursor_row;
                let col = self.cursor_col;
                for c in col..TERM_COLS {
                    let src = if c + n < TERM_COLS { self.cell_at(row, c + n) } else { Cell::blank() };
                    self.set_cell(row, c, src);
                }
            }
            b'S' => { // scroll up
                let n = p0.max(1) as usize;
                for _ in 0..n { self.scroll_up_one(); }
            }
            b'T' => { // scroll down
                let n = p0.max(1) as usize;
                for _ in 0..n {
                    for r in (1..TERM_ROWS).rev() {
                        let src = (r - 1) * TERM_COLS;
                        let dst = r * TERM_COLS;
                        for c in 0..TERM_COLS { let v = self.cells[src+c]; self.cells[dst+c] = v; }
                    }
                    for c in 0..TERM_COLS { self.set_cell(0, c, Cell::blank()); }
                }
            }
            b'd' => { // line position absolute (1-based)
                self.cursor_row = if p0 == 0 { 0 } else { (p0 as usize - 1).min(TERM_ROWS - 1) };
            }
            b'm' => { // SGR
                let mut tmp = [0u32; 8];
                tmp[..nparams].copy_from_slice(&self.csi_params[..nparams]);
                self.apply_sgr(&tmp[..nparams]);
            }
            b'r' => { /* DECSTBM — set scroll region: ignore */ }
            b'l' | b'h' => { /* mode set/reset — ignore */ }
            b'n' => { // DSR — device status report
                // We don't actually respond since there's no output channel back to process
            }
            _ => {}
        }

        // Reset CSI accumulation state
        self.csi_params  = [0u32; 8];
        self.csi_nparams = 0;
        self.csi_cur     = 0;
    }

    fn apply_sgr(&mut self, params: &[u32]) {
        if params.is_empty() || (params.len() == 1 && params[0] == 0) {
            self.fg = DEFAULT_FG;
            self.bg = DEFAULT_BG;
            return;
        }
        let mut i = 0;
        while i < params.len() {
            match params[i] {
                0  => { self.fg = DEFAULT_FG; self.bg = DEFAULT_BG; }
                1  => { /* bold — no-op (would need bold font) */ }
                2  => { /* dim  — no-op */ }
                3  => { /* italic — no-op */ }
                4  => { /* underline — no-op (no underline pixel support) */ }
                7  => { core::mem::swap(&mut self.fg, &mut self.bg); }  // reverse
                22 => { /* normal intensity — no-op */ }
                24 => { /* underline off — no-op */ }
                27 => { core::mem::swap(&mut self.fg, &mut self.bg); }  // reverse off
                30..=37 => { self.fg = PALETTE[(params[i] - 30) as usize]; }
                38 => {
                    if i + 2 < params.len() && params[i + 1] == 5 {
                        // 256-colour fg: 38;5;n
                        self.fg = ansi256(params[i + 2] as u8);
                        i += 2;
                    } else if i + 4 < params.len() && params[i + 1] == 2 {
                        // 24-bit fg: 38;2;r;g;b
                        let r = params[i + 2]; let g = params[i + 3]; let b = params[i + 4];
                        self.fg = (r << 16) | (g << 8) | b;
                        i += 4;
                    }
                }
                39  => { self.fg = DEFAULT_FG; }
                40..=47 => { self.bg = PALETTE[(params[i] - 40) as usize]; }
                48 => {
                    if i + 2 < params.len() && params[i + 1] == 5 {
                        self.bg = ansi256(params[i + 2] as u8);
                        i += 2;
                    } else if i + 4 < params.len() && params[i + 1] == 2 {
                        let r = params[i + 2]; let g = params[i + 3]; let b = params[i + 4];
                        self.bg = (r << 16) | (g << 8) | b;
                        i += 4;
                    }
                }
                49  => { self.bg = DEFAULT_BG; }
                90..=97  => { self.fg = PALETTE[(params[i] - 90 + 8) as usize]; }
                100..=107 => { self.bg = PALETTE[(params[i] - 100 + 8) as usize]; }
                _ => {}
            }
            i += 1;
        }
    }

    // ── Keyboard input ring ─────────────────────────────────────────────────

    fn push_input(&mut self, ch: u8) {
        self.input_ring.push(ch);
    }

    fn read_input(&mut self, buf: &mut [u8]) -> usize {
        let mut n = 0;
        while n < buf.len() {
            match self.input_ring.pop() {
                Some(b) => { buf[n] = b; n += 1; }
                None    => break,
            }
        }
        n
    }
}

// ── 256-colour cube ──────────────────────────────────────────────────────────
fn ansi256(n: u8) -> u32 {
    match n {
        0..=15  => PALETTE[n as usize],
        16..=231 => {
            let n = n - 16;
            let b = (n % 6) as u32 * 51;
            let g = ((n / 6) % 6) as u32 * 51;
            let r = (n / 36) as u32 * 51;
            (r << 16) | (g << 8) | b
        }
        232..=255 => {
            let v = (n - 232) as u32 * 10 + 8;
            (v << 16) | (v << 8) | v
        }
    }
}

// ── Global state ──────────────────────────────────────────────────────────────
static TERM_WIN: Once<Mutex<TermWindow>> = Once::new();

// ── Public API ────────────────────────────────────────────────────────────────

/// Open the terminal window (no-op if already open).
pub fn term_window_init() {
    if TERM_WIN.get().is_some() { return; }
    let id = crate::desktop::wm_create_window(60, 50, WIN_W, WIN_H, "Terminal");
    TERM_WIN.call_once(|| Mutex::new(TermWindow::new(id)));
    if let Some(tw) = TERM_WIN.get() {
        tw.lock().redraw();
    }
    crate::desktop::wm_composite();
}

/// Returns `true` if the terminal window has been opened.
pub fn term_window_is_open() -> bool {
    TERM_WIN.get().is_some()
}

/// Write bytes to the terminal (VT100 processing + repaint).
pub fn term_window_write(data: &[u8]) {
    if let Some(tw) = TERM_WIN.get() {
        let mut t = tw.lock();
        for &b in data {
            t.write_byte(b);
        }
        t.redraw();
    }
}

/// Feed a keystroke into the terminal's keyboard-input ring (for /dev/tty reads).
pub fn term_window_key(ch: u8) {
    if let Some(tw) = TERM_WIN.get() {
        tw.lock().push_input(ch);
    }
}

/// Read pending keyboard input from the terminal (used by /dev/tty).
/// Returns the number of bytes placed in `buf`.
pub fn tty_read(buf: &mut [u8]) -> usize {
    if let Some(tw) = TERM_WIN.get() {
        tw.lock().read_input(buf)
    } else {
        0
    }
}
