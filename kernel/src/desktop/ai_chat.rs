//! AI Chat — Phase 26. Native UI for querying the kernel AI / system state.
//!
//! Type a question or command and press Enter. The chat assistant can:
//!  - Report system status (uptime, memory, tasks)
//!  - Answer common system questions
//!  - Relay AI engine decisions from the event log

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use spin::{Mutex, Once};
use crate::desktop::compositor::{
    wm_create_window, wm_fill_window_rect, wm_draw_text_cell, wm_flip,
};

const WIN_W:   u32 = 700;
const WIN_H:   u32 = 500;
const COLS:    usize = 80;
const ROWS:    usize = 27;
const FONT_W:  u32 = 8;
const FONT_H:  u32 = 16;
const PAD_X:   u32 = 6;
const PAD_Y:   u32 = 4;

const BG:       u32 = 0xFF0F131A;
const MSG_FG:   u32 = 0xFFCCCCCC;
const USR_FG:   u32 = 0xFF6DBFFF;
const BOT_FG:   u32 = 0xFF7EC8A0;
const INPUT_BG: u32 = 0xFF1E2836;
const INPUT_FG: u32 = 0xFFEEEEEE;
const HDR_BG:   u32 = 0xFF0D2244;
const HDR_FG:   u32 = 0xFF88CCFF;

struct ChatMsg {
    from_user: bool,
    text: String,
}

struct AiChat {
    win_id: u32,
    history: Vec<ChatMsg>,
    input: Vec<u8>,
    input_cur: usize,
    scroll: usize,
}

static AI_CHAT: Once<Mutex<AiChat>> = Once::new();

impl AiChat {
    fn render(&self) {
        let id = self.win_id;
        wm_fill_window_rect(id, 0, 0, WIN_W, WIN_H, BG);

        // Header
        wm_fill_window_rect(id, 0, 0, WIN_W, FONT_H + 4, HDR_BG);
        let hdr = b"NodeAI Assistant  -  ask about system, memory, AI, or anything";
        for (i, &b) in hdr.iter().enumerate().take((WIN_W / FONT_W) as usize) {
            wm_draw_text_cell(id, PAD_X + i as u32 * FONT_W, 3, b, HDR_FG, HDR_BG);
        }

        // Messages
        let total = self.history.len();
        let vis = ROWS;
        let start = if total > vis + self.scroll { total - vis - self.scroll } else { 0 };
        for (row, msg) in self.history[start..].iter().take(vis).enumerate() {
            let py = FONT_H + 4 + PAD_Y + row as u32 * FONT_H;
            wm_fill_window_rect(id, 0, py, WIN_W, FONT_H, BG);
            let (prefix, fg) = if msg.from_user {
                (b"You: " as &[u8], USR_FG)
            } else {
                (b"AI:  " as &[u8], BOT_FG)
            };
            let mut col = 0usize;
            for &b in prefix {
                wm_draw_text_cell(id, PAD_X + col as u32 * FONT_W, py, b, fg, BG);
                col += 1;
            }
            for b in msg.text.bytes().take(COLS - prefix.len()) {
                wm_draw_text_cell(id, PAD_X + col as u32 * FONT_W, py, b, MSG_FG, BG);
                col += 1;
            }
        }

        // Input line
        let iy = WIN_H - FONT_H - 6;
        wm_fill_window_rect(id, 0, iy, WIN_W, FONT_H + 4, INPUT_BG);
        let prompt = b"> ";
        for (i, &b) in prompt.iter().enumerate() {
            wm_draw_text_cell(id, PAD_X + i as u32 * FONT_W, iy + 2, b, INPUT_FG, INPUT_BG);
        }
        let tx = PAD_X + prompt.len() as u32 * FONT_W;
        for (i, &b) in self.input.iter().enumerate().take(COLS - 2) {
            let bg = if i == self.input_cur { INPUT_FG } else { INPUT_BG };
            let fg = if i == self.input_cur { INPUT_BG } else { INPUT_FG };
            wm_draw_text_cell(id, tx + i as u32 * FONT_W, iy + 2, b, fg, bg);
        }
        // Cursor at end if input_cur == input.len()
        if self.input_cur == self.input.len() {
            wm_draw_text_cell(id, tx + self.input.len() as u32 * FONT_W, iy + 2, b' ', INPUT_BG, INPUT_FG);
        }

        wm_flip(id);
    }

    fn send(&mut self) {
        if self.input.is_empty() { return; }
        let text = String::from_utf8_lossy(&self.input).to_string();
        self.history.push(ChatMsg { from_user: true, text: text.clone() });
        self.input.clear();
        self.input_cur = 0;
        let response = self.process_query(&text);
        self.history.push(ChatMsg { from_user: false, text: response });
        // Scroll to bottom
        self.scroll = 0;
    }

    fn process_query(&self, q: &str) -> String {
        let q_lower = q.to_lowercase();
        // Pattern match on common queries
        if q_lower.contains("uptime") || q_lower.contains("time") {
            let ms = crate::scheduler::uptime_ms();
            return alloc::format!("System uptime: {}ms ({} seconds)", ms, ms / 1000);
        }
        if q_lower.contains("memory") || q_lower.contains("mem") || q_lower.contains("ram") {
            let free = crate::scheduler::free_mb();
            return alloc::format!("Free memory: {} MB", free);
        }
        if q_lower.contains("version") || q_lower.contains("kernel") || q_lower.contains("nodeai") {
            return String::from("NodeAI OS v0.26.0 — AI-native kernel, x86_64 bare metal");
        }
        if q_lower.contains("help") || q_lower.contains("what can") {
            return String::from("I can answer: uptime, memory, version, tasks. More AI coming in Phase 29!");
        }
        if q_lower.contains("hello") || q_lower.contains("hi") {
            return String::from("Hello! I'm the NodeAI assistant. How can I help?");
        }
        if q_lower.contains("cpu") || q_lower.contains("processor") {
            return String::from("Running on x86_64. NodeAI AI engine active (event bus + scheduler model).");
        }
        if q_lower.contains("audio") || q_lower.contains("sound") {
            let avail = if crate::audio::is_available() { "AC97 controller detected and active." } else { "No audio hardware detected." };
            return String::from(avail);
        }
        if q_lower.contains("task") || q_lower.contains("process") {
            return String::from("Process management: kernel tasks + user processes via ELF loader.");
        }
        // Default: echo back
        alloc::format!("I don't have specific data on '{}'. Try: uptime, memory, version, tasks.", q)
    }

    fn handle_key(&mut self, ch: u8) {
        match ch {
            0x0D => { self.send(); }
            0x08 => {
                if self.input_cur > 0 {
                    self.input_cur -= 1;
                    self.input.remove(self.input_cur);
                }
            }
            0x43 => { if self.input_cur < self.input.len() { self.input_cur += 1; } } // right
            0x44 => { if self.input_cur > 0 { self.input_cur -= 1; } }               // left
            0x41 => { self.scroll = self.scroll.saturating_add(1).min(self.history.len()); } // up scroll
            0x42 => { self.scroll = self.scroll.saturating_sub(1); }                         // down scroll
            b if b >= 0x20 => {
                self.input.insert(self.input_cur, b);
                self.input_cur += 1;
            }
            _ => {}
        }
    }
}

// ── Public API ─────────────────────────────────────────────────────────────────

pub fn ai_chat_open() {
    AI_CHAT.call_once(|| {
        let id = wm_create_window(100, 60, WIN_W, WIN_H, "AI Chat");
        let mut chat = AiChat {
            win_id: id, history: Vec::new(),
            input: Vec::new(), input_cur: 0, scroll: 0,
        };
        // Welcome message
        chat.history.push(ChatMsg {
            from_user: false,
            text: String::from("Welcome to NodeAI Assistant! Type a question and press Enter."),
        });
        Mutex::new(chat)
    });
    if let Some(app) = AI_CHAT.get() {
        app.lock().render();
    }
}

pub fn ai_chat_is_open() -> bool { AI_CHAT.get().is_some() }

pub fn ai_chat_key(ch: u8) {
    if let Some(app) = AI_CHAT.get() {
        let mut g = app.lock();
        g.handle_key(ch);
        g.render();
    }
}
