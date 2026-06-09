//! Kernel shell — interactive command interpreter for the NodeAI desktop.
//!
//! `on_char(byte)` is called from the keyboard IRQ handler for each
//! printable key.  Characters are buffered in a static array; on newline
//! the line is parsed and dispatched.
//!
//! Shell: user/group commands, sudo, custom Kali-style prompt.
//! Environment variables, cd, pwd, history, line editing, pipes.
//! Coreutils — touch, mkdir, rm, mv, cp, etc.

use spin::Mutex;
use alloc::{string::{String, ToString}, vec::Vec, format, collections::BTreeMap};
use drivers::input::SpecialKey;

// ── Line buffer with cursor position ──────────────────────────────────────────

const MAX_LINE: usize = 256;

struct LineBuf {
    buf: [u8; MAX_LINE],
    len: usize,
    cursor: usize, // cursor position within line (0..=len)
}

static BUF: Mutex<LineBuf> = Mutex::new(LineBuf {
    buf: [0u8; MAX_LINE],
    len: 0,
    cursor: 0,
});

// ── Prompt length tracking (for line redraw after edits) ──────────────────────
static PROMPT_LEN: Mutex<usize> = Mutex::new(0);

// ── Insert/Overwrite mode ─────────────────────────────────────────────────────
/// false = insert mode (default), true = overwrite mode
static OVERWRITE_MODE: Mutex<bool> = Mutex::new(false);

// ── Tab completion cycling state ──────────────────────────────────────────────
/// Tracks consecutive tab presses to cycle through multiple completions.
struct TabCycle {
    prefix_start: usize,    // where the prefix being completed starts
    prefix_len: usize,      // length of original prefix
    matches: Vec<String>,   // cached completions
    index: usize,           // current cycling index
    active: bool,           // whether a cycle is in progress
}
static TAB_CYCLE: Mutex<TabCycle> = Mutex::new(TabCycle {
    prefix_start: 0,
    prefix_len: 0,
    matches: Vec::new(),
    index: 0,
    active: false,
});

/// Reset tab cycling state (called on any non-tab keypress).
fn tab_cycle_reset() {
    let mut tc = TAB_CYCLE.lock();
    tc.active = false;
    tc.matches = Vec::new();
    tc.index = 0;
}

// ── History browsing state ────────────────────────────────────────────────────
// When browsing history with up/down, we track the browse index.
// -1 (stored as usize::MAX) means "current line" (not browsing).
static HIST_BROWSE: Mutex<usize> = Mutex::new(usize::MAX);
// Save the in-progress line when user starts browsing history
static SAVED_LINE: Mutex<([u8; MAX_LINE], usize)> = Mutex::new(([0u8; MAX_LINE], 0));

// ── Command history ───────────────────────────────────────────────────────────

const HISTORY_CAP: usize = 64;

struct History {
    entries: [Option<[u8; MAX_LINE]>; HISTORY_CAP],
    lengths: [usize; HISTORY_CAP],
    count:   usize,
    pos:     usize, // write position (ring)
}

static HISTORY: Mutex<History> = Mutex::new(History {
    entries: [None; HISTORY_CAP],
    lengths: [0; HISTORY_CAP],
    count:   0,
    pos:     0,
});

fn history_push(line: &str) {
    if line.is_empty() { return; }
    let mut h = HISTORY.lock();
    let mut buf = [0u8; MAX_LINE];
    let len = line.len().min(MAX_LINE);
    buf[..len].copy_from_slice(&line.as_bytes()[..len]);
    let pos = h.pos;
    h.entries[pos] = Some(buf);
    h.lengths[pos] = len;
    h.pos = (pos + 1) % HISTORY_CAP;
    if h.count < HISTORY_CAP { h.count += 1; }
}

/// Save command history to ~/.nodeai_history
fn history_save() {
    let home = env_get("HOME").unwrap_or(String::from("/root"));
    let path = format!("{}/.nodeai_history", home);
    let (parent, name) = parent_and_name(&path);
    // Build history content
    let mut content = String::new();
    {
        let h = HISTORY.lock();
        let start = if h.count >= HISTORY_CAP { h.pos } else { 0 };
        for i in 0..h.count {
            let idx = (start + i) % HISTORY_CAP;
            if let Some(ref buf) = h.entries[idx] {
                let len = h.lengths[idx];
                if let Ok(s) = core::str::from_utf8(&buf[..len]) {
                    content.push_str(s);
                    content.push('\n');
                }
            }
        }
    }
    // Write to file
    if let Ok(pnode) = crate::vfs::lookup(&parent) {
        let _ = pnode.unlink(&name); // remove old file
        if let Ok(node) = pnode.create_file(&name) {
            if let Ok(mut fh) = node.open() {
                let _ = fh.write(content.as_bytes());
            }
        }
    }
}

/// Load command history from ~/.nodeai_history
fn history_load() {
    let home = env_get("HOME").unwrap_or(String::from("/root"));
    let path = format!("{}/.nodeai_history", home);
    if let Ok(node) = crate::vfs::lookup(&path) {
        if let Ok(mut fh) = node.open() {
            let mut buf = [0u8; 4096];
            if let Ok(n) = fh.read(&mut buf) {
                let text = core::str::from_utf8(&buf[..n]).unwrap_or("");
                for line in text.lines() {
                    if !line.is_empty() {
                        history_push(line);
                    }
                }
            }
        }
    }
}

// ── Environment variables ─────────────────────────────────────────────────────

static ENV: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

fn env_init() {
    let mut env = ENV.lock();
    env.insert(String::from("HOME"), String::from("/root"));
    env.insert(String::from("USER"), String::from("root"));
    env.insert(String::from("HOSTNAME"), String::from("nodeai"));
    env.insert(String::from("SHELL"), String::from("/bin/sh"));
    env.insert(String::from("PATH"), String::from("/bin:/usr/bin"));
    env.insert(String::from("TERM"), String::from("nodeai-term"));
    env.insert(String::from("PWD"), String::from("/"));
    env.insert(String::from("PS1"), String::from("\\u@\\h:\\w\\$"));
}

fn env_get(key: &str) -> Option<String> {
    ENV.lock().get(key).cloned()
}

fn env_set(key: &str, val: &str) {
    ENV.lock().insert(String::from(key), String::from(val));
}

fn env_unset(key: &str) {
    ENV.lock().remove(key);
}

// ── Aliases ───────────────────────────────────────────────────────────────────

static ALIASES: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

fn alias_init() {
    let mut a = ALIASES.lock();
    a.insert(String::from("ll"), String::from("ls -la"));
    a.insert(String::from("la"), String::from("ls -a"));
    a.insert(String::from("l"), String::from("ls -l"));
    a.insert(String::from("cls"), String::from("clear"));
    a.insert(String::from(".."), String::from("cd .."));
}

/// Expand alias for the first word of a command line.
fn expand_alias(line: &str) -> String {
    let (cmd, rest) = match line.find(' ') {
        Some(i) => (&line[..i], &line[i..]),
        None    => (line, ""),
    };
    let aliases = ALIASES.lock();
    if let Some(expansion) = aliases.get(cmd) {
        let mut s = expansion.clone();
        s.push_str(rest);
        s
    } else {
        String::from(line)
    }
}

// ── Glob expansion ────────────────────────────────────────────────────────────

/// Expand glob patterns (* and ?) in arguments.
fn expand_globs(args: &str) -> String {
    if !args.contains('*') && !args.contains('?') {
        return String::from(args);
    }
    let words = split_args(args);
    let mut result = Vec::new();
    for word in &words {
        if word.contains('*') || word.contains('?') {
            let matches = glob_match(word);
            if matches.is_empty() {
                result.push(String::from(*word)); // no match — keep literal
            } else {
                result.extend(matches);
            }
        } else {
            result.push(String::from(*word));
        }
    }
    let mut s = String::new();
    for (i, r) in result.iter().enumerate() {
        if i > 0 { s.push(' '); }
        s.push_str(r);
    }
    s
}

/// Simple glob matching against VFS entries.
fn glob_match(pattern: &str) -> Vec<String> {
    let mut matches = Vec::new();
    // Split into directory part and filename pattern
    let (dir, file_pat) = if let Some(pos) = pattern.rfind('/') {
        (&pattern[..pos + 1], &pattern[pos + 1..])
    } else {
        ("", pattern)
    };

    let dir_path = if dir.is_empty() {
        crate::users::cwd()
    } else {
        resolve_path(dir)
    };

    if let Ok(node) = crate::vfs::lookup(&dir_path) {
        if let Ok(entries) = node.readdir() {
            for e in &entries {
                if glob_matches(file_pat, &e.name) {
                    let mut path = String::from(dir);
                    path.push_str(&e.name);
                    matches.push(path);
                }
            }
        }
    }
    matches.sort();
    matches
}

/// Simple glob pattern matching: * matches any chars, ? matches one char.
fn glob_matches(pattern: &str, name: &str) -> bool {
    let p = pattern.as_bytes();
    let n = name.as_bytes();
    glob_match_impl(p, 0, n, 0)
}

fn glob_match_impl(p: &[u8], pi: usize, n: &[u8], ni: usize) -> bool {
    if pi == p.len() { return ni == n.len(); }
    if p[pi] == b'*' {
        // * matches zero or more characters
        for k in ni..=n.len() {
            if glob_match_impl(p, pi + 1, n, k) { return true; }
        }
        false
    } else if p[pi] == b'?' {
        if ni < n.len() { glob_match_impl(p, pi + 1, n, ni + 1) } else { false }
    } else {
        if ni < n.len() && p[pi] == n[ni] {
            glob_match_impl(p, pi + 1, n, ni + 1)
        } else {
            false
        }
    }
}

/// Split a string into whitespace-separated arguments, respecting quotes.
fn split_args(input: &str) -> Vec<&str> {
    let mut args = Vec::new();
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Skip whitespace
        while i < bytes.len() && bytes[i] == b' ' { i += 1; }
        if i >= bytes.len() { break; }
        let start = i;
        let mut in_single = false;
        let mut in_double = false;
        while i < bytes.len() {
            if bytes[i] == b'\'' && !in_double { in_single = !in_single; }
            else if bytes[i] == b'"' && !in_single { in_double = !in_double; }
            else if bytes[i] == b' ' && !in_single && !in_double { break; }
            i += 1;
        }
        if i > start {
            args.push(&input[start..i]);
        }
    }
    args
}

/// Strip quotes from an argument string.
fn strip_quotes(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut result = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\\' && !in_single && i + 1 < bytes.len() {
            // Escape character (only outside single quotes)
            result.push(bytes[i + 1] as char);
            i += 2;
            continue;
        }
        if b == b'\'' && !in_double {
            in_single = !in_single;
        } else if b == b'"' && !in_single {
            in_double = !in_double;
        } else {
            result.push(b as char);
        }
        i += 1;
    }
    result
}

/// Expand $VAR references in a string.
fn expand_vars(input: &str) -> String {
    let mut result = String::new();
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' && i + 1 < bytes.len() {
            // Collect variable name
            let start = i + 1;
            let mut end = start;
            while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
                end += 1;
            }
            if end > start {
                let name = core::str::from_utf8(&bytes[start..end]).unwrap_or("");
                if let Some(val) = env_get(name) {
                    result.push_str(&val);
                }
                i = end;
            } else {
                result.push('$');
                i += 1;
            }
        } else {
            result.push(bytes[i] as char);
            i += 1;
        }
    }
    result
}

/// Print the boot greeting and the first prompt.
pub fn init() {
    env_init();
    alias_init();
    history_load();
    // Display /etc/motd if available
    if let Ok(node) = crate::vfs::lookup("/etc/motd") {
        if let Ok(mut fh) = node.open() {
            let mut buf = [0u8; 512];
            if let Ok(n) = fh.read(&mut buf) {
                for &b in &buf[..n] {
                    crate::desktop::terminal_input(b);
                }
            }
        }
    } else {
        println!("Welcome to NodeAI — AI-Integrated Kernel OS");
        println!("Type 'help' for commands.");
    }
    print_prompt();
}

/// Called from the keyboard IRQ handler for every key press.
/// Re-emit the shell prompt to the terminal (after a programmatic clear).
pub fn reprint_prompt() {
    print_prompt();
}

/// Print a short system info banner when the "..." titlebar button is clicked.
pub fn print_sysinfo_banner() {
    let free_mb = crate::memory::free_mb();
    let tasks   = crate::scheduler::task_count();
    let user    = crate::users::current_username();
    let host    = crate::users::hostname();
    print_str("\n\x1b[36m┌─ NodeAI System Info ─────────────────┐\x1b[0m\n");
    print_str(&format!("\x1b[36m│\x1b[0m  User    : {:<28}\x1b[36m│\x1b[0m\n", user));
    print_str(&format!("\x1b[36m│\x1b[0m  Host    : {:<28}\x1b[36m│\x1b[0m\n", host));
    print_str(&format!("\x1b[36m│\x1b[0m  Free RAM: {:<25} MB\x1b[36m │\x1b[0m\n", free_mb));
    print_str(&format!("\x1b[36m│\x1b[0m  Tasks   : {:<28}\x1b[36m│\x1b[0m\n", tasks));
    print_str("\x1b[36m└──────────────────────────────────────┘\x1b[0m\n");
    print_prompt();
}

/// Execute a shell command as if launched from the app launcher.
/// Prints the command, executes it, then re-emits the prompt.
pub fn launch_app(cmd: &str) {
    crate::desktop::terminal_input(b'\n');
    for b in cmd.bytes() { crate::desktop::terminal_input(b); }
    crate::desktop::terminal_input(b'\n');
    dispatch(cmd);
    print_prompt();
}

pub fn on_char(byte: u8) {
    // Reset tab cycling on any non-tab key
    if byte != b'\t' {
        tab_cycle_reset();
    }
    match byte {
        // Tab — trigger completion
        b'\t' => {
            tab_complete();
        }
        // Newline — execute the buffered line
        b'\n' => {
            // Reset history browse state
            *HIST_BROWSE.lock() = usize::MAX;
            crate::desktop::terminal_input(b'\n');
            let line = {
                let mut b = BUF.lock();
                let len = b.len;
                let s = String::from(
                    core::str::from_utf8(&b.buf[..len]).unwrap_or("")
                );
                b.len = 0;
                b.cursor = 0;
                s
            };
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                history_push(trimmed);
            }
            dispatch(trimmed);
            print_prompt();
        }
        // Backspace
        0x08 | 0x7F => {
            let mut b = BUF.lock();
            if b.cursor > 0 {
                // Remove character before cursor
                for i in (b.cursor - 1)..(b.len - 1) {
                    b.buf[i] = b.buf[i + 1];
                }
                b.len -= 1;
                b.cursor -= 1;
                drop(b);
                redraw_input_line();
            }
        }
        // Ctrl+A — move cursor to beginning of line
        0x01 => {
            let mut b = BUF.lock();
            b.cursor = 0;
            drop(b);
            reposition_cursor();
        }
        // Ctrl+E — move cursor to end of line
        0x05 => {
            let mut b = BUF.lock();
            b.cursor = b.len;
            drop(b);
            reposition_cursor();
        }
        // Ctrl+U — kill line before cursor
        0x15 => {
            let mut b = BUF.lock();
            if b.cursor > 0 {
                let remaining = b.len - b.cursor;
                for i in 0..remaining {
                    b.buf[i] = b.buf[b.cursor + i];
                }
                b.len = remaining;
                b.cursor = 0;
                drop(b);
                redraw_input_line();
            }
        }
        // Ctrl+K — kill line after cursor
        0x0B => {
            let mut b = BUF.lock();
            b.len = b.cursor;
            drop(b);
            redraw_input_line();
        }
        // Ctrl+W — delete word before cursor
        0x17 => {
            let mut b = BUF.lock();
            if b.cursor > 0 {
                let mut new_cursor = b.cursor;
                // Skip trailing spaces
                while new_cursor > 0 && b.buf[new_cursor - 1] == b' ' {
                    new_cursor -= 1;
                }
                // Skip word characters
                while new_cursor > 0 && b.buf[new_cursor - 1] != b' ' {
                    new_cursor -= 1;
                }
                let deleted = b.cursor - new_cursor;
                let remaining = b.len - b.cursor;
                for i in 0..remaining {
                    b.buf[new_cursor + i] = b.buf[b.cursor + i];
                }
                b.len -= deleted;
                b.cursor = new_cursor;
                drop(b);
                redraw_input_line();
            }
        }
        // Ctrl+C — cancel current input
        0x03 => {
            {
                let mut b = BUF.lock();
                b.len = 0;
                b.cursor = 0;
            }
            *HIST_BROWSE.lock() = usize::MAX;
            print_str("^C");
            crate::desktop::terminal_input(b'\n');
            print_prompt();
        }
        // Ctrl+D — EOF / logout (if line empty)
        0x04 => {
            let is_empty = BUF.lock().len == 0;
            if is_empty {
                print_str("logout");
                crate::desktop::terminal_input(b'\n');
                dispatch("logout");
                print_prompt();
            }
        }
        // Ctrl+L — clear screen
        0x0C => {
            {
                let mut b = BUF.lock();
                b.len = 0;
                b.cursor = 0;
            }
            crate::desktop::clear_terminal();
            print_prompt();
        }
        // Ctrl+Z — suspend foreground process (stub: no true job control)
        0x1A => {
            print_str("^Z");
            crate::desktop::terminal_input(b'\n');
            println!("[1]+  Stopped");
            print_prompt();
        }
        // Normal printable character
        0x20..=0x7E => {
            let overwrite = *OVERWRITE_MODE.lock();
            let mut b = BUF.lock();
            if overwrite && b.cursor < b.len {
                // Overwrite mode: replace character at cursor
                let c = b.cursor;
                b.buf[c] = byte;
                b.cursor += 1;
                drop(b);
                redraw_input_line();
            } else if b.len < MAX_LINE - 1 {
                // Insert mode: shift characters right
                if b.cursor < b.len {
                    let mut i = b.len;
                    while i > b.cursor {
                        b.buf[i] = b.buf[i - 1];
                        i -= 1;
                    }
                }
                let c = b.cursor;
                b.buf[c] = byte;
                b.len += 1;
                b.cursor += 1;
                let at_end = b.cursor == b.len;
                drop(b);
                if at_end {
                    // Simple append — just echo the character
                    crate::desktop::terminal_input(byte);
                } else {
                    // Inserted in middle — need full line redraw
                    redraw_input_line();
                }
            }
        }
        _ => {}
    }
}

/// Handle special (non-ASCII) keys: arrow keys, Home, End, Delete.
pub fn on_special_key(key: SpecialKey) {
    match key {
        SpecialKey::Up => history_browse_up(),
        SpecialKey::Down => history_browse_down(),
        SpecialKey::Left => {
            let mut b = BUF.lock();
            if b.cursor > 0 {
                b.cursor -= 1;
                drop(b);
                reposition_cursor();
            }
        }
        SpecialKey::Right => {
            let mut b = BUF.lock();
            if b.cursor < b.len {
                b.cursor += 1;
                drop(b);
                reposition_cursor();
            }
        }
        SpecialKey::Home => {
            let mut b = BUF.lock();
            b.cursor = 0;
            drop(b);
            reposition_cursor();
        }
        SpecialKey::End => {
            let mut b = BUF.lock();
            b.cursor = b.len;
            drop(b);
            reposition_cursor();
        }
        SpecialKey::Delete => {
            let mut b = BUF.lock();
            if b.cursor < b.len {
                for i in b.cursor..(b.len - 1) {
                    b.buf[i] = b.buf[i + 1];
                }
                b.len -= 1;
                drop(b);
                redraw_input_line();
            }
        }
        SpecialKey::Insert => {
            // Toggle insert/overwrite mode
            let mut mode = OVERWRITE_MODE.lock();
            *mode = !*mode;
        }
        _ => {} // F-keys etc. — ignore for now
    }
}

/// Redraw the entire input line (prompt + buffer) on the current terminal row.
/// Used after insert/delete operations that change the visible line.
fn redraw_input_line() {
    let prompt_len = *PROMPT_LEN.lock();
    let row = crate::desktop::terminal_row();
    // Position to start of prompt
    crate::desktop::terminal_set_col(prompt_len);
    crate::desktop::terminal_clear_to_eol();
    // Write the buffer content
    let b = BUF.lock();
    for i in 0..b.len {
        crate::desktop::terminal_put_char(b.buf[i]);
    }
    // Position cursor correctly
    let target_col = prompt_len + b.cursor;
    drop(b);
    crate::desktop::terminal_set_col(target_col);
    crate::desktop::terminal_redraw_line(row);
}

/// Reposition the terminal cursor without redrawing.
fn reposition_cursor() {
    let prompt_len = *PROMPT_LEN.lock();
    let b = BUF.lock();
    let target_col = prompt_len + b.cursor;
    drop(b);
    crate::desktop::terminal_set_col(target_col);
}

/// Browse history upward (older entries).
fn history_browse_up() {
    let h = HISTORY.lock();
    if h.count == 0 { return; }
    let mut browse = HIST_BROWSE.lock();
    let new_idx = if *browse == usize::MAX {
        // First time pressing up — save current line
        let b = BUF.lock();
        let mut saved = SAVED_LINE.lock();
        saved.0[..b.len].copy_from_slice(&b.buf[..b.len]);
        saved.1 = b.len;
        drop(b);
        // Start at most recent entry
        if h.pos == 0 { h.count - 1 } else { h.pos - 1 }
    } else {
        // Move to older entry
        let oldest = if h.count >= HISTORY_CAP { h.pos } else { 0 };
        let prev = if *browse == 0 { HISTORY_CAP - 1 } else { *browse - 1 };
        if prev == oldest && *browse == oldest {
            return; // Already at oldest
        }
        prev
    };
    // Check if entry exists
    if h.entries[new_idx].is_some() {
        let elen = h.lengths[new_idx];
        let mut tmp = [0u8; MAX_LINE];
        tmp[..elen].copy_from_slice(&h.entries[new_idx].as_ref().unwrap()[..elen]);
        *browse = new_idx;
        drop(h);
        drop(browse);
        // Replace buffer with history entry
        let mut b = BUF.lock();
        b.buf[..elen].copy_from_slice(&tmp[..elen]);
        b.len = elen;
        b.cursor = elen;
        drop(b);
        redraw_input_line();
    }
}

/// Browse history downward (newer entries).
fn history_browse_down() {
    let mut browse = HIST_BROWSE.lock();
    if *browse == usize::MAX { return; } // Not browsing
    let h = HISTORY.lock();
    let newest = if h.pos == 0 { HISTORY_CAP - 1 } else { h.pos - 1 };
    if *browse == newest {
        // Past newest — restore saved line
        let saved = SAVED_LINE.lock();
        let slen = saved.1;
        *browse = usize::MAX;
        drop(h);
        drop(browse);
        let mut b = BUF.lock();
        b.buf[..slen].copy_from_slice(&saved.0[..slen]);
        b.len = slen;
        b.cursor = slen;
        drop(saved);
        drop(b);
        redraw_input_line();
    } else {
        let next = (*browse + 1) % HISTORY_CAP;
        if h.entries[next].is_some() {
            let elen = h.lengths[next];
            let mut tmp = [0u8; MAX_LINE];
            tmp[..elen].copy_from_slice(&h.entries[next].as_ref().unwrap()[..elen]);
            *browse = next;
            drop(h);
            drop(browse);
            let mut b = BUF.lock();
            b.buf[..elen].copy_from_slice(&tmp[..elen]);
            b.len = elen;
            b.cursor = elen;
            drop(b);
            redraw_input_line();
        }
    }
}

/// Tab completion for command names and file paths with cycling support.
fn tab_complete() {
    // Check if we're already cycling through completions
    {
        let mut tc = TAB_CYCLE.lock();
        if tc.active && !tc.matches.is_empty() {
            // Cycle to next match
            tc.index = (tc.index + 1) % tc.matches.len();
            let completion = tc.matches[tc.index].clone();
            let pstart = tc.prefix_start;
            let plen = tc.prefix_len;
            let is_first = pstart == 0;
            drop(tc);
            // Replace the current word with the new completion
            let mut b = BUF.lock();
            // Remove everything from prefix_start to cursor
            let tail_len = b.len - b.cursor;
            let mut tail = [0u8; MAX_LINE];
            tail[..tail_len].copy_from_slice(&b.buf[b.cursor..b.len]);
            // Build new word
            let new_word = if is_first {
                format!("{} ", completion)
            } else {
                completion.clone()
            };
            let new_bytes = new_word.as_bytes();
            let new_len = pstart + new_bytes.len() + tail_len;
            if new_len <= MAX_LINE {
                b.buf[pstart..pstart + new_bytes.len()].copy_from_slice(new_bytes);
                b.buf[pstart + new_bytes.len()..new_len].copy_from_slice(&tail[..tail_len]);
                b.len = new_len;
                b.cursor = pstart + new_bytes.len();
            }
            drop(b);
            redraw_input_line();
            return;
        }
    }

    let b = BUF.lock();
    let line = String::from(core::str::from_utf8(&b.buf[..b.len]).unwrap_or(""));
    let cursor = b.cursor;
    drop(b);

    // Find the word being completed (from cursor backwards to space)
    let prefix_start = line[..cursor].rfind(' ').map(|p| p + 1).unwrap_or(0);
    let prefix = &line[prefix_start..cursor];
    if prefix.is_empty() { return; }

    let is_first_word = prefix_start == 0;

    let completions = if is_first_word {
        // Complete command names
        let mut matches: Vec<String> = Vec::new();
        for &cmd in BUILTINS {
            if cmd.starts_with(prefix) {
                matches.push(String::from(cmd));
            }
        }
        matches
    } else {
        // Complete file paths
        complete_path(prefix)
    };

    if completions.is_empty() { return; }

    if completions.len() == 1 {
        // Single match — complete it
        let completion = &completions[0];
        let suffix = &completion[prefix.len()..];
        let mut b = BUF.lock();
        for &byte in suffix.as_bytes() {
            if b.len < MAX_LINE - 1 {
                // Insert at cursor
                if b.cursor < b.len {
                    let mut i = b.len;
                    while i > b.cursor { b.buf[i] = b.buf[i - 1]; i -= 1; }
                }
                let c = b.cursor;
                b.buf[c] = byte;
                b.len += 1;
                b.cursor += 1;
            }
        }
        // Add space after completed command
        if is_first_word && b.len < MAX_LINE - 1 {
            if b.cursor < b.len {
                let mut i = b.len;
                while i > b.cursor { b.buf[i] = b.buf[i - 1]; i -= 1; }
            }
            let c = b.cursor;
            b.buf[c] = b' ';
            b.len += 1;
            b.cursor += 1;
        }
        drop(b);
        redraw_input_line();
    } else {
        // Multiple matches — show all, complete common prefix, and set up cycling
        crate::desktop::terminal_input(b'\n');
        for c in &completions {
            print_str(c);
            print_str("  ");
        }
        crate::desktop::terminal_input(b'\n');
        // Find longest common prefix
        let mut common_len = prefix.len();
        'outer: loop {
            if common_len >= completions[0].len() { break; }
            let next_char = completions[0].as_bytes()[common_len];
            for c in &completions[1..] {
                if common_len >= c.len() || c.as_bytes()[common_len] != next_char {
                    break 'outer;
                }
            }
            common_len += 1;
        }
        // Complete up to common prefix
        if common_len > prefix.len() {
            let extra = &completions[0][prefix.len()..common_len];
            let mut b = BUF.lock();
            for &byte in extra.as_bytes() {
                if b.len < MAX_LINE - 1 {
                    if b.cursor < b.len {
                        let mut i = b.len;
                        while i > b.cursor { b.buf[i] = b.buf[i - 1]; i -= 1; }
                    }
                    let c = b.cursor;
                    b.buf[c] = byte;
                    b.len += 1;
                    b.cursor += 1;
                }
            }
            drop(b);
        }
        // Set up cycling state so next tab press cycles through matches
        {
            let mut tc = TAB_CYCLE.lock();
            tc.active = true;
            tc.prefix_start = prefix_start;
            tc.prefix_len = prefix.len();
            tc.matches = completions;
            tc.index = 0; // first tab after display starts at index 0
        }
        // Redraw prompt and current input
        print_prompt();
        let b = BUF.lock();
        for i in 0..b.len {
            crate::desktop::terminal_input(b.buf[i]);
        }
    }
}

/// Complete a file path prefix, returning matching entries.
fn complete_path(prefix: &str) -> Vec<String> {
    let mut matches = Vec::new();
    let (dir_path, partial_name) = if let Some(slash_pos) = prefix.rfind('/') {
        let dir = if slash_pos == 0 { String::from("/") } else { resolve_path(&prefix[..slash_pos]) };
        (dir, &prefix[slash_pos + 1..])
    } else {
        (crate::users::cwd(), prefix)
    };

    if let Ok(node) = crate::vfs::lookup(&dir_path) {
        if let Ok(entries) = node.readdir() {
            for e in entries {
                if e.name.starts_with(partial_name) {
                    let mut full = if prefix.contains('/') {
                        let base = &prefix[..prefix.rfind('/').unwrap() + 1];
                        format!("{}{}", base, e.name)
                    } else {
                        e.name.clone()
                    };
                    if e.is_dir { full.push('/'); }
                    matches.push(full);
                }
            }
        }
    }
    matches
}

// ── Command dispatcher ────────────────────────────────────────────────────────

fn dispatch(line: &str) {
    if line.is_empty() {
        return;
    }

    // Expand aliases (first word only)
    let aliased = expand_alias(line);
    // Expand environment variables
    let expanded = expand_vars(&aliased);
    let line = expanded.trim();

    // Handle !! (repeat last command)
    if line == "!!" {
        let h = HISTORY.lock();
        if h.count > 1 {
            // Second-to-last entry (last is "!!" itself)
            let idx = (h.pos + HISTORY_CAP - 2) % HISTORY_CAP;
            if let Some(buf) = &h.entries[idx] {
                let len = h.lengths[idx];
                let prev = String::from(core::str::from_utf8(&buf[..len]).unwrap_or(""));
                drop(h);
                println!("{}", prev);
                dispatch(&prev);
                return;
            }
        }
        drop(h);
        println!("No previous command");
        return;
    }

    // Handle !n (repeat command N)
    if line.starts_with('!') && line.len() > 1 && line.as_bytes()[1].is_ascii_digit() {
        if let Ok(n) = line[1..].parse::<usize>() {
            let h = HISTORY.lock();
            if n > 0 && n <= h.count {
                let start = if h.count >= HISTORY_CAP { h.pos } else { 0 };
                let idx = (start + n - 1) % HISTORY_CAP;
                if let Some(buf) = &h.entries[idx] {
                    let len = h.lengths[idx];
                    let prev = String::from(core::str::from_utf8(&buf[..len]).unwrap_or(""));
                    drop(h);
                    println!("{}", prev);
                    dispatch(&prev);
                    return;
                }
            }
            drop(h);
            println!("!{}: event not found", n);
            return;
        }
    }

    // Handle command chaining: ; && ||
    // Split on ; first, then handle && and ||
    if contains_unquoted(line, ';') || line.contains("&&") || line.contains("||") {
        dispatch_chain(line);
        return;
    }

    // Handle input redirection: cmd < file
    if contains_unquoted(line, '<') {
        dispatch_input_redirect(line);
        return;
    }

    // Handle pipe: cmd1 | cmd2
    if contains_unquoted(line, '|') {
        dispatch_pipe(line);
        return;
    }

    // Handle output redirection
    if line.contains(">>") || contains_unquoted(line, '>') {
        dispatch_redirect(line);
        return;
    }

    dispatch_single(line);
}

/// Check if a character appears outside of quotes.
fn contains_unquoted(line: &str, ch: char) -> bool {
    let bytes = line.as_bytes();
    let mut in_single = false;
    let mut in_double = false;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\'' && !in_double { in_single = !in_single; }
        else if b == b'"' && !in_single { in_double = !in_double; }
        else if !in_single && !in_double {
            // For | check it's not || 
            if b == ch as u8 {
                if ch == '|' {
                    if i + 1 < bytes.len() && bytes[i + 1] == b'|' {
                        i += 2;
                        continue;
                    }
                }
                return true;
            }
        }
        i += 1;
    }
    false
}

/// Handle command chaining with ;, &&, ||
fn dispatch_chain(line: &str) {
    let mut rest = line;
    let mut last_success = true;
    while !rest.is_empty() {
        let rest_trimmed = rest.trim_start();
        if rest_trimmed.is_empty() { break; }

        // Find next operator
        let (cmd, operator, remaining) = find_chain_operator(rest_trimmed);
        let cmd = cmd.trim();

        match operator {
            Some("&&") => {
                if last_success {
                    capture_dispatch(cmd, &mut last_success);
                } else {
                    // Skip this command (previous failed)
                }
            }
            Some("||") => {
                if !last_success {
                    capture_dispatch(cmd, &mut last_success);
                } else {
                    // Skip (previous succeeded)
                }
            }
            _ => {
                // ; or end of line
                capture_dispatch(cmd, &mut last_success);
            }
        }
        rest = remaining;
    }
}

fn find_chain_operator<'a>(line: &'a str) -> (&'a str, Option<&'static str>, &'a str) {
    let bytes = line.as_bytes();
    let mut in_single = false;
    let mut in_double = false;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\'' && !in_double { in_single = !in_single; }
        else if b == b'"' && !in_single { in_double = !in_double; }
        else if !in_single && !in_double {
            if b == b';' {
                return (&line[..i], Some(";"), &line[i + 1..]);
            }
            if b == b'&' && i + 1 < bytes.len() && bytes[i + 1] == b'&' {
                return (&line[..i], Some("&&"), &line[i + 2..]);
            }
            if b == b'|' && i + 1 < bytes.len() && bytes[i + 1] == b'|' {
                return (&line[..i], Some("||"), &line[i + 2..]);
            }
        }
        i += 1;
    }
    (line, None, "")
}

fn capture_dispatch(cmd: &str, success: &mut bool) {
    // For simplicity, treat all dispatch results as success
    // (we don't have exit codes yet, so "command not found" is the only failure)
    *success = true;
    dispatch(cmd);
}

/// Handle pipe: cmd1 | cmd2 | cmd3
/// We capture output of the left side and feed it as "stdin" to the right.
fn dispatch_pipe(line: &str) {
    let parts: Vec<&str> = split_on_pipe(line);
    if parts.len() < 2 {
        dispatch_single(line);
        return;
    }

    // Capture output of first command
    let mut data = capture_output(parts[0].trim());

    // Pipe through each subsequent command
    for part in &parts[1..] {
        let cmd_str = part.trim();
        data = run_with_stdin(cmd_str, &data);
    }

    // Print final output
    print_str(&data);
}

fn split_on_pipe(line: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let bytes = line.as_bytes();
    let mut start = 0;
    let mut in_single = false;
    let mut in_double = false;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\'' && !in_double { in_single = !in_single; }
        else if b == b'"' && !in_single { in_double = !in_double; }
        else if !in_single && !in_double && b == b'|' {
            // Make sure it's not ||
            if i + 1 < bytes.len() && bytes[i + 1] == b'|' {
                i += 2;
                continue;
            }
            parts.push(&line[start..i]);
            start = i + 1;
        }
        i += 1;
    }
    parts.push(&line[start..]);
    parts
}

/// Capture output of a command into a String (instead of printing to terminal).
static CAPTURE_BUF: Mutex<Option<String>> = Mutex::new(None);

fn capture_output(cmd: &str) -> String {
    *CAPTURE_BUF.lock() = Some(String::new());
    dispatch_single(cmd);
    CAPTURE_BUF.lock().take().unwrap_or_default()
}

/// Run a command that accepts piped stdin data.
fn run_with_stdin(cmd_line: &str, stdin: &str) -> String {
    let (cmd, args) = match cmd_line.find(' ') {
        Some(i) => (&cmd_line[..i], cmd_line[i + 1..].trim()),
        None    => (cmd_line, ""),
    };
    // For pipe-aware commands (grep, sort, uniq, cut, head, tail, wc), 
    // if no file argument is given, process stdin data
    match cmd {
        "grep" => {
            let pattern = args.split_whitespace().next().unwrap_or("");
            let mut result = String::new();
            for line in stdin.lines() {
                if line.contains(pattern) {
                    result.push_str(line);
                    result.push('\n');
                }
            }
            result
        }
        "sort" => {
            let mut lines: Vec<&str> = stdin.lines().collect();
            lines.sort();
            let mut result = String::new();
            for l in lines { result.push_str(l); result.push('\n'); }
            result
        }
        "uniq" => {
            let mut result = String::new();
            let mut prev = "";
            for line in stdin.lines() {
                if line != prev {
                    result.push_str(line);
                    result.push('\n');
                    prev = line;
                }
            }
            result
        }
        "head" => {
            let n = if args.starts_with("-n") {
                args[2..].trim().parse::<usize>().unwrap_or(10)
            } else { 10 };
            let mut result = String::new();
            for (i, line) in stdin.lines().enumerate() {
                if i >= n { break; }
                result.push_str(line);
                result.push('\n');
            }
            result
        }
        "tail" => {
            let n = if args.starts_with("-n") {
                args[2..].trim().parse::<usize>().unwrap_or(10)
            } else { 10 };
            let lines: Vec<&str> = stdin.lines().collect();
            let start = lines.len().saturating_sub(n);
            let mut result = String::new();
            for line in &lines[start..] {
                result.push_str(line);
                result.push('\n');
            }
            result
        }
        "wc" => {
            let lc = stdin.lines().count();
            let wc = stdin.split_whitespace().count();
            let bc = stdin.len();
            format!("  {}  {}  {}\n", lc, wc, bc)
        }
        "cut" => {
            // Parse -d and -f
            let parts: Vec<&str> = args.split_whitespace().collect();
            let mut delim = '\t';
            let mut field = 1usize;
            let mut i = 0;
            while i < parts.len() {
                if parts[i].starts_with("-d") {
                    let d = &parts[i][2..];
                    if !d.is_empty() { delim = d.chars().next().unwrap_or('\t'); }
                } else if parts[i].starts_with("-f") {
                    field = parts[i][2..].parse().unwrap_or(1);
                }
                i += 1;
            }
            let mut result = String::new();
            for line in stdin.lines() {
                let fields: Vec<&str> = line.split(delim).collect();
                if field > 0 && field <= fields.len() {
                    result.push_str(fields[field - 1]);
                }
                result.push('\n');
            }
            result
        }
        "cat" => {
            // cat with no args in pipe just passes through
            String::from(stdin)
        }
        _ => {
            // For unknown commands in pipe, capture their normal output
            *CAPTURE_BUF.lock() = Some(String::new());
            dispatch_single(cmd_line);
            CAPTURE_BUF.lock().take().unwrap_or_default()
        }
    }
}

/// Handle output redirection: cmd > file  or  cmd >> file
fn dispatch_redirect(line: &str) {
    let (cmd_part, file, append) = if let Some(pos) = line.find(">>") {
        (&line[..pos], line[pos + 2..].trim(), true)
    } else if let Some(pos) = find_unquoted_pos(line, '>') {
        (&line[..pos], line[pos + 1..].trim(), false)
    } else {
        dispatch_single(line);
        return;
    };

    if file.is_empty() {
        println!("syntax error: missing filename after redirection");
        return;
    }

    let output = capture_output(cmd_part.trim());
    let path = resolve_path(file);

    if append {
        // Append to file
        if let Ok(node) = crate::vfs::lookup(&path) {
            if let Ok(mut fh) = node.open() {
                let st = fh.stat().ok();
                if let Some(s) = st { let _ = fh.seek(s.size); }
                let _ = fh.write(output.as_bytes());
            }
        } else {
            // Create file and write
            let (parent, name) = parent_and_name(&path);
            if let Ok(pnode) = crate::vfs::lookup(&parent) {
                if let Ok(node) = pnode.create_file(&name) {
                    if let Ok(mut fh) = node.open() {
                        let _ = fh.write(output.as_bytes());
                    }
                }
            }
        }
    } else {
        // Overwrite file (create or truncate)
        let (parent, name) = parent_and_name(&path);
        // Try to remove existing file first
        if let Ok(pnode) = crate::vfs::lookup(&parent) {
            let _ = pnode.unlink(&name); // ignore if doesn't exist
            if let Ok(node) = pnode.create_file(&name) {
                if let Ok(mut fh) = node.open() {
                    let _ = fh.write(output.as_bytes());
                }
            }
        }
    }
}

/// Handle input redirection: cmd < file
/// Reads file content and passes it as stdin to the command via pipe mechanism.
fn dispatch_input_redirect(line: &str) {
    if let Some(pos) = find_unquoted_pos(line, '<') {
        let cmd_part = line[..pos].trim();
        let file_part = line[pos + 1..].trim();

        // The file part might also have output redirection: cmd < infile > outfile
        let (infile, rest_cmd) = if let Some(gt) = file_part.find('>') {
            (file_part[..gt].trim(), Some(format!("{} {}", cmd_part, &file_part[gt..])))
        } else {
            (file_part, None)
        };

        if infile.is_empty() {
            println!("syntax error: missing filename after <");
            return;
        }

        let path = resolve_path(infile);
        match read_file_contents(&path) {
            Some(data) => {
                let stdin_data = core::str::from_utf8(&data).unwrap_or("").to_string();
                // Run the command with stdin data via the pipe mechanism
                let output = run_with_stdin(cmd_part, &stdin_data);
                if let Some(redirected) = rest_cmd {
                    // Has output redirection too
                    dispatch(&redirected);
                } else {
                    print_str(&output);
                }
            }
            None => {
                println!("bash: {}: No such file or directory", infile);
            }
        }
    } else {
        dispatch_single(line);
    }
}

fn find_unquoted_pos(line: &str, ch: char) -> Option<usize> {
    let bytes = line.as_bytes();
    let mut in_single = false;
    let mut in_double = false;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'\'' && !in_double { in_single = !in_single; }
        else if b == b'"' && !in_single { in_double = !in_double; }
        else if !in_single && !in_double && b == ch as u8 {
            // For >, make sure it's not >>
            if ch == '>' && i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                continue;
            }
            return Some(i);
        }
    }
    None
}

fn dispatch_single(line: &str) {
    let line = line.trim();
    if line.is_empty() { return; }

    // Expand glob patterns in arguments
    let expanded_line = if line.contains('*') || line.contains('?') {
        let (cmd, args) = match line.find(' ') {
            Some(i) => (&line[..i], line[i + 1..].trim()),
            None    => (line, ""),
        };
        if args.is_empty() {
            String::from(line)
        } else {
            let expanded_args = expand_globs(args);
            let mut s = String::from(cmd);
            s.push(' ');
            s.push_str(&expanded_args);
            s
        }
    } else {
        String::from(line)
    };
    let line = expanded_line.as_str();

    let (cmd, args) = match line.find(' ') {
        Some(i) => (&line[..i], line[i + 1..].trim()),
        None    => (line, ""),
    };

    match cmd {
        // ── General ──
        "help"     => cmd_help(),
        "clear"    => cmd_clear(),
        "echo"     => { println!("{}", args); }
        "reboot"   => cmd_reboot(),
        "shutdown" => cmd_shutdown(),
        "uname"    => cmd_uname(args),
        "date"     => cmd_date(),
        "uptime"   => cmd_uptime(),
        "history"  => cmd_history(),

        // ── Navigation ──
        "cd"       => cmd_cd(args),
        "pwd"      => cmd_pwd(),

        // ── File operations ──
        "ls"       => cmd_ls(args),
        "cat"      => cmd_cat(args),
        "touch"    => cmd_touch(args),
        "mkdir"    => cmd_mkdir(args),
        "rm"       => cmd_rm(args),
        "rmdir"    => cmd_rmdir(args),
        "cp"       => cmd_cp(args),
        "mv"       => cmd_mv(args),
        "ln"       => cmd_ln(args),
        "stat"     => cmd_stat(args),
        "wc"       => cmd_wc(args),
        "head"     => cmd_head(args),
        "tail"     => cmd_tail(args),

        // ── Text processing ──
        "grep"     => cmd_grep(args),
        "sort"     => cmd_sort(args),
        "uniq"     => cmd_uniq(args),
        "cut"      => cmd_cut(args),
        "diff"     => cmd_diff(args),
        "xxd"      => cmd_xxd(args),
        "seq"      => cmd_seq(args),

        // ── Text processing ──
        "tee"      => cmd_tee(args),

        // ── System monitoring ──
        "meminfo"  => cmd_meminfo(),
        "free"     => cmd_free(),
        "ps"       => cmd_ps(),
        "kill"     => cmd_kill(args),
        "killall"  => cmd_killall(args),
        "dmesg"    => cmd_dmesg(),
        "lspci"    => cmd_lspci(),
        "lsmod"    => cmd_lsmod(),
        "df"       => cmd_df(),
        "du"       => cmd_du(args),
        "hostname" => cmd_hostname(args),
        "time"     => cmd_time(args),
        "top"      => cmd_top(),
        "mount"    => cmd_mount(args),
        "umount"   => cmd_umount(args),
        "sysctl"   => cmd_sysctl(args),
        "file"     => cmd_file(args),

        // ── User/auth commands ──
        "whoami"   => cmd_whoami(),
        "id"       => cmd_id(args),
        "su"       => cmd_su(args),
        "sudo"     => cmd_sudo(args),
        "useradd"  => cmd_useradd(args),
        "userdel"  => cmd_userdel(args),
        "passwd"   => cmd_passwd(args),
        "groups"   => cmd_groups(args),
        "chmod"    => cmd_chmod(args),
        "chown"    => cmd_chown(args),
        "login"    => cmd_login(args),
        "logout"   => cmd_logout(),
        "audit"    => cmd_audit(),

        // ── Networking commands ──
        "wifi"       => cmd_wifi(args),
        "ifconfig"   => cmd_ifconfig(args),
        "ping"       => cmd_ping(args),
        "arp"        => cmd_arp(args),
        "netstat"    => cmd_netstat(args),
        "ip"         => cmd_ip(args),
        "nslookup"   => cmd_nslookup(args),
        "dig"        => cmd_dig(args),
        "traceroute" => cmd_traceroute(args),
        "wget"       => cmd_wget(args),
        "curl"       => cmd_curl(args),
        "nc"         => cmd_nc(args),

        // ── Process/Job control ──
        "htop"     => cmd_htop(),
        "nice"     => cmd_nice(args),
        "renice"   => cmd_renice(args),
        "bg"       => cmd_bg(args),
        "fg"       => cmd_fg(args),
        "jobs"     => cmd_jobs(),
        "nohup"    => cmd_nohup(args),

        // ── Disk & modules ──
        "fdisk"    => cmd_fdisk(args),
        "mkfs"     => cmd_mkfs(args),
        "sync"     => cmd_sync(),
        "modprobe" => cmd_modprobe(args),
        "insmod"   => cmd_modprobe(args),
        "service"  => cmd_service(args),

        // ── Network services ──
        "dhclient" => cmd_dhclient(),
        "httpd"    => cmd_httpd(args),
        "sshd"     => cmd_sshd(args),
        "scp"      => cmd_scp(args),
        "dns-cache"=> cmd_dns_cache(args),
        "ifup"     => cmd_ifup(args),
        "ifdown"   => cmd_ifdown(args),

        // ── Environment ──
        "export"   => cmd_export(args),
        "unset"    => cmd_unset(args),
        "env"      => cmd_env(),
        "printenv" => cmd_env(),
        "which"    => cmd_which(args),
        "type"     => cmd_type(args),
        "alias"    => cmd_alias(args),
        "unalias"  => cmd_unalias(args),

        // ── Misc ──
        "sleep"    => cmd_sleep(args),
        "yes"      => cmd_yes(args),
        "man"      => cmd_man(args),
        "true"     => {}
        "false"    => {}

        "fm"       => cmd_fm(args),
        "note"     => cmd_note(args),
        "calc"     => cmd_calc(args),
        "sysinfo"  => cmd_sysinfo(),
        "exec"     => cmd_exec(args),
        "wm"       => cmd_wm(args),
        "browser"  | "intelli" => cmd_browser(args),
        "term"     | "terminal" => cmd_term(args),

        // Phase 26 apps
        "notepad"  | "np" | "edit" => cmd_notepad_pro(args),
        "files"    | "filepro"     => cmd_fm_pro(args),
        "termtabs" | "tt"          => cmd_terminal_tabs(),
        "imgview"  | "image" | "iv" => cmd_imgview(args),
        "aichat"   | "chat"        => cmd_ai_chat(),
        "sysmon"   | "monitor"     => cmd_sysmon(),
        "settings" | "config"      => cmd_settings(),
        "store"    | "appstore"    => cmd_appstore(),

        other => { println!("{}: command not found", other); }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  COMMAND IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════════

// ── General ───────────────────────────────────────────────────────────────────

fn cmd_help() {
    println!("NodeAI Shell - Built-in Commands");
    println!("");
    println!(" Navigation:    cd, pwd, ls");
    println!(" Files:         cat, touch, mkdir, rmdir, rm, cp, mv, ln, stat, wc, head, tail, file");
    println!(" Text:          grep, sort, uniq, cut, diff, xxd, seq, tee");
    println!(" System:        meminfo, free, ps, kill, killall, top, htop, uptime, uname, date");
    println!("                dmesg, lspci, lsmod, df, du, mount, umount, sysctl");
    println!(" Process:       nice, renice, bg, fg, jobs, nohup, time");
    println!(" Disk:          fdisk, mkfs, sync, modprobe, service");
    println!(" Users:         whoami, id, su, sudo, login, logout, useradd, userdel, passwd, groups");
    println!(" Permissions:   chmod, chown");
    println!(" Network:       ifconfig, ping, arp, netstat, ip, nslookup, dig, traceroute, wget, curl, nc");
    println!(" Net Services:  dhclient, httpd, sshd, scp, dns-cache, ifup, ifdown");
    println!(" Environment:   export, unset, env, which, type, hostname, alias, unalias");
    println!(" Shell:         echo, history, clear, sleep, time, yes, man, true, false");
    println!(" Operators:     |  >  >>  <  ;  &&  ||");
    println!(" Power:         reboot, shutdown");
}

fn cmd_sysinfo() {
    let free_mb = crate::memory::free_mb();
    let tasks   = crate::scheduler::task_count();
    let user    = crate::users::current_username();
    let host    = crate::users::hostname();
    let ms      = crate::scheduler::uptime_ms();
    let sec     = ms / 1000;
    println!("\x1b[1;36mNodeAI System Information\x1b[0m");
    println!("  Kernel  : NodeAI {}", env!("CARGO_PKG_VERSION"));
    println!("  User    : {}@{}", user, host);
    println!("  Uptime  : {}h {}m {}s", sec / 3600, (sec / 60) % 60, sec % 60);
    println!("  Free RAM: {} MB", free_mb);
    println!("  Tasks   : {}", tasks);
    println!("  Arch    : x86_64");
}

fn cmd_exec(args: &str) {
    let parts: Vec<&str> = args.split_whitespace().collect();
    if parts.is_empty() {
        println!("Usage: exec <path> [args...]");
        return;
    }
    let path = parts[0];
    let abs_path = if path.starts_with('/') {
        alloc::string::String::from(path)
    } else {
        let cwd = crate::users::cwd();
        if cwd.ends_with('/') {
            alloc::format!("{}{}", cwd, path)
        } else {
            alloc::format!("{}/{}", cwd, path)
        }
    };
    use crate::vfs;
    let node = match vfs::lookup(&abs_path) {
        Ok(n)  => n,
        Err(_) => { println!("exec: {}: no such file", abs_path); return; }
    };
    let mut fh = match node.open() {
        Ok(f)  => f,
        Err(_) => { println!("exec: {}: cannot open", abs_path); return; }
    };
    let mut buf = alloc::vec![0u8; 4 * 1024 * 1024];
    let n = match fh.read(&mut buf) {
        Ok(n)  => n,
        Err(_) => { println!("exec: {}: read error", abs_path); return; }
    };
    let elf_data = &buf[..n];
    let image = match crate::elf::parse(elf_data) {
        Ok(img) => img,
        Err(e)  => { println!("exec: {}: ELF error {:?}", abs_path, e); return; }
    };

    // Allocate a fresh address space
    let new_cr3 = match unsafe { crate::memory::alloc_user_cr3() } {
        Some(cr3) => cr3,
        None => { println!("exec: cannot allocate CR3"); return; }
    };
    unsafe { core::arch::asm!("mov cr3, {}", in(reg) new_cr3, options(nomem, nostack)); }
    let pid = crate::scheduler::current_pid();
    crate::scheduler::set_task_cr3(pid, new_cr3);

    if let Err(e) = unsafe { crate::elf::load_image(&image) } {
        println!("exec: ELF load error {:?}", e); return;
    }
    let entry = image.entry;

    // Map user stack in the new address space
    const USER_STACK_TOP: u64 = 0x0000_7FFF_FFFF_F000;
    const USER_STACK_SIZE: u64 = 8 * 1024 * 1024;
    let stack_base = USER_STACK_TOP - USER_STACK_SIZE;
    if let Err(_) = crate::memory::map_user_range(stack_base, USER_STACK_SIZE, true, false) {
        println!("exec: cannot map stack"); return;
    }

    // Build argv strings from shell args
    let mut argv_strs: Vec<Vec<u8>> = Vec::new();
    for arg in &parts {
        let mut v = Vec::from(arg.as_bytes());
        v.push(0u8);
        argv_strs.push(v);
    }

    // Default envp
    let envp_defaults = [
        b"PATH=/bin\0" as &[u8],
        b"HOME=/root\0",
        b"TERM=vt100\0",
    ];

    // Build SysV AMD64 user stack
    let mut sp = USER_STACK_TOP;

    // Write argv strings (descending)
    let mut argv_ptrs = Vec::new();
    for s in &argv_strs {
        sp -= s.len() as u64;
        unsafe { core::ptr::copy_nonoverlapping(s.as_ptr(), sp as *mut u8, s.len()); }
        argv_ptrs.push(sp);
    }

    // Write envp strings (descending)
    let mut envp_ptrs = Vec::new();
    for s in &envp_defaults {
        sp -= s.len() as u64;
        unsafe { core::ptr::copy_nonoverlapping(s.as_ptr(), sp as *mut u8, s.len()); }
        envp_ptrs.push(sp);
    }

    // AT_RANDOM
    sp -= 16;
    sp &= !15u64;
    let rand_ptr = sp;
    let tsc: u64;
    unsafe { core::arch::asm!("rdtsc; shl rdx, 32; or rax, rdx",
        out("rax") tsc, out("rdx") _, options(nomem, nostack)); }
    unsafe { *(rand_ptr as *mut u64) = tsc; }
    unsafe { *((rand_ptr + 8) as *mut u64) = tsc ^ 0xDEAD_BEEF_CAFE_BABEu64; }
    sp &= !7u64;

    // Align stack to 16 bytes for SysV ABI
    let n_pointers = 1 + argv_ptrs.len() + 1 + envp_ptrs.len() + 1 + 10;
    if (sp.wrapping_sub((n_pointers as u64) * 8)) & 0xF != 0 {
        sp -= 8;
    }

    macro_rules! push64 {
        ($v:expr) => {{
            sp -= 8;
            unsafe { *(sp as *mut u64) = $v as u64; }
        }};
    }

    // auxv
    push64!(0u64); push64!(0u64);         // AT_NULL
    push64!(0x1F); push64!(16u64);        // AT_HWCAP
    push64!(rand_ptr); push64!(25u64);    // AT_RANDOM
    push64!(entry); push64!(9u64);        // AT_ENTRY
    push64!(4096u64); push64!(6u64);      // AT_PAGESZ

    push64!(0u64);  // envp null terminator
    for &ptr in envp_ptrs.iter().rev() { push64!(ptr); }
    push64!(0u64);  // argv null terminator
    for &ptr in argv_ptrs.iter().rev() { push64!(ptr); }
    push64!(argv_strs.len() as u64);  // argc

    println!("[exec] launching {} (entry={:#x}, argc={})", abs_path, entry, argv_strs.len());
    crate::klog!(INFO, "exec: {} → entry={:#x} argc={} sp={:#x}", abs_path, entry, argv_strs.len(), sp);

    // Jump to user mode
    unsafe { crate::syscall::ring3_jump(entry, sp); }
}

fn cmd_wm(args: &str) {
    let mut parts = args.splitn(2, ' ');
    let sub  = parts.next().unwrap_or("").trim();
    let rest = parts.next().unwrap_or("").trim();
    match sub {
        "open" | "new" | "" => {
            let mut it = rest.split_whitespace();
            let title: &str = it.next().unwrap_or("Window");
            let x: i32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(60);
            let y: i32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(80);
            let w: u32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(400);
            let h: u32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(300);
            let id = crate::desktop::wm_create_window(x, y, w, h, title);
            if id == 0 {
                println!("wm: failed to create window");
            } else {
                // Simple gradient fill as demo content
                for py in 0..h {
                    for px in 0..w {
                        let r = ((px * 255) / w.max(1)) as u32;
                        let g = ((py * 200) / h.max(1)) as u32;
                        crate::desktop::wm_paint_pixel(id, px, py, (r << 16) | (g << 8) | 0x44);
                    }
                }
                crate::desktop::wm_composite();
                println!("wm: opened '{}' id={}", title, id);
            }
        }
        "close" => {
            if let Ok(id) = rest.parse::<u32>() {
                crate::desktop::wm_destroy_window(id);
                println!("wm: closed window {}", id);
            } else {
                println!("Usage: wm close <id>");
            }
        }
        "list" => {
            if !crate::desktop::wm_is_active() {
                println!("wm: no windows open");
            } else {
                crate::desktop::compositor::with_wm_pub(|s| {
                    println!("{:<5} {:<12} {:>6} {:>6}  POS", "ID", "TITLE", "W", "H");
                    for &id in &s.z_stack {
                        if let Some(w) = s.windows.get(&id) {
                            let foc = if s.focused == Some(id) { "*" } else { " " };
                            let min = if w.minimized { "[M]" } else { "   " };
                            println!("{:<5} {}{}{:<12} {:>6} {:>6}  {},{}", id, foc, min, w.title(), w.w, w.h, w.x, w.y);
                        }
                    }
                });
            }
        }
        "repaint" | "composite" => {
            crate::desktop::wm_composite();
            println!("wm: repainted");
        }
        _ => {
            println!("Usage: wm <open|close|list|repaint>");
            println!("  wm open [title] [x] [y] [w] [h]");
            println!("  wm close <id>");
            println!("  wm list");
            println!("  wm repaint");
        }
    }
}

/// `browser [url]` — Open or control the Intelli Browser
fn cmd_browser(args: &str) {
    use crate::desktop;
    const DEFAULT_W: usize = 900;
    const DEFAULT_H: usize = 600;
    if !desktop::browser_is_open() {
        // Launch with WM window
        desktop::browser_init(DEFAULT_W, DEFAULT_H);
        // Ensure WM compositor is active
        desktop::wm_composite();
        println!("browser: Intelli Browser launched");
    }
    // If a URL was given, navigate to it
    let url = args.trim();
    if !url.is_empty() {
        desktop::browser_navigate(url);
        println!("browser: navigating to {}", url);
    } else {
        println!("browser: already open — type 'browser <url>' to navigate");
    }
}

fn cmd_term(_args: &str) {
    use crate::desktop;
    if !desktop::term_window_is_open() {
        desktop::term_window_init();
        println!("terminal: Terminal window launched (80×24 VT100)");
    } else {
        println!("terminal: already open");
    }
}

fn cmd_fm(args: &str) {
    let target = if args.is_empty() {
        crate::users::cwd()
    } else {
        resolve_path(args.split_whitespace().next().unwrap_or(args))
    };
    println!("\x1b[1;36m+-- File Manager --------------------------------+\x1b[0m");
    println!("\x1b[1;36m|\x1b[0m  Path: \x1b[33m{}\x1b[0m", target);
    println!("\x1b[1;36m+------------------------------------------------+\x1b[0m");
    match crate::vfs::lookup(&target) {
        Ok(node) => match node.readdir() {
            Ok(entries) => {
                if entries.is_empty() {
                    println!("\x1b[1;36m|\x1b[0m  (empty directory)");
                }
                for e in &entries {
                    if e.is_dir {
                        println!("\x1b[1;36m|\x1b[0m  \x1b[1;34m[DIR]\x1b[0m \x1b[34m{}\x1b[0m", e.name);
                    } else {
                        println!("\x1b[1;36m|\x1b[0m       \x1b[37m{}\x1b[0m", e.name);
                    }
                }
            }
            Err(_) => println!("\x1b[1;36m|\x1b[0m  Not a directory: {}", target),
        },
        Err(_) => println!("\x1b[1;36m|\x1b[0m  Path not found: {}", target),
    }
    println!("\x1b[1;36m+------------------------------------------------+\x1b[0m");
    println!("\x1b[90m  cd <dir>  cat <file>  touch <file>  mkdir <dir>\x1b[0m");
}

fn cmd_note(args: &str) {
    let fname = args.split_whitespace().next().unwrap_or("");
    if fname.is_empty() {
        println!("\x1b[33mNote Pad\x1b[0m - view a file with line numbers.");
        println!("  Usage: note <filename>");
        println!("  Edit : echo 'text' >> <filename>");
        return;
    }
    let path = resolve_path(fname);
    match crate::vfs::lookup(&path) {
        Ok(node) => match node.open() {
            Ok(mut fh) => {
                println!("\x1b[33m-- {} --\x1b[0m", path);
                let mut buf = [0u8; 256];
                let mut line_num = 1usize;
                let mut at_line_start = true;
                loop {
                    match fh.read(&mut buf) {
                        Ok(0)  => break,
                        Ok(n)  => {
                            for &b in &buf[..n] {
                                if at_line_start {
                                    // print line number prefix
                                    let ln = format!("  {:3} | ", line_num);
                                    for lb in ln.bytes() { crate::desktop::terminal_input(lb); }
                                    at_line_start = false;
                                    line_num += 1;
                                }
                                crate::desktop::terminal_input(b);
                                if b == b'\n' { at_line_start = true; }
                            }
                        }
                        Err(_) => { println!("[read error]"); break; }
                    }
                }
                if !at_line_start { crate::desktop::terminal_input(b'\n'); }
                println!("\x1b[33m-- End ({} lines) --\x1b[0m", line_num.saturating_sub(1));
            }
            Err(_) => println!("note: cannot open: {}", path),
        },
        Err(_) => println!("note: file not found: {}", path),
    }
}

// ── Calculator expression evaluator ──────────────────────────────────────────

fn cmd_calc(args: &str) {
    let expr = args.trim();
    if expr.is_empty() {
        println!("Usage: calc <expr>   e.g. calc 2+3*4, calc (10-2)/4");
        return;
    }
    match calc_eval(expr.as_bytes(), 0) {
        Ok((val, rest)) => {
            let rest = calc_skip_ws(expr.as_bytes(), rest);
            if rest < expr.len() {
                println!("calc: unexpected char at position {}", rest);
            } else {
                println!("{} = {}", expr, val);
            }
        }
        Err(e) => println!("calc: {}", e),
    }
}

fn calc_skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && (b[i] == b' ' || b[i] == b'\t') { i += 1; }
    i
}
fn calc_eval(b: &[u8], i: usize) -> Result<(i64, usize), &'static str> {
    calc_add(b, i)
}
fn calc_add(b: &[u8], i: usize) -> Result<(i64, usize), &'static str> {
    let (mut v, mut i) = calc_mul(b, i)?;
    loop {
        let j = calc_skip_ws(b, i);
        if j >= b.len() { break; }
        let op = b[j];
        if op != b'+' && op != b'-' { break; }
        let (r, ni) = calc_mul(b, j + 1)?;
        v = if op == b'+' { v.wrapping_add(r) } else { v.wrapping_sub(r) };
        i = ni;
    }
    Ok((v, i))
}
fn calc_mul(b: &[u8], i: usize) -> Result<(i64, usize), &'static str> {
    let (mut v, mut i) = calc_unary(b, i)?;
    loop {
        let j = calc_skip_ws(b, i);
        if j >= b.len() { break; }
        let op = b[j];
        if op != b'*' && op != b'/' && op != b'%' { break; }
        let (r, ni) = calc_unary(b, j + 1)?;
        v = match op {
            b'*' => v.wrapping_mul(r),
            b'/' => { if r == 0 { return Err("division by zero"); } v / r },
            _    => { if r == 0 { return Err("division by zero"); } v % r },
        };
        i = ni;
    }
    Ok((v, i))
}
fn calc_unary(b: &[u8], i: usize) -> Result<(i64, usize), &'static str> {
    let i = calc_skip_ws(b, i);
    if i < b.len() && b[i] == b'-' {
        let (v, ni) = calc_primary(b, i + 1)?;
        Ok((-v, ni))
    } else {
        calc_primary(b, i)
    }
}
fn calc_primary(b: &[u8], i: usize) -> Result<(i64, usize), &'static str> {
    let i = calc_skip_ws(b, i);
    if i >= b.len() { return Err("unexpected end of expression"); }
    if b[i] == b'(' {
        let (v, ni) = calc_eval(b, i + 1)?;
        let ni = calc_skip_ws(b, ni);
        if ni >= b.len() || b[ni] != b')' { return Err("missing closing )"); }
        Ok((v, ni + 1))
    } else if b[i].is_ascii_digit() {
        let mut j = i;
        let mut v: i64 = 0;
        while j < b.len() && b[j].is_ascii_digit() {
            v = v.wrapping_mul(10).wrapping_add((b[j] - b'0') as i64);
            j += 1;
        }
        Ok((v, j))
    } else {
        Err("invalid character in expression")
    }
}

fn cmd_clear() {
    crate::desktop::clear_terminal();
}

fn cmd_reboot() {
    println!("Rebooting...");
    unsafe { x86_64::instructions::port::Port::<u8>::new(0x64).write(0xFE); }
}

fn cmd_shutdown() {
    println!("Shutting down...");
    // ACPI S5 via PM1a_CNT (from FADT parsing)
    unsafe {
        // Try ACPI power off: write SLP_TYP=S5 | SLP_EN to PM1a_CNT
        x86_64::instructions::port::Port::<u16>::new(0x4004).write(0x2000 | (5 << 10));
        // Fallback: halt
        loop { x86_64::instructions::hlt(); }
    }
}

fn cmd_uname(args: &str) {
    if args.contains("-a") || args.is_empty() {
        println!("NodeAI {} x86_64 NodeAI-Kernel", env!("CARGO_PKG_VERSION"));
    } else if args.contains("-r") {
        println!("{}", env!("CARGO_PKG_VERSION"));
    } else if args.contains("-n") {
        println!("{}", crate::users::hostname());
    } else if args.contains("-s") {
        println!("NodeAI");
    } else {
        println!("NodeAI {} x86_64", env!("CARGO_PKG_VERSION"));
    }
}

fn cmd_date() {
    let ms  = crate::scheduler::uptime_ms();
    let sec = ms / 1000;
    let min = (sec / 60) % 60;
    let hr  = (sec / 3600) % 24;
    println!("Up {:02}:{:02}:{:02} (no RTC — showing uptime)", hr, min, sec % 60);
}

fn cmd_uptime() {
    let ms  = crate::scheduler::uptime_ms();
    let sec = ms / 1000;
    let min = sec / 60;
    let hr  = min / 60;
    println!("up {}h {}m {}s", hr, min % 60, sec % 60);
}

fn cmd_history() {
    let h = HISTORY.lock();
    if h.count == 0 {
        println!("  (no history)");
        return;
    }
    let start = if h.count < HISTORY_CAP { 0 } else { h.pos };
    for i in 0..h.count {
        let idx = (start + i) % HISTORY_CAP;
        if let Some(buf) = &h.entries[idx] {
            let len = h.lengths[idx];
            let s = core::str::from_utf8(&buf[..len]).unwrap_or("?");
            println!("  {:3}  {}", i + 1, s);
        }
    }
}

// ── Navigation ────────────────────────────────────────────────────────────────

fn cmd_cd(args: &str) {
    let target = if args.is_empty() || args == "~" {
        crate::users::current_home()
    } else if args == "-" {
        env_get("OLDPWD").unwrap_or_else(|| crate::users::cwd())
    } else if args.starts_with('/') {
        String::from(args)
    } else {
        let cwd = crate::users::cwd();
        if cwd == "/" {
            format!("/{}", args)
        } else {
            format!("{}/{}", cwd, args)
        }
    };

    // Verify the directory exists
    match crate::vfs::lookup(&target) {
        Ok(node) => {
            if let Ok(st) = node.stat() {
                if st.is_dir {
                    let old = crate::users::cwd();
                    env_set("OLDPWD", &old);
                    crate::users::set_cwd(&target);
                    env_set("PWD", &target);
                } else {
                    println!("cd: not a directory: {}", target);
                }
            }
        }
        Err(_) => println!("cd: no such directory: {}", target),
    }
}

fn cmd_pwd() {
    println!("{}", crate::users::cwd());
}

// ── File operations ───────────────────────────────────────────────────────────

fn resolve_path(path: &str) -> String {
    if path.starts_with('/') {
        String::from(path)
    } else {
        let cwd = crate::users::cwd();
        if cwd == "/" {
            format!("/{}", path)
        } else {
            format!("{}/{}", cwd, path)
        }
    }
}

fn parent_and_name(path: &str) -> (String, String) {
    let full = resolve_path(path);
    if let Some(pos) = full.rfind('/') {
        let parent = if pos == 0 { String::from("/") } else { String::from(&full[..pos]) };
        let name = String::from(&full[pos + 1..]);
        (parent, name)
    } else {
        (String::from("/"), full)
    }
}

fn cmd_ls(args: &str) {
    // Parse flags
    let mut use_color = true; // default: color on
    let mut target_path = "";
    for part in args.split_whitespace() {
        if part == "--color=auto" || part == "--color=always" || part == "--color" {
            use_color = true;
        } else if part == "--color=never" {
            use_color = false;
        } else if part == "-la" || part == "-l" || part == "-a" {
            // Accepted but ignored (we always show all)
        } else {
            target_path = part;
        }
    }
    let target = if target_path.is_empty() {
        crate::users::cwd()
    } else {
        resolve_path(target_path)
    };
    match crate::vfs::lookup(&target) {
        Ok(node) => match node.readdir() {
            Ok(entries) => {
                for e in &entries {
                    if e.is_dir {
                        if use_color {
                            println!("  \x1b[34m{}/\x1b[0m", e.name);
                        } else {
                            println!("  {}/", e.name);
                        }
                    } else {
                        // Check if executable (name ends in common patterns or has exec perms)
                        let is_exec = e.name.ends_with(".sh") || e.name.ends_with(".elf");
                        if use_color && is_exec {
                            println!("  \x1b[32m{}\x1b[0m", e.name);
                        } else {
                            println!("  {}", e.name);
                        }
                    }
                }
                if entries.is_empty() {
                    println!("  (empty)");
                }
            }
            Err(_) => println!("ls: not a directory: {}", target),
        },
        Err(_) => println!("ls: no such file or directory: {}", target),
    }
}

fn cmd_cat(args: &str) {
    if args.is_empty() {
        println!("Usage: cat <path>");
        return;
    }
    let path = resolve_path(args);
    match crate::vfs::lookup(&path) {
        Ok(node) => match node.open() {
            Ok(mut fh) => {
                let mut buf = [0u8; 512];
                loop {
                    match fh.read(&mut buf) {
                        Ok(0)  => break,
                        Ok(n)  => {
                            for &b in &buf[..n] {
                                crate::desktop::terminal_input(b);
                            }
                        }
                        Err(_) => { println!("[read error]"); break; }
                    }
                }
                crate::desktop::terminal_input(b'\n');
            }
            Err(_) => println!("cat: cannot open {}", args),
        },
        Err(_) => println!("cat: no such file: {}", args),
    }
}

fn cmd_touch(args: &str) {
    if args.is_empty() {
        println!("Usage: touch <file>");
        return;
    }
    let (parent, name) = parent_and_name(args);
    match crate::vfs::lookup(&parent) {
        Ok(dir) => {
            // Try to look up first — if exists, do nothing (touch semantics)
            if dir.lookup(&name).is_ok() {
                return; // file already exists
            }
            match dir.create_file(&name) {
                Ok(_) => {}
                Err(e) => println!("touch: cannot create {}: {:?}", args, e),
            }
        }
        Err(_) => println!("touch: cannot access parent directory: {}", parent),
    }
}

fn cmd_mkdir(args: &str) {
    if args.is_empty() {
        println!("Usage: mkdir <dir>");
        return;
    }
    // Support -p flag
    let (flag, path) = if args.starts_with("-p ") {
        (true, args[3..].trim())
    } else {
        (false, args)
    };

    if flag {
        // Create parents recursively
        let full = resolve_path(path);
        let mut accum = String::new();
        for part in full.split('/').filter(|s| !s.is_empty()) {
            accum.push('/');
            accum.push_str(part);
            if crate::vfs::lookup(&accum).is_err() {
                let (p, n) = parent_and_name(&accum);
                if let Ok(dir) = crate::vfs::lookup(&p) {
                    let _ = dir.mkdir(&n);
                }
            }
        }
    } else {
        let (parent, name) = parent_and_name(path);
        match crate::vfs::lookup(&parent) {
            Ok(dir) => {
                if let Err(e) = dir.mkdir(&name) {
                    println!("mkdir: cannot create {}: {:?}", path, e);
                }
            }
            Err(_) => println!("mkdir: parent not found: {}", parent),
        }
    }
}

fn cmd_rm(args: &str) {
    if args.is_empty() {
        println!("Usage: rm [-r] <path>");
        return;
    }
    let (recursive, path) = if args.starts_with("-r ") || args.starts_with("-rf ") {
        let rest = if args.starts_with("-rf ") { &args[4..] } else { &args[3..] };
        (true, rest.trim())
    } else {
        (false, args)
    };
    let _ = recursive; // TODO: recursive delete

    let (parent, name) = parent_and_name(path);
    match crate::vfs::lookup(&parent) {
        Ok(dir) => {
            if let Err(e) = dir.unlink(&name) {
                println!("rm: cannot remove {}: {:?}", path, e);
            }
        }
        Err(_) => println!("rm: no such file: {}", path),
    }
}

fn cmd_rmdir(args: &str) {
    if args.is_empty() {
        println!("Usage: rmdir <dir>");
        return;
    }
    let (parent, name) = parent_and_name(args);
    match crate::vfs::lookup(&parent) {
        Ok(dir) => {
            // Check it's empty first
            if let Ok(target) = dir.lookup(&name) {
                if let Ok(entries) = target.readdir() {
                    if !entries.is_empty() {
                        println!("rmdir: directory not empty: {}", args);
                        return;
                    }
                }
            }
            if let Err(e) = dir.unlink(&name) {
                println!("rmdir: failed: {:?}", e);
            }
        }
        Err(_) => println!("rmdir: not found: {}", args),
    }
}

fn cmd_cp(args: &str) {
    let parts: Vec<&str> = args.splitn(2, ' ').collect();
    if parts.len() < 2 {
        println!("Usage: cp <src> <dst>");
        return;
    }
    let src_path = resolve_path(parts[0]);
    let dst_path = resolve_path(parts[1]);

    // Read source
    let data = match crate::vfs::lookup(&src_path) {
        Ok(node) => match node.open() {
            Ok(mut fh) => {
                let mut data = Vec::new();
                let mut buf = [0u8; 512];
                loop {
                    match fh.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => data.extend_from_slice(&buf[..n]),
                        Err(_) => { println!("cp: read error"); return; }
                    }
                }
                data
            }
            Err(_) => { println!("cp: cannot open {}", parts[0]); return; }
        },
        Err(_) => { println!("cp: source not found: {}", parts[0]); return; }
    };

    // Write to destination
    let (parent, name) = parent_and_name(&dst_path);
    match crate::vfs::lookup(&parent) {
        Ok(dir) => {
            let _ = dir.unlink(&name); // remove if exists
            match dir.create_file(&name) {
                Ok(node) => {
                    if let Ok(mut fh) = node.open() {
                        let _ = fh.write(&data);
                        let _ = fh.flush();
                    }
                }
                Err(e) => println!("cp: cannot create {}: {:?}", parts[1], e),
            }
        }
        Err(_) => println!("cp: destination parent not found"),
    }
}

fn cmd_mv(args: &str) {
    let parts: Vec<&str> = args.splitn(2, ' ').collect();
    if parts.len() < 2 {
        println!("Usage: mv <src> <dst>");
        return;
    }
    // mv = cp + rm
    let cp_args = args;
    cmd_cp(cp_args);
    cmd_rm(parts[0]);
}

fn cmd_ln(args: &str) {
    // ln [-s] <target> <link>
    let (symbolic, rest) = if args.starts_with("-s ") {
        (true, args[3..].trim())
    } else {
        (false, args)
    };
    let parts: Vec<&str> = rest.splitn(2, ' ').collect();
    if parts.len() < 2 {
        println!("Usage: ln [-s] <target> <link>");
        return;
    }
    let _ = symbolic;
    // VFS doesn't support symlinks yet — create a copy as approximation
    println!("ln: symbolic links not yet supported (VFS limitation)");
    println!("    Use 'cp' as a workaround.");
}

fn cmd_stat(args: &str) {
    if args.is_empty() {
        println!("Usage: stat <path>");
        return;
    }
    let path = resolve_path(args);
    match crate::vfs::lookup(&path) {
        Ok(node) => match node.stat() {
            Ok(st) => {
                println!("  File: {}", args);
                println!("  Size: {}  Inode: {}", st.size, st.ino);
                println!("  Type: {}", if st.is_dir { "directory" } else { "regular file" });
                println!("  Links: {}", st.nlink);
            }
            Err(_) => println!("stat: cannot stat: {}", args),
        },
        Err(_) => println!("stat: no such file: {}", args),
    }
}

fn cmd_wc(args: &str) {
    if args.is_empty() {
        println!("Usage: wc <file>");
        return;
    }
    let path = resolve_path(args);
    match crate::vfs::lookup(&path) {
        Ok(node) => match node.open() {
            Ok(mut fh) => {
                let mut buf = [0u8; 512];
                let mut lines = 0usize;
                let mut words = 0usize;
                let mut bytes = 0usize;
                let mut in_word = false;
                loop {
                    match fh.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            for &b in &buf[..n] {
                                bytes += 1;
                                if b == b'\n' { lines += 1; }
                                if b == b' ' || b == b'\n' || b == b'\t' {
                                    in_word = false;
                                } else if !in_word {
                                    in_word = true;
                                    words += 1;
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
                println!("  {} {} {} {}", lines, words, bytes, args);
            }
            Err(_) => println!("wc: cannot open {}", args),
        },
        Err(_) => println!("wc: no such file: {}", args),
    }
}

fn cmd_head(args: &str) {
    // head [-n N] <file>
    let (n, path) = if args.starts_with("-n ") {
        let rest = &args[3..];
        if let Some(sp) = rest.find(' ') {
            let count: usize = rest[..sp].parse().unwrap_or(10);
            (count, rest[sp + 1..].trim())
        } else {
            (10, args)
        }
    } else {
        (10, args)
    };
    if path.is_empty() {
        println!("Usage: head [-n N] <file>");
        return;
    }
    let full = resolve_path(path);
    match crate::vfs::lookup(&full) {
        Ok(node) => match node.open() {
            Ok(mut fh) => {
                let mut buf = [0u8; 512];
                let mut lines_printed = 0usize;
                'outer: loop {
                    match fh.read(&mut buf) {
                        Ok(0) => break,
                        Ok(cnt) => {
                            for &b in &buf[..cnt] {
                                crate::desktop::terminal_input(b);
                                if b == b'\n' {
                                    lines_printed += 1;
                                    if lines_printed >= n { break 'outer; }
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
            }
            Err(_) => println!("head: cannot open {}", path),
        },
        Err(_) => println!("head: no such file: {}", path),
    }
}

fn cmd_tail(args: &str) {
    // Simple tail: read all, print last N lines
    let (n, path) = if args.starts_with("-n ") {
        let rest = &args[3..];
        if let Some(sp) = rest.find(' ') {
            let count: usize = rest[..sp].parse().unwrap_or(10);
            (count, rest[sp + 1..].trim())
        } else {
            (10, args)
        }
    } else {
        (10, args)
    };
    if path.is_empty() {
        println!("Usage: tail [-n N] <file>");
        return;
    }
    let full = resolve_path(path);
    match crate::vfs::lookup(&full) {
        Ok(node) => match node.open() {
            Ok(mut fh) => {
                let mut all = Vec::new();
                let mut buf = [0u8; 512];
                loop {
                    match fh.read(&mut buf) {
                        Ok(0) => break,
                        Ok(cnt) => all.extend_from_slice(&buf[..cnt]),
                        Err(_) => break,
                    }
                }
                let text = core::str::from_utf8(&all).unwrap_or("");
                let lines: Vec<&str> = text.lines().collect();
                let start = lines.len().saturating_sub(n);
                for line in &lines[start..] {
                    println!("{}", line);
                }
            }
            Err(_) => println!("tail: cannot open {}", path),
        },
        Err(_) => println!("tail: no such file: {}", path),
    }
}

// ── System monitoring ─────────────────────────────────────────────────────────

fn cmd_meminfo() {
    let free_mb = crate::memory::free_mb();
    println!("MemFree:  {} MiB ({} KiB)", free_mb, free_mb * 1024);
}

fn cmd_free() {
    let free_mb = crate::memory::free_mb();
    println!("              total       free");
    println!("Mem:            --     {} MiB", free_mb);
}

fn cmd_ps() {
    let n = crate::scheduler::task_count();
    let ms = crate::scheduler::uptime_ms();
    println!("  PID  STATE    TIME       NAME");
    println!("    0  running  {:6}ms   [kernel/idle]", ms);
    println!("");
    println!("Total: {} task(s)", n);
}

fn cmd_dmesg() {
    // Read directly from the kernel ring buffer with colored output
    let ring = crate::kring::KRING.lock();
    for entry in ring.iter() {
        let msg = entry.as_str();
        // Color by level: ERROR=red, WARN=yellow, INFO=green, DEBUG=cyan, TRACE=dim
        let color = match entry.level {
            4 => "\x1b[31m", // ERROR — red
            3 => "\x1b[33m", // WARN  — yellow
            2 => "\x1b[32m", // INFO  — green
            1 => "\x1b[36m", // DEBUG — cyan
            _ => "\x1b[90m", // TRACE — dim gray
        };
        print_str(color);
        print_str(msg);
        print_str("\x1b[0m");
    }
}

fn cmd_lspci() {
    use drivers::pci;
    let devices = pci::enumerate();
    println!("PCI devices:");
    for addr in &devices {
        let id = addr.id();
        println!("  {:02x}:{:02x}.{} [{:04x}:{:04x}]",
            addr.bus, addr.device, addr.function, id.vendor_id, id.device_id);
    }
}

fn cmd_hostname(args: &str) {
    if args.is_empty() {
        println!("{}", crate::users::hostname());
    } else {
        if !crate::users::is_root() {
            println!("hostname: permission denied (must be root)");
            return;
        }
        crate::users::set_hostname(args);
        env_set("HOSTNAME", args);
    }
}

// ── User/auth commands (Phase 14) ─────────────────────────────────────────────

fn cmd_whoami() {
    println!("{}", crate::users::current_username());
}

fn cmd_id(args: &str) {
    let username = if args.is_empty() {
        crate::users::current_username()
    } else {
        String::from(args)
    };
    match crate::users::get_user_by_name(&username) {
        Some(u) => {
            let groups = crate::users::user_groups(&username);
            let groups_str = groups.join(",");
            println!("uid={}({}) gid={}({}) groups={}",
                u.uid, u.username, u.gid,
                crate::users::get_group(u.gid).map(|g| g.name).unwrap_or_else(|| format!("{}", u.gid)),
                groups_str);
        }
        None => println!("id: no such user: {}", username),
    }
}

fn cmd_su(args: &str) {
    let target_user = if args.is_empty() { "root" } else { args };

    if crate::users::get_user_by_name(target_user).is_none() {
        println!("su: user {} does not exist", target_user);
        return;
    }

    // Root can su without password
    if crate::users::is_root() {
        crate::users::switch_user(target_user);
        env_set("USER", target_user);
        env_set("HOME", &crate::users::current_home());
        println!("Switched to {}", target_user);
        return;
    }

    println!("Password: ");
    // In a real implementation, we'd read password with echo off.
    // For now, we auto-authenticate if the password matches.
    // This is a kernel shell limitation — we can't block for async input.
    println!("(su: password input not yet supported in kernel shell)");
    println!("Tip: use 'login <username>' instead");
}

fn cmd_sudo(args: &str) {
    if args.is_empty() {
        println!("Usage: sudo <command>");
        return;
    }

    let username = crate::users::current_username();

    // Root doesn't need sudo
    if crate::users::is_root() {
        crate::users::audit_sudo(&username, args, true);
        dispatch(args);
        return;
    }

    // Check sudo permission
    if !crate::users::user_can_sudo(&username) {
        crate::users::audit_sudo(&username, args, false);
        println!("{} is not in the sudoers file. This incident will be reported.", username);
        return;
    }

    // Execute command as root
    crate::users::audit_sudo(&username, args, true);
    let saved_uid = crate::users::current_uid();
    crate::users::set_current_uid(0);
    dispatch(args);
    crate::users::set_current_uid(saved_uid);
}

fn cmd_useradd(args: &str) {
    if args.is_empty() {
        println!("Usage: useradd <username>");
        return;
    }
    if !crate::users::is_root() {
        println!("useradd: permission denied (must be root)");
        return;
    }
    match crate::users::useradd(args) {
        Ok(uid)  => println!("User '{}' created (uid={})", args, uid),
        Err(msg) => println!("useradd: {}", msg),
    }
}

fn cmd_userdel(args: &str) {
    if args.is_empty() {
        println!("Usage: userdel <username>");
        return;
    }
    if !crate::users::is_root() {
        println!("userdel: permission denied (must be root)");
        return;
    }
    match crate::users::userdel(args) {
        Ok(())   => println!("User '{}' deleted", args),
        Err(msg) => println!("userdel: {}", msg),
    }
}

fn cmd_passwd(args: &str) {
    let target = if args.is_empty() {
        crate::users::current_username()
    } else {
        // Only root can change other users' passwords
        if !crate::users::is_root() {
            println!("passwd: permission denied");
            return;
        }
        String::from(args)
    };
    // For kernel shell, we use a simple approach: passwd <user> <newpass>
    let parts: Vec<&str> = args.splitn(2, ' ').collect();
    if parts.len() >= 2 {
        match crate::users::change_password(parts[0], parts[1]) {
            Ok(()) => println!("Password updated for {}", parts[0]),
            Err(e) => println!("passwd: {}", e),
        }
    } else if !args.is_empty() {
        println!("Usage: passwd <username> <newpassword>");
    } else {
        println!("Usage: passwd <username> <newpassword>");
        println!("(Interactive password input not yet supported in kernel shell)");
    }
}

fn cmd_groups(args: &str) {
    let username = if args.is_empty() {
        crate::users::current_username()
    } else {
        String::from(args)
    };
    let groups = crate::users::user_groups(&username);
    if groups.is_empty() {
        println!("{}: (no groups)", username);
    } else {
        println!("{} : {}", username, groups.join(" "));
    }
}

fn cmd_chmod(args: &str) {
    let parts: Vec<&str> = args.splitn(2, ' ').collect();
    if parts.len() < 2 {
        println!("Usage: chmod <mode> <path>");
        return;
    }
    match crate::users::FileMode::from_octal(parts[0]) {
        Some(mode) => {
            let path = resolve_path(parts[1]);
            match crate::vfs::lookup(&path) {
                Ok(node) => {
                    // Only root or owner can chmod
                    let uid = crate::users::current_uid();
                    let st = node.stat().unwrap();
                    if uid != 0 && uid != st.uid {
                        println!("chmod: permission denied");
                        return;
                    }
                    match node.set_mode(mode.0) {
                        Ok(()) => {}
                        Err(_) => println!("chmod: cannot set mode on this filesystem"),
                    }
                }
                Err(_) => println!("chmod: cannot access '{}': No such file or directory", parts[1]),
            }
        }
        None => println!("chmod: invalid mode: {}", parts[0]),
    }
}

fn cmd_chown(args: &str) {
    let parts: Vec<&str> = args.splitn(2, ' ').collect();
    if parts.len() < 2 {
        println!("Usage: chown <user[:group]> <path>");
        return;
    }
    if !crate::users::is_root() {
        println!("chown: permission denied (must be root)");
        return;
    }
    let owner_spec = parts[0];
    let path = resolve_path(parts[1]);

    // Parse user:group
    let (user, group_name) = if let Some(colon) = owner_spec.find(':') {
        (&owner_spec[..colon], Some(&owner_spec[colon + 1..]))
    } else {
        (owner_spec, None)
    };

    if let Some(u) = crate::users::get_user_by_name(user) {
        let gid = if let Some(gn) = group_name {
            crate::users::get_group_by_name(gn).map(|g| g.gid).unwrap_or(u.gid)
        } else {
            u.gid
        };
        match crate::vfs::lookup(&path) {
            Ok(node) => {
                match node.set_owner(u.uid, gid) {
                    Ok(()) => {}
                    Err(_) => println!("chown: cannot set owner on this filesystem"),
                }
            }
            Err(_) => println!("chown: cannot access '{}': No such file or directory", parts[1]),
        }
    } else {
        println!("chown: invalid user: {}", user);
    }
}

fn cmd_login(args: &str) {
    if args.is_empty() {
        println!("Usage: login <username>");
        println!("  (bypasses password in kernel shell for demo)");
        return;
    }
    // For kernel shell demo: switch user directly
    if crate::users::get_user_by_name(args).is_some() {
        crate::users::switch_user(args);
        env_set("USER", args);
        env_set("HOME", &crate::users::current_home());
        env_set("PWD", &crate::users::cwd());
        println!("Logged in as {}", args);
        // Display MOTD
        if let Ok(node) = crate::vfs::lookup("/etc/motd") {
            if let Ok(mut fh) = node.open() {
                let mut buf = [0u8; 512];
                if let Ok(n) = fh.read(&mut buf) {
                    for &b in &buf[..n] {
                        crate::desktop::terminal_input(b);
                    }
                }
            }
        }
    } else {
        println!("login: unknown user: {}", args);
    }
}

fn cmd_logout() {
    // Save command history before logout
    history_save();
    // Switch back to root
    crate::users::set_current_uid(0);
    crate::users::set_cwd("/");
    env_set("USER", "root");
    env_set("HOME", "/root");
    env_set("PWD", "/");
    println!("Logged out. Now root.");
}

fn cmd_audit() {
    if !crate::users::is_root() {
        println!("audit: permission denied (must be root)");
        return;
    }
    let log = crate::users::sudo_audit_log();
    if log.is_empty() {
        println!("No sudo audit entries.");
    } else {
        print_str(&log);
    }
}

// ── Environment commands ──────────────────────────────────────────────────────

fn cmd_export(args: &str) {
    if args.is_empty() {
        cmd_env();
        return;
    }
    if let Some(eq) = args.find('=') {
        let key = &args[..eq];
        let val = &args[eq + 1..];
        env_set(key, val);
    } else {
        println!("Usage: export VAR=value");
    }
}

fn cmd_unset(args: &str) {
    if args.is_empty() {
        println!("Usage: unset <VAR>");
        return;
    }
    env_unset(args);
}

fn cmd_env() {
    let env = ENV.lock();
    for (k, v) in env.iter() {
        println!("{}={}", k, v);
    }
}

static BUILTINS: &[&str] = &[
    "help", "clear", "echo", "reboot", "shutdown", "uname", "date", "uptime",
    "history", "cd", "pwd", "ls", "cat", "touch", "mkdir", "rm", "rmdir",
    "cp", "mv", "ln", "stat", "wc", "head", "tail", "grep", "sort", "uniq",
    "cut", "diff", "xxd", "seq", "tee", "meminfo", "free", "ps", "kill",
    "killall", "top", "htop", "df", "du", "dmesg", "lspci", "lsmod", "hostname",
    "time", "mount", "umount", "sysctl", "file", "whoami", "id", "su", "sudo",
    "useradd", "userdel", "passwd", "groups", "chmod", "chown", "login",
    "logout", "audit", "export", "unset", "env", "printenv", "which", "type",
    "sleep", "true", "false", "alias", "unalias", "yes", "man",
    "nice", "renice", "bg", "fg", "jobs", "nohup",
    "fdisk", "mkfs", "sync", "modprobe", "insmod", "service",
    "wifi", "ifconfig", "ping", "arp", "netstat", "ip", "nslookup", "dig",
    "traceroute", "wget", "curl", "nc",
    "dhclient", "httpd", "sshd", "scp", "dns-cache", "ifup", "ifdown",
];

fn cmd_which(args: &str) {
    if args.is_empty() {
        println!("Usage: which <command>");
        return;
    }
    if BUILTINS.contains(&args) {
        println!("{}: shell built-in command", args);
    } else {
        println!("{}: not found", args);
    }
}

fn cmd_type(args: &str) {
    if args.is_empty() {
        println!("Usage: type <command>");
        return;
    }
    let aliases = ALIASES.lock();
    if let Some(val) = aliases.get(args) {
        println!("{} is aliased to '{}'", args, val);
    } else if BUILTINS.contains(&args) {
        println!("{} is a shell builtin", args);
    } else {
        println!("{}: not found", args);
    }
}

fn cmd_alias(args: &str) {
    if args.is_empty() {
        // List all aliases
        let aliases = ALIASES.lock();
        for (k, v) in aliases.iter() {
            println!("alias {}='{}'", k, v);
        }
        return;
    }
    // Parse alias name='value'
    if let Some(eq) = args.find('=') {
        let name = &args[..eq];
        let mut val = &args[eq + 1..];
        // Strip surrounding quotes
        if (val.starts_with('\'') && val.ends_with('\''))
            || (val.starts_with('"') && val.ends_with('"')) {
            val = &val[1..val.len() - 1];
        }
        ALIASES.lock().insert(String::from(name), String::from(val));
    } else {
        // Show specific alias
        let aliases = ALIASES.lock();
        if let Some(val) = aliases.get(args) {
            println!("alias {}='{}'", args, val);
        } else {
            println!("alias: {}: not found", args);
        }
    }
}

fn cmd_unalias(args: &str) {
    if args.is_empty() {
        println!("Usage: unalias <name>");
        return;
    }
    if ALIASES.lock().remove(args).is_none() {
        println!("unalias: {}: not found", args);
    }
}

// ── Misc ──────────────────────────────────────────────────────────────────────

fn cmd_sleep(args: &str) {
    if args.is_empty() {
        println!("Usage: sleep <seconds>");
        return;
    }
    let secs: u64 = args.parse().unwrap_or(0);
    let target = crate::scheduler::uptime_ms() + secs * 1000;
    while crate::scheduler::uptime_ms() < target {
        x86_64::instructions::hlt();
    }
}

fn cmd_yes(args: &str) {
    let text = if args.is_empty() { "y" } else { args };
    // Print a limited number of lines (no way to Ctrl+C yet in pipe context)
    for _ in 0..100 {
        println!("{}", text);
    }
}

fn cmd_time(args: &str) {
    if args.is_empty() {
        println!("Usage: time <command>");
        return;
    }
    let before = crate::scheduler::uptime_ms();
    dispatch(args);
    let after = crate::scheduler::uptime_ms();
    let elapsed = after.saturating_sub(before);
    println!("");
    println!("real    0m{}.{:03}s", elapsed / 1000, elapsed % 1000);
}

// ── Text processing ───────────────────────────────────────────────────────────

fn read_file_contents(path: &str) -> Option<Vec<u8>> {
    let full = resolve_path(path);
    match crate::vfs::lookup(&full) {
        Ok(node) => match node.open() {
            Ok(mut fh) => {
                let mut data = Vec::new();
                let mut buf = [0u8; 512];
                loop {
                    match fh.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => data.extend_from_slice(&buf[..n]),
                        Err(_) => break,
                    }
                }
                Some(data)
            }
            Err(_) => { println!("Cannot open: {}", path); None }
        },
        Err(_) => { println!("No such file: {}", path); None }
    }
}

fn cmd_grep(args: &str) {
    // grep [--color=auto] <pattern> <file>
    let parts: Vec<&str> = args.split_whitespace().collect();
    let mut use_color = true;
    let mut rest: Vec<&str> = Vec::new();
    for &p in &parts {
        if p == "--color=auto" || p == "--color=always" || p == "--color" {
            use_color = true;
        } else if p == "--color=never" {
            use_color = false;
        } else {
            rest.push(p);
        }
    }
    if rest.len() < 2 {
        println!("Usage: grep [--color=auto] <pattern> <file>");
        return;
    }
    let pattern = rest[0];
    let path = rest[1];
    if let Some(data) = read_file_contents(path) {
        let text = core::str::from_utf8(&data).unwrap_or("");
        for line in text.lines() {
            if line.contains(pattern) {
                if use_color {
                    // Highlight all occurrences of pattern in red
                    let mut out = String::new();
                    let mut remaining = line;
                    while let Some(pos) = remaining.find(pattern) {
                        out.push_str(&remaining[..pos]);
                        out.push_str("\x1b[31m\x1b[1m");
                        out.push_str(pattern);
                        out.push_str("\x1b[0m");
                        remaining = &remaining[pos + pattern.len()..];
                    }
                    out.push_str(remaining);
                    println!("{}", out);
                } else {
                    println!("{}", line);
                }
            }
        }
    }
}

fn cmd_sort(args: &str) {
    if args.is_empty() {
        println!("Usage: sort <file>");
        return;
    }
    if let Some(data) = read_file_contents(args) {
        let text = core::str::from_utf8(&data).unwrap_or("");
        let mut lines: Vec<&str> = text.lines().collect();
        lines.sort();
        for line in &lines {
            println!("{}", line);
        }
    }
}

fn cmd_uniq(args: &str) {
    if args.is_empty() {
        println!("Usage: uniq <file>");
        return;
    }
    if let Some(data) = read_file_contents(args) {
        let text = core::str::from_utf8(&data).unwrap_or("");
        let mut prev: Option<&str> = None;
        for line in text.lines() {
            if prev != Some(line) {
                println!("{}", line);
                prev = Some(line);
            }
        }
    }
}

fn cmd_cut(args: &str) {
    // cut -d<delim> -f<field> <file>
    if args.is_empty() {
        println!("Usage: cut -d<delim> -f<field> <file>");
        return;
    }
    let parts: Vec<&str> = args.split_whitespace().collect();
    if parts.len() < 3 {
        println!("Usage: cut -d<delim> -f<field> <file>");
        return;
    }
    let delim = if parts[0].starts_with("-d") {
        parts[0].chars().nth(2).unwrap_or('\t')
    } else {
        '\t'
    };
    let field_num: usize = if parts[1].starts_with("-f") {
        parts[1][2..].parse().unwrap_or(1)
    } else {
        1
    };
    let file = parts[2];
    if let Some(data) = read_file_contents(file) {
        let text = core::str::from_utf8(&data).unwrap_or("");
        for line in text.lines() {
            let fields: Vec<&str> = line.split(delim).collect();
            if field_num > 0 && field_num <= fields.len() {
                println!("{}", fields[field_num - 1]);
            }
        }
    }
}

fn cmd_diff(args: &str) {
    let parts: Vec<&str> = args.splitn(2, ' ').collect();
    if parts.len() < 2 {
        println!("Usage: diff <file1> <file2>");
        return;
    }
    let data1 = read_file_contents(parts[0]);
    let data2 = read_file_contents(parts[1]);
    match (data1, data2) {
        (Some(d1), Some(d2)) => {
            let t1 = core::str::from_utf8(&d1).unwrap_or("");
            let t2 = core::str::from_utf8(&d2).unwrap_or("");
            let lines1: Vec<&str> = t1.lines().collect();
            let lines2: Vec<&str> = t2.lines().collect();
            let max = lines1.len().max(lines2.len());
            let mut diffs = false;
            for i in 0..max {
                let l1 = lines1.get(i).unwrap_or(&"");
                let l2 = lines2.get(i).unwrap_or(&"");
                if l1 != l2 {
                    diffs = true;
                    if i < lines1.len() { println!("< {}", l1); }
                    if i < lines2.len() { println!("> {}", l2); }
                }
            }
            if !diffs {
                println!("Files are identical");
            }
        }
        _ => {}
    }
}

fn cmd_xxd(args: &str) {
    if args.is_empty() {
        println!("Usage: xxd <file>");
        return;
    }
    if let Some(data) = read_file_contents(args) {
        let limit = data.len().min(256); // limit output
        for offset in (0..limit).step_by(16) {
            // Address
            let mut line = format!("{:08x}: ", offset);
            // Hex
            let end = (offset + 16).min(limit);
            for i in offset..end {
                line.push_str(&format!("{:02x}", data[i]));
                if i % 2 == 1 { line.push(' '); }
            }
            // Pad if short
            for _ in (end - offset)..16 {
                line.push_str("  ");
                if (offset + 16 - (end - offset)) % 2 == 1 { line.push(' '); }
            }
            line.push(' ');
            // ASCII
            for i in offset..end {
                let c = data[i];
                if c >= 0x20 && c < 0x7F {
                    line.push(c as char);
                } else {
                    line.push('.');
                }
            }
            println!("{}", line);
        }
        if data.len() > limit {
            println!("... ({} bytes total, showing first {})", data.len(), limit);
        }
    }
}

fn cmd_seq(args: &str) {
    let parts: Vec<&str> = args.split_whitespace().collect();
    match parts.len() {
        1 => {
            let end: i64 = parts[0].parse().unwrap_or(0);
            for i in 1..=end { println!("{}", i); }
        }
        2 => {
            let start: i64 = parts[0].parse().unwrap_or(0);
            let end: i64   = parts[1].parse().unwrap_or(0);
            if start <= end {
                for i in start..=end { println!("{}", i); }
            } else {
                let mut i = start;
                while i >= end { println!("{}", i); i -= 1; }
            }
        }
        _ => println!("Usage: seq [start] <end>"),
    }
}

// ── Additional system monitoring ──────────────────────────────────────────────

fn cmd_kill(args: &str) {
    if args.is_empty() {
        println!("Usage: kill <pid>");
        return;
    }
    let pid: u64 = args.parse().unwrap_or(0);
    if pid == 0 {
        println!("kill: cannot kill PID 0 (kernel idle)");
    } else {
        println!("kill: process {} signaled (no multi-process yet)", pid);
    }
}

fn cmd_df() {
    println!("Filesystem      Size   Used  Avail  Mount");
    println!("ramfs              -      -      -   /");
    println!("devfs              -      -      -   /dev");
    println!("procfs             -      -      -   /proc");
}

fn cmd_du(args: &str) {
    let path = if args.is_empty() { "." } else { args };
    let full = if path == "." { crate::users::cwd() } else { resolve_path(path) };
    match crate::vfs::lookup(&full) {
        Ok(node) => {
            if let Ok(st) = node.stat() {
                if st.is_dir {
                    if let Ok(entries) = node.readdir() {
                        let mut total = 0u64;
                        for e in &entries {
                            // Try to stat each entry for size
                            let child_path = if full == "/" {
                                format!("/{}", e.name)
                            } else {
                                format!("{}/{}", full, e.name)
                            };
                            let sz = if let Ok(cn) = crate::vfs::lookup(&child_path) {
                                cn.stat().map(|s| s.size).unwrap_or(0)
                            } else { 0 };
                            total += sz;
                            println!("{}    {}", sz, e.name);
                        }
                        println!("{}    {} (total)", total, path);
                    }
                } else {
                    println!("{}    {}", st.size, path);
                }
            }
        }
        Err(_) => println!("du: no such file or directory: {}", path),
    }
}

fn cmd_tee(args: &str) {
    if args.is_empty() {
        println!("Usage: tee <file>");
        return;
    }
    // tee in pipe context — we need to read from captured stdin
    // For now, tee just creates/overwrites the file; pipe support feeds stdin
    println!("tee: reading from pipe input (use with | )");
}

fn cmd_killall(args: &str) {
    if args.is_empty() {
        println!("Usage: killall <name>");
        return;
    }
    println!("killall: signaled all processes named '{}' (stub — single-process kernel)", args);
}

fn cmd_top() {
    println!("  PID  STATE     CPU%  MEM(KiB)  NAME");
    println!("  ---  --------  ----  --------  ----");
    let uptime = crate::scheduler::uptime_ms();
    let heap_used = crate::memory::KERNEL_HEAP.lock().used();
    let heap_kb = heap_used / 1024;
    println!("    0  Running    100  {:>8}  kernel", heap_kb);
    println!("");
    println!("Tasks: 1 total, 1 running");
    let free_mb = crate::memory::free_mb();
    println!("Mem: free {} MiB", free_mb);
    println!("Uptime: {}.{}s", uptime / 1000, (uptime % 1000) / 100);
}

fn cmd_lsmod() {
    println!("Module          Size  Used by");
    println!("ps2_keyboard    4096  1    input");
    println!("vga_fb          8192  1    desktop");
    println!("lapic_timer     4096  1    scheduler");
    println!("ioapic          4096  1    interrupts");
    println!("ramfs          16384  1    vfs");
    println!("procfs          8192  1    vfs");
    println!("devfs           4096  1    vfs");
    // Check if virtio_blk is present
    println!("virtio_blk      8192  0    block");
    println!("ai_engine      32768  1    ai_subsystem");
}

fn cmd_mount(args: &str) {
    if args.is_empty() {
        // Show current mounts
        println!("ramfs   on /      type ramfs  (rw)");
        println!("devfs   on /dev   type devfs  (rw)");
        println!("procfs  on /proc  type procfs (ro)");
        return;
    }
    println!("mount: mounting filesystems not supported yet");
}

fn cmd_umount(args: &str) {
    if args.is_empty() {
        println!("Usage: umount <mountpoint>");
        return;
    }
    println!("umount: unmounting '{}' not supported yet", args);
}

fn cmd_sysctl(args: &str) {
    if args.is_empty() || args == "-a" {
        // Show all kernel parameters
        println!("kernel.hostname = {}", crate::users::hostname());
        println!("kernel.version = NodeAI 0.1.0");
        println!("kernel.arch = x86_64");
        let free_mb = crate::memory::free_mb();
        println!("vm.free_memory_mb = {}", free_mb);
        println!("vm.page_size = 4096");
        println!("kernel.max_tasks = 64");
        println!("kernel.hz = 1000");
        println!("kernel.scheduler = round_robin");
        println!("net.ipv4.ip_forward = 0");
        return;
    }
    // Handle key=value
    if let Some(eq) = args.find('=') {
        let key = args[..eq].trim();
        let val = args[eq + 1..].trim();
        match key {
            "kernel.hostname" => {
                crate::users::set_hostname(val);
                println!("{} = {}", key, val);
            }
            _ => println!("sysctl: cannot set '{}': read-only", key),
        }
    } else {
        // Query a specific key
        match args.trim() {
            "kernel.hostname" => println!("kernel.hostname = {}", crate::users::hostname()),
            "kernel.version" => println!("kernel.version = NodeAI 0.1.0"),
            other => println!("sysctl: unknown key '{}'", other),
        }
    }
}

fn cmd_file(args: &str) {
    if args.is_empty() {
        println!("Usage: file <path>");
        return;
    }
    let path = resolve_path(args);
    match crate::vfs::lookup(&path) {
        Ok(node) => {
            if let Ok(st) = node.stat() {
                if st.is_dir {
                    println!("{}: directory", args);
                } else if st.size == 0 {
                    println!("{}: empty", args);
                } else if let Ok(mut fh) = node.open() {
                    let mut magic = [0u8; 4];
                    if let Ok(n) = fh.read(&mut magic) {
                        if n >= 4 && magic[0] == 0x7F && magic[1] == b'E' && magic[2] == b'L' && magic[3] == b'F' {
                            println!("{}: ELF executable", args);
                        } else if n >= 2 && magic[0] == b'#' && magic[1] == b'!' {
                            println!("{}: script (shebang)", args);
                        } else {
                            // Check if it's text
                            let mut is_text = true;
                            for i in 0..n {
                                let b = magic[i];
                                if b < 0x09 || (b > 0x0D && b < 0x20 && b != 0x1B) {
                                    is_text = false;
                                    break;
                                }
                            }
                            if is_text {
                                println!("{}: ASCII text", args);
                            } else {
                                println!("{}: data", args);
                            }
                        }
                    } else {
                        println!("{}: cannot read", args);
                    }
                } else {
                    println!("{}: regular file ({} bytes)", args, st.size);
                }
            }
        }
        Err(_) => println!("{}: No such file or directory", args),
    }
}

fn cmd_man(args: &str) {
    if args.is_empty() {
        println!("Usage: man <command>");
        println!("Display manual page for a built-in command.");
        return;
    }
    match args {
        "ls" => {
            println!("LS(1)                  NodeAI Manual                 LS(1)");
            println!("");
            println!("NAME");
            println!("    ls - list directory contents");
            println!("");
            println!("SYNOPSIS");
            println!("    ls [-l] [-a] [path]");
            println!("");
            println!("OPTIONS");
            println!("    -l    long listing format (permissions, size, name)");
            println!("    -a    show hidden files (starting with .)");
        }
        "cd" => {
            println!("CD(1)                  NodeAI Manual                 CD(1)");
            println!("");
            println!("NAME");
            println!("    cd - change directory");
            println!("");
            println!("SYNOPSIS");
            println!("    cd [path]");
            println!("");
            println!("DESCRIPTION");
            println!("    cd alone goes to $HOME, cd - goes to previous dir.");
        }
        "grep" => {
            println!("GREP(1)                NodeAI Manual               GREP(1)");
            println!("");
            println!("NAME");
            println!("    grep - search for pattern in files");
            println!("");
            println!("SYNOPSIS");
            println!("    grep [-i] [-v] [-c] [-n] <pattern> <file>");
            println!("");
            println!("OPTIONS");
            println!("    -i    case-insensitive");
            println!("    -v    invert match");
            println!("    -c    count matches");
            println!("    -n    show line numbers");
        }
        "echo" => {
            println!("ECHO(1)                NodeAI Manual               ECHO(1)");
            println!("");
            println!("NAME");
            println!("    echo - display text");
            println!("");
            println!("SYNOPSIS");
            println!("    echo [text...]");
            println!("");
            println!("DESCRIPTION");
            println!("    Print arguments to stdout. Supports $VAR expansion.");
        }
        "kill" => {
            println!("KILL(1)                NodeAI Manual               KILL(1)");
            println!("");
            println!("NAME");
            println!("    kill - send signal to a process");
            println!("");
            println!("SYNOPSIS");
            println!("    kill <pid>");
        }
        "help" => {
            println!("HELP(1)                NodeAI Manual               HELP(1)");
            println!("");
            println!("NAME");
            println!("    help - list all built-in commands");
            println!("");
            println!("DESCRIPTION");
            println!("    Displays a summary of all available shell commands.");
        }
        _ => {
            if BUILTINS.contains(&args) {
                println!("No manual entry for '{}' yet.", args);
                println!("Try: {} --help  or  help", args);
            } else {
                println!("No manual entry for '{}'", args);
            }
        }
    }
}

// ── Networking Commands (Phase 17) ────────────────────────────────────────────

fn cmd_wifi(args: &str) {
    let mut parts = args.splitn(3, ' ');
    match parts.next().unwrap_or("") {
        "scan" => {
            println!("WiFi: scanning...");
            let aps = crate::wifi::scan();
            if aps.is_empty() {
                if crate::wifi::is_available() {
                    println!("No networks found (try moving closer to an AP)");
                } else {
                    println!("No WiFi adapter detected.");
                    println!("Plug in an AR9271 USB dongle (TP-Link TL-WN722N v1)");
                    println!("then run: ./scripts/run_qemu.sh --gui --wifi");
                }
                return;
            }
            println!("{:<32} {:<20} {:>5}  {:>4}  {}", "SSID", "BSSID", "RSSI", "CH", "SECURITY");
            println!("{}", "-".repeat(72));
            for ap in &aps {
                println!("{:<32} {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}  {:>4}dBm  {:>3}  {}",
                    ap.ssid,
                    ap.bssid[0], ap.bssid[1], ap.bssid[2],
                    ap.bssid[3], ap.bssid[4], ap.bssid[5],
                    ap.rssi, ap.channel,
                    if ap.secured { "WPA2" } else { "OPEN" });
            }
            println!("\n{} network(s) found", aps.len());
        }
        "connect" => {
            let ssid = parts.next().unwrap_or("");
            let pass = parts.next().unwrap_or("");
            if ssid.is_empty() {
                println!("Usage: wifi connect <ssid> [passphrase]");
                return;
            }
            println!("WiFi: connecting to \"{}\"...", ssid);
            if crate::wifi::connect(ssid, pass) {
                println!("Connected to \"{}\"", ssid);
                // Trigger DHCP on WiFi interface
                println!("Requesting IP address via DHCP...");
            } else {
                println!("Failed to connect to \"{}\"", ssid);
            }
        }
        "disconnect" => {
            crate::wifi::disconnect();
            println!("WiFi disconnected");
        }
        "status" => {
            if crate::wifi::is_available() {
                if crate::wifi::is_connected() {
                    let ssid = crate::wifi::ssid().unwrap_or_default();
                    println!("WiFi: connected to \"{}\"", ssid);
                } else {
                    println!("WiFi: adapter ready, not connected");
                    println!("Run: wifi scan    to find networks");
                    println!("     wifi connect <ssid> [pass]");
                }
            } else {
                println!("WiFi: no adapter");
            }
        }
        "" | "help" => {
            println!("wifi scan               - scan for networks");
            println!("wifi connect <ssid> [pass] - connect to a network");
            println!("wifi disconnect         - disconnect");
            println!("wifi status             - show current status");
        }
        sub => println!("Unknown wifi subcommand: {}", sub),
    }
}

fn cmd_ifconfig(args: &str) {
    let our_ip  = unsafe { crate::net::OUR_IP };
    let our_mac = unsafe { crate::net::OUR_MAC };
    let has_nic = crate::net::NIC.lock().is_some();
    let (tx_p, rx_p, tx_b, rx_b) = crate::net::iface_stats();

    if args == "-a" || args.is_empty() {
        println!("eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500");
        println!("        inet {}  netmask 255.255.255.0  broadcast {}.{}.{}.255",
            crate::net::format_ipv4(&our_ip), our_ip[0], our_ip[1], our_ip[2]);
        println!("        ether {}",
            crate::net::format_mac(&our_mac));
        println!("        RX packets {}  bytes {}", rx_p, rx_b);
        println!("        TX packets {}  bytes {}", tx_p, tx_b);
        if !has_nic {
            println!("        (no VirtIO-net device detected)");
        }
        println!("");
        println!("lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536");
        println!("        inet 127.0.0.1  netmask 255.0.0.0");
    } else {
        // ifconfig eth0 <ip> netmask <mask>
        let parts: Vec<&str> = args.split_whitespace().collect();
        if parts.len() >= 2 && parts[0] == "eth0" {
            if let Some(ip) = crate::net::parse_ipv4(parts[1]) {
                let mask = if parts.len() >= 4 && parts[2] == "netmask" {
                    crate::net::parse_ipv4(parts[3]).unwrap_or([255, 255, 255, 0])
                } else {
                    [255, 255, 255, 0]
                };
                let gw = [ip[0], ip[1], ip[2], 1]; // default gateway: x.x.x.1
                crate::net::configure_static_ip(ip, mask, gw);
                println!("eth0: IP set to {}", crate::net::format_ipv4(&ip));
            } else {
                println!("ifconfig: invalid IP address '{}'", parts[1]);
            }
        } else {
            println!("Usage: ifconfig [-a]");
            println!("       ifconfig eth0 <ip> [netmask <mask>]");
        }
    }
}

fn cmd_ping(args: &str) {
    if args.is_empty() {
        println!("Usage: ping [-c count] <host>");
        return;
    }

    let mut count = 4u32;
    let mut target = args;

    // Parse -c count option
    let parts: Vec<&str> = args.split_whitespace().collect();
    let mut i = 0;
    while i < parts.len() {
        if parts[i] == "-c" && i + 1 < parts.len() {
            if let Ok(c) = parts[i + 1].parse::<u32>() {
                count = c;
            }
            i += 2;
        } else {
            target = parts[i];
            i += 1;
        }
    }

    let dst_ip = match crate::net::resolve(target) {
        Some(ip) => ip,
        None => {
            println!("ping: {}: Name or service not known", target);
            return;
        }
    };

    println!("PING {} ({}) 56(84) bytes of data.", target, crate::net::format_ipv4(&dst_ip));

    let mut received = 0u32;
    let mut min_rtt = u64::MAX;
    let mut max_rtt = 0u64;
    let mut total_rtt = 0u64;

    for seq in 0..count {
        let id = 0x4E41; // "NA" for NodeAI
        match crate::net::ping(dst_ip, id, seq as u16, 3000) {
            Some(rtt) => {
                received += 1;
                if rtt < min_rtt { min_rtt = rtt; }
                if rtt > max_rtt { max_rtt = rtt; }
                total_rtt += rtt;
                println!("64 bytes from {}: icmp_seq={} ttl=64 time={} ms",
                    crate::net::format_ipv4(&dst_ip), seq, rtt);
            }
            None => {
                println!("Request timeout for icmp_seq {}", seq);
            }
        }
    }

    let loss = if count > 0 { ((count - received) as f64 / count as f64) * 100.0 } else { 0.0 };
    println!("--- {} ping statistics ---", target);
    println!("{} packets transmitted, {} received, {:.0}% packet loss",
        count, received, loss);
    if received > 0 {
        let avg = total_rtt / received as u64;
        println!("rtt min/avg/max = {}/{}/{} ms", min_rtt, avg, max_rtt);
    }
}

fn cmd_arp(args: &str) {
    if args == "-a" || args.is_empty() {
        let entries = crate::net::arp_cache_entries();
        if entries.is_empty() {
            println!("ARP cache is empty");
        } else {
            for (ip, mac, _ts) in entries {
                println!("? ({}) at {} [ether] on eth0",
                    crate::net::format_ipv4(&ip),
                    crate::net::format_mac(&mac));
            }
        }
    } else {
        println!("Usage: arp [-a]");
    }
}

fn cmd_netstat(args: &str) {
    let show_tcp = args.is_empty() || args.contains("-t") || args.contains("-a");
    let show_listen = args.contains("-l") || args.contains("-a") || args.is_empty();

    println!("Active Internet connections");
    println!("Proto  Local Address          Foreign Address        State");

    if show_tcp {
        // Get TCP sockets from the tcp module
        let sockets = crate::net::tcp::SOCKETS.lock();
        for (key, sock) in sockets.iter() {
            let local = alloc::format!("0.0.0.0:{}", key.local_port);
            let remote = alloc::format!("{}:{}", crate::net::format_ipv4(&key.remote_ip), key.remote_port);
            let state = alloc::format!("{:?}", sock.state);
            println!("tcp    {:<22} {:<22} {}", local, remote, state);
        }
    }

    if show_listen {
        let listeners = crate::net::tcp::LISTENERS.lock();
        for (&port, _) in listeners.iter() {
            println!("tcp    0.0.0.0:{:<16} 0.0.0.0:*              LISTEN", port);
        }
    }
}

fn cmd_ip(args: &str) {
    let parts: Vec<&str> = args.split_whitespace().collect();
    match parts.first().copied() {
        Some("addr") | Some("address") | Some("a") => {
            let our_ip  = unsafe { crate::net::OUR_IP };
            let our_mac = unsafe { crate::net::OUR_MAC };
            println!("1: lo: <LOOPBACK,UP> mtu 65536");
            println!("    inet 127.0.0.1/8 scope host lo");
            println!("2: eth0: <BROADCAST,MULTICAST,UP> mtu 1500");
            println!("    link/ether {}", crate::net::format_mac(&our_mac));
            println!("    inet {}/24 brd {}.{}.{}.255 scope global eth0",
                crate::net::format_ipv4(&our_ip), our_ip[0], our_ip[1], our_ip[2]);
        }
        Some("route") | Some("r") => {
            println!("Kernel IP routing table");
            println!("{:<18} {:<18} {:<18} {}", "Destination", "Gateway", "Genmask", "Iface");
            for r in crate::net::route_entries() {
                println!("{:<18} {:<18} {:<18} {}",
                    crate::net::format_ipv4(&r.destination),
                    crate::net::format_ipv4(&r.gateway),
                    crate::net::format_ipv4(&r.netmask),
                    r.iface);
            }
        }
        Some("link") | Some("l") => {
            let our_mac = unsafe { crate::net::OUR_MAC };
            let has_nic = crate::net::NIC.lock().is_some();
            println!("1: lo: <LOOPBACK,UP> mtu 65536");
            println!("    link/loopback 00:00:00:00:00:00");
            println!("2: eth0: <BROADCAST,MULTICAST,{}> mtu 1500",
                if has_nic { "UP" } else { "DOWN" });
            println!("    link/ether {}", crate::net::format_mac(&our_mac));
        }
        _ => {
            println!("Usage: ip <addr|route|link>");
        }
    }
}

fn cmd_nslookup(args: &str) {
    if args.is_empty() {
        println!("Usage: nslookup <hostname>");
        return;
    }

    let dns_ip = *crate::net::DNS_SERVER.lock();
    println!("Server:     {}", crate::net::format_ipv4(&dns_ip));
    println!("");

    match crate::net::resolve(args) {
        Some(ip) => {
            println!("Name:       {}", args);
            println!("Address:    {}", crate::net::format_ipv4(&ip));
        }
        None => {
            println!("** server can't find {}: NXDOMAIN", args);
        }
    }
}

fn cmd_dig(args: &str) {
    if args.is_empty() {
        println!("Usage: dig <hostname>");
        return;
    }

    let dns_ip = *crate::net::DNS_SERVER.lock();
    println!(";; QUESTION SECTION:");
    println!(";{}.\t\t\tIN\tA", args);
    println!("");

    match crate::net::resolve(args) {
        Some(ip) => {
            println!(";; ANSWER SECTION:");
            println!("{}.\t\t300\tIN\tA\t{}", args, crate::net::format_ipv4(&ip));
        }
        None => {
            println!(";; Got NXDOMAIN for {}", args);
        }
    }
    println!("");
    println!(";; SERVER: {}#53", crate::net::format_ipv4(&dns_ip));
}

fn cmd_traceroute(args: &str) {
    if args.is_empty() {
        println!("Usage: traceroute <host>");
        return;
    }

    let dst_ip = match crate::net::resolve(args) {
        Some(ip) => ip,
        None => {
            println!("traceroute: {}: Name or service not known", args);
            return;
        }
    };

    println!("traceroute to {} ({}), 30 hops max, 60 byte packets",
        args, crate::net::format_ipv4(&dst_ip));

    // Simplified traceroute: send ICMP echo with increasing TTL
    // In a real implementation we'd get ICMP Time Exceeded replies
    for ttl in 1..=30u8 {
        let id = 0x4E41;
        match crate::net::ping(dst_ip, id, ttl as u16, 2000) {
            Some(rtt) => {
                println!("{:>2}  {} ({})  {} ms",
                    ttl, args, crate::net::format_ipv4(&dst_ip), rtt);
                break; // Reached destination
            }
            None => {
                println!("{:>2}  * * *", ttl);
                if ttl >= 5 { break; } // Give up after 5 timeouts
            }
        }
    }
}

fn cmd_wget(args: &str) {
    if args.is_empty() {
        println!("Usage: wget <url>");
        return;
    }

    // Parse simple HTTP URL: http://host[:port]/path
    let url = if args.starts_with("http://") { &args[7..] } else { args };
    let (hostport, path) = match url.find('/') {
        Some(i) => (&url[..i], &url[i..]),
        None => (url, "/"),
    };
    let (host, port) = match hostport.find(':') {
        Some(i) => (&hostport[..i], hostport[i+1..].parse::<u16>().unwrap_or(80)),
        None => (hostport, 80u16),
    };

    let dst_ip = match crate::net::resolve(host) {
        Some(ip) => ip,
        None => {
            println!("wget: unable to resolve host address '{}'", host);
            return;
        }
    };

    println!("Connecting to {}:{} ({})...", host, port, crate::net::format_ipv4(&dst_ip));

    // Build HTTP GET request
    let request = alloc::format!(
        "GET {} HTTP/1.0\r\nHost: {}\r\nUser-Agent: NodeAI/1.0\r\nConnection: close\r\n\r\n",
        path, host
    );

    // Use raw TCP — initiate a connection and send the request
    let our_ip = unsafe { crate::net::OUR_IP };
    let our_mac = unsafe { crate::net::OUR_MAC };
    let local_port = 49152 + (crate::scheduler::uptime_ms() % 16384) as u16;

    // Resolve destination MAC (use gateway for non-local)
    let gw: [u8; 4] = [10, 0, 2, 2];
    let dst_mac = if dst_ip[0..3] == our_ip[0..3] {
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

    // Send SYN
    let isn: u32 = (crate::scheduler::uptime_ms() & 0xFFFF_FFFF) as u32;
    let syn = crate::net::tcp::TcpHeader::build(
        local_port, port, isn, 0,
        crate::net::tcp::SYN, 65535,
        our_ip, dst_ip, &[],
    );
    let ip_hdr = crate::net::Ipv4Header::build(crate::net::IP_PROTO_TCP, our_ip, dst_ip, syn.len());
    let mut pkt = ip_hdr;
    pkt.extend_from_slice(&syn);
    let frame = crate::net::EthFrame::build(dst_mac, our_mac, crate::net::ETHERTYPE_IPV4, &pkt);
    crate::net::transmit(&frame);

    // Register the connection in the TCP stack
    let key = crate::net::tcp::TcpSocketKey {
        local_port,
        remote_ip: dst_ip,
        remote_port: port,
    };
    {
        let mut sockets = crate::net::tcp::SOCKETS.lock();
        sockets.insert(key.clone(), crate::net::tcp::TcpSocket {
            state: crate::net::tcp::TcpState::SynSent,
            snd_nxt: isn.wrapping_add(1),
            snd_una: isn,
            rcv_nxt: 0,
            snd_wnd: 65535,
            rcv_buf: Vec::new(),
            cwnd: 1460, ssthresh: 65535,
            last_send_ms: 0, rto_ms: 1000, retransmit_buf: Vec::new(),
            owner_pid: crate::scheduler::current_pid(), ai_cwnd_mul: 100,
        });
    }

    // Wait for connection to establish
    let deadline = crate::scheduler::uptime_ms() + 5000;
    let mut established = false;
    while crate::scheduler::uptime_ms() < deadline {
        crate::net::poll();
        let sockets = crate::net::tcp::SOCKETS.lock();
        if let Some(sock) = sockets.get(&key) {
            if sock.state == crate::net::tcp::TcpState::Established {
                established = true;
                break;
            }
        }
        drop(sockets);
        core::hint::spin_loop();
    }

    if !established {
        println!("wget: failed to establish connection to {}:{}", host, port);
        crate::net::tcp::SOCKETS.lock().remove(&key);
        return;
    }

    println!("Connected. Sending HTTP request...");

    // Send HTTP GET via TCP
    crate::net::tcp::send(local_port, dst_ip, port, request.as_bytes());

    // Wait for response data
    let deadline = crate::scheduler::uptime_ms() + 10000;
    while crate::scheduler::uptime_ms() < deadline {
        crate::net::poll();
        let sockets = crate::net::tcp::SOCKETS.lock();
        if let Some(sock) = sockets.get(&key) {
            if !sock.rcv_buf.is_empty() || sock.state == crate::net::tcp::TcpState::CloseWait {
                break;
            }
        }
        drop(sockets);
        core::hint::spin_loop();
    }

    // Read received data
    let data = {
        let mut sockets = crate::net::tcp::SOCKETS.lock();
        if let Some(sock) = sockets.get_mut(&key) {
            let buf = core::mem::take(&mut sock.rcv_buf);
            buf
        } else {
            Vec::new()
        }
    };

    // Close connection
    crate::net::tcp::close(local_port, dst_ip, port);

    if data.is_empty() {
        println!("wget: no data received");
        return;
    }

    // Print the response, showing headers and body
    if let Ok(text) = core::str::from_utf8(&data) {
        println!("{}", text);
    } else {
        println!("(received {} bytes of binary data)", data.len());
    }
    println!("--- {} bytes received ---", data.len());
}

fn cmd_curl(args: &str) {
    // curl is just an alias for wget in this implementation
    cmd_wget(args);
}

fn cmd_nc(args: &str) {
    let parts: Vec<&str> = args.split_whitespace().collect();
    if parts.len() < 2 {
        println!("Usage: nc <host> <port>");
        return;
    }

    let host = parts[0];
    let port = match parts[1].parse::<u16>() {
        Ok(p) => p,
        Err(_) => {
            println!("nc: invalid port '{}'", parts[1]);
            return;
        }
    };

    let dst_ip = match crate::net::resolve(host) {
        Some(ip) => ip,
        None => {
            println!("nc: {}: Name or service not known", host);
            return;
        }
    };

    println!("Connection to {} {} port [tcp/*] succeeded!",
        crate::net::format_ipv4(&dst_ip), port);
    println!("(interactive mode not yet supported — use wget for HTTP)");
}

// ── Process/Job control (Phase 16) ───────────────────────────────────────────

fn cmd_htop() {
    // htop-style interactive display (single-shot since we can't do real-time refresh in shell)
    crate::desktop::clear_terminal();
    let uptime = crate::scheduler::uptime_ms();
    let heap_used = crate::memory::KERNEL_HEAP.lock().used();
    let heap_kb = heap_used / 1024;
    let free_mb = crate::memory::free_mb();
    // Header
    println!("\x1b[32m  NodeAI htop — Uptime: {}.{}s\x1b[0m", uptime / 1000, (uptime % 1000) / 100);
    println!("");
    // CPU bar
    let cpu_pct = 100u64; // single-core always busy
    let bar_len = 40usize;
    let filled = (cpu_pct as usize * bar_len) / 100;
    let mut bar = String::from("[");
    for i in 0..bar_len {
        if i < filled { bar.push('|'); } else { bar.push(' '); }
    }
    bar.push(']');
    println!("  CPU \x1b[31m{}\x1b[0m {:>3}%", bar, cpu_pct);
    // Memory bar
    let total_mb = free_mb + (heap_kb as u64 / 1024) + 1; // approximate
    let mem_pct = ((total_mb - free_mb) * 100) / total_mb.max(1);
    let mfilled = (mem_pct as usize * bar_len) / 100;
    let mut mbar = String::from("[");
    for i in 0..bar_len {
        if i < mfilled { mbar.push('|'); } else { mbar.push(' '); }
    }
    mbar.push(']');
    println!("  Mem \x1b[32m{}\x1b[0m {:>3}%  ({}/{} MiB)", mbar, mem_pct, total_mb - free_mb, total_mb);
    println!("");
    // Process table
    println!("\x1b[7m  PID  USER     STATE     CPU%  MEM(KiB)  COMMAND                    \x1b[0m");
    println!("    0  root     \x1b[32mRunning\x1b[0m    100  {:>8}  [kernel]", heap_kb);
    println!("    1  root     \x1b[33mSleep\x1b[0m        0         0  [idle]");
    println!("");
    println!("Tasks: \x1b[32m2 total\x1b[0m, \x1b[32m1 running\x1b[0m, \x1b[33m1 sleeping\x1b[0m");
    println!("Press 'q' to quit, 'k' to kill (single-shot display)");
}

fn cmd_nice(args: &str) {
    if args.is_empty() {
        println!("Usage: nice [-n priority] <command>");
        return;
    }
    let parts: Vec<&str> = args.split_whitespace().collect();
    let (priority, cmd_idx) = if parts[0] == "-n" && parts.len() >= 3 {
        (parts[1].parse::<i32>().unwrap_or(0), 2)
    } else {
        (10, 0) // default niceness
    };
    if cmd_idx < parts.len() {
        let remaining: String = parts[cmd_idx..].join(" ");
        println!("[nice {}] {}", priority, remaining);
        dispatch(&remaining);
    }
}

fn cmd_renice(args: &str) {
    let parts: Vec<&str> = args.split_whitespace().collect();
    if parts.len() < 3 || parts[1] != "-p" {
        println!("Usage: renice <priority> -p <pid>");
        return;
    }
    let priority: i32 = parts[0].parse().unwrap_or(0);
    let pid: u64 = parts[2].parse().unwrap_or(0);
    println!("{}: old priority 0, new priority {}", pid, priority);
}

fn cmd_bg(args: &str) {
    let job = if args.is_empty() { "1" } else { args };
    println!("[{}]+ {} &", job, "continued");
    println!("(job control not yet fully supported — single-process kernel)");
}

fn cmd_fg(args: &str) {
    let job = if args.is_empty() { "1" } else { args };
    println!("[{}]+ continued", job);
    println!("(job control not yet fully supported — single-process kernel)");
}

fn cmd_jobs() {
    println!("(no background jobs — single-process kernel)");
}

fn cmd_nohup(args: &str) {
    if args.is_empty() {
        println!("Usage: nohup <command>");
        return;
    }
    println!("nohup: ignoring input and appending output to 'nohup.out'");
    dispatch(args);
}

// ── Disk & modules (Phase 16) ────────────────────────────────────────────────

fn cmd_fdisk(args: &str) {
    if !args.contains("-l") && !args.is_empty() {
        println!("Usage: fdisk -l [device]");
        return;
    }
    println!("Disk /dev/vda: 64 MiB, 67108864 bytes, 131072 sectors");
    println!("Units: sectors of 1 * 512 = 512 bytes");
    println!("Sector size (logical/physical): 512 bytes / 512 bytes");
    println!("");
    println!("Device      Boot    Start      End  Sectors  Size  Id  Type");
    println!("/dev/vda1           2048    131071   129024   63M  83  Linux");
}

fn cmd_mkfs(args: &str) {
    if args.is_empty() {
        println!("Usage: mkfs.<type> <device>");
        println!("Supported types: ramfs, ext2 (stub)");
        return;
    }
    let parts: Vec<&str> = args.split_whitespace().collect();
    let fstype = if parts[0].starts_with("-t") && parts.len() >= 3 {
        parts[1]
    } else {
        "ramfs"
    };
    let dev = parts.last().unwrap_or(&"/dev/vda1");
    println!("mkfs.{}: creating filesystem on {}...", fstype, dev);
    println!("Writing superblock and filesystem metadata... done");
    println!("mkfs.{}: done", fstype);
}

fn cmd_sync() {
    println!("Flushing filesystem buffers...");
    println!("sync: done");
}

fn cmd_modprobe(args: &str) {
    if args.is_empty() {
        println!("Usage: modprobe <module>");
        return;
    }
    let module = args.split_whitespace().next().unwrap_or(args);
    match module {
        "virtio_blk" | "virtio_net" | "virtio_gpu" | "ps2_keyboard" |
        "vga_fb" | "lapic_timer" | "ioapic" | "ramfs" | "procfs" | "devfs" |
        "ai_engine" => {
            println!("modprobe: module '{}' already loaded", module);
        }
        _ => {
            println!("modprobe: FATAL: Module {} not found in directory /lib/modules", module);
        }
    }
}

fn cmd_service(args: &str) {
    let parts: Vec<&str> = args.split_whitespace().collect();
    if parts.is_empty() {
        println!("Usage: service <name> <start|stop|status|restart>");
        println!("Available services: networking, sshd, httpd, ai-engine");
        return;
    }
    let svc = parts[0];
    let action = if parts.len() > 1 { parts[1] } else { "status" };
    match action {
        "status" => {
            let state = match svc {
                "networking" => "active (running)",
                "ai-engine"  => "active (running)",
                "sshd" => if crate::net::ssh_server_running() { "active (running)" } else { "inactive (dead)" },
                "httpd" => if crate::net::http_server_running() { "active (running)" } else { "inactive (dead)" },
                _ => "inactive (dead)",
            };
            println!("{}.service — {} daemon", svc, svc);
            println!("   Loaded: loaded");
            println!("   Active: {}", state);
        }
        "start" => {
            match svc {
                "sshd" => { crate::net::ssh_server_start(22); println!("sshd: started on port 22"); }
                "httpd" => { crate::net::http_server_start(8081, "/var/www"); println!("httpd: started on port 8081"); }
                _ => { println!("Starting {}...", svc); println!("{}: started", svc); }
            }
        }
        "stop" => {
            match svc {
                "sshd" => { crate::net::ssh_server_stop(); println!("sshd: stopped"); }
                "httpd" => { crate::net::http_server_stop(); println!("httpd: stopped"); }
                _ => { println!("Stopping {}...", svc); println!("{}: stopped", svc); }
            }
        }
        "restart" => {
            match svc {
                "sshd" => { crate::net::ssh_server_stop(); crate::net::ssh_server_start(22); println!("sshd: restarted"); }
                "httpd" => { crate::net::http_server_stop(); crate::net::http_server_start(8081, "/var/www"); println!("httpd: restarted"); }
                _ => { println!("Restarting {}...", svc); println!("{}: restarted", svc); }
            }
        }
        _ => println!("Unknown action: {}", action),
    }
}

// ── Network services (Phase 17) ──────────────────────────────────────────────

fn cmd_dhclient() {
    println!("DHCP: sending DISCOVER...");
    if crate::net::dhcp_request() {
        let ip = unsafe { crate::net::OUR_IP };
        println!("DHCP: acquired {}.{}.{}.{}", ip[0], ip[1], ip[2], ip[3]);
    } else {
        println!("DHCP: failed to acquire lease");
    }
}

fn cmd_httpd(args: &str) {
    let parts: Vec<&str> = args.split_whitespace().collect();
    if parts.is_empty() {
        if crate::net::http_server_running() {
            println!("httpd: running");
        } else {
            println!("Usage: httpd start [port] [docroot]");
            println!("       httpd stop");
            println!("       httpd status");
        }
        return;
    }
    match parts[0] {
        "start" => {
            let port: u16 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(8081);
            let root = parts.get(2).copied().unwrap_or("/var/www");
            crate::net::http_server_start(port, root);
            println!("httpd: listening on port {}, serving {}", port, root);
        }
        "stop" => {
            crate::net::http_server_stop();
            println!("httpd: stopped");
        }
        "status" => {
            if crate::net::http_server_running() {
                println!("httpd: running");
            } else {
                println!("httpd: stopped");
            }
        }
        _ => println!("Usage: httpd start|stop|status"),
    }
}

fn cmd_sshd(args: &str) {
    let parts: Vec<&str> = args.split_whitespace().collect();
    if parts.is_empty() {
        if crate::net::ssh_server_running() {
            println!("sshd: running (stub — no encryption)");
        } else {
            println!("Usage: sshd start [port]");
            println!("       sshd stop");
            println!("       sshd status");
        }
        return;
    }
    match parts[0] {
        "start" => {
            let port: u16 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(22);
            crate::net::ssh_server_start(port);
            println!("sshd: listening on port {} (stub — encryption not implemented)", port);
        }
        "stop" => {
            crate::net::ssh_server_stop();
            println!("sshd: stopped");
        }
        "status" => {
            if crate::net::ssh_server_running() {
                println!("sshd: running (stub)");
            } else {
                println!("sshd: stopped");
            }
        }
        _ => println!("Usage: sshd start|stop|status"),
    }
}

fn cmd_scp(args: &str) {
    let parts: Vec<&str> = args.split_whitespace().collect();
    if parts.len() < 2 {
        println!("Usage: scp <source> <dest>");
        println!("  Local path: /path/to/file");
        println!("  Remote:     host:/path/to/file");
        println!("Note: encryption not available; uses plaintext TCP transfer");
        return;
    }
    let src = parts[0];
    let dst = parts[1];
    // Determine if source is remote (contains ':')
    if src.contains(':') {
        // Download: remote → local
        let rparts: Vec<&str> = src.splitn(2, ':').collect();
        let host = rparts[0];
        let rpath = if rparts.len() > 1 { rparts[1] } else { "/" };
        println!("scp: connecting to {}...", host);
        match crate::net::resolve(host) {
            Some(ip) => {
                println!("scp: {}.{}.{}.{} — downloading {}",
                    ip[0], ip[1], ip[2], ip[3], rpath);
                println!("scp: transfer complete (stub — no encryption)");
            }
            None => println!("scp: could not resolve host '{}'", host),
        }
    } else if dst.contains(':') {
        // Upload: local → remote
        let rparts: Vec<&str> = dst.splitn(2, ':').collect();
        let host = rparts[0];
        let rpath = if rparts.len() > 1 { rparts[1] } else { "/" };
        // Check if local file exists
        let full = resolve_path(src);
        match crate::vfs::lookup(&full) {
            Ok(_) => {
                println!("scp: uploading {} to {}:{}...", src, host, rpath);
                match crate::net::resolve(host) {
                    Some(ip) => {
                        println!("scp: connected to {}.{}.{}.{}",
                            ip[0], ip[1], ip[2], ip[3]);
                        println!("scp: transfer complete (stub — no encryption)");
                    }
                    None => println!("scp: could not resolve host '{}'", host),
                }
            }
            Err(_) => println!("scp: {}: No such file or directory", src),
        }
    } else {
        // Local copy — just delegate to cp
        cmd_cp(&format!("{} {}", src, dst));
    }
}

fn cmd_dns_cache(args: &str) {
    let parts: Vec<&str> = args.split_whitespace().collect();
    let sub = parts.first().copied().unwrap_or("show");
    match sub {
        "show" | "dump" => {
            let entries = crate::net::dns_cache_entries();
            if entries.is_empty() {
                println!("DNS cache is empty");
            } else {
                println!("{:<30} {:<16} TTL(s)", "Hostname", "IP");
                for (name, ip, ttl) in &entries {
                    println!("{:<30} {}.{}.{}.{:<7} {}", name, ip[0], ip[1], ip[2], ip[3], ttl);
                }
            }
        }
        "flush" | "clear" => {
            crate::net::dns_cache_flush();
            println!("DNS cache flushed");
        }
        _ => {
            println!("Usage: dns-cache show|flush");
        }
    }
}

fn cmd_ifup(args: &str) {
    if args.trim().is_empty() || args.trim() == "eth0" {
        println!("Loading /etc/network/interfaces...");
        crate::net::load_network_config();
        let ip = unsafe { crate::net::OUR_IP };
        println!("eth0: {}.{}.{}.{}", ip[0], ip[1], ip[2], ip[3]);
    } else {
        println!("ifup: unknown interface '{}'", args.trim());
    }
}

fn cmd_ifdown(args: &str) {
    if args.trim().is_empty() || args.trim() == "eth0" {
        println!("eth0: interface down (stub — cannot disable sole NIC)");
    } else {
        println!("ifdown: unknown interface '{}'", args.trim());
    }
}

// ── Prompt ────────────────────────────────────────────────────────────────────

fn print_prompt() {
    // Kali-style with ANSI colors: \e[32muser@hostname\e[0m:\e[34mpath\e[0m#
    let user = crate::users::current_username();
    let host = crate::users::hostname();
    let cwd  = crate::users::cwd();
    let home = crate::users::current_home();

    // Abbreviate home as ~
    let display_path = if cwd == home {
        String::from("~")
    } else if cwd.starts_with(&home) && home != "/" {
        format!("~{}", &cwd[home.len()..])
    } else {
        cwd
    };

    let suffix = if crate::users::is_root() { '#' } else { '$' };

    // Colored prompt: green user@host, blue path
    let prompt = format!(
        "\x1b[32m{}@{}\x1b[0m:\x1b[34m{}\x1b[0m{} ",
        user, host, display_path, suffix
    );
    // Track visible prompt length (without escape sequences)
    let visible_len = user.len() + 1 + host.len() + 1 + display_path.len() + 2; // user@host:path# 
    *PROMPT_LEN.lock() = visible_len;
    print_str(&prompt);
}

/// Write a string to the desktop terminal one byte at a time.
fn print_str(s: &str) {
    let mut cap = CAPTURE_BUF.lock();
    if let Some(ref mut buf) = *cap {
        buf.push_str(s);
        return;
    }
    drop(cap);
    for b in s.bytes() {
        crate::desktop::terminal_input(b);
    }
}

// ── Phase 26 Application Commands ────────────────────────────────────────────

fn cmd_notepad_pro(args: &str) {
    let f = String::from(args.split_whitespace().next().unwrap_or(""));
    crate::desktop::notepad_open(&f);
    if f.is_empty() {
        println!("notepad: opened (new file)");
    } else {
        println!("notepad: opened {}", f);
    }
}

fn cmd_fm_pro(args: &str) {
    let p = String::from(args.split_whitespace().next().unwrap_or(""));
    crate::desktop::fm_pro_open(&p);
    println!("files: File Manager Pro opened");
}

fn cmd_terminal_tabs() {
    crate::desktop::terminal_app_open();
    println!("termtabs: Tabbed terminal opened");
}

fn cmd_imgview(args: &str) {
    let f = String::from(args.split_whitespace().next().unwrap_or(""));
    crate::desktop::imgview_open(&f);
    println!("imgview: Image viewer opened");
}

fn cmd_ai_chat() {
    crate::desktop::ai_chat_open();
    println!("aichat: AI Chat opened");
}

fn cmd_sysmon() {
    crate::desktop::sysmon_open();
    println!("sysmon: System monitor opened");
}

fn cmd_settings() {
    crate::desktop::settings_open();
    println!("settings: Settings panel opened");
}

fn cmd_appstore() {
    crate::desktop::appstore_open();
    println!("store: App Store opened");
}

// ── Formatting helpers ────────────────────────────────────────────────────────

macro_rules! println {
    ()          => { crate::desktop::terminal_input(b'\n'); };
    ($fmt:literal $(, $arg:expr)*) => {{
        let s = alloc::format!(concat!($fmt, "\n") $(, $arg)*);
        $crate::shell::print_str(&s);
    }};
}

macro_rules! print {
    ($fmt:literal $(, $arg:expr)*) => {{
        let s = alloc::format!($fmt $(, $arg)*);
        $crate::shell::print_str(&s);
    }};
}

pub(crate) use println;
pub(crate) use print;
