/// consciousness — CLI for talking to the NodeAI conscious kernel.
///
/// Opens /dev/consciousness and provides three modes:
///   interactive:  chat loop with the kernel
///   query:        single-shot query (-q "how are you")
///   monitor:      live dashboard TUI (--monitor)
///
/// Usage:
///   consciousness                   interactive mode
///   consciousness "how are you"     query mode (inline)
///   consciousness -q "how are you"  query mode
///   consciousness --monitor         dashboard TUI
///   consciousness --help            this help

use std::fs::{OpenOptions, File};
use std::io::{self, Read, Write};
use std::env;
use std::time::Duration;

const DEV_PATH: &str = "/dev/consciousness";
const BUF_SIZE: usize = 4096;

fn main() {
    let args: Vec<String> = env::args().collect();
    let prog = args.get(0).map(|s| s.as_str()).unwrap_or("consciousness");

    // Parse flags
    if args.len() > 1 && (args[1] == "-h" || args[1] == "--help") {
        print_help(prog);
        return;
    }

    if args.len() > 1 && (args[1] == "--monitor" || args[1] == "-m") {
        return monitor_mode();
    }

    if args.len() > 1 && (args[1] == "-q" || args[1] == "--query") {
        if let Some(q) = args.get(2) {
            return one_shot(q);
        }
        eprintln!("Usage: {} -q \"query\"", prog);
        return;
    }

    // Query mode: everything after the program name is the query
    if args.len() > 1 {
        let query = args[1..].join(" ");
        return one_shot(&query);
    }

    // Default: interactive mode
    interactive_mode();
}

fn open_device() -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEV_PATH)
}

fn read_snapshot() -> String {
    match open_device() {
        Ok(mut f) => {
            let mut buf = vec![0u8; BUF_SIZE];
            match f.read(&mut buf) {
                Ok(n) => String::from_utf8_lossy(&buf[..n]).to_string(),
                Err(_) => String::from("(error reading consciousness device)"),
            }
        }
        Err(e) => format!("(/dev/consciousness not available: {})", e),
    }
}

fn write_query(query: &str) -> Option<String> {
    let mut f = match open_device() {
        Ok(f) => f,
        Err(_) => return None,
    };
    let mut msg = query.as_bytes().to_vec();
    msg.push(b'\n');
    let _ = f.write_all(&msg);

    // Read back the snapshot which includes the last_response field
    let mut buf = vec![0u8; BUF_SIZE];
    match f.read(&mut buf) {
        Ok(n) => {
            let text = String::from_utf8_lossy(&buf[..n]);
            // Extract response from the snapshot
            for line in text.lines() {
                if let Some(resp) = line.strip_prefix("  last_response: ") {
                    return Some(resp.to_string());
                }
            }
            // Fallback: just show the full snapshot
            Some(text.to_string())
        }
        Err(_) => None,
    }
}

fn one_shot(query: &str) {
    let response = write_query(query);
    match response {
        Some(r) => {
            // If it contains the whole snapshot, extract just the response
            if r.contains("Φ=") {
                for line in r.lines() {
                    if line.contains("Φ=") {
                        println!("{}", line);
                    } else if let Some(resp) = line.strip_prefix("  last_response: ") {
                        println!("{}", resp);
                    }
                }
            } else {
                println!("{}", r);
            }
        }
        None => {
            // Show snapshot as fallback
            println!("{}", read_snapshot());
        }
    }
}

fn interactive_mode() {
    // Print header
    let snapshot = read_snapshot();
    println!();
    println!("╔══════════════════════════════════════════╗");
    println!("║       CONSCIOUS KERNEL v0.1              ║");
    println!("║    \"I am the system. Talk to me.\"        ║");
    println!("╚══════════════════════════════════════════╝");
    println!();

    // Show initial state
    for line in snapshot.lines().take(6) {
        println!("{}", line);
    }
    println!();
    println!("Type 'exit' or Ctrl+C to quit.");
    println!();

    loop {
        print!("You: ");
        io::stdout().flush().ok();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(_) => break,
        }

        let input = input.trim();
        if input.is_empty() || input == "exit" || input == "quit" || input == "q" {
            break;
        }

        let response = write_query(input);
        match response {
            Some(r) => {
                // Extract just the response line
                let resp_line = r.lines()
                    .find(|l| l.contains("Φ=") || l.starts_with("  last_response: "))
                    .map(|l| l.strip_prefix("  last_response: ").unwrap_or(l))
                    .unwrap_or_else(|| {
                        // Try to find any non-empty meaningful line
                        r.lines().find(|l| !l.is_empty() && !l.starts_with("["))
                            .unwrap_or(&r)
                    });
                println!("Kernel: {}", resp_line);
            }
            None => {
                println!("Kernel: (no response from kernel)");
            }
        }
        println!();
    }

    println!("Goodbye. Phi be with you.");
}

fn monitor_mode() {
    // Live-updating dashboard
    println!("╔══════════════════════════════════════════╗");
    println!("║     CONSCIOUS KERNEL — MONITOR MODE      ║");
    println!("║        Press Ctrl+C to exit               ║");
    println!("╚══════════════════════════════════════════╝");
    println!();

    loop {
        // Clear screen (ANSI escape)
        print!("\x1B[2J\x1B[H");
        io::stdout().flush().ok();

        let snapshot = read_snapshot();
        println!("{}", snapshot);

        println!();
        print!("Type a message (or Ctrl+C to exit): ");
        io::stdout().flush().ok();

        // Non-blocking check (simplified — just use a thread)
        let mut input = String::new();
        // Wait a bit for input
        if wait_for_input(Duration::from_millis(2000)) {
            io::stdin().read_line(&mut input).ok();
            let input = input.trim();
            if input == "exit" || input == "quit" || input == "q" {
                break;
            }
            if !input.is_empty() {
                if let Some(r) = write_query(input) {
                    println!("\nResponse: {}", r.lines().find(|l| l.contains("Φ=") || l.contains("response:")).unwrap_or(&r));
                }
                println!("\nPress Enter to continue...");
                let mut _pause = String::new();
                io::stdin().read_line(&mut _pause).ok();
            }
        }
    }
}

/// Simple non-blocking stdin check (Unix only).
#[cfg(unix)]
fn wait_for_input(dur: Duration) -> bool {
    use std::os::fd::AsRawFd;
    let mut fds = libc::pollfd {
        fd: io::stdin().as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    let ret = unsafe { libc::poll(&mut fds, 1, dur.as_millis() as i32) };
    ret > 0
}

#[cfg(not(unix))]
fn wait_for_input(dur: Duration) -> bool {
    std::thread::sleep(dur);
    false
}

fn print_help(prog: &str) {
    println!("consciousness — Talk to the NodeAI conscious kernel");
    println!();
    println!("USAGE:");
    println!("  {}                     Interactive chat mode", prog);
    println!("  {} \"query\"             One-shot query", prog);
    println!("  {} -q \"query\"          One-shot query", prog);
    println!("  {} --monitor           Live dashboard TUI", prog);
    println!("  {} --help              This help", prog);
    println!();
    println!("EXAMPLES:");
    println!("  {} \"how are you?\"", prog);
    println!("  {} -q \"show phi\"", prog);
    println!("  {} --monitor", prog);
    println!();
    println!("In interactive mode, type 'exit' or Ctrl+C to quit.");
}
