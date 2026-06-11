//! projectk-daemon — Userspace Project-K neural inference daemon.
//!
//! Communicates with the kernel through /dev/llm:
//!   1. Kernel writes a prompt to /dev/llm
//!   2. Daemon reads the prompt from /dev/llm
//!   3. Daemon runs Project-K inference (its own static muts, isolated address space)
//!   4. Daemon writes the response to /dev/llm
//!   5. Kernel reads the response
//!
//! This completely eliminates LLVM static mut aliasing bugs because the
//! inference engine runs in a SEPARATE PROCESS with its own virtual address space.
//! LLVM cannot alias memory across process boundaries.

mod inference;
mod tok;

use std::fs::OpenOptions;
use std::io::{self, Read, Write};
use std::path::Path;

const DEV_PATH: &str = "/dev/llm";
const BUF_SIZE: usize = 4096;

fn main() {
    eprintln!("projectk-daemon: starting...");

    // Initialize Project-K model (loads 1.8MB MHSI binary)
    inference::init();
    if !inference::is_loaded() {
        eprintln!("projectk-daemon: ERROR — model failed to load");
        std::process::exit(1);
    }
    eprintln!("projectk-daemon: model loaded, {} generations ready", inference::gen_count());

    // Main daemon loop: read query → infer → write response
    let mut buf = vec![0u8; BUF_SIZE];
    loop {
        // Read a query from /dev/llm
        let query = match read_query(&mut buf) {
            Some(q) => q,
            None => {
                std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
        };

        if query.is_empty() || query == "exit" {
            continue;
        }

        eprintln!("projectk-daemon: inferring for: {}", &query[..query.len().min(60)]);

        // Run inference
        match inference::generate(&query) {
            Some(response) => {
                eprintln!("projectk-daemon: response ({} chars)", response.len());
                // Write response to /dev/llm for kernel to read
                write_response(&response);
            }
            None => {
                eprintln!("projectk-daemon: inference failed");
            }
        }
    }
}

/// Read a query from /dev/llm. Returns None if no query available.
fn read_query(buf: &mut [u8]) -> Option<String> {
    let mut f = match OpenOptions::new().read(true).write(true).open(DEV_PATH) {
        Ok(f) => f,
        Err(_) => return None,
    };

    // Set non-blocking read
    // Read available data
    let n = match f.read(buf) {
        Ok(n) if n > 0 => n,
        _ => return None,
    };

    let text = String::from_utf8_lossy(&buf[..n]).trim().to_string();
    if text.is_empty() { None } else { Some(text) }
}

/// Write a response to /dev/llm for the kernel to read.
fn write_response(response: &str) {
    if let Ok(mut f) = OpenOptions::new().write(true).open(DEV_PATH) {
        let _ = f.write_all(response.as_bytes());
        let _ = f.write_all(b"\n");
    }
}
