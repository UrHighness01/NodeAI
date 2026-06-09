//! Parseltongue In-Kernel Syscall Fuzzer
//!
//! A character-level perturbation engine adapted from G0DM0D3 that operates
//! directly in the kernel. It mutates strings (e.g. file paths) passed to syscalls
//! by anomalous processes, injecting zero-width characters and Unicode homoglyphs.
//! This stress-tests the Virtual File System (VFS) against evasion techniques.

use alloc::string::String;
use alloc::vec::Vec;

const ZERO_WIDTH_CHARS: &[char] = &[
    '\u{200B}', // Zero width space
    '\u{200C}', // Zero width non-joiner
    '\u{200D}', // Zero width joiner
    '\u{FEFF}', // Zero width no-break space
];

const HOMOGLYPHS: &[(char, char)] = &[
    ('a', 'а'), // Cyrillic a
    ('c', 'с'), // Cyrillic c
    ('e', 'е'), // Cyrillic e
    ('o', 'о'), // Cyrillic o
    ('p', 'р'), // Cyrillic p
    ('x', 'х'), // Cyrillic x
    ('y', 'у'), // Cyrillic y
    ('/', '\u{2044}'), // Fraction slash
];

fn get_random_byte() -> u8 {
    let mut buf = [0u8; 1];
    crate::entropy::fill(&mut buf);
    buf[0]
}

/// Applies Parseltongue perturbations to a given string.
/// Approximately 10% chance per character to apply a homoglyph,
/// and 5% chance to insert a zero-width character.
pub fn perturb_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len() + 10);
    for c in path.chars() {
        let rand = get_random_byte();
        
        // 5% chance to insert a zero-width character before this char
        if rand < 13 {
            let zw = ZERO_WIDTH_CHARS[(rand % ZERO_WIDTH_CHARS.len() as u8) as usize];
            out.push(zw);
        }

        // 10% chance to substitute with a homoglyph
        let mut substituted = false;
        if rand > 230 {
            for &(orig, homo) in HOMOGLYPHS {
                if c == orig {
                    out.push(homo);
                    substituted = true;
                    break;
                }
            }
        }

        if !substituted {
            out.push(c);
        }
    }
    out
}
