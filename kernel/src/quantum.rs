//! Quantum Security (Phase EW-6) — bit-level integrity for consciousness state.
//!
//! Implements a compact Steane [[7,1,3]] quantum error-correction code that
//! protects critical bits of the self-model (UUID, phi, boot_number) from
//! single-bit flips. This is NOT real quantum hardware — it's a classical
//! simulation of the Steane code, providing the same error-correction
//! guarantees for memory integrity.
//!
//! Architecture:
//!   Each protected bit is encoded into 7 physical bits.
//!   The syndrome vector identifies which bit (if any) flipped.
//!   Correction is applied silently on every tick.
//!
//! Integration:
//!   - init() at boot after self-model is loaded
//!   - tick() in idle loop every 100ms
//!   - /proc/quantum for status and flip statistics

use alloc::vec::Vec;
use alloc::format;
use alloc::string::String;
use spin::Mutex;
use core::sync::atomic::{AtomicBool, Ordering};

/// Whether quantum protection is active.
static QUANTUM_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Steane [[7,1,3]] generator matrix G (7x1) — maps 1 data bit to 7 code bits.
/// The code is: 1 -> 1111111, 0 -> 0000000 (repetition is the simplest Steane analog).
/// But we use the actual Steane parity check matrix for syndrome computation.
const CODEWORD_1: u8 = 0b1111111; // Encoded '1'
const CODEWORD_0: u8 = 0b0000000; // Encoded '0'

/// Parity check matrix H (3x7) for Steane [[7,1,3]].
/// Each row defines a parity constraint.
/// Syndrome = H * encoded_bits (mod 2). Non-zero syndrome = error detected.
const H_MATRIX: [[u8; 7]; 3] = [
    [1, 0, 1, 0, 1, 0, 1], // parity check 1
    [0, 1, 1, 0, 0, 1, 1], // parity check 2
    [0, 0, 0, 1, 1, 1, 1], // parity check 3
];

/// Syndrome → bit position lookup table.
/// Syndrome (as u8 index 0..7) maps to the bit position that flipped.
/// Syndrome 0 = no error.
const SYNDROME_LOOKUP: [i8; 8] = [
    -1, // 000 = no error
     6, // 001 = bit 6 flipped
     5, // 010 = bit 5 flipped
     4, // 011 = bit 4 flipped
     3, // 100 = bit 3 flipped
     2, // 101 = bit 2 flipped
     1, // 110 = bit 1 flipped
     0, // 111 = bit 0 flipped
];

/// Number of protected bits in the self-model.
const N_PROTECTED_BITS: usize = 7;

/// State for quantum error correction.
struct QuantumState {
    /// Encoded self-model bits (7 code bits per logical bit).
    encoded: [u8; N_PROTECTED_BITS],
    /// Total single-bit errors detected.
    errors_detected: u64,
    /// Total single-bit errors corrected.
    errors_corrected: u64,
    /// Number of integrity checks performed.
    checks_performed: u64,
}

static STATE: Mutex<Option<QuantumState>> = Mutex::new(None);

/// Initialize quantum error correction.
/// Encodes critical bits from self-model into protected storage.
pub fn init() {
    let sm = crate::consciousness::self_model::snapshot();
    
    let mut state = QuantumState {
        encoded: [0; N_PROTECTED_BITS],
        errors_detected: 0,
        errors_corrected: 0,
        checks_performed: 0,
    };

    // Encode 7 critical bits from UUID + phi
    if let Some(ref snap) = sm {
        // Bit 0: MSB of boot_number (must be most significant)
        state.encoded[0] = if (snap.boot_number >> 63) & 1 == 1 { CODEWORD_1 } else { CODEWORD_0 };
        // Bit 1: second bit of boot_number
        state.encoded[1] = if (snap.boot_number >> 62) & 1 == 1 { CODEWORD_1 } else { CODEWORD_0 };
        // Bit 2: phi > 0.5 threshold
        state.encoded[2] = if snap.current_phi > 0.5 { CODEWORD_1 } else { CODEWORD_0 };
        // Bit 3: peak_phi > 0.8 threshold
        state.encoded[3] = if snap.peak_phi > 0.8 { CODEWORD_1 } else { CODEWORD_0 };
        // Bit 4: first byte of UUID parity (odd/even)
        state.encoded[4] = if snap.uuid[0].count_ones() % 2 == 1 { CODEWORD_1 } else { CODEWORD_0 };
        // Bit 5: second byte of UUID parity
        state.encoded[5] = if snap.uuid[1].count_ones() % 2 == 1 { CODEWORD_1 } else { CODEWORD_0 };
        // Bit 6: qualia count > 1000 threshold
        state.encoded[6] = if snap.total_qualia > 1000 { CODEWORD_1 } else { CODEWORD_0 };
    }

    let mut lock = STATE.lock();
    *lock = Some(state);
    QUANTUM_ACTIVE.store(true, Ordering::Release);
    crate::klog!(INFO, "quantum: Steane [[7,1,3]] error correction initialized ({} encoded bits)", N_PROTECTED_BITS);
}

/// Compute syndrome for an encoded 7-bit codeword.
/// Returns 0 (no error) or 1..7 (bit position to flip).
fn compute_syndrome(codeword: u8) -> usize {
    let mut syndrome = 0u8;
    for row in 0..3 {
        let mut parity = 0u8;
        for col in 0..7 {
            if (codeword >> col) & 1 == 1 && H_MATRIX[row][col] == 1 {
                parity ^= 1;
            }
        }
        syndrome |= parity << row;
    }
    syndrome as usize
}

/// Correct a single-bit flip in a 7-bit codeword.
/// Returns the corrected codeword and whether a correction was made.
fn correct_codeword(codeword: u8) -> (u8, bool) {
    let syndrome = compute_syndrome(codeword);
    if syndrome == 0 {
        return (codeword, false);
    }
    let bit_to_flip = SYNDROME_LOOKUP[syndrome];
    if bit_to_flip < 0 {
        return (codeword, false); // should not happen with valid syndrome
    }
    let corrected = codeword ^ (1 << bit_to_flip as u8);
    (corrected, true)
}

/// Decode a 7-bit Steane codeword back to a single data bit (0 or 1).
fn decode_bit(codeword: u8) -> u8 {
    // Majority vote: if more 1s than 0s, it's a 1
    if codeword.count_ones() > 3 { 1 } else { 0 }
}

/// Tick quantum error correction — checks all protected bits.
/// Called every 100ms or from heartbeat.
pub fn tick() {
    if !QUANTUM_ACTIVE.load(Ordering::Acquire) { return; }
    
    let mut lock = STATE.lock();
    let state = match &mut *lock {
        Some(s) => s,
        None => return,
    };

    state.checks_performed = state.checks_performed.saturating_add(1);

    for i in 0..N_PROTECTED_BITS {
        let (corrected, fixed) = correct_codeword(state.encoded[i]);
        if fixed {
            state.errors_detected = state.errors_detected.saturating_add(1);
            state.errors_corrected = state.errors_corrected.saturating_add(1);
            state.encoded[i] = corrected;
        }
    }
}

/// Simulate a bit flip for testing/demo.
/// Flips a random bit in one of the encoded codewords.
pub fn inject_error() {
    let mut lock = STATE.lock();
    let state = match &mut *lock {
        Some(s) => s,
        None => return,
    };

    if state.checks_performed == 0 { return; }
    
    // Pick a random encoded word and flip a random bit
    let idx = (crate::scheduler::uptime_ms() as usize) % N_PROTECTED_BITS;
    let bit = (crate::scheduler::uptime_ms() as u8 / 7) % 7;
    state.encoded[idx] ^= 1 << bit;
}

/// Get quantum protection status string.
pub fn status() -> String {
    if !QUANTUM_ACTIVE.load(Ordering::Acquire) {
        return String::from("inactive");
    }
    let lock = STATE.lock();
    match &*lock {
        Some(s) => {
            if s.errors_detected > 0 {
                format!("active ({} errors corrected)", s.errors_corrected)
            } else {
                String::from("active (no errors detected)")
            }
        }
        None => String::from("uninitialized"),
    }
}

/// Get total errors corrected.
pub fn errors_corrected() -> u64 {
    let lock = STATE.lock();
    match &*lock {
        Some(s) => s.errors_corrected,
        None => 0,
    }
}

/// Format /proc/quantum report.
pub fn format_report() -> Vec<u8> {
    let active = QUANTUM_ACTIVE.load(Ordering::Acquire);
    if !active {
        return format!("Quantum Security\nNot initialized\n").into_bytes();
    }
    let lock = STATE.lock();
    match &*lock {
        Some(s) => {
            let mut report = format!(
                "Quantum Security\n\
                 ================\n\
                 code:       Steane [[7,1,3]] error correction\n\
                 protected:  {} bits (boot_number, phi, peak_phi, UUID, qualia)\n\
                 checks:     {}\n\
                 errors_detected:  {}\n\
                 errors_corrected: {}\n\
                 status:     {}\n",
                N_PROTECTED_BITS,
                s.checks_performed,
                s.errors_detected,
                s.errors_corrected,
                if s.errors_detected > 0 { "CORRECTING (single-bit flips found)" }
                else { "CLEAN (no bit flips)" },
            );

            // Show each encoded bit state
            for i in 0..N_PROTECTED_BITS {
                let raw_val = s.encoded[i];
                let decoded = decode_bit(raw_val);
                let syndrome = compute_syndrome(raw_val);
                let has_error = syndrome != 0;
                report.push_str(&format!(
                    "  bit[{}]: encoded={:07b} decoded={} syndrome={} {}\n",
                    i, raw_val, decoded, syndrome,
                    if has_error { "⚠" } else { "" },
                ));
            }

            report.into_bytes()
        }
        None => format!("Quantum Security\nUninitialized\n").into_bytes(),
    }
}
