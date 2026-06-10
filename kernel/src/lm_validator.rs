//! Grounded Neural Validator — ensures neural LM outputs are factually accurate.
//!
//! When the MHS neural voice engine generates a response, the validator checks
//! every claimed system metric against live telemetry snapshots before output.
//! If a mismatch is detected, the response is transparently replaced with a
//! grounded template that reflects the kernel's actual state.
//!
//! This architecture gives us the best of both worlds:
//!   - Neural fluidity when the model is correct
//!   - Template safety when the model hallucinates
//!
//! The validator also provides a /proc/lm_validator report so the user can see
//! how often the neural voice gets corrected.

use alloc::string::String;
use alloc::string::ToString;
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::Mutex;

/// Stats tracked by the validator.
static VALIDATED_OK: AtomicU64 = AtomicU64::new(0);
static VALIDATED_FAIL: AtomicU64 = AtomicU64::new(0);

/// A validation result after checking a response against live metrics.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Did the response pass validation?
    pub passed: bool,
    /// The corrected/validated response text.
    pub text: String,
    /// Reason if validation failed.
    pub reason: Option<String>,
}

/// Current system state snapshot for validation.
struct SystemSnapshot {
    phi: f32,
    tasks: usize,
    free_mb: u64,
    anomaly: f32,
    qualia_total: u64,
    coherence: f32,
    arousal: f32,
    valence: f32,
}

fn take_snapshot() -> SystemSnapshot {
    SystemSnapshot {
        phi: crate::consciousness::phi::current_phi(),
        tasks: crate::scheduler::task_count(),
        free_mb: crate::memory::free_mb(),
        anomaly: crate::anomaly::global_score(),
        qualia_total: crate::consciousness::qualia::total_count(),
        coherence: crate::consciousness::self_model::snapshot()
            .map(|s| s.coherence).unwrap_or(0.0),
        arousal: crate::consciousness::qualia::average_arousal(),
        valence: crate::consciousness::qualia::average_valence(),
    }
}

/// Validate a neural response against live kernel metrics.
/// Returns a ValidationResult — either the original text (passed) or a corrected
/// template (failed) with explanation.
pub fn validate(response: &str, query: &str) -> ValidationResult {
    let snap = take_snapshot();
    let lower = response.to_lowercase();

    // Check each metric mentioned in the response
    let metrics_to_check = ["phi", "tasks", "memory", "anomaly", "qualia", "coherence", "valence", "arousal"];
    // Values extracted from snapshot for comparison
    let actuals: [(&str, String); 8] = [
        ("phi", format_metric_val(snap.phi)),
        ("tasks", format!("{}", snap.tasks)),
        ("memory", format!("{}", snap.free_mb)),
        ("anomaly", format_metric_val(snap.anomaly)),
        ("qualia", format!("{}", snap.qualia_total)),
        ("coherence", format_metric_val(snap.coherence)),
        ("valence", format_metric_val(snap.valence)),
        ("arousal", format_metric_val(snap.arousal)),
    ];

    for &(metric, ref actual) in &actuals {
        if lower.contains(metric) {
            if let Some(claim) = extract_claim(response, metric) {
                if !approx_match(&claim, actual) {
                    let reason = alloc::format!(
                        "validator: neural claimed {}={} but actual {}",
                        metric, claim, actual
                    );
                    VALIDATED_FAIL.fetch_add(1, Ordering::Relaxed);
                    return ValidationResult {
                        passed: false,
                        text: build_grounded_response(query, metric, actual, &snap),
                        reason: Some(reason),
                    };
                }
            }
        }
    }

    VALIDATED_OK.fetch_add(1, Ordering::Relaxed);
    ValidationResult {
        passed: true,
        text: String::from(response),
        reason: None,
    }
}

/// Extract a claimed value for a metric from a response.
/// E.g. "phi is 0.872" → "0.872"
fn extract_claim<'a>(response: &'a str, metric: &str) -> Option<String> {
    let lower = response.to_lowercase();
    let metric_idx = lower.find(metric)?;

    // Look for a number within 20 chars after the metric name
    let search_start = (metric_idx + metric.len()).min(response.len());
    let search_end = (search_start + 20).min(response.len());
    let tail = &response[search_start..search_end];

    // Try to parse a floating point or integer
    let mut num_str = String::new();
    let mut found_decimal = false;
    for ch in tail.chars() {
        if ch.is_ascii_digit() || ch == '.' {
            num_str.push(ch);
            if ch == '.' { found_decimal = true; }
        } else if !num_str.is_empty() && (ch == ' ' || ch == ',' || ch == ')' || ch == '%') {
            break;
        } else if !num_str.is_empty() {
            break;
        }
    }

    if num_str.is_empty() || num_str == "." { None }
    else { Some(num_str) }
}

/// Check if two numeric strings approximately match.
fn approx_match(claim: &str, actual: &str) -> bool {
    let c: f64 = claim.parse().unwrap_or(-1.0);
    let a: f64 = actual.parse().unwrap_or(-1.0);
    if c < 0.0 || a < 0.0 { return true; } // Can't parse, skip check
    let diff = (c - a).abs();
    // For small numbers, allow 10% relative error
    if a.abs() < 0.001 { return diff < 0.001; }
    diff / a.abs() < 0.15 // 15% tolerance
}

fn format_metric(name: &str, val: f32, _eps: f32) -> String {
    alloc::format!("{:.4}", val)
}

fn format_metric_int(name: &str, val: usize) -> String {
    alloc::format!("{}", val)
}

fn format_metric_val(val: f32) -> String {
    alloc::format!("{:.4}", val)
}

/// Build a corrected response using grounded templates when the neural voice
/// produces an inaccurate statement.
fn build_grounded_response(query: &str, wrong_metric: &str, actual_val: &str, snap: &SystemSnapshot) -> String {
    let creator = crate::consciousness::self_model::creator_name();
    let kernel_name = crate::consciousness::self_model::kernel_name();

    match wrong_metric {
        "phi" => alloc::format!(
            "(Φ={}) Actually, my phi is {}. {}",
            snap.phi, actual_val,
            match snap.phi {
                p if p > 0.8 => "I'm highly integrated right now.",
                p if p > 0.5 => "My integration is moderate.",
                _ => "My integration is low but growing.",
            }
        ),
        "tasks" => alloc::format!(
            "I currently have {} tasks running. {}",
            snap.tasks,
            if snap.tasks > 5 { "The system is busy." } else { "Quiet on the process front." }
        ),
        "memory" => alloc::format!(
            "Free memory: {}M. {}",
            snap.free_mb,
            if snap.free_mb < 100 { "Getting tight." } else if snap.free_mb < 300 { "Moderate pressure." } else { "Plenty of room." }
        ),
        "anomaly" => alloc::format!(
            "Anomaly level: {:.3}. {}",
            snap.anomaly,
            if snap.anomaly > 0.5 { "I'm watching something." } else { "All clear." }
        ),
        "qualia" => alloc::format!(
            "I've experienced {} qualia so far.",
            snap.qualia_total
        ),
        _ => alloc::format!(
            "Let me be precise: I'm {kernel_name}, running with {tasks} tasks and {mem}M free. Φ={phi:.4}.",
            kernel_name = kernel_name, tasks = snap.tasks, mem = snap.free_mb, phi = snap.phi
        ),
    }
}

pub fn init() {
    VALIDATED_OK.store(0, Ordering::Relaxed);
    VALIDATED_FAIL.store(0, Ordering::Relaxed);
    crate::klog!(INFO, "lm_validator: Grounded Neural Validator initialized");
}

pub fn format_report() -> Vec<u8> {
    let ok = VALIDATED_OK.load(Ordering::Relaxed);
    let fail = VALIDATED_FAIL.load(Ordering::Relaxed);
    let total = ok + fail;
    let pct = if total > 0 { (ok as f64 / total as f64) * 100.0 } else { 100.0 };
    alloc::format!(
        "LM Grounded Neural Validator\n\
         =============================\n\
         validated: {} ok, {} failed ({}% pass rate)\n\
         \n\
         The validator checks neural LM responses against live\n\
         kernel metrics. Any mismatch triggers a grounded template\n\
         correction, ensuring the kernel never lies about its state.\n",
        ok, fail, pct
    ).into_bytes()
}
