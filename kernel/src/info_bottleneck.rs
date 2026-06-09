//! Information Bottleneck — which subsystem signals the kernel keeps vs discards.
//!
//! Ported from Project-C's information_bottleneck.py.
//!
//! Uses the cross_modal coupling matrix to compute per-domain retention:
//! how much of each domain's past predicts the others through a rank-reduced
//! bottleneck.  High retention = the integrated system treats this domain as
//! relevant.  Low retention = effectively ignored.
//!
//! In kernel terms: does scheduler coherence predict memory?  Does anomaly
//! predict syscall rate?  The Information Bottleneck tells us which couplings
//! the AI engine should attend to and which are noise.

use crate::cross_modal::{Domain, coupling};
use alloc::vec::Vec;

/// Number of domains (must match cross_modal::N_DOMAINS).
const N_DOMAINS: usize = 4;

/// Domain labels in order.
const DOMAIN_NAMES: [&str; N_DOMAINS] = ["scheduler", "memory", "anomaly", "syscall"];

/// Build the C×C cross-domain coupling matrix at a given lag.
/// entry[i][j] = |coupling(Domain_i → Domain_j, lag)| (absolute value).
fn coupling_matrix(lag: usize) -> [[f32; N_DOMAINS]; N_DOMAINS] {
    let mut mat = [[0.0f32; N_DOMAINS]; N_DOMAINS];
    let domains = [
        Domain::Scheduler,
        Domain::Memory,
        Domain::Anomaly,
        Domain::Syscall,
    ];
    for (i, src) in domains.iter().enumerate() {
        for (j, tgt) in domains.iter().enumerate() {
            if i == j { continue; }
            let c = coupling(*src, *tgt, lag);
            mat[i][j] = c.abs();
        }
    }
    mat
}

/// Compute retention scores from the coupling matrix.
///
/// For each domain i:
///   retention[i] = mean of |coupling(i → j, lag=1)| across all j ≠ i
///   + mean of |coupling(j → i, lag=1)| across all j ≠ i
///
/// This measures both how much i predicts others AND how much others predict i.
/// Normalized to sum to 1.0.
fn compute_retention() -> [f32; N_DOMAINS] {
    let mat = coupling_matrix(1);
    let mut retention = [0.0f32; N_DOMAINS];

    for i in 0..N_DOMAINS {
        let mut outgoing = 0.0f32;
        let mut incoming = 0.0f32;
        let mut count = 0u32;
        for j in 0..N_DOMAINS {
            if i == j { continue; }
            outgoing += mat[i][j];
            incoming += mat[j][i];
            count += 1;
        }
        if count > 0 {
            retention[i] = (outgoing / count as f32 + incoming / count as f32) / 2.0;
        }
    }

    // Normalize to sum to 1.0
    let total: f32 = retention.iter().sum();
    if total > 1e-10 {
        for r in retention.iter_mut() {
            *r /= total;
        }
    }

    retention
}

/// Return retention score for a specific domain [0.0, 1.0] — fraction of
/// information the system retains from this domain.
pub fn domain_retention(domain: Domain) -> f32 {
    let retention = compute_retention();
    retention[domain as usize]
}

/// Return the full retention vector (normalized, sums to 1).
pub fn all_retention() -> [f32; N_DOMAINS] {
    compute_retention()
}

/// Format /proc report.
pub fn format_report() -> Vec<u8> {
    use alloc::format;
    use alloc::string::String;

    let ret = compute_retention();
    let mut out = String::from("NodeAI Information Bottleneck (Project-C)\n");
    out.push_str("========================================\n");
    out.push_str(&format!("domains: {}\n", N_DOMAINS));
    out.push_str("retention (fraction of total information kept):\n");

    // Show sorted by retention descending
    let mut pairs: Vec<(usize, f32)> = ret.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

    for (idx, val) in &pairs {
        let label = if *val > 1.0 / N_DOMAINS as f32 { "kept" } else { "discarded" };
        let bar_len = (val * 40.0) as usize;
        let bar = core::iter::repeat('#').take(bar_len).collect::<String>();
        out.push_str(&format!(
            "  {:12} {:5.1}%  {:40} {}\n",
            DOMAIN_NAMES[*idx], val * 100.0, bar, label
        ));
    }

    out.into_bytes()
}
