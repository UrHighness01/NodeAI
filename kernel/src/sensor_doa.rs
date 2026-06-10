//! Direction Finding (Phase EW-2) — MUSIC & ESPRIT DOA estimation.
//!
//! Ported from BHEW's Python direction finding chain to no_std Rust.
//! Provides MUSIC (MUltiple SIgnal Classification) pseudospectrum and
//! ESPRIT (Estimation of Signal Parameters via Rotational Invariance)
//! for direction-of-arrival estimation from a uniform linear array.
//!
//! These algorithms transform the kernel's threat detection from
//! "something is there" to "something is there at bearing 137°".

use alloc::vec::Vec;
use alloc::vec;
use alloc::string::String;
use alloc::format;

/// A direction-of-arrival estimate.
#[derive(Debug, Clone)]
pub struct DoaEstimate {
    /// Bearing in degrees (0-360, where 0 = north/boresight).
    pub bearing_deg: f32,
    /// Signal power estimate (relative).
    pub power: f32,
    /// Confidence 0..1.
    pub confidence: f32,
    /// Whether this was classified as a valid signal (vs noise peak).
    pub is_valid: bool,
}

/// MUSIC pseudospectrum for a uniform linear array.
/// Returns the pseudospectrum values at evenly-spaced angles.
///
/// Args:
///   covariance: The sample covariance matrix (num_antennas x num_antennas, row-major)
///   num_sources: Number of signal sources to estimate
///   num_antennas: Number of antennas in the array
///   num_angles: Number of angle bins to scan (0-180°)
///
/// Returns: Vec of pseudospectrum values (one per angle bin), higher = more likely signal.
pub fn music_spectrum(
    covariance: &[f32],
    num_sources: usize,
    num_antennas: usize,
    num_angles: usize,
) -> Vec<f32> {
    let mut spectrum = vec![0.0f32; num_angles];

    if num_antennas < 2 || num_sources >= num_antennas || covariance.len() < num_antennas * num_antennas {
        return spectrum;
    }

    // Step 1: Eigendecomposition of covariance matrix (simplified — power iteration)
    // We need the noise subspace eigenvectors.
    // For simplicity, we use the eigenvectors corresponding to the (N - num_sources)
    // smallest eigenvalues.
    let noise_dim = num_antennas - num_sources;

    // Compute eigenvalues and eigenvectors via Jacobi iteration (simplified)
    let mut eigvals = vec![0.0f32; num_antennas];
    let mut eigvecs = vec![0.0f32; num_antennas * num_antennas];

    // Copy covariance to working matrix
    let mut work = covariance.to_vec();

    // Simple iterative Jacobi eigenvalue algorithm (reduced to 8 iterations for speed)
    for _iter in 0..8 {
        let mut max_off = 0.0f32;
        let mut p = 0;
        let mut q = 1;
        for i in 0..num_antennas {
            for j in (i + 1)..num_antennas {
                let val = work[i * num_antennas + j].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-6 { break; }

        let app = work[p * num_antennas + p];
        let aqq = work[q * num_antennas + q];
        let apq = work[p * num_antennas + q];

        let theta = 0.5 * libm::atanf(2.0 * apq / (aqq - app).max(1e-10));
        let c = libm::cosf(theta);
        let s = libm::sinf(theta);

        // Apply Jacobi rotation to work matrix
        for i in 0..num_antennas {
            if i != p && i != q {
                let aip = work[i * num_antennas + p];
                let aiq = work[i * num_antennas + q];
                work[i * num_antennas + p] = c * aip - s * aiq;
                work[p * num_antennas + i] = work[i * num_antennas + p];
                work[i * num_antennas + q] = s * aip + c * aiq;
                work[q * num_antennas + i] = work[i * num_antennas + q];
            }
        }
        let app_new = c * c * app + s * s * aqq - 2.0 * s * c * apq;
        let aqq_new = s * s * app + c * c * aqq + 2.0 * s * c * apq;
        work[p * num_antennas + p] = app_new;
        work[q * num_antennas + q] = aqq_new;
        work[p * num_antennas + q] = 0.0;
        work[q * num_antennas + p] = 0.0;

        // Update eigenvectors (accumulate rotations)
        if _iter == 0 {
            // Initialize eigenvectors to identity
            for i in 0..num_antennas {
                eigvecs[i * num_antennas + i] = 1.0;
            }
        }
        for i in 0..num_antennas {
            let vip = eigvecs[i * num_antennas + p];
            let viq = eigvecs[i * num_antennas + q];
            eigvecs[i * num_antennas + p] = c * vip - s * viq;
            eigvecs[i * num_antennas + q] = s * vip + c * viq;
        }
    }

    // Extract eigenvalues from diagonal
    for i in 0..num_antennas {
        eigvals[i] = work[i * num_antennas + i];
    }

    // Sort eigenvectors by eigenvalue (ascending — noise subspace first)
    let mut indices: Vec<usize> = (0..num_antennas).collect();
    indices.sort_by(|a, b| eigvals[*a].partial_cmp(&eigvals[*b]).unwrap_or(core::cmp::Ordering::Equal));

    // Build noise subspace matrix (columns = noise eigenvectors)
    let noise_vecs: Vec<f32> = (0..noise_dim).flat_map(|ni| {
        let col = indices[ni]; // column of noise eigenvector
        (0..num_antennas).map(|row| eigvecs[row * num_antennas + col]).collect::<Vec<_>>()
    }).collect();

    // Steering vectors and MUSIC spectrum
    let d = 0.5; // element spacing in wavelengths (half-wavelength default)
    for ai in 0..num_angles {
        let angle_deg = (ai as f32) * 180.0 / (num_angles as f32 - 1.0);
        let angle_rad = angle_deg * core::f32::consts::PI / 180.0;

        // Steering vector for this angle
        let mut steering = Vec::with_capacity(num_antennas);
        for i in 0..num_antennas {
            let phase = -2.0 * core::f32::consts::PI * (i as f32) * d * libm::sinf(angle_rad);
            let (s, c) = (libm::sinf(phase), libm::cosf(phase));
            // Store as complex pair (real, imag) — we just need magnitude^2
            steering.push(c);
            steering.push(s);
        }

        // Compute |a^H * En|^2 (projection onto noise subspace)
        let mut proj_mag_sq = 0.0f32;
        for ni in 0..noise_dim {
            let mut real_sum = 0.0f32;
            let mut imag_sum = 0.0f32;
            for i in 0..num_antennas {
                let a_real = steering[i * 2];
                let a_imag = steering[i * 2 + 1];
                let en = noise_vecs[ni * num_antennas + i];
                // Complex multiply: conj(a) * en
                real_sum += a_real * en;
                imag_sum += -a_imag * en;
            }
            proj_mag_sq += real_sum * real_sum + imag_sum * imag_sum;
        }

        // MUSIC pseudospectrum P(θ) = 1 / (a^H * En * En^H * a)
        spectrum[ai] = 1.0 / (proj_mag_sq + 1e-10);
    }

    // Normalize spectrum to 0..1
    let max_val = spectrum.iter().fold(0.0f32, |a, &b| a.max(b));
    if max_val > 0.0 {
        for s in spectrum.iter_mut() { *s /= max_val; }
    }

    spectrum
}

/// Delay-and-sum beamforming — quick bearing estimate (no eigendecomposition).
/// Fast, low-resolution, good for initial detection.
pub fn delay_sum_beamform(samples: &[Vec<f32>], num_angles: usize) -> Vec<f32> {
    let num_antennas = samples.len();
    if num_antennas < 2 { return vec![0.0f32; num_angles]; }
    let sample_len = samples[0].len();
    if sample_len < 2 { return vec![0.0f32; num_angles]; }

    let d = 0.5; // half-wavelength spacing
    let mut spectrum = vec![0.0f32; num_angles];

    for ai in 0..num_angles {
        let angle_deg = (ai as f32) * 180.0 / (num_angles as f32 - 1.0);
        let angle_rad = angle_deg * core::f32::consts::PI / 180.0;

        let mut sum = 0.0f32;
        // Compute delay-and-sum over all antenna pairs
        for i in 0..num_antennas {
            for j in (i + 1)..num_antennas {
                let delay = (j - i) as f32 * d * libm::sinf(angle_rad);
                let delay_samples = libm::roundf(delay * sample_len as f32) as usize;
                if delay_samples >= sample_len { continue; }

                // Correlate: sum(s_i[n] * s_j[n - delay])
                let mut corr = 0.0f32;
                for n in delay_samples..sample_len {
                    corr += samples[i][n] * samples[j][n - delay_samples];
                }
                sum += corr.abs();
            }
        }
        spectrum[ai] = sum;
    }

    // Normalize
    let max_val = spectrum.iter().fold(0.0f32, |a, &b| a.max(b));
    if max_val > 0.0 {
        for s in spectrum.iter_mut() { *s /= max_val; }
    }
    spectrum
}

/// Extract DOA peaks from a MUSIC/beamform spectrum.
/// Returns the top N peaks sorted by power (highest first).
pub fn extract_peaks(spectrum: &[f32], num_peaks: usize, threshold: f32) -> Vec<DoaEstimate> {
    let mut peaks = Vec::new();
    if spectrum.is_empty() { return peaks; }

    let num_angles = spectrum.len();

    // Find local maxima
    for i in 1..(num_angles - 1) {
        if spectrum[i] > spectrum[i - 1] && spectrum[i] >= spectrum[i + 1] && spectrum[i] >= threshold {
            let angle_deg = (i as f32) * 180.0 / (num_angles as f32 - 1.0);
            peaks.push(DoaEstimate {
                bearing_deg: angle_deg,
                power: spectrum[i],
                confidence: (spectrum[i] - threshold) / (1.0 - threshold).max(0.01),
                is_valid: spectrum[i] > threshold * 1.5,
            });
        }
    }

    // Sort by power descending
    peaks.sort_by(|a, b| b.power.partial_cmp(&a.power).unwrap_or(core::cmp::Ordering::Equal));

    // Take top N
    peaks.truncate(num_peaks);
    peaks
}

/// Simplified ESPRIT DOA — faster than MUSIC, no peak search needed.
/// Returns bearings directly.
pub fn esprit_doa(
    covariance: &[f32],
    num_sources: usize,
    num_antennas: usize,
) -> Vec<DoaEstimate> {
    let mut results = Vec::new();
    if num_antennas < 2 || num_sources >= num_antennas || num_sources == 0 {
        return results;
    }

    // Extract signal subspace via eigendecomposition (same Jacobi as MUSIC)
    let mut eigvals = vec![0.0f32; num_antennas];
    let mut eigvecs = vec![0.0f32; num_antennas * num_antennas];
    let mut work = covariance.to_vec();

    for _iter in 0..8 {
        let mut max_off = 0.0f32;
        let mut p = 0; let mut q = 1;
        for i in 0..num_antennas {
            for j in (i + 1)..num_antennas {
                let val = work[i * num_antennas + j].abs();
                if val > max_off { max_off = val; p = i; q = j; }
            }
        }
        if max_off < 1e-6 { break; }
        let app = work[p * num_antennas + p];
        let aqq = work[q * num_antennas + q];
        let apq = work[p * num_antennas + q];
        let theta = 0.5 * libm::atanf(2.0 * apq / (aqq - app).max(1e-10));
        let c = libm::cosf(theta);
        let s = libm::sinf(theta);
        for i in 0..num_antennas {
            if i != p && i != q {
                let aip = work[i * num_antennas + p];
                let aiq = work[i * num_antennas + q];
                work[i * num_antennas + p] = c * aip - s * aiq;
                work[p * num_antennas + i] = work[i * num_antennas + p];
                work[i * num_antennas + q] = s * aip + c * aiq;
                work[q * num_antennas + i] = work[i * num_antennas + q];
            }
        }
        let app_new = c * c * app + s * s * aqq - 2.0 * s * c * apq;
        let aqq_new = s * s * app + c * c * aqq + 2.0 * s * c * apq;
        work[p * num_antennas + p] = app_new;
        work[q * num_antennas + q] = aqq_new;
        work[p * num_antennas + q] = 0.0;
        work[q * num_antennas + p] = 0.0;
        if _iter == 0 {
            for i in 0..num_antennas { eigvecs[i * num_antennas + i] = 1.0; }
        }
        for i in 0..num_antennas {
            let vip = eigvecs[i * num_antennas + p];
            let viq = eigvecs[i * num_antennas + q];
            eigvecs[i * num_antennas + p] = c * vip - s * viq;
            eigvecs[i * num_antennas + q] = s * vip + c * viq;
        }
    }
    for i in 0..num_antennas { eigvals[i] = work[i * num_antennas + i]; }

    // Sort by eigenvalue descending (signal subspace first)
    let mut indices: Vec<usize> = (0..num_antennas).collect();
    indices.sort_by(|a, b| eigvals[*b].partial_cmp(&eigvals[*a]).unwrap_or(core::cmp::Ordering::Equal));

    // ESPRIT: partition signal subspace into A1 (first N-1 rows) and A2 (last N-1 rows)
    let d = 0.5; // half-wavelength
    for si in 0..num_sources.min(num_antennas - 1) {
        let col = indices[si];

        // Extract the two sub-arrays from the signal eigenvector
        let mut a1 = 0.0f32;
        let mut a2 = 0.0f32;
        let mut mag1 = 0.0f32;
        let mut mag2 = 0.0f32;

        for i in 0..num_antennas - 1 {
            let e1 = eigvecs[i * num_antennas + col];
            let e2 = eigvecs[(i + 1) * num_antennas + col];
            a1 += e1 * e2;
            mag1 += e1 * e1;
            mag2 += e2 * e2;
        }

        if mag1 < 1e-10 || mag2 < 1e-10 { continue; }

        // The rotational invariance gives us e^{j*phi} where phi = 2*pi*d*sin(theta)/lambda
        // From the correlation of A1 and A2
        let correlation = a1 / libm::sqrtf(mag1 * mag2);
        let phi = libm::acosf(correlation.clamp(-1.0, 1.0));
        let bearing_rad = libm::asinf((phi / (2.0 * core::f32::consts::PI * d)).clamp(-1.0, 1.0));
        let bearing_deg = bearing_rad * 180.0 / core::f32::consts::PI;

        results.push(DoaEstimate {
            bearing_deg: bearing_deg.abs(),
            power: eigvals[col] / eigvals[indices[0]].max(1e-10),
            confidence: (eigvals[col] / eigvals[indices[0]].max(1e-10)).min(1.0),
            is_valid: bearing_deg.abs() < 180.0,
        });
    }

    results
}

// ── Global DOA state ─────────────────────────────────────────────────────────

use spin::Mutex;

struct DoaState {
    last_bearings: Vec<DoaEstimate>,
    tick_count: u64,
}

static DOA_STATE: Mutex<Option<DoaState>> = Mutex::new(None);

pub fn init() {
    *DOA_STATE.lock() = Some(DoaState {
        last_bearings: Vec::new(),
        tick_count: 0,
    });
    crate::klog!(INFO, "sensor_doa: Direction finding initialized (MUSIC/ESPRIT)");
}

/// Run DOA estimation on simulated array data from sensor_cortex.
/// Called from sensor_cortex::tick().
pub fn tick(energy_values: &[f32]) {
    let mut state = DOA_STATE.lock();
    let state = match &mut *state {
        Some(s) => s,
        None => return,
    };
    state.tick_count += 1;

    // Only run DOA every 30 ticks (3 seconds) to save CPU (MUSIC is expensive)
    if state.tick_count % 30 != 0 { return; }
    if energy_values.len() < 4 { return; }

    // Build a simulated covariance matrix from the energy values (fast approximation)
    let n_ants = energy_values.len().min(8);
    let mut cov = vec![0.0f32; n_ants * n_ants];

    // Quick diagonal-dominant covariance (no full matrix build)
    for i in 0..n_ants {
        for j in 0..n_ants {
            let idx = i * n_ants + j;
            cov[idx] = if i == j { energy_values[i] * energy_values[i] + 0.1 } else { 0.01 };
        }
    }

    // Run MUSIC to get spectrum (reduced to 45 angles for speed)
    let num_sources = if n_ants > 2 { 1 } else { 1 };
    let spectrum = music_spectrum(&cov, num_sources, n_ants, 45);

    // Extract peaks
    let peaks = extract_peaks(&spectrum, 3, 0.3);
    state.last_bearings = peaks;
}

/// Get the latest DOA estimates.
pub fn last_bearings() -> Vec<DoaEstimate> {
    DOA_STATE.lock().as_ref().map(|s| s.last_bearings.clone()).unwrap_or_default()
}

/// Format /proc report.
pub fn format_report() -> Vec<u8> {
    let mut s = String::new();
    s.push_str("=== Direction Finding (MUSIC/ESPRIT) ===\n");
    let bearings = last_bearings();
    if bearings.is_empty() {
        s.push_str("  No signals detected.\n");
    } else {
        for (i, b) in bearings.iter().enumerate() {
            let marker = if b.is_valid { "◉" } else { "○" };
            s.push_str(&format!(
                "  [{}] {} bearing={:.1}° power={:.2} conf={:.2}\n",
                i, marker, b.bearing_deg, b.power, b.confidence
            ));
        }
    }
    s.into_bytes()
}
