//! Spectrum Sensing Algorithms (Phase EW-1).
//!
//! Ported from BHEW's Python spectrum sensing chain to no_std Rust.
//! Provides cyclostationary feature extraction, Gabor time-frequency analysis,
//! and energy detection for the kernel's electromagnetic sense.
//!
//! All algorithms operate on f32 IQ samples (interleaved I, Q, I, Q, ...).

use alloc::vec::Vec;

/// A spectral feature extracted from the signal.
#[derive(Debug, Clone)]
pub struct SpectralFeature {
    /// Feature frequency bin (normalized 0..1).
    pub alpha: f32,
    /// Cyclic autoconrelation magnitude.
    pub magnitude: f32,
    /// Modulation class hint (if identifiable).
    pub modulation_hint: ModulationClass,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModulationClass {
    Unknown,
    Bpsk,
    Qpsk,
    Fsk,
    Am,
    Fm,
    Noise,
}

/// Energy detection — fast path for "is anything there?"
/// Returns true if the RMS power exceeds the threshold.
pub fn energy_detect(samples: &[f32], threshold: f32) -> bool {
    if samples.is_empty() {
        return false;
    }
    let mut sum_sq = 0.0_f64;
    for &s in samples {
        sum_sq += (s as f64) * (s as f64);
    }
    let rms = libm::sqrtf((sum_sq / samples.len() as f64) as f32);
    rms > threshold
}

/// Simple single-cycle cyclic autoconrelation function (CAF).
/// A peak at non-zero alpha indicates cyclostationarity — useful for
/// distinguishing modulated signals from noise.
///
/// Simplified: computes |sum(s[n] * s[n+tau] * exp(-j*2*pi*alpha*n))|
pub fn caf_single(samples: &[f32], alpha: f32, tau: usize) -> f32 {
    if samples.len() < tau + 2 {
        return 0.0;
    }
    let n = samples.len() - tau;
    let mut real_sum = 0.0_f64;
    let mut imag_sum = 0.0_f64;
    for i in 0..n {
        // IQ samples are interleaved: s[2i] = I, s[2i+1] = Q
        let i_idx = 2 * i;
        let q_idx = 2 * i + 1;
        let tau_i_idx = 2 * (i + tau);
        let tau_q_idx = 2 * (i + tau) + 1;

        if q_idx >= samples.len() || tau_q_idx >= samples.len() {
            break;
        }

        let s_i = samples[i_idx] as f64;
        let s_q = samples[q_idx] as f64;
        let s_tau_i = samples[tau_i_idx] as f64;
        let s_tau_q = samples[tau_q_idx] as f64;

        // s[n] * conj(s[n+tau])
        let prod_real = s_i * s_tau_i + s_q * s_tau_q;
        let prod_imag = s_q * s_tau_i - s_i * s_tau_q;

        // exp(-j*2*pi*alpha*n)
        let theta = -2.0 * core::f64::consts::PI * (alpha as f64) * (i as f64);
        let (s_t, c) = libm::sincosf(theta as f32);

        real_sum += (prod_real as f64) * (c as f64) - (prod_imag as f64) * (s_t as f64);
        imag_sum += (prod_real as f64) * (s_t as f64) + (prod_imag as f64) * (c as f64);
    }
    libm::sqrtf((real_sum * real_sum + imag_sum * imag_sum) as f32) / (n as f32)
}

/// Extract cyclostationary features by scanning a range of alpha values.
/// Returns features sorted by magnitude (highest first).
pub fn cyclostationary_features(samples: &[f32], num_alphas: usize) -> Vec<SpectralFeature> {
    let mut features = Vec::with_capacity(num_alphas);
    let step = 1.0 / (num_alphas as f32 + 1.0);
    for i in 0..num_alphas {
        let alpha = (i as f32 + 1.0) * step;
        let mag = caf_single(samples, alpha, 1);
        let hint = if mag > 0.3 {
            if alpha < 0.3 {
                ModulationClass::Bpsk
            } else if alpha < 0.6 {
                ModulationClass::Qpsk
            } else {
                ModulationClass::Fsk
            }
        } else if mag > 0.1 {
            ModulationClass::Am
        } else {
            ModulationClass::Noise
        };
        features.push(SpectralFeature {
            alpha,
            magnitude: mag,
            modulation_hint: hint,
        });
    }
    // Sort by magnitude descending
    features.sort_by(|a, b| b.magnitude.partial_cmp(&a.magnitude).unwrap_or(core::cmp::Ordering::Equal));
    features
}

/// Gabor transform — STFT with a Gaussian window.
/// Returns a time-frequency matrix: [time_bins][freq_bins].
pub fn gabor_tf(samples: &[f32], window_size: usize, sigma: f32) -> Vec<Vec<f32>> {
    if samples.len() < window_size {
        return Vec::new();
    }
    let num_windows = samples.len() / window_size;
    let num_freq = window_size / 2 + 1;
    let mut result = Vec::with_capacity(num_windows);

    // Pre-compute Gaussian window
    let mut window = Vec::with_capacity(window_size);
    let center = (window_size as f32 - 1.0) / 2.0;
    let sigma_sq = sigma * sigma;
    for i in 0..window_size {
        let x = i as f32 - center;
        let w = libm::expf(-0.5 * x * x / sigma_sq);
        window.push(w);
    }

    for w in 0..num_windows {
        let start = w * window_size;
        let end = core::cmp::min(start + window_size, samples.len());
        if end - start < 2 {
            break;
        }

        // Apply window to this segment
        let mut segment = Vec::with_capacity(end - start);
        for i in start..end {
            segment.push(samples[i] * window[i - start]);
        }

        // Simple DFT (for real signals — take IQ magnitude first)
        let n = segment.len();
        let mut freqs = Vec::with_capacity(num_freq);
        for k in 0..num_freq {
            let mut real = 0.0_f64;
            let mut imag = 0.0_f64;
            for (i, &val) in segment.iter().enumerate() {
                let theta = -2.0 * core::f64::consts::PI * (k as f64) * (i as f64) / (n as f64);
                let (s, c) = libm::sincosf(theta as f32);
                real += (val as f64) * (c as f64);
                imag += (val as f64) * (s as f64);
            }
            let mag = libm::sqrtf((real * real + imag * imag) as f32) / (n as f32);
            freqs.push(mag);
        }
        result.push(freqs);
    }
    result
}

/// Compute the power spectral density estimate via periodogram.
/// Returns frequency bins with their power magnitudes.
pub fn periodogram(samples: &[f32]) -> Vec<(f32, f32)> {
    let n = samples.len();
    if n < 2 {
        return Vec::new();
    }
    let num_freq = n / 2 + 1;
    let mut result = Vec::with_capacity(num_freq);

    for k in 0..num_freq {
        let mut real = 0.0_f64;
        let mut imag = 0.0_f64;
        for (i, &val) in samples.iter().enumerate() {
            let theta = -2.0 * core::f64::consts::PI * (k as f64) * (i as f64) / (n as f64);
            let (s, c) = libm::sincosf(theta as f32);
            real += (val as f64) * (c as f64);
            imag += (val as f64) * (s as f64);
        }
        let power = (real * real + imag * imag) as f32;
        let freq = k as f32 / n as f32;
        result.push((freq, power));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_detect_silence() {
        let samples = vec![0.0_f32; 100];
        assert!(!energy_detect(&samples, 0.01));
    }

    #[test]
    fn test_energy_detect_signal() {
        let samples: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        assert!(energy_detect(&samples, 0.01));
    }

    #[test]
    fn test_caf_noise() {
        let samples: Vec<f32> = vec![0.0; 64];
        let mag = caf_single(&samples, 0.1, 1);
        // All-zero input should give ~0
        assert!(mag < 0.001);
    }

    #[test]
    fn test_cyclostationary_features_noise() {
        let samples: Vec<f32> = vec![0.0; 128];
        let features = cyclostationary_features(&samples, 5);
        // All-zero input → all magnitudes near 0
        for f in &features {
            assert!(f.magnitude < 0.001);
        }
    }

    #[test]
    fn test_periodogram_sanity() {
        let samples: Vec<f32> = vec![1.0; 16];
        let psd = periodogram(&samples);
        assert!(!psd.is_empty());
        // DC component should dominate
        assert!(psd[0].1 > 0.0);
    }

    #[test]
    fn test_gabor_tf_empty() {
        let result = gabor_tf(&[], 4, 1.0);
        assert!(result.is_empty());
    }
}
