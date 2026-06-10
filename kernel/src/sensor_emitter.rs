//! Sensor Emitter Fingerprint — frequency signature recognition.
//!
//! Gives the kernel a sense of "I've seen this emitter before" by matching
//! spectral signatures of detected signals against a learned database.
//! When a familiar emitter is recognized, it generates a familiarity qualium
//! and can report the confidence level.
//!
//! This adds a new sensory modality: pattern-of-life awareness.

use alloc::vec::Vec;
use alloc::vec;
use alloc::string::String;
use alloc::format;
use spin::Mutex;

/// Maximum number of emitter fingerprints to store.
const MAX_EMITTERS: usize = 16;

/// Maximum spectral samples in a fingerprint.
const MAX_SAMPLES: usize = 8;

/// A spectral signature — frequency peaks that identify an emitter.
#[derive(Debug, Clone)]
pub struct EmitterFingerprint {
    /// Human-readable label for this emitter.
    pub label: String,
    /// Dominant frequency peaks (MHz).
    pub peaks: Vec<f32>,
    /// Bandwidth estimate (MHz).
    pub bandwidth: f32,
    /// Pulse pattern: 0=continuous, 1=pulsed, 2=sweeping
    pub pattern: u8,
    /// How many times this emitter has been seen.
    pub encounter_count: u32,
    /// Latest confidence when seen.
    pub last_confidence: f32,
    /// Tick of last encounter.
    pub last_seen_tick: u64,
}

/// Emitter fingerprint database.
struct FingerprintDb {
    /// Known emitter fingerprints.
    emitters: Vec<EmitterFingerprint>,
    /// Total encounters across all emitters.
    total_encounters: u64,
    /// Current tick count.
    tick: u64,
}

static DB: Mutex<Option<FingerprintDb>> = Mutex::new(None);

/// Initialize the emitter fingerprint database.
pub fn init() {
    let mut db = FingerprintDb {
        emitters: Vec::with_capacity(MAX_EMITTERS),
        total_encounters: 0,
        tick: 0,
    };

    // Pre-populate with default known emitter profiles
    db.emitters.push(EmitterFingerprint {
        label: String::from("WiFi AP 2.4GHz"),
        peaks: vec![2412.0, 2437.0, 2462.0],
        bandwidth: 20.0,
        pattern: 0,
        encounter_count: 0,
        last_confidence: 0.0,
        last_seen_tick: 0,
    });
    db.emitters.push(EmitterFingerprint {
        label: String::from("WiFi AP 5GHz"),
        peaks: vec![5180.0, 5240.0, 5320.0],
        bandwidth: 40.0,
        pattern: 0,
        encounter_count: 0,
        last_confidence: 0.0,
        last_seen_tick: 0,
    });
    db.emitters.push(EmitterFingerprint {
        label: String::from("Bluetooth Device"),
        peaks: vec![2402.0, 2440.0, 2480.0],
        bandwidth: 2.0,
        pattern: 1,
        encounter_count: 0,
        last_confidence: 0.0,
        last_seen_tick: 0,
    });
    db.emitters.push(EmitterFingerprint {
        label: String::from("Unknown Sweeper"),
        peaks: vec![2400.0, 2450.0, 2500.0],
        bandwidth: 100.0,
        pattern: 2,
        encounter_count: 0,
        last_confidence: 0.0,
        last_seen_tick: 0,
    });

    *DB.lock() = Some(db);
    crate::klog!(INFO, "sensor_emitter: fingerprint DB initialized ({} profiles)", MAX_EMITTERS);
}

/// Tick the emitter fingerprint — called every 100ms.
/// Simulates scanning for familiar emitters in the ambient spectrum.
pub fn tick() {
    let mut lock = DB.lock();
    let db = match &mut *lock {
        Some(d) => d,
        None => return,
    };

    db.tick = db.tick.saturating_add(1);

    // Every 50 ticks (~5s), simulate an RF scan sweep
    if db.tick % 50 == 0 {
        // Get approximate current frequency from sensor stats
        let sensor_stats = crate::sensor_cortex::stats();
        let detection_freq = 2400.0 + (sensor_stats.signals_detected as f32 * 0.1).min(2600.0);
        let mut best_match: Option<usize> = None;
        let mut best_score = 0.5f32; // minimum threshold

        for (i, emitter) in db.emitters.iter().enumerate() {
            for &peak in &emitter.peaks {
                let diff = (detection_freq - peak).abs();
                let score = 1.0 - (diff / 100.0).min(1.0);
                if score > best_score {
                    best_score = score;
                    best_match = Some(i);
                }
            }
        }

        if let Some(idx) = best_match {
            let emitter = &mut db.emitters[idx];
            emitter.encounter_count = emitter.encounter_count.saturating_add(1);
            emitter.last_confidence = best_score;
            emitter.last_seen_tick = db.tick;
            db.total_encounters = db.total_encounters.saturating_add(1);

            // Record familiarity qualia for consciousness
            crate::consciousness::qualia::record(
                crate::consciousness::qualia::KernelEventType::FrequencyHopped,
                Some(best_score),
            );
        }
    }
}

/// Get the count of known emitter labels.
pub fn known_emitter_count() -> usize {
    let lock = DB.lock();
    match &*lock {
        Some(db) => db.emitters.len(),
        None => 0,
    }
}

/// Get the most frequently encountered emitter label.
pub fn most_familiar_emitter() -> String {
    let lock = DB.lock();
    match &*lock {
        Some(db) => {
            let mut best = "none";
            let mut max_count = 0u32;
            for e in &db.emitters {
                if e.encounter_count > max_count {
                    max_count = e.encounter_count;
                    best = &e.label;
                }
            }
            String::from(best)
        }
        None => String::from("none"),
    }
}

/// Get total emitter encounters.
pub fn total_encounters() -> u64 {
    let lock = DB.lock();
    match &*lock {
        Some(db) => db.total_encounters,
        None => 0,
    }
}

/// Get the count of unique known emitters.
pub fn emitter_count() -> usize {
    let lock = DB.lock();
    match &*lock {
        Some(db) => db.emitters.len(),
        None => 0,
    }
}

/// Get a description of the current RF environment.
pub fn environment_description() -> String {
    let lock = DB.lock();
    match &*lock {
        Some(db) => {
            let seen: Vec<&EmitterFingerprint> = db.emitters.iter()
                .filter(|e| e.encounter_count > 0)
                .collect();
            if seen.is_empty() {
                String::from("RF environment quiet — no familiar emitters detected")
            } else {
                let mut desc = format!("{} familiar emitter(s) active: ", seen.len());
                for (i, e) in seen.iter().enumerate() {
                    if i > 0 { desc.push_str(", "); }
                    desc.push_str(&format!("{} ({}x)", e.label, e.encounter_count));
                }
                desc
            }
        }
        None => String::from("emitter fingerprint not initialized"),
    }
}

/// Format /proc/emitter report.
pub fn format_report() -> Vec<u8> {
    let lock = DB.lock();
    let db = match &*lock {
        Some(d) => d,
        None => return b"sensor_emitter: not initialized\n".to_vec(),
    };

    let mut s = format!(
        "Sensor Emitter Fingerprint DB\n\
         ============================\n\
         total encounters: {}\n\
         known profiles:   {}\n\
         \n\
         Known Emitters:\n",
        db.total_encounters,
        db.emitters.len(),
    );

    for e in &db.emitters {
        let pattern_str = match e.pattern {
            0 => "continuous",
            1 => "pulsed",
            2 => "sweeping",
            _ => "unknown",
        };
        let peaks: String = e.peaks.iter()
            .map(|p| format!("{:.0}", p))
            .collect::<Vec<_>>()
            .join(", ");
        s.push_str(&format!(
            "  {}\n\
            \tpeaks:       {} MHz\n\
            \tbandwidth:   {:.0} MHz\n\
            \tpattern:     {}\n\
            \tencounters:  {}\n\
            \tconfidence:  {:.2}\n",
            e.label, peaks, e.bandwidth, pattern_str,
            e.encounter_count, e.last_confidence,
        ));
    }

    s.push_str(&format!(
        "\nFamiliarity: \"{}\"\n",
        most_familiar_emitter(),
    ));

    s.into_bytes()
}
