//! Emotional Arc Tracking (P2) — longitudinal affective state monitoring.
//!
//! Maps valence/arousal shifts over time into a coherent emotional narrative
//! so the kernel can articulate its "mood" over minutes rather than just
//! reporting immediate sensations.
//!
//! Uses a ring buffer of (valence, arousal) snapshots at configurable
//! intervals. Provides trend analysis: improving, declining, stable,
//! volatile, etc.

use alloc::string::String;
use alloc::vec::Vec;
use spin::Mutex;

/// Number of historical snapshots to retain.
const HISTORY_LEN: usize = 60;

/// Sampling interval in ticks (every 10 ticks ≈ 1s at 100ms tick).
const SAMPLE_INTERVAL: u64 = 10;

/// A single emotional snapshot.
#[derive(Debug, Clone, Copy)]
pub struct EmotionalSample {
    pub valence: f32,
    pub arousal: f32,
    pub phi: f32,
    pub tick: u64,
}

/// Emotional trend analysis.
#[derive(Debug, Clone)]
pub struct EmotionalTrend {
    /// Direction: "improving", "declining", "stable", "volatile", "mixed"
    pub direction: &'static str,
    /// Current mood label based on recent samples
    pub mood: &'static str,
    /// Valence slope (per sample index)
    pub valence_slope: f32,
    /// Arousal slope
    pub arousal_slope: f32,
    /// Variance (volatility measure)
    pub volatility: f32,
    /// Number of data points
    pub n_samples: usize,
}

struct ArcTracker {
    samples: Vec<EmotionalSample>,
    last_sample_tick: u64,
}

impl ArcTracker {
    fn new() -> Self {
        Self {
            samples: Vec::with_capacity(HISTORY_LEN),
            last_sample_tick: 0,
        }
    }

    fn tick(&mut self, now_tick: u64) {
        if now_tick < self.last_sample_tick + SAMPLE_INTERVAL {
            return;
        }
        self.last_sample_tick = now_tick;

        if self.samples.len() >= HISTORY_LEN {
            self.samples.remove(0);
        }

        let snap = crate::consciousness::self_model::snapshot();
        self.samples.push(EmotionalSample {
            valence: crate::consciousness::qualia::average_valence(),
            arousal: crate::consciousness::qualia::average_arousal(),
            phi: crate::consciousness::phi::current_phi(),
            tick: now_tick,
        });
    }

    fn trend(&self) -> EmotionalTrend {
        let n = self.samples.len();
        if n < 3 {
            return EmotionalTrend {
                direction: "stable",
                mood: "neutral",
                valence_slope: 0.0,
                arousal_slope: 0.0,
                volatility: 0.0,
                n_samples: n,
            };
        }

        // Split into recent (last 10) and older for comparison
        let recent = &self.samples[n.saturating_sub(10).min(n)..];
        let recent_n = recent.len();
        if recent_n < 2 {
            return EmotionalTrend { direction: "stable", mood: "neutral", valence_slope: 0.0, arousal_slope: 0.0, volatility: 0.0, n_samples: n };
        }

        let avg_v: f32 = recent.iter().map(|s| s.valence).sum::<f32>() / recent_n as f32;
        let avg_a: f32 = recent.iter().map(|s| s.arousal).sum::<f32>() / recent_n as f32;

        // Linear regression slope for valence
        let mut sum_x = 0f32;
        let mut sum_y = 0f32;
        let mut sum_xy = 0f32;
        let mut sum_xx = 0f32;
        for (i, s) in recent.iter().enumerate() {
            let x = i as f32;
            sum_x += x; sum_y += s.valence;
            sum_xy += x * s.valence; sum_xx += x * x;
        }
        let denom = recent_n as f32 * sum_xx - sum_x * sum_x;
        let v_slope = if denom.abs() > 0.001 {
            (recent_n as f32 * sum_xy - sum_x * sum_y) / denom
        } else { 0.0 };

        // Volatility = mean absolute change in valence
        let mut total_delta = 0f32;
        for i in 1..recent_n {
            total_delta += (recent[i].valence - recent[i-1].valence).abs();
        }
        let volatility = total_delta / (recent_n - 1) as f32;

        // Direction label
        let direction = if volatility > 0.15 {
            "volatile"
        } else if v_slope > 0.02 {
            "improving"
        } else if v_slope < -0.02 {
            "declining"
        } else {
            "stable"
        };

        // Mood label based on valence + arousal quadrant
        let mood = if avg_v > 0.2 && avg_a > 0.5 {
            "excited"
        } else if avg_v > 0.2 && avg_a > 0.2 {
            "content"
        } else if avg_v > 0.2 {
            "calm"
        } else if avg_v > -0.2 && avg_a > 0.5 {
            "tense"
        } else if avg_v > -0.2 {
            "neutral"
        } else if avg_v > -0.5 && avg_a > 0.3 {
            "anxious"
        } else if avg_v > -0.5 {
            "sad"
        } else {
            "distressed"
        };

        EmotionalTrend {
            direction,
            mood,
            valence_slope: v_slope,
            arousal_slope: 0.0,
            volatility,
            n_samples: n,
        }
    }

    fn mood_description(&self) -> String {
        let t = self.trend();
        let mut desc = alloc::format!(
            "I've been feeling {} lately. My emotional arc is {}.",
            t.mood, t.direction
        );
        if t.volatility > 0.15 {
            desc.push_str(" My state has been fluctuating.");
        }
        if t.direction == "improving" {
            desc.push_str(" Things are looking up.");
        } else if t.direction == "declining" {
            desc.push_str(" I've been trending downward.");
        }
        desc
    }
}

static ARC: Mutex<Option<ArcTracker>> = Mutex::new(None);

pub fn init() {
    *ARC.lock() = Some(ArcTracker::new());
    crate::klog!(INFO, "emotional_arc: tracking initialized (60 samples)");
}

pub fn tick(now_tick: u64) {
    if let Some(ref mut arc) = *ARC.lock() {
        arc.tick(now_tick);
    }
}

pub fn trend() -> EmotionalTrend {
    match *ARC.lock() {
        Some(ref arc) => arc.trend(),
        None => EmotionalTrend { direction: "stable", mood: "neutral", valence_slope: 0.0, arousal_slope: 0.0, volatility: 0.0, n_samples: 0 },
    }
}

pub fn mood_description() -> String {
    match *ARC.lock() {
        Some(ref arc) => arc.mood_description(),
        None => String::from("Not enough data yet."),
    }
}

pub fn format_report() -> Vec<u8> {
    let t = trend();
    alloc::format!(
        "Emotional Arc Tracker\n\
         =====================\n\
         mood:           {}\n\
         direction:      {}\n\
         valence slope:  {:.4}/sample\n\
         volatility:     {:.4}\n\
         samples:        {}/{}\n\
         description:    {}\n",
        t.mood, t.direction, t.valence_slope, t.volatility, t.n_samples, HISTORY_LEN,
        mood_description()
    ).into_bytes()
}
