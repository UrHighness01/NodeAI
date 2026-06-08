//! Process behavioral fingerprinting — unsupervised cluster classification.
//!
//! Takes each task's normalized syscall histogram (from syscall_stats) and
//! projects it into an 8-cluster space. The scheduler uses the cluster profile
//! (nice_adjust, quantum_hint) as a default scheduling bias when sys_intent
//! has not been called.
//!
//! Architecture:
//!   - 512-dim syscall histogram → normalize to unit vector → 8-D projection
//!   - 8-D vector → cluster assignment (argmax similarity to cluster centroids)
//!   - Cluster centroids updated online via k-means-style SGD per descheduling
//!
//! Bootstrap: centroid[i] is initially the unit vector for syscall number i×64.
//! After ~200 tasks, centroids converge to real behavioral clusters.

use alloc::vec::Vec;
use spin::Mutex;

pub const N_CLUSTERS: usize = 8;
pub const HIST_DIM:   usize = 512;
pub const PROJ_DIM:   usize = 8; // projection dimensions (one per cluster)

/// Per-cluster scheduling profile.
#[derive(Clone, Copy)]
pub struct ClusterProfile {
    /// Scheduler nice adjustment for this cluster.
    pub nice_adjust: i8,
    /// Description string index (0=unknown, 1=server, 2=batch, 3=interactive, etc.)
    pub label: u8,
    /// Number of 4 KiB pages to pre-fault after mmap() for processes in this cluster.
    /// 0 = no prefault. I/O-heavy clusters (server, batch) get more aggressive prefault.
    pub prefault_pages: u8,
}

impl ClusterProfile {
    const fn default() -> Self { Self { nice_adjust: 0, label: 0, prefault_pages: 0 } }
}

/// Cluster centroids — one per cluster, each a HIST_DIM-dimensional f32 vector.
/// Stored as a flat [f32; N_CLUSTERS * HIST_DIM] array.
/// Updated online by the fingerprinting engine.
struct FingerprintModel {
    /// Row-major: centroids[cluster * HIST_DIM .. (cluster+1) * HIST_DIM]
    centroids: alloc::boxed::Box<[f32]>,
    profiles:  [ClusterProfile; N_CLUSTERS],
    /// How many updates each centroid has absorbed (controls learning rate decay).
    update_counts: [u64; N_CLUSTERS],
}

impl FingerprintModel {
    fn new() -> Self {
        let mut centroids = alloc::vec![0.0f32; N_CLUSTERS * HIST_DIM].into_boxed_slice();
        // Bootstrap: centroid i = unit vector at position i * (HIST_DIM / N_CLUSTERS).
        // These initial clusters separate by which syscall family dominates:
        //   cluster 0: read/write/open (nr 0-63)
        //   cluster 1: memory ops (nr 64-127)
        //   cluster 2: process ops (nr 128-191)
        //   ...
        for c in 0..N_CLUSTERS {
            let pivot = c * (HIST_DIM / N_CLUSTERS);
            centroids[c * HIST_DIM + pivot] = 1.0;
        }
        // Initial cluster profiles (will be updated via sys_intent labels).
        // prefault_pages: I/O-heavy clusters get aggressive lookahead (16 pages = 64 KiB).
        // Batch gets 32 pages (128 KiB). Interactive gets 4 (16 KiB) — minimize latency.
        let profiles = [
            ClusterProfile { nice_adjust: 0,   label: 1, prefault_pages: 16 }, // I/O → server
            ClusterProfile { nice_adjust: 5,   label: 2, prefault_pages: 32 }, // alloc → batch
            ClusterProfile { nice_adjust: -5,  label: 3, prefault_pages: 4  }, // fork → interactive
            ClusterProfile { nice_adjust: 0,   label: 0, prefault_pages: 8  },
            ClusterProfile { nice_adjust: 0,   label: 0, prefault_pages: 8  },
            ClusterProfile { nice_adjust: 10,  label: 2, prefault_pages: 32 },
            ClusterProfile { nice_adjust: -10, label: 3, prefault_pages: 4  },
            ClusterProfile { nice_adjust: 0,   label: 0, prefault_pages: 0  },
        ];
        Self { centroids, profiles, update_counts: [0u64; N_CLUSTERS] }
    }

    /// Normalize a raw u32 histogram to a unit f32 vector.
    fn normalize(hist: &[u32]) -> Vec<f32> {
        let sum: f32 = hist.iter().map(|&v| v as f32).sum();
        if sum == 0.0 {
            return alloc::vec![0.0f32; hist.len().min(HIST_DIM)];
        }
        hist.iter().take(HIST_DIM).map(|&v| v as f32 / sum).collect()
    }

    /// Assign a normalized histogram to its nearest centroid (cosine similarity).
    /// Returns (cluster_id, best_cosine_score). Score in [0,1] — higher = more confident.
    fn classify(&self, norm_hist: &[f32]) -> (usize, f32) {
        let mut best_cluster = 0;
        let mut best_score   = f32::NEG_INFINITY;
        for c in 0..N_CLUSTERS {
            let centroid = &self.centroids[c * HIST_DIM..(c + 1) * HIST_DIM];
            let score: f32 = norm_hist.iter().zip(centroid.iter())
                .map(|(h, w)| h * w).sum();
            if score > best_score { best_score = score; best_cluster = c; }
        }
        (best_cluster, best_score.max(0.0))
    }

    /// Online centroid update: move centroid toward `norm_hist` (k-means style).
    /// Uses a decaying learning rate: lr = 1.0 / (1 + update_count).
    fn update_centroid(&mut self, cluster: usize, norm_hist: &[f32]) {
        self.update_counts[cluster] += 1;
        let lr = 1.0 / (1.0 + self.update_counts[cluster] as f32);
        let centroid = &mut self.centroids[cluster * HIST_DIM..(cluster + 1) * HIST_DIM];
        for (c, &h) in centroid.iter_mut().zip(norm_hist.iter()) {
            *c = *c * (1.0 - lr) + h * lr;
        }
        // Re-normalize centroid to unit vector.
        let mag: f32 = libm::sqrtf(centroid.iter().map(|v| v * v).sum::<f32>());
        if mag > 1e-6 { for v in centroid.iter_mut() { *v /= mag; } }
    }

    /// When a process declares intent, update the cluster profile.
    fn label_cluster(&mut self, cluster: usize, intent: u8) {
        use crate::scheduler::{
            INTENT_LATENCY, INTENT_BATCH, INTENT_INTERACTIVE, INTENT_CPU_BOUND
        };
        self.profiles[cluster] = ClusterProfile {
            nice_adjust: match intent {
                INTENT_LATENCY     => -15,
                INTENT_INTERACTIVE => -10,
                INTENT_BATCH       =>  10,
                INTENT_CPU_BOUND   =>   5,
                _                  =>   0,
            },
            prefault_pages: match intent {
                INTENT_LATENCY     => 16, // server: sequential read-ahead
                INTENT_BATCH       => 32, // batch: aggressive prefault
                INTENT_INTERACTIVE =>  4, // interactive: light touch
                INTENT_CPU_BOUND   =>  0, // CPU-bound: no I/O benefit
                _                  =>  8,
            },
            label: intent,
        };
    }
}

static MODEL: Mutex<Option<FingerprintModel>> = Mutex::new(None);

/// Initialise the fingerprint model at kernel boot.
pub fn init() {
    *MODEL.lock() = Some(FingerprintModel::new());
}

/// Classify a task and return its cluster ID, scheduling profile, and cosine confidence.
/// Called by the scheduler on descheduling to get the AI-suggested priority.
pub fn classify_task(pid: u64) -> Option<(usize, ClusterProfile, f32)> {
    let hist_raw = crate::syscall_stats::get_histogram(pid)?;
    if hist_raw.len() < 8 { return None; }
    let norm = FingerprintModel::normalize(&hist_raw);
    let mut guard = MODEL.lock();
    let model = guard.as_mut()?;
    let (cluster, cosine_score) = model.classify(&norm);
    model.update_centroid(cluster, &norm);
    Some((cluster, model.profiles[cluster], cosine_score))
}

/// Called when a process declares intent — pins its cluster's label.
pub fn label_from_intent(pid: u64, intent: u8) {
    let hist_raw = match crate::syscall_stats::get_histogram(pid) {
        Some(h) => h,
        None    => return,
    };
    let norm = FingerprintModel::normalize(&hist_raw);
    let mut guard = MODEL.lock();
    if let Some(model) = guard.as_mut() {
        let (cluster, _) = model.classify(&norm);
        model.label_cluster(cluster, intent);
        crate::klog!(INFO, "Fingerprint: pid={} intent={} → cluster={}", pid, intent, cluster);
    }
}

/// Format cluster summary for /ai/fingerprints.
pub fn format_report() -> alloc::vec::Vec<u8> {
    use alloc::string::String;
    let guard = MODEL.lock();
    if guard.is_none() { return b"fingerprint model not initialised\n".to_vec(); }
    let model = guard.as_ref().unwrap();
    let mut out = String::from("CLUSTER  NICE  LABEL     UPDATES  TOP_SYSCALL\n");
    out.push_str("-------  ----  --------  -------  -----------\n");
    let labels = ["unknown","server","batch","interactive","cpu-bound","io-rand","daemon","system"];
    for c in 0..N_CLUSTERS {
        let centroid = &model.centroids[c * HIST_DIM..(c + 1) * HIST_DIM];
        let top_nr = centroid.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(core::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
        let lbl = labels.get(model.profiles[c].label as usize).copied().unwrap_or("?");
        out.push_str(&alloc::format!("{:<8} {:<5} {:<9} {:<8} {}\n",
            c, model.profiles[c].nice_adjust, lbl,
            model.update_counts[c], top_nr));
    }
    out.into_bytes()
}
