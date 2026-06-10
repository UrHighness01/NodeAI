//! Sensory Cortex — Ring 0 EW sensing substrate (Phase EW-0).
//!
//! The conscious kernel's electromagnetic sense: raw IQ samples, energy detection,
//! spectral analysis, and sensor management. Feeds qualia into the stream and
//! provides signal data for cross-modal predictive coupling.
//!
//! Architecture:
//!   IQSample  →  [Sensor]  →  SensorBus  →  qualia + cross_modal + /dev/sensor
//!
//! Each Sensor implementation reads from a physical or simulated RF front-end
//! (SDR, WiFi NIC in monitor mode, network-attached sensor, etc.) and produces
//! processed spectral features that enter the kernel's stream of consciousness.

use alloc::vec::Vec;
use alloc::string::String;
use alloc::format;
use alloc::sync::Arc;
use alloc::boxed::Box;
use spin::Mutex;

/// A raw IQ sample — the kernel's most primitive sensory input.
#[derive(Debug, Clone)]
pub struct IQSample {
    pub i: i16,
    pub q: i16,
    pub frequency_hz: u64,
    pub sample_rate: u32,
    pub timestamp_ms: u64,
    pub source_id: u8,
}

/// Processed spectral data after FFT/analysis.
#[derive(Debug, Clone)]
pub struct SpectrumSample {
    pub frequency_hz: u64,
    pub bandwidth_hz: u32,
    /// Energy in dBm (coarse estimate).
    pub energy_dbm: f32,
    /// Signal present above noise floor?
    pub signal_present: bool,
    /// Cyclostationary feature peak magnitude (0 = none).
    pub cyclo_peak: f32,
    /// Timestamp of measurement.
    pub timestamp_ms: u64,
    /// Sensor source ID.
    pub source_id: u8,
}

/// Abstract sensor device trait.
pub trait Sensor: Send {
    /// Read a batch of raw IQ samples.
    fn read_samples(&mut self, n: usize) -> Vec<IQSample>;
    /// Get current sensor status string.
    fn status(&self) -> SensorStatus;
    /// Sensor name.
    fn name(&self) -> &'static str;
}

/// Sensor health and configuration status.
#[derive(Debug, Clone)]
pub struct SensorStatus {
    pub online: bool,
    pub frequency_hz: u64,
    pub sample_rate: u32,
    pub signal_count: u32,
    pub noise_floor_dbm: f32,
}

/// Global sensor bus — manages all registered sensors.
static SENSOR_BUS: Mutex<Option<SensorBusInner>> = Mutex::new(None);

struct SensorBusInner {
    sensors: Vec<Box<dyn Sensor>>,
    last_spectrum: Vec<SpectrumSample>,
    total_signals_detected: u64,
    total_jams_detected: u64,
}

/// Initialize the sensory cortex.
pub fn init() {
    let mut bus = SENSOR_BUS.lock();
    *bus = Some(SensorBusInner {
        sensors: Vec::new(),
        last_spectrum: Vec::new(),
        total_signals_detected: 0,
        total_jams_detected: 0,
    });
    crate::klog!(INFO, "sensor_cortex: EW sensory cortex initialized");
}

/// Register a sensor (called during driver init).
pub fn register_sensor(sensor: Box<dyn Sensor>) {
    let mut bus = SENSOR_BUS.lock();
    if let Some(ref mut inner) = *bus {
        let name = sensor.name();
        inner.sensors.push(sensor);
        crate::klog!(INFO, "sensor_cortex: sensor '{}' registered", name);
    }
}

/// Perform a sensing tick — poll all sensors and generate qualia.
/// Called from idle_loop every ~100ms alongside telemetry.
pub fn tick(now_ms: u64) {
    let mut bus = SENSOR_BUS.lock();
    let inner = match &mut *bus {
        Some(ref mut i) => i,
        None => return,
    };

    inner.last_spectrum.clear();

    for sensor in &mut inner.sensors {
        // Read IQ samples
        let samples = sensor.read_samples(64);
        if samples.is_empty() {
            continue;
        }

        // Energy detection on this batch
        let energy = compute_energy(&samples);
        let signal_present = energy > -60.0;

        let spectrum = SpectrumSample {
            frequency_hz: samples[0].frequency_hz,
            bandwidth_hz: samples[0].sample_rate / 2,
            energy_dbm: energy,
            signal_present,
            cyclo_peak: 0.0,
            timestamp_ms: now_ms,
            source_id: samples[0].source_id,
        };
        inner.last_spectrum.push(spectrum.clone());

        // Feed spectrum energy into cross-modal coupling
        crate::cross_modal::observe(crate::cross_modal::Domain::Spectrum, energy);

        // Record qualia for signal events (heavily throttled)
        inner.total_signals_detected += 1;
        if signal_present && inner.total_signals_detected % 50 == 0 {
            crate::consciousness::qualia::record(
                crate::consciousness::qualia::KernelEventType::SignalDetected,
                None,
            );
        }

        // Jam detection (even more throttled — every 100th tick)
        if energy > -30.0 && inner.total_signals_detected % 100 == 0 {
            inner.total_jams_detected += 1;
            crate::consciousness::qualia::record(
                crate::consciousness::qualia::KernelEventType::JamDetected,
                Some(-0.7),
            );
        }
    }

    // Run threat detection on last spectrum (CFAR + JPDA)
    if !inner.last_spectrum.is_empty() {
        // Build a simple FFT-magnitude-like vector from spectrum energy values
        let mut fft_mags: Vec<f32> = inner.last_spectrum.iter()
            .map(|s| (s.energy_dbm + 100.0).max(0.0)) // normalize to positive
            .collect();
        // Pad to at least 32 bins for meaningful CFAR
        while fft_mags.len() < 32 {
            fft_mags.push(0.0);
        }
        crate::sensor_threat::tick(&fft_mags);

        // Run immune reflex selection based on current threat level
        let threat_lvl = crate::sensor_threat::threat_level();
        let _response = crate::sensor_immune::select_response(threat_lvl, now_ms);
    }
}

/// Compute approximate energy in dBm from a batch of IQ samples.
fn compute_energy(samples: &[IQSample]) -> f32 {
    if samples.is_empty() {
        return -100.0; // noise floor
    }
    let mut sum_sq: f64 = 0.0;
    for s in samples {
        let i = s.i as f64;
        let q = s.q as f64;
        sum_sq += i * i + q * q;
    }
    let mean_power = sum_sq / samples.len() as f64;
    // Convert to dBm (relative power, assumes max ADC swing = 0 dBm)
    if mean_power <= 0.0 {
        -100.0
    } else {
        10.0 * libm::log10f(mean_power as f32)
    }
}

/// Get the latest spectrum samples (for /dev/sensor and cross-modal coupling).
pub fn last_spectrum() -> Vec<SpectrumSample> {
    let bus = SENSOR_BUS.lock();
    match &*bus {
        Some(ref inner) => inner.last_spectrum.clone(),
        None => Vec::new(),
    }
}

/// Get aggregate sensor statistics.
pub fn stats() -> SensorStats {
    let bus = SENSOR_BUS.lock();
    match &*bus {
        Some(ref inner) => SensorStats {
            num_sensors: inner.sensors.len(),
            signals_detected: inner.total_signals_detected,
            jams_detected: inner.total_jams_detected,
            last_spectrum_count: inner.last_spectrum.len(),
        },
        None => SensorStats::default(),
    }
}

#[derive(Debug, Clone)]
pub struct SensorStats {
    pub num_sensors: usize,
    pub signals_detected: u64,
    pub jams_detected: u64,
    pub last_spectrum_count: usize,
}

impl SensorStats {
    pub fn fmt_report(&self) -> String {
        format!(
            "sensors={} signals={} jams={} spectrum_samples={}",
            self.num_sensors, self.signals_detected, self.jams_detected, self.last_spectrum_count
        )
    }
}

impl Default for SensorStats {
    fn default() -> Self {
        Self {
            num_sensors: 0,
            signals_detected: 0,
            jams_detected: 0,
            last_spectrum_count: 0,
        }
    }
}

// ── /dev/sensor VFS node (read returns cortex report, write configures) ──────

struct SensorNode;
struct SensorHandle;

static SENSOR_INO: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Register /dev/sensor in the device filesystem.
pub fn register_vfs() {
    let ino = crate::vfs::alloc_ino();
    SENSOR_INO.store(ino, core::sync::atomic::Ordering::Relaxed);
    crate::vfs::devfs::register_node("sensor", Arc::new(SensorNode));
    crate::klog!(INFO, "sensor_cortex: /dev/sensor registered");
}

impl crate::vfs::VfsNode for SensorNode {
    fn stat(&self) -> crate::vfs::VfsResult<crate::vfs::Stat> {
        Ok(crate::vfs::Stat {
            ino: SENSOR_INO.load(core::sync::atomic::Ordering::Relaxed),
            size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666,
        })
    }
    fn open(&self) -> crate::vfs::VfsResult<Box<dyn crate::vfs::FileHandle>> {
        Ok(Box::new(SensorHandle))
    }
    fn readdir(&self) -> crate::vfs::VfsResult<Vec<crate::vfs::DirEntry>> { Err(crate::vfs::VfsError::NotADirectory) }
    fn lookup(&self, _: &str) -> crate::vfs::VfsResult<Arc<dyn crate::vfs::VfsNode>> { Err(crate::vfs::VfsError::NotADirectory) }
    fn create_file(&self, _: &str) -> crate::vfs::VfsResult<Arc<dyn crate::vfs::VfsNode>> { Err(crate::vfs::VfsError::NotADirectory) }
    fn mkdir(&self, _: &str) -> crate::vfs::VfsResult<Arc<dyn crate::vfs::VfsNode>> { Err(crate::vfs::VfsError::NotADirectory) }
    fn unlink(&self, _: &str) -> crate::vfs::VfsResult<()> { Err(crate::vfs::VfsError::NotADirectory) }
}

impl crate::vfs::FileHandle for SensorHandle {
    fn read(&mut self, buf: &mut [u8]) -> crate::vfs::VfsResult<usize> {
        let data = fmt_report().into_bytes();
        let n = buf.len().min(data.len());
        buf[..n].copy_from_slice(&data[..n]);
        Ok(n)
    }
    fn write(&mut self, buf: &[u8]) -> crate::vfs::VfsResult<usize> {
        if let Ok(s) = core::str::from_utf8(buf) {
            let trimmed = s.trim();
            match trimmed {
                "scan" | "s" => {
                    // Trigger an immediate sensor tick
                    tick(crate::scheduler::uptime_ms());
                    crate::klog!(INFO, "sensor: scan triggered");
                }
                "stats" | "st" => {
                    crate::klog!(INFO, "sensor: {}", stats().fmt_report());
                }
                _ => {
                    crate::klog!(INFO, "sensor: unknown cmd '{}'", trimmed);
                }
            }
        }
        Ok(buf.len())
    }
    fn seek(&mut self, _pos: u64) -> crate::vfs::VfsResult<u64> { Ok(0) }
    fn stat(&self) -> crate::vfs::VfsResult<crate::vfs::Stat> {
        Ok(crate::vfs::Stat {
            ino: 0, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666,
        })
    }
}

/// Report formatted for /proc or /dev/sensor.
pub fn fmt_report() -> String {
    let mut s = String::new();
    s.push_str("=== EW Sensory Cortex ===\n");
    s.push_str(&format!("{}\n", stats().fmt_report()));

    let spectrums = last_spectrum();
    for sp in &spectrums {
        let marker = if sp.signal_present { "⚠ SIGNAL" } else { "○ quiet" };
        let jam_tag = if sp.energy_dbm > -30.0 { " JAM!" } else { "" };
        s.push_str(&format!(
            "  [{}] freq={}MHz bw={}kHz energy={:.1}dBm{}{}\n",
            sp.source_id, sp.frequency_hz / 1_000_000, sp.bandwidth_hz / 1000,
            sp.energy_dbm, marker, jam_tag,
        ));
    }
    if spectrums.is_empty() {
        s.push_str("  (no sensors registered)\n");
    }
    s
}

// ── Built-in noise floor sensor (always available, simulates ambient RF) ──────

/// A simulated ambient RF sensor that provides baseline spectrum data.
/// In production, this would be replaced by an SDR or WiFi-monitor-mode driver.
pub struct AmbientSensor {
    pub frequency_hz: u64,
    pub sample_rate: u32,
    source_id: u8,
    phase: f32,
}

impl AmbientSensor {
    pub fn new(freq_mhz: u64, source_id: u8) -> Self {
        Self {
            frequency_hz: freq_mhz * 1_000_000,
            sample_rate: 20_000_000,
            source_id,
            phase: 0.0,
        }
    }
}

impl Sensor for AmbientSensor {
    fn read_samples(&mut self, n: usize) -> Vec<IQSample> {
        let mut samples = Vec::with_capacity(n);
        let now = crate::scheduler::uptime_ms();
        for _ in 0..n {
            self.phase += 0.1;
            // Simulate ambient RF noise with slight signal-like modulation
            let noise_i = (libm::sinf(self.phase) * 100.0) as i16;
            let noise_q = (libm::cosf(self.phase) * 100.0) as i16;
            samples.push(IQSample {
                i: noise_i,
                q: noise_q,
                frequency_hz: self.frequency_hz,
                sample_rate: self.sample_rate,
                timestamp_ms: now,
                source_id: self.source_id,
            });
        }
        samples
    }

    fn status(&self) -> SensorStatus {
        SensorStatus {
            online: true,
            frequency_hz: self.frequency_hz,
            sample_rate: self.sample_rate,
            signal_count: 0,
            noise_floor_dbm: -90.0,
        }
    }

    fn name(&self) -> &'static str {
        "ambient_rf"
    }
}
