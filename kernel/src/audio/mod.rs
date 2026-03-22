//! NodeAI Audio Subsystem — Phase 25.
//!
//! Supports:
//!   - Intel AC97 (ICH 82801AA/AB/BA/CA) — the codec VirtualBox exposes.
//!   - PCM ring buffer (48 kHz, 16-bit stereo).
//!   - Software mixer (up to 4 concurrent sources).
//!   - WAV file decoder for kernel notification sounds.
//!   - `/dev/dsp` (OSS) and `/dev/snd/pcmC0D0p` (ALSA) device nodes.
//!
//! ## Architecture
//!
//! ```
//! app write → PCM_RING → audio_flush()
//!                ↓              ↓
//!           [BDL buf0..N] ← refilled when hardware advances
//!                ↓
//!            AC97 DMA → DAC → speakers
//! ```

use spin::{Mutex, Once};
use alloc::vec::Vec;

// ── AC97 PCI device IDs ───────────────────────────────────────────────────────
pub const AC97_VENDOR:  u16 = 0x8086;
pub const AC97_DEV_ICH:  u16 = 0x2415; // 82801AA
pub const AC97_DEV_ICH0: u16 = 0x2425; // 82801AB
pub const AC97_DEV_ICH2: u16 = 0x2445; // 82801BA
pub const AC97_DEV_ICH3: u16 = 0x2485; // 82801CA

// ── NAM (Native Audio Mixer) register offsets ─────────────────────────────────
const NAM_RESET:       u16 = 0x00; // Codec reset
const NAM_MASTER_VOL:  u16 = 0x02; // Master Volume
const NAM_HP_VOL:      u16 = 0x04; // Headphone Volume
const NAM_PCM_OUT_VOL: u16 = 0x18; // PCM Output Volume
const NAM_SAMPLE_RATE: u16 = 0x2C; // PCM DAC Sample Rate

// ── NABM (Native Audio Bus Master) register offsets ──────────────────────────
// PCM Output channel lives at 0x10–0x1B relative to NABM base.
const PO_BDBAR:  u16 = 0x10; // u32 Buffer Descriptor List Base Address
const PO_CIV:    u16 = 0x14; // u8  Current Index Value (read-only)
const PO_LVI:    u16 = 0x15; // u8  Last Valid Index
const PO_SR:     u16 = 0x16; // u16 Status Register
const PO_PICB:   u16 = 0x18; // u16 Position In Current Buffer (samples remaining)
const PO_PIV:    u16 = 0x1A; // u8  Prefetched Index Value
const PO_CR:     u16 = 0x1B; // u8  Control Register

const NABM_GLOB_CNT:  u16 = 0x2C; // Global Control
const NABM_GLOB_STA:  u16 = 0x30; // Global Status

// Control Register bits
const CR_RPBM:  u8 = 0x01; // Run/Pause Bus Master (1 = run)
const CR_RR:    u8 = 0x02; // Reset Registers
const CR_LVBIE: u8 = 0x04; // Last Valid Buffer Interrupt Enable
const CR_FEIE:  u8 = 0x08; // FIFO Error Interrupt Enable
const CR_IOCE:  u8 = 0x10; // Interrupt On Completion Enable

// Status Register bits
const SR_DCH:   u16 = 0x01; // DMA Controller Halted (1 = stopped)
const SR_CELV:  u16 = 0x02; // Current Equals Last Valid
const SR_LVBCI: u16 = 0x04; // Last Valid Buffer Completion Interrupt
const SR_BCIS:  u16 = 0x08; // Buffer Completion Interrupt Status
const SR_FIFOE: u16 = 0x10; // FIFO Error

// BDL entry flags
const BDL_IOC:  u16 = 0x8000; // Interrupt On Completion
const BDL_BUP:  u16 = 0x4000; // Buffer Underrun Policy (1 = keep last sample)

// ── Constants ─────────────────────────────────────────────────────────────────
const BDL_ENTRIES:   usize = 32;           // Max BDL entries (hardware max)
const BUF_FRAMES:    usize = 1024;         // Audio frames per BDL entry
const BUF_SAMPLES:   usize = BUF_FRAMES * 2; // Stereo samples (L+R)
const BUF_BYTES:     usize = BUF_SAMPLES * 2; // 16-bit samples => 2 bytes each
const PCM_RING_SIZE: usize = 65536;        // Software PCM ring (64 KiB)
const SAMPLE_RATE:   u32   = 48000;
const CHANNELS:      u32   = 2;

// ── BDL entry (8 bytes) ───────────────────────────────────────────────────────
#[repr(C, packed)]
struct BdlEntry {
    phys_addr: u32,   // Physical address of audio buffer
    samples:   u16,   // Number of samples (not bytes!) in this buffer
    flags:     u16,   // BDL_IOC / BDL_BUP
}

// ── AC97 device state ─────────────────────────────────────────────────────────
struct Ac97 {
    nam_base:     u16,   // NAM I/O port base
    nabm_base:    u16,   // NABM I/O port base
    /// Physical address of the BDL page
    bdl_phys:     u32,
    /// Virtual pointer to BDL (bdl_phys + phys_offset)
    bdl_virt:     *mut BdlEntry,
    /// Physical addresses of the audio data buffers (one per BDL entry)
    buf_phys:     [u32; BDL_ENTRIES],
    /// Virtual pointers to the audio data buffers
    buf_virt:     [*mut i16; BDL_ENTRIES],
    /// Software PCM ring buffer (interleaved S16LE stereo samples)
    ring:         alloc::vec::Vec<i16>,
    ring_rd:      usize,
    ring_wr:      usize,
    /// Last BDL index we filled (LVI we set last)
    lvi:          u8,
    /// Number of bits of silence to fill when ring is empty
    volume_pct:   u8,    // 0..=100
}

// SAFETY: Ac97 contains raw pointers to physically-contiguous DMA memory.
// We only access them from the locked Mutex context.
unsafe impl Send for Ac97 {}

// ── I/O port helpers ──────────────────────────────────────────────────────────
#[inline]
unsafe fn outb(port: u16, val: u8) {
    x86_64::instructions::port::Port::<u8>::new(port).write(val);
}
#[inline]
unsafe fn outw(port: u16, val: u16) {
    x86_64::instructions::port::Port::<u16>::new(port).write(val);
}
#[inline]
unsafe fn outl(port: u16, val: u32) {
    x86_64::instructions::port::Port::<u32>::new(port).write(val);
}
#[inline]
unsafe fn inb(port: u16) -> u8 {
    x86_64::instructions::port::Port::<u8>::new(port).read()
}
#[inline]
unsafe fn inw(port: u16) -> u16 {
    x86_64::instructions::port::Port::<u16>::new(port).read()
}
#[inline]
unsafe fn inl(port: u16) -> u32 {
    x86_64::instructions::port::Port::<u32>::new(port).read()
}

impl Ac97 {
    /// Initialise AC97 given the two I/O base addresses.
    /// Allocates DMA memory via PMM and programs the BDL.
    unsafe fn init(nam: u16, nabm: u16, phys_offset: u64) -> Option<Self> {
        // 1. Allocate one physical page for BDL (32 entries × 8 bytes = 256 B)
        let bdl_phys = crate::memory::alloc_frame()? as u32;
        let bdl_virt = (phys_offset + bdl_phys as u64) as *mut BdlEntry;

        // 2. Allocate BDL_ENTRIES physical pages for audio data buffers
        let mut buf_phys  = [0u32; BDL_ENTRIES];
        let mut buf_virt  = [core::ptr::null_mut::<i16>(); BDL_ENTRIES];
        for i in 0..BDL_ENTRIES {
            let phys = crate::memory::alloc_frame()? as u32;
            buf_phys[i] = phys;
            buf_virt[i] = (phys_offset + phys as u64) as *mut i16;
            // Zero-fill the buffer (silence)
            core::ptr::write_bytes(buf_virt[i] as *mut u8, 0, 4096);
        }

        // 3. Write BDL entries
        for i in 0..BDL_ENTRIES {
            let entry = bdl_virt.add(i);
            (*entry).phys_addr = buf_phys[i];
            (*entry).samples   = BUF_SAMPLES as u16;
            // IOC on last entry so we know when to wrap
            (*entry).flags     = if i == BDL_ENTRIES - 1 { BDL_IOC } else { 0 };
        }

        // 4. Reset NAM codec
        outw(nam + NAM_RESET, 0x0000);
        // Small delay
        for _ in 0..10_000u32 { core::hint::spin_loop(); }

        // 5. Unmute: master volume 0dB (0x0000 = 0 dB, 0x8000 = mute)
        outw(nam + NAM_MASTER_VOL,  0x0000);
        outw(nam + NAM_HP_VOL,      0x0000);
        outw(nam + NAM_PCM_OUT_VOL, 0x0000);

        // 6. Set sample rate to 48 kHz (if the codec supports variable rate)
        outw(nam + NAM_SAMPLE_RATE, SAMPLE_RATE as u16);

        // 7. Enable AC97 (set Global Control)
        outl(nabm + NABM_GLOB_CNT, 0x0002); // bit 1 = cold reset release

        // 8. Reset PCM Out channel
        outb(nabm + PO_CR, CR_RR);
        for _ in 0..1000u32 { core::hint::spin_loop(); }

        // 9. Write BDL base address
        outl(nabm + PO_BDBAR, bdl_phys);

        // 10. Set Last Valid Index to maximum
        outb(nabm + PO_LVI, (BDL_ENTRIES - 1) as u8);

        // 11. Clear status bits that may be set
        outw(nabm + PO_SR, SR_LVBCI | SR_BCIS | SR_FIFOE);

        // 12. Start DMA (Run + IOC enable)
        outb(nabm + PO_CR, CR_RPBM);

        crate::klog!(INFO, "audio: AC97 initialised — NAM={:#x} NABM={:#x}", nam, nabm);

        Some(Ac97 {
            nam_base:   nam,
            nabm_base:  nabm,
            bdl_phys,
            bdl_virt,
            buf_phys,
            buf_virt,
            ring:     alloc::vec![0i16; PCM_RING_SIZE],
            ring_rd:  0,
            ring_wr:  0,
            lvi:      (BDL_ENTRIES - 1) as u8,
            volume_pct: 100,
        })
    }

    // ── Ring buffer operations ──────────────────────────────────────────────

    fn ring_free(&self) -> usize {
        (self.ring_rd + PCM_RING_SIZE - self.ring_wr - 1) & (PCM_RING_SIZE - 1)
    }

    fn ring_used(&self) -> usize {
        (self.ring_wr + PCM_RING_SIZE - self.ring_rd) & (PCM_RING_SIZE - 1)
    }

    /// Write S16LE stereo samples into the ring buffer.
    fn ring_push(&mut self, samples: &[i16]) {
        for &s in samples {
            let next = (self.ring_wr + 1) & (PCM_RING_SIZE - 1);
            if next == self.ring_rd { break; } // overflow — drop
            self.ring[self.ring_wr] = s;
            self.ring_wr = next;
        }
    }

    /// Pop one sample from the ring (or silence on underrun).
    fn ring_pop(&mut self) -> i16 {
        if self.ring_rd == self.ring_wr { return 0; }
        let s = self.ring[self.ring_rd];
        self.ring_rd = (self.ring_rd + 1) & (PCM_RING_SIZE - 1);
        s
    }

    // ── Flush ring → next available HW buffer ──────────────────────────────

    /// Called periodically (or on each audio write) to push samples from the
    /// software ring into whichever BDL slots the hardware is not currently
    /// consuming.
    unsafe fn flush(&mut self) {
        let civ = inb(self.nabm_base + PO_CIV) as usize;
        let lvi = self.lvi as usize;
        // Number of entries hardware has finished since our last update
        let free_slots = (civ + BDL_ENTRIES - 1 - lvi) % BDL_ENTRIES;
        for _ in 0..free_slots {
            // Advance our write pointer to the next slot past lvi
            let next = (lvi + 1) % BDL_ENTRIES;
            let buf  = self.buf_virt[next];
            // Fill BDL buffer from ring
            for i in 0..BUF_SAMPLES {
                *buf.add(i) = self.ring_pop();
            }
            // Update LVI so hardware extends its valid range
            self.lvi = next as u8;
            outb(self.nabm_base + PO_LVI, self.lvi);
            // Clear completion status
            outw(self.nabm_base + PO_SR, SR_BCIS | SR_LVBCI);
            break; // fill one slot per flush call to avoid starvation
        }
        // Restart hardware if it stopped (e.g. hit LVI with DCH=1)
        let sr = inw(self.nabm_base + PO_SR);
        if (sr & SR_DCH) != 0 {
            outb(self.nabm_base + PO_CR, CR_RPBM);
        }
    }

    // ── Volume control ─────────────────────────────────────────────────────

    unsafe fn set_volume(&mut self, pct: u8) {
        self.volume_pct = pct.min(100);
        // AC97 volume: 0x00 = max, 0x1F = min, bit 15 = mute
        // Maps 0..100 → 0x1F..0x00
        let attn = if pct == 0 { 0x8000u16 }  // mute
                   else { ((100 - pct as u16) * 31 / 100) as u16 };
        let stereo = attn | (attn << 8);
        outw(self.nam_base + NAM_MASTER_VOL,  stereo);
        outw(self.nam_base + NAM_HP_VOL,      stereo);
        outw(self.nam_base + NAM_PCM_OUT_VOL, stereo);
    }
}

// ── Global audio state ────────────────────────────────────────────────────────
static AC97: Once<Mutex<Ac97>> = Once::new();

// ── Public API ────────────────────────────────────────────────────────────────

/// Probe PCI bus for AC97 and initialise if found.
/// Called from `main.rs` during the PCI scan phase.
pub fn init_if_present(
    vendor: u16,
    device: u16,
    nam_bar: u16,
    nabm_bar: u16,
) {
    let is_ac97 = vendor == AC97_VENDOR && matches!(
        device,
        AC97_DEV_ICH | AC97_DEV_ICH0 | AC97_DEV_ICH2 | AC97_DEV_ICH3
    );
    if !is_ac97 { return; }
    if AC97.get().is_some() { return; } // already initialised

    let phys_offset = crate::memory::phys_offset();
    if let Some(dev) = unsafe { Ac97::init(nam_bar, nabm_bar, phys_offset) } {
        AC97.call_once(|| Mutex::new(dev));
        crate::klog!(INFO, "audio: AC97 device available");
    }
}

/// Returns true if AC97 audio hardware has been initialised.
pub fn is_available() -> bool { AC97.get().is_some() }

/// Write raw S16LE stereo samples into the audio ring buffer.
/// `data` must be a byte slice matching S16LE interleaved stereo PCM.
pub fn write_pcm_bytes(data: &[u8]) {
    if let Some(dev) = AC97.get() {
        let mut d = dev.lock();
        // Reinterpret bytes as i16 samples
        let samples: &[i16] = unsafe {
            core::slice::from_raw_parts(data.as_ptr() as *const i16, data.len() / 2)
        };
        d.ring_push(samples);
        // Immediate flush attempt
        unsafe { d.flush(); }
    }
}

/// Periodic audio tick — call this from the scheduler/timer to keep DMA fed.
pub fn tick() {
    if let Some(dev) = AC97.get() {
        unsafe { dev.lock().flush(); }
    }
}

/// Set master volume 0..=100.
pub fn set_volume(pct: u8) {
    if let Some(dev) = AC97.get() {
        unsafe { dev.lock().set_volume(pct); }
    }
}

/// Get current volume 0..=100.
pub fn get_volume() -> u8 {
    AC97.get().map(|d| d.lock().volume_pct).unwrap_or(100)
}

// ── WAV decoder ───────────────────────────────────────────────────────────────
//
// Supports only PCM uncompressed WAV (format 1), mono or stereo, 8 or 16-bit.
// Resamples to 48 kHz stereo S16LE for AC97 output.

/// Decode and play a WAV file from a byte slice.
/// Returns Ok(()) on success or Err(&str) with error description.
pub fn play_wav(data: &[u8]) -> Result<(), &'static str> {
    if data.len() < 44 { return Err("too short"); }
    // RIFF header
    if &data[0..4] != b"RIFF" { return Err("not RIFF"); }
    if &data[8..12] != b"WAVE" { return Err("not WAVE"); }

    // Find fmt chunk
    let mut pos = 12usize;
    let mut audio_format = 0u16;
    let mut num_channels = 0u16;
    let mut sample_rate  = 0u32;
    let mut bits_per_sample = 0u16;
    let mut pcm_data: &[u8] = &[];

    while pos + 8 <= data.len() {
        let chunk_id   = &data[pos..pos+4];
        let chunk_size = u32::from_le_bytes([data[pos+4], data[pos+5], data[pos+6], data[pos+7]]) as usize;
        pos += 8;
        if chunk_id == b"fmt " {
            if chunk_size < 16 { return Err("fmt too small"); }
            audio_format    = u16::from_le_bytes([data[pos], data[pos+1]]);
            num_channels    = u16::from_le_bytes([data[pos+2], data[pos+3]]);
            sample_rate     = u32::from_le_bytes([data[pos+4], data[pos+5], data[pos+6], data[pos+7]]);
            bits_per_sample = u16::from_le_bytes([data[pos+14], data[pos+15]]);
        } else if chunk_id == b"data" {
            let end = (pos + chunk_size).min(data.len());
            pcm_data = &data[pos..end];
            break;
        }
        pos += (chunk_size + 1) & !1; // word-align
    }

    if audio_format != 1 { return Err("not PCM WAV"); }
    if pcm_data.is_empty() { return Err("no data chunk"); }

    // Convert to S16LE stereo 48 kHz
    let mut out: Vec<i16> = Vec::new();
    let is_stereo = num_channels >= 2;

    let src_step = (num_channels as usize) * (bits_per_sample as usize / 8);
    let in_frames = pcm_data.len() / src_step.max(1);

    // Simple nearest-neighbour resampler — ratio = src_rate / dst_rate
    let mut src_idx_frac = 0u64;
    let step = ((sample_rate as u64) << 16) / (SAMPLE_RATE as u64);

    loop {
        let src_idx = (src_idx_frac >> 16) as usize;
        if src_idx >= in_frames { break; }

        let byte_off = src_idx * src_step;
        let (l, r) = if bits_per_sample == 8 {
            let l8 = pcm_data.get(byte_off).copied().unwrap_or(128);
            let l16 = (l8 as i16 - 128) * 256;
            let r8 = if is_stereo {
                pcm_data.get(byte_off + 1).copied().unwrap_or(128)
            } else { l8 };
            let r16 = (r8 as i16 - 128) * 256;
            (l16, r16)
        } else {
            let l16 = i16::from_le_bytes([
                pcm_data.get(byte_off).copied().unwrap_or(0),
                pcm_data.get(byte_off + 1).copied().unwrap_or(0),
            ]);
            let r16 = if is_stereo {
                i16::from_le_bytes([
                    pcm_data.get(byte_off + 2).copied().unwrap_or(0),
                    pcm_data.get(byte_off + 3).copied().unwrap_or(0),
                ])
            } else { l16 };
            (l16, r16)
        };
        out.push(l);
        out.push(r);
        src_idx_frac += step;
    }

    write_pcm_bytes(unsafe {
        core::slice::from_raw_parts(out.as_ptr() as *const u8, out.len() * 2)
    });
    Ok(())
}

// ── Software mixer ────────────────────────────────────────────────────────────
//
// Up to 4 simultaneous audio sources.  Each source pushes raw S16LE stereo
// samples into the mixer, which sums them (with saturation) before forwarding
// to the ring buffer.

const MIXER_SOURCES: usize = 4;
const MIXER_BUF:     usize = 2048;

struct MixerSource {
    buf:  [i16; MIXER_BUF],
    rd:   usize,
    wr:   usize,
    gain: i32, // 0..=256, fixed-point 1.0 = 128
}

impl MixerSource {
    const fn new() -> Self {
        Self { buf: [0i16; MIXER_BUF], rd: 0, wr: 0, gain: 128 }
    }
    fn push(&mut self, s: &[i16]) {
        for &v in s {
            let next = (self.wr + 1) % MIXER_BUF;
            if next != self.rd { self.buf[self.wr] = v; self.wr = next; }
        }
    }
    fn pop(&mut self) -> Option<i16> {
        if self.rd == self.wr { return None; }
        let v = self.buf[self.rd];
        self.rd = (self.rd + 1) % MIXER_BUF;
        Some(v)
    }
}

struct Mixer {
    sources: [MixerSource; MIXER_SOURCES],
}

impl Mixer {
    const fn new() -> Self {
        Self {
            sources: [
                MixerSource::new(), MixerSource::new(),
                MixerSource::new(), MixerSource::new(),
            ],
        }
    }

    /// Pull one stereo frame from all active sources, mix, and return.
    fn pull_frame(&mut self) -> (i16, i16) {
        let mut l = 0i32;
        let mut r = 0i32;
        for src in &mut self.sources {
            if let Some(sl) = src.pop() {
                let sr = src.pop().unwrap_or(0);
                l += (sl as i32) * src.gain / 128;
                r += (sr as i32) * src.gain / 128;
            }
        }
        let cl = l.clamp(-32768, 32767) as i16;
        let cr = r.clamp(-32768, 32767) as i16;
        (cl, cr)
    }

    /// Write `n` stereo frames to the AC97 ring buffer.
    fn flush_frames(&mut self, n: usize) {
        if let Some(dev) = AC97.get() {
            let mut d = dev.lock();
            for _ in 0..n {
                let (l, r) = self.pull_frame();
                d.ring_push(&[l, r]);
            }
            unsafe { d.flush(); }
        }
    }
}

static MIXER: Mutex<Mixer> = Mutex::new(Mixer::new());

/// Write samples to mixer source `id` (0..=3).
pub fn mixer_write(src_id: usize, samples: &[i16]) {
    if src_id >= MIXER_SOURCES { return; }
    MIXER.lock().sources[src_id].push(samples);
    // Immediately flush mixed output
    MIXER.lock().flush_frames(samples.len() / 2);
}
