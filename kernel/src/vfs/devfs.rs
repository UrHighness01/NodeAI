//! devfs — synthetic device filesystem at /dev — Phase 7.
//!
//! Provides:
//!   /dev/null  — discards all writes, returns EOF on read
//!   /dev/zero  — returns infinite zeros
//!   /dev/kmsg  — reads from the kernel ring buffer (KRING)

use alloc::{boxed::Box, collections::BTreeMap, string::String, sync::Arc, vec::Vec};
use spin::Mutex;
use super::{alloc_ino, DirEntry, FileHandle, Stat, VfsError, VfsNode, VfsResult};

// ── /dev/null ─────────────────────────────────────────────────────────────────

struct NullNode(u64);
struct NullHandle;

impl FileHandle for NullHandle {
    fn read(&mut self, _: &mut [u8]) -> VfsResult<usize> { Ok(0) } // EOF
    fn write(&mut self, buf: &[u8])  -> VfsResult<usize> { Ok(buf.len()) }
    fn seek(&mut self, pos: u64)     -> VfsResult<u64>   { Ok(pos) }
    fn stat(&self)                   -> VfsResult<Stat>   {
        Ok(Stat { ino: 1, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666 })
    }
}

impl VfsNode for NullNode {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.0, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666 })
    }
    fn open(&self) -> VfsResult<Box<dyn FileHandle>> { Ok(Box::new(NullHandle)) }
    fn readdir(&self) -> VfsResult<Vec<DirEntry>> { Err(VfsError::NotADirectory) }
    fn lookup(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::NotADirectory) }
}

// ── /dev/zero ─────────────────────────────────────────────────────────────────

struct ZeroNode(u64);
struct ZeroHandle;

impl FileHandle for ZeroHandle {
    fn read(&mut self, buf: &mut [u8]) -> VfsResult<usize> {
        buf.fill(0);
        Ok(buf.len())
    }
    fn write(&mut self, buf: &[u8]) -> VfsResult<usize>  { Ok(buf.len()) }
    fn seek(&mut self, pos: u64)    -> VfsResult<u64>    { Ok(pos) }
    fn stat(&self)                  -> VfsResult<Stat>    {
        Ok(Stat { ino: 2, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666 })
    }
}

impl VfsNode for ZeroNode {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.0, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666 })
    }
    fn open(&self) -> VfsResult<Box<dyn FileHandle>> { Ok(Box::new(ZeroHandle)) }
    fn readdir(&self) -> VfsResult<Vec<DirEntry>> { Err(VfsError::NotADirectory) }
    fn lookup(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::NotADirectory) }
}

// ── /dev/kmsg ─────────────────────────────────────────────────────────────────

struct KmsgNode(u64);

/// On open, snapshot the current ring buffer contents.
struct KmsgHandle {
    data: Vec<u8>,
    pos:  usize,
    ino:  u64,
}

impl FileHandle for KmsgHandle {
    fn read(&mut self, buf: &mut [u8]) -> VfsResult<usize> {
        let avail = self.data.len().saturating_sub(self.pos);
        let n = buf.len().min(avail);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
    fn write(&mut self, _: &[u8])   -> VfsResult<usize> { Err(VfsError::ReadOnly) }
    fn seek(&mut self, pos: u64)    -> VfsResult<u64>   { self.pos = pos as usize; Ok(pos) }
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.ino, size: self.data.len() as u64, is_dir: false, nlink: 1,
                  uid: 0, gid: 0, mode: 0o440 })
    }
}

impl VfsNode for KmsgNode {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.0, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o440 })
    }
    fn open(&self) -> VfsResult<Box<dyn FileHandle>> {
        // Snapshot KRING entries as newline-separated strings
        let mut data: Vec<u8> = Vec::new();
        {
            static LEVELS: [&[u8]; 5] = [b"TRACE", b"DEBUG", b"INFO ", b"WARN ", b"ERROR"];
            let ring = crate::kring::KRING.lock();
            for entry in ring.iter() {
                let lvl = LEVELS.get(entry.level as usize).copied().unwrap_or(b"?????");
                data.extend_from_slice(b"[");
                data.extend_from_slice(lvl);
                data.extend_from_slice(b"] ");
                data.extend_from_slice(&entry.data[..entry.len]);
                data.push(b'\n');
            }
        }
        Ok(Box::new(KmsgHandle { data, pos: 0, ino: self.0 }))
    }
    fn readdir(&self) -> VfsResult<Vec<DirEntry>> { Err(VfsError::NotADirectory) }
    fn lookup(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::NotADirectory) }
}

// ── /dev/random and /dev/urandom ─────────────────────────────────────────────
// Both use the behavioral entropy pool. /dev/random is identical to /dev/urandom
// on this kernel (no blocking — the pool is always stirred by idle_loop).

struct RandomNode(u64);

struct RandomHandle(u64);

impl FileHandle for RandomHandle {
    fn bytes_available(&self) -> usize { usize::MAX } // always ready
    fn read(&mut self, buf: &mut [u8]) -> VfsResult<usize> {
        crate::entropy::fill(buf);
        Ok(buf.len())
    }
    fn write(&mut self, data: &[u8]) -> VfsResult<usize> {
        // Writing to /dev/random adds entropy to the pool.
        let mut val: u64 = 0;
        for &b in data.iter().take(8) { val = (val << 8) | b as u64; }
        crate::entropy::stir(val);
        Ok(data.len())
    }
    fn seek(&mut self, _: u64) -> VfsResult<u64> { Ok(0) }
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.0, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666 })
    }
    fn clone_box(&self) -> Option<Box<dyn FileHandle>> {
        Some(Box::new(RandomHandle(self.0)))
    }
}

impl VfsNode for RandomNode {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.0, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666 })
    }
    fn open(&self) -> VfsResult<Box<dyn FileHandle>> { Ok(Box::new(RandomHandle(self.0))) }
    fn readdir(&self) -> VfsResult<Vec<DirEntry>> { Err(VfsError::NotADirectory) }
    fn lookup(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::NotADirectory) }
}

// ── /dev/composer ─────────────────────────────────────────────────────────────
//
// Window-compositor IPC device.  Userspace opens this file and issues
// write() calls containing binary command packets, or calls ioctl().
//
// Write packet format (all values little-endian):
//   [cmd: u32][window_id: u32][payload...]
//
// Supported commands:
//   1 (CREATE)  — payload: [x:i32][y:i32][w:u32][h:u32][title_len:u32][title_bytes…]
//   2 (DESTROY) — payload: (none; window_id identifies target)
//   3 (FLIP)    — payload: (none; blits window pixel buffer to screen)
//   4 (FILL)    — payload: [rx:u32][ry:u32][rw:u32][rh:u32][rgba:u32]

struct ComposerNode(u64);
struct ComposerHandle { ino: u64 }

/// Interpret a raw write buffer as a composer command.
fn composer_dispatch(buf: &[u8]) {
    if buf.len() < 8 { return; }
    let cmd = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let wid = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let payload = &buf[8..];
    match cmd {
        1 => {
            // CREATE_WINDOW: x i32, y i32, w u32, h u32, title_len u32, title bytes
            if payload.len() < 20 { return; }
            let x = i32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
            let y = i32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
            let w = u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]);
            let h = u32::from_le_bytes([payload[12], payload[13], payload[14], payload[15]]);
            let tl = u32::from_le_bytes([payload[16], payload[17], payload[18], payload[19]]) as usize;
            let title = if payload.len() >= 20 + tl {
                core::str::from_utf8(&payload[20..20 + tl]).unwrap_or("window")
            } else { "window" };
            crate::desktop::wm_create_window(x, y, w, h, title);
            crate::desktop::wm_composite();
        }
        2 => {
            // DESTROY_WINDOW
            crate::desktop::wm_destroy_window(wid);
        }
        3 => {
            // FLIP — blit window buffer to screen
            crate::desktop::wm_flip(wid);
        }
        4 => {
            // FILL_RECT
            if payload.len() < 20 { return; }
            let rx = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
            let ry = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
            let rw = u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]);
            let rh = u32::from_le_bytes([payload[12], payload[13], payload[14], payload[15]]);
            let rgba = u32::from_le_bytes([payload[16], payload[17], payload[18], payload[19]]);
            crate::desktop::wm_fill_window_rect(wid, rx, ry, rw, rh, rgba);
        }
        _ => {}
    }
}

impl FileHandle for ComposerHandle {
    fn read(&mut self, _buf: &mut [u8]) -> VfsResult<usize> { Ok(0) }
    fn write(&mut self, buf: &[u8]) -> VfsResult<usize> {
        composer_dispatch(buf);
        Ok(buf.len())
    }
    fn seek(&mut self, pos: u64) -> VfsResult<u64> { Ok(pos) }
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.ino, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666 })
    }
}

impl VfsNode for ComposerNode {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.0, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666 })
    }
    fn open(&self) -> VfsResult<Box<dyn FileHandle>> {
        Ok(Box::new(ComposerHandle { ino: self.0 }))
    }
    fn readdir(&self) -> VfsResult<Vec<DirEntry>> { Err(VfsError::NotADirectory) }
    fn lookup(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::NotADirectory) }
}

// ── /dev/tty ─────────────────────────────────────────────────────────────────
// Reads return pending keyboard input from the terminal window input ring.
// Writes send data to the terminal window (VT100 processed).

struct TtyNode(u64);
struct TtyHandle { ino: u64 }

impl FileHandle for TtyHandle {
    fn read(&mut self, buf: &mut [u8]) -> VfsResult<usize> {
        Ok(crate::desktop::tty_read(buf))
    }
    fn write(&mut self, buf: &[u8]) -> VfsResult<usize> {
        crate::desktop::term_window_write(buf);
        Ok(buf.len())
    }
    fn seek(&mut self, _pos: u64) -> VfsResult<u64> { Ok(0) }
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.ino, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666 })
    }
}

impl VfsNode for TtyNode {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.0, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666 })
    }
    fn open(&self) -> VfsResult<Box<dyn FileHandle>> { Ok(Box::new(TtyHandle { ino: self.0 })) }
    fn readdir(&self) -> VfsResult<Vec<DirEntry>> { Err(VfsError::NotADirectory) }
    fn lookup(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::NotADirectory) }
}

// ── /dev/ptmx ─────────────────────────────────────────────────────────────────
// Simplified PTY master — routes to the kernel terminal window.

struct PtmxNode(u64);
struct PtmxHandle { ino: u64 }

impl FileHandle for PtmxHandle {
    fn read(&mut self, buf: &mut [u8]) -> VfsResult<usize> {
        Ok(crate::desktop::tty_read(buf))
    }
    fn write(&mut self, buf: &[u8]) -> VfsResult<usize> {
        crate::desktop::term_window_write(buf);
        Ok(buf.len())
    }
    fn seek(&mut self, _pos: u64) -> VfsResult<u64> { Ok(0) }
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.ino, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666 })
    }
}

impl VfsNode for PtmxNode {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.0, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666 })
    }
    fn open(&self) -> VfsResult<Box<dyn FileHandle>> { Ok(Box::new(PtmxHandle { ino: self.0 })) }
    fn readdir(&self) -> VfsResult<Vec<DirEntry>> { Err(VfsError::NotADirectory) }
    fn lookup(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::NotADirectory) }
}

// ── /dev/dsp ─────────────────────────────────────────────────────────────────
// OSS-compatible PCM output device.  write() feeds raw 16-bit stereo PCM at
// whatever rate the caller uses; ioctl() stubs return success.

struct DspNode(u64);
struct DspHandle { ino: u64 }

impl FileHandle for DspHandle {
    fn read(&mut self, _buf: &mut [u8]) -> VfsResult<usize> { Ok(0) }
    fn write(&mut self, buf: &[u8]) -> VfsResult<usize> {
        crate::audio::write_pcm_bytes(buf);
        Ok(buf.len())
    }
    fn seek(&mut self, _pos: u64) -> VfsResult<u64> { Ok(0) }
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.ino, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666 })
    }
}

impl VfsNode for DspNode {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.0, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o666 })
    }
    fn open(&self) -> VfsResult<Box<dyn FileHandle>> { Ok(Box::new(DspHandle { ino: self.0 })) }
    fn readdir(&self) -> VfsResult<Vec<DirEntry>> { Err(VfsError::NotADirectory) }
    fn lookup(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::NotADirectory) }
}

// ── /dev/snd/pcmC0D0p ────────────────────────────────────────────────────────
// ALSA-compatible PCM playback node: card 0, device 0, playback.

struct PcmNode(u64);
struct PcmHandle { ino: u64 }

impl FileHandle for PcmHandle {
    fn read(&mut self, _buf: &mut [u8]) -> VfsResult<usize> { Ok(0) }
    fn write(&mut self, buf: &[u8]) -> VfsResult<usize> {
        crate::audio::write_pcm_bytes(buf);
        Ok(buf.len())
    }
    fn seek(&mut self, _pos: u64) -> VfsResult<u64> { Ok(0) }
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.ino, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o660 })
    }
}

impl VfsNode for PcmNode {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.0, size: 0, is_dir: false, nlink: 1, uid: 0, gid: 0, mode: 0o660 })
    }
    fn open(&self) -> VfsResult<Box<dyn FileHandle>> { Ok(Box::new(PcmHandle { ino: self.0 })) }
    fn readdir(&self) -> VfsResult<Vec<DirEntry>> { Err(VfsError::NotADirectory) }
    fn lookup(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::NotADirectory) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::NotADirectory) }
}

// ── /dev/snd/ (directory) ─────────────────────────────────────────────────────
struct SndDir {
    ino:      u64,
    children: Mutex<BTreeMap<String, Arc<dyn VfsNode>>>,
}

impl SndDir {
    fn new() -> Self {
        let mut ch: BTreeMap<String, Arc<dyn VfsNode>> = BTreeMap::new();
        ch.insert(String::from("pcmC0D0p"), Arc::new(PcmNode(alloc_ino())));
        Self { ino: alloc_ino(), children: Mutex::new(ch) }
    }
}

impl VfsNode for SndDir {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.ino, size: 0, is_dir: true, nlink: 2, uid: 0, gid: 0, mode: 0o755 })
    }
    fn open(&self) -> VfsResult<Box<dyn FileHandle>> { Err(VfsError::NotAFile) }
    fn readdir(&self) -> VfsResult<Vec<DirEntry>> {
        let ch = self.children.lock();
        Ok(ch.iter().map(|(name, node)| {
            let is_dir = node.stat().map(|s| s.is_dir).unwrap_or(false);
            let ino    = node.stat().map(|s| s.ino).unwrap_or(0);
            DirEntry { name: name.clone(), is_dir, ino }
        }).collect())
    }
    fn lookup(&self, name: &str) -> VfsResult<Arc<dyn VfsNode>> {
        self.children.lock().get(name).cloned().ok_or(VfsError::NotFound)
    }
    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::ReadOnly) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::ReadOnly) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::ReadOnly) }
}

// ── DevDir (the /dev directory itself) ───────────────────────────────────────

/// Global reference to the mounted devfs root — set by `DevDir::root()`.
static DEVFS_ROOT: spin::Once<Arc<DevDir>> = spin::Once::new();

/// Register a device node in /dev at runtime (called by block device init).
pub fn register_node(name: &str, node: Arc<dyn VfsNode>) {
    if let Some(root) = DEVFS_ROOT.get() {
        root.children.lock().insert(String::from(name), node);
    }
}

pub struct DevDir {
    ino:      u64,
    children: Mutex<BTreeMap<String, Arc<dyn VfsNode>>>,
}

impl DevDir {
    pub fn new() -> Self {
        let ino = alloc_ino();
        let mut ch: BTreeMap<String, Arc<dyn VfsNode>> = BTreeMap::new();
        ch.insert(String::from("null"),    Arc::new(NullNode(alloc_ino())));
        ch.insert(String::from("zero"),    Arc::new(ZeroNode(alloc_ino())));
        ch.insert(String::from("kmsg"),    Arc::new(KmsgNode(alloc_ino())));
        ch.insert(String::from("random"),  Arc::new(RandomNode(alloc_ino())));
        ch.insert(String::from("urandom"), Arc::new(RandomNode(alloc_ino())));
        ch.insert(String::from("composer"),Arc::new(ComposerNode(alloc_ino())));
        ch.insert(String::from("tty"),     Arc::new(TtyNode(alloc_ino())));
        ch.insert(String::from("ptmx"),    Arc::new(PtmxNode(alloc_ino())));
        ch.insert(String::from("dsp"),     Arc::new(DspNode(alloc_ino())));
        ch.insert(String::from("snd"),     Arc::new(SndDir::new()));
        Self { ino, children: Mutex::new(ch) }
    }

    /// Create the DevDir, store it as the global devfs root, and return an Arc.
    pub fn root() -> Arc<Self> {
        let dir = Arc::new(Self::new());
        DEVFS_ROOT.call_once(|| dir.clone());
        dir
    }
}

impl VfsNode for DevDir {
    fn stat(&self) -> VfsResult<Stat> {
        Ok(Stat { ino: self.ino, size: 0, is_dir: true, nlink: 2, uid: 0, gid: 0, mode: 0o755 })
    }
    fn open(&self) -> VfsResult<Box<dyn FileHandle>> { Err(VfsError::NotAFile) }
    fn readdir(&self) -> VfsResult<Vec<DirEntry>> {
        let ch = self.children.lock();
        Ok(ch.iter().map(|(name, node)| {
            let is_dir = node.stat().map(|s| s.is_dir).unwrap_or(false);
            let ino    = node.stat().map(|s| s.ino).unwrap_or(0);
            DirEntry { name: name.clone(), is_dir, ino }
        }).collect())
    }
    fn lookup(&self, name: &str) -> VfsResult<Arc<dyn VfsNode>> {
        self.children.lock().get(name).cloned().ok_or(VfsError::NotFound)
    }
    fn create_file(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::ReadOnly) }
    fn mkdir(&self, _: &str) -> VfsResult<Arc<dyn VfsNode>> { Err(VfsError::ReadOnly) }
    fn unlink(&self, _: &str) -> VfsResult<()> { Err(VfsError::ReadOnly) }
}
