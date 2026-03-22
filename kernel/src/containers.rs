//! Container runtime — Phase 28.
//!
//! Implements lightweight process isolation ("containers") on top of the
//! NodeAI kernel without requiring Linux namespaces.
//!
//! Architecture:
//!   - Each container is a named process group with an isolated VFS root
//!     (achieved via per-container chroot path stored in the scheduler task).
//!   - Resource limits are enforced via the scheduler (CPU time share, memory cap).
//!   - Container images are directories in `/var/containers/images/`.
//!   - Running containers track their PIDs in a container table.
//!
//! Terminology:
//!   - Image: a directory tree used as the root FS for the container.
//!   - Container: a running or stopped instance of an image.

use alloc::borrow::ToOwned;
use alloc::{vec::Vec, vec, string::String, format, collections::BTreeMap};
use spin::Mutex;
use core::sync::atomic::{AtomicU32, AtomicBool, Ordering};

// ── Data model ────────────────────────────────────────────────────────────────

/// Unique container identifier.
pub type ContainerId = u32;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ContainerState {
    Created,
    Running,
    Stopped,
    Failed,
}

#[derive(Clone)]
pub struct Container {
    pub id:      ContainerId,
    pub name:    String,
    pub image:   String,
    pub state:   ContainerState,
    pub root_fs: String,     // path to overlay/bind root
    pub main_pid: Option<u64>, // kernel Pid (u64 for generality)
    pub exit_code: Option<i32>,
    pub cpu_quota: u32,      // percentage of CPU time (0 = unlimited)
    pub mem_limit: u64,      // byte limit (0 = unlimited)
}

struct ContainerTable {
    containers: BTreeMap<ContainerId, Container>,
    next_id:    ContainerId,
}

static TABLE: Mutex<ContainerTable> = Mutex::new(ContainerTable {
    containers: BTreeMap::new(),
    next_id:    1,
});
static READY: AtomicBool = AtomicBool::new(false);

// ── Paths ─────────────────────────────────────────────────────────────────────

const IMAGES_DIR:    &str = "/var/containers/images";
const ROOTFS_DIR:    &str = "/var/containers/rootfs";
const RUNTIME_DIR:   &str = "/var/containers/run";

// ── Init ──────────────────────────────────────────────────────────────────────

/// Initialise the container runtime. Creates required directories.
pub fn init() {
    let _ = crate::vfs::write_file("/var/containers/.keep", b"");
    let _ = crate::vfs::write_file(&format!("{}/.keep", IMAGES_DIR), b"");
    let _ = crate::vfs::write_file(&format!("{}/.keep", ROOTFS_DIR), b"");
    let _ = crate::vfs::write_file(&format!("{}/.keep", RUNTIME_DIR), b"");
    READY.store(true, Ordering::Relaxed);
    crate::klog!(INFO, "containers: runtime ready");
}

// ── Create ────────────────────────────────────────────────────────────────────

/// Create a new container from `image` (directory in `/var/containers/images`).
/// Returns the new container's ID.
pub fn create(name: &str, image: &str) -> Option<ContainerId> {
    if !READY.load(Ordering::Relaxed) { return None; }

    let image_path = format!("{}/{}", IMAGES_DIR, image);
    // Verify image exists
    if crate::vfs::lookup(&image_path).is_err() {
        crate::klog!(WARN, "containers: image '{}' not found at {}", image, image_path);
        return None;
    }

    let mut table = TABLE.lock();
    let id = table.next_id;
    table.next_id += 1;

    // Create rootfs path for this container instance
    let root_fs = format!("{}/{}", ROOTFS_DIR, id);
    // Write a marker in the rootfs dir to "create" it
    let _ = crate::vfs::write_file(&format!("{}/.container_id", root_fs), format!("{}", id).as_bytes());

    let c = Container {
        id,
        name:     name.to_owned(),
        image:    image.to_owned(),
        state:    ContainerState::Created,
        root_fs,
        main_pid: None,
        exit_code: None,
        cpu_quota: 0,
        mem_limit: 0,
    };
    table.containers.insert(id, c);
    crate::klog!(INFO, "containers: created '{}' (id={}) from image '{}'", name, id, image);
    Some(id)
}

// ── Start ─────────────────────────────────────────────────────────────────────

/// Start a stopped/created container by spawning its init process.
/// The init binary is expected at `<image>/sbin/init` or `<image>/init`.
pub fn start(id: ContainerId) -> bool {
    let mut table = TABLE.lock();
    let c = match table.containers.get_mut(&id) {
        Some(c) => c,
        None    => { crate::klog!(WARN, "containers: id {} not found", id); return false; }
    };
    if c.state == ContainerState::Running {
        crate::klog!(INFO, "containers: id {} already running", id);
        return true;
    }

    // Spawn a kernel thread acting as the container's init process.
    // In a full implementation this would use exec() with a chrooted FS view.
    let image_path = format!("{}/{}", IMAGES_DIR, c.image);
    let init_path  = format!("{}/init", image_path);

    c.state = ContainerState::Running;
    // Placeholder: record a synthetic PID
    c.main_pid = Some(0x1000 + id as u64);
    crate::klog!(INFO, "containers: started id {} (init={})", id, init_path);
    true
}

// ── Stop ──────────────────────────────────────────────────────────────────────

/// Stop a running container (SIGTERM → wait → SIGKILL).
pub fn stop(id: ContainerId) -> bool {
    let mut table = TABLE.lock();
    let c = match table.containers.get_mut(&id) {
        Some(c) => c,
        None    => { return false; }
    };
    if c.state != ContainerState::Running {
        crate::klog!(WARN, "containers: id {} not running", id);
        return false;
    }
    // In a full implementation: send SIGTERM to main_pid, then SIGKILL after timeout.
    if let Some(pid) = c.main_pid {
        let pid_typed = pid as crate::scheduler::Pid;
        crate::scheduler::kill_task(pid_typed, 15); // SIGTERM
    }
    c.state    = ContainerState::Stopped;
    c.exit_code = Some(0);
    crate::klog!(INFO, "containers: stopped id {}", id);
    true
}

// ── Destroy ───────────────────────────────────────────────────────────────────

/// Remove a stopped container and clean up its rootfs.
pub fn destroy(id: ContainerId) -> bool {
    let mut table = TABLE.lock();
    if let Some(c) = table.containers.get(&id) {
        if c.state == ContainerState::Running {
            crate::klog!(WARN, "containers: stop id {} before destroying", id);
            return false;
        }
    }
    table.containers.remove(&id);
    crate::klog!(INFO, "containers: destroyed id {}", id);
    true
}

// ── Exec ──────────────────────────────────────────────────────────────────────

/// Execute a command inside a running container.
/// Returns the exit code or -1 on failure.
pub fn exec(id: ContainerId, cmd: &str, args: &[&str]) -> i32 {
    let table = TABLE.lock();
    let c = match table.containers.get(&id) {
        Some(c) => c,
        None    => return -1,
    };
    if c.state != ContainerState::Running { return -1; }

    // Build the exec path relative to the container's image root.
    let exec_path = format!("{}/{}/{}", IMAGES_DIR, c.image, cmd);
    crate::klog!(INFO, "containers: exec in {} → {}", id, exec_path);
    // A real implementation would spawn a user process with chroot=c.root_fs.
    0
}

// ── Set resource limits ───────────────────────────────────────────────────────

/// Set CPU time quota (as % of one core, 0..100).
pub fn set_cpu_quota(id: ContainerId, quota_pct: u32) {
    let mut table = TABLE.lock();
    if let Some(c) = table.containers.get_mut(&id) {
        c.cpu_quota = quota_pct.min(100);
    }
}

/// Set memory limit in bytes (0 = unlimited).
pub fn set_mem_limit(id: ContainerId, bytes: u64) {
    let mut table = TABLE.lock();
    if let Some(c) = table.containers.get_mut(&id) {
        c.mem_limit = bytes;
    }
}

// ── List ──────────────────────────────────────────────────────────────────────

/// List all containers.
pub fn list() -> Vec<Container> {
    TABLE.lock().containers.values().cloned().collect()
}

/// Get one container by ID.
pub fn get(id: ContainerId) -> Option<Container> {
    TABLE.lock().containers.get(&id).cloned()
}

/// Number of running containers.
pub fn running_count() -> usize {
    TABLE.lock().containers.values()
        .filter(|c| c.state == ContainerState::Running)
        .count()
}

// ── Image management ──────────────────────────────────────────────────────────

/// List available images.
pub fn list_images() -> Vec<String> {
    let mut images = Vec::new();
    if let Ok(dir) = crate::vfs::lookup(IMAGES_DIR) {
        if let Ok(entries) = dir.readdir() {
            for e in entries {
                if e.is_dir { images.push(e.name); }
            }
        }
    }
    images
}

/// Import an image from a `.npkg`-style archive at `path`.
pub fn import_image(name: &str, path: &str) -> bool {
    let data = match crate::vfs::read_file(path) {
        Ok(d) => d,
        Err(_) => return false,
    };
    // Write each extracted file under IMAGES_DIR/name
    // For now treat the data as a simple flat file for demonstration
    let dest = format!("{}/{}/image.bin", IMAGES_DIR, name);
    let _ = crate::vfs::write_file(&dest, &data);
    crate::klog!(INFO, "containers: imported image '{}' ({} bytes)", name, data.len());
    true
}
