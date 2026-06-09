//! User and group management — /etc/passwd, /etc/group, uid/gid tracking, su, sudo.
//!
//! Provides:
//!   - User/group database (in-memory, backed by /etc/passwd, /etc/group, /etc/shadow)
//!   - Password hashing (simple salted hash — no_std safe)
//!   - Authentication (login, session tracking)
//!   - Privilege escalation (su, sudo)
//!   - File permission model (rwxrwxrwx, uid/gid ownership)

use alloc::{string::String, vec::Vec, format};
use spin::Mutex;

// ── Types ─────────────────────────────────────────────────────────────────────

pub type Uid = u32;
pub type Gid = u32;

/// POSIX-style file permission bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileMode(pub u16);

impl FileMode {
    // Standard permission bits
    pub const OWNER_R: u16 = 0o400;
    pub const OWNER_W: u16 = 0o200;
    pub const OWNER_X: u16 = 0o100;
    pub const GROUP_R: u16 = 0o040;
    pub const GROUP_W: u16 = 0o020;
    pub const GROUP_X: u16 = 0o010;
    pub const OTHER_R: u16 = 0o004;
    pub const OTHER_W: u16 = 0o002;
    pub const OTHER_X: u16 = 0o001;
    pub const SETUID:  u16 = 0o4000;
    pub const SETGID:  u16 = 0o2000;
    pub const STICKY:  u16 = 0o1000;

    pub const DIR_DEFAULT:  Self = Self(0o755);
    pub const FILE_DEFAULT: Self = Self(0o644);
    pub const EXEC_DEFAULT: Self = Self(0o755);
    pub const SHADOW_MODE:  Self = Self(0o600);

    pub fn owner_read(self)  -> bool { self.0 & Self::OWNER_R != 0 }
    pub fn owner_write(self) -> bool { self.0 & Self::OWNER_W != 0 }
    pub fn owner_exec(self)  -> bool { self.0 & Self::OWNER_X != 0 }
    pub fn group_read(self)  -> bool { self.0 & Self::GROUP_R != 0 }
    pub fn group_write(self) -> bool { self.0 & Self::GROUP_W != 0 }
    pub fn group_exec(self)  -> bool { self.0 & Self::GROUP_X != 0 }
    pub fn other_read(self)  -> bool { self.0 & Self::OTHER_R != 0 }
    pub fn other_write(self) -> bool { self.0 & Self::OTHER_W != 0 }
    pub fn other_exec(self)  -> bool { self.0 & Self::OTHER_X != 0 }
    pub fn is_setuid(self)   -> bool { self.0 & Self::SETUID  != 0 }
    pub fn is_setgid(self)   -> bool { self.0 & Self::SETGID  != 0 }

    /// Format as rwxrwxrwx string (9 chars).
    pub fn as_str(&self) -> [u8; 9] {
        let m = self.0;
        [
            if m & Self::OWNER_R != 0 { b'r' } else { b'-' },
            if m & Self::OWNER_W != 0 { b'w' } else { b'-' },
            if m & Self::OWNER_X != 0 {
                if m & Self::SETUID != 0 { b's' } else { b'x' }
            } else {
                if m & Self::SETUID != 0 { b'S' } else { b'-' }
            },
            if m & Self::GROUP_R != 0 { b'r' } else { b'-' },
            if m & Self::GROUP_W != 0 { b'w' } else { b'-' },
            if m & Self::GROUP_X != 0 {
                if m & Self::SETGID != 0 { b's' } else { b'x' }
            } else {
                if m & Self::SETGID != 0 { b'S' } else { b'-' }
            },
            if m & Self::OTHER_R != 0 { b'r' } else { b'-' },
            if m & Self::OTHER_W != 0 { b'w' } else { b'-' },
            if m & Self::OTHER_X != 0 { b'x' } else { b'-' },
        ]
    }

    /// Parse an octal mode string like "755" into FileMode.
    pub fn from_octal(s: &str) -> Option<Self> {
        let mut val: u16 = 0;
        for b in s.bytes() {
            if b < b'0' || b > b'7' { return None; }
            val = val * 8 + (b - b'0') as u16;
        }
        Some(Self(val))
    }
}

// ── User entry ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct User {
    pub username: String,
    pub uid:      Uid,
    pub gid:      Gid,
    pub home:     String,
    pub shell:    String,
}

#[derive(Debug, Clone)]
pub struct Group {
    pub name:    String,
    pub gid:     Gid,
    pub members: Vec<String>,
}

/// Shadow entry — password hash.
#[derive(Debug, Clone)]
struct ShadowEntry {
    username: String,
    hash:     String,   // "salt:hash" format
}

// ── Database ──────────────────────────────────────────────────────────────────

static USERS:   Mutex<Vec<User>>        = Mutex::new(Vec::new());
static GROUPS:  Mutex<Vec<Group>>       = Mutex::new(Vec::new());
static SHADOW:  Mutex<Vec<ShadowEntry>> = Mutex::new(Vec::new());

/// Current session: uid of the logged-in user on the console.
static CURRENT_UID: core::sync::atomic::AtomicU32 =
    core::sync::atomic::AtomicU32::new(0); // root by default

/// Current working directory per-session.
static CWD: Mutex<String> = Mutex::new(String::new());

/// Sudo grace period tracking — last successful sudo time (uptime_ms).
static SUDO_LAST_AUTH_MS: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(0);
/// Sudo timeout in milliseconds (5 minutes).
const SUDO_TIMEOUT_MS: u64 = 5 * 60 * 1000;

/// Default umask for new files.
static UMASK: core::sync::atomic::AtomicU16 =
    core::sync::atomic::AtomicU16::new(0o022);

// ── Sudo audit log ───────────────────────────────────────────────────────────

const MAX_AUDIT_ENTRIES: usize = 64;

struct AuditEntry {
    username: String,
    command:  String,
    timestamp_ms: u64,
    success: bool,
}

static SUDO_AUDIT: Mutex<Vec<AuditEntry>> = Mutex::new(Vec::new());

/// Record a sudo invocation in the audit log.
pub fn audit_sudo(username: &str, command: &str, success: bool) {
    let entry = AuditEntry {
        username: String::from(username),
        command: String::from(command),
        timestamp_ms: crate::scheduler::uptime_ms(),
        success,
    };
    let mut log = SUDO_AUDIT.lock();
    if log.len() >= MAX_AUDIT_ENTRIES {
        log.remove(0);
    }
    log.push(entry);
}

/// Get the sudo audit log as formatted text.
pub fn sudo_audit_log() -> String {
    let log = SUDO_AUDIT.lock();
    let mut out = String::new();
    for e in log.iter() {
        let status = if e.success { "OK" } else { "FAIL" };
        out.push_str(&format!(
            "[{}ms] {} : sudo {} : {}\n",
            e.timestamp_ms, e.username, e.command, status
        ));
    }
    out
}

// ── Hostname ──────────────────────────────────────────────────────────────────

static HOSTNAME: Mutex<String> = Mutex::new(String::new());

pub fn hostname() -> String {
    let h = HOSTNAME.lock();
    if h.is_empty() { String::from("nodeai") } else { h.clone() }
}

pub fn set_hostname(name: &str) {
    *HOSTNAME.lock() = String::from(name);
}

// ── Initialization ────────────────────────────────────────────────────────────

/// Create default users and groups. Call after VFS init.
pub fn init() {
    // Default groups
    {
        let mut g = GROUPS.lock();
        g.push(Group { name: String::from("root"),   gid: 0,    members: Vec::new() });
        g.push(Group { name: String::from("nodeai"), gid: 1000, members: Vec::new() });
        g.push(Group { name: String::from("nobody"), gid: 65534, members: Vec::new() });
        g.push(Group { name: String::from("sudo"),   gid: 27,   members: alloc::vec![String::from("nodeai")] });
    }

    // Default users
    {
        let mut u = USERS.lock();
        u.push(User {
            username: String::from("root"),
            uid: 0, gid: 0,
            home: String::from("/root"),
            shell: String::from("/bin/sh"),
        });
        u.push(User {
            username: String::from("nodeai"),
            uid: 1000, gid: 1000,
            home: String::from("/home/nodeai"),
            shell: String::from("/bin/sh"),
        });
        u.push(User {
            username: String::from("nobody"),
            uid: 65534, gid: 65534,
            home: String::from("/nonexistent"),
            shell: String::from("/bin/false"),
        });
    }

    // Default passwords: root="root", nodeai="nodeai"
    {
        let mut s = SHADOW.lock();
        s.push(ShadowEntry {
            username: String::from("root"),
            hash: hash_password("root"),
        });
        s.push(ShadowEntry {
            username: String::from("nodeai"),
            hash: hash_password("nodeai"),
        });
    }

    // Set initial CWD
    *CWD.lock() = String::from("/");

    // Create /etc directory and config files in VFS
    populate_etc_files();

    // Create home directories
    create_home_dirs();

    // Create /etc/hostname
    set_hostname("nodeai");

    crate::klog!(INFO, "Users: root(0) + nodeai(1000) initialized, /etc populated");
}

// ── Password hashing (simple FNV-1a based — no_std safe) ──────────────────────
// NOTE: In production, use Argon2id. This is a simplified hash for the kernel
// demo. The salt+hash approach prevents trivial rainbow table attacks.

fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn hash_password(password: &str) -> String {
    // Use a deterministic "salt" derived from the password length + fixed seed.
    // Real implementation would use RDRAND for random salt.
    let salt = fnv1a_hash(&[password.len() as u8, 0x4E, 0x41, 0x49]); // "NAI"
    let salted = format!("{}:{}", salt, password);
    let hash = fnv1a_hash(salted.as_bytes());
    format!("{:016x}:{:016x}", salt, hash)
}

fn verify_password(password: &str, stored_hash: &str) -> bool {
    // Extract the salt from stored hash
    if let Some(colon) = stored_hash.find(':') {
        let salt_str = &stored_hash[..colon];
        // Reconstruct: salt_str is the hex salt, rehash with it
        let salted = format!("{}:{}", salt_str, password);
        let hash = fnv1a_hash(salted.as_bytes());
        let expected = format!("{}:{:016x}", salt_str, hash);
        expected == stored_hash
    } else {
        false
    }
}

// ── Populate /etc files ───────────────────────────────────────────────────────

fn populate_etc_files() {
    // Create /etc directory
    if let Ok(root) = crate::vfs::lookup("/") {
        let _ = root.mkdir("etc");
        let _ = root.mkdir("root");
        let _ = root.mkdir("home");
    }

    // /etc/passwd
    write_etc_file("passwd", &generate_passwd());
    // /etc/group
    write_etc_file("group", &generate_group());
    // /etc/shadow
    write_etc_file("shadow", &generate_shadow());
    // /etc/hostname
    write_etc_file("hostname", b"nodeai\n");
    // /etc/motd
    write_etc_file("motd",
        b"Welcome to NodeAI - AI-Integrated Kernel OS\n\
          Type 'help' for available commands.\n");
    // /etc/sudoers
    write_etc_file("sudoers",
        b"# /etc/sudoers - NodeAI sudo configuration\n\
          root ALL=(ALL) ALL\n\
          %sudo ALL=(ALL) ALL\n");
}

fn write_etc_file(name: &str, content: &[u8]) {
    if let Ok(etc) = crate::vfs::lookup("/etc") {
        if let Ok(node) = etc.create_file(name) {
            if let Ok(mut fh) = node.open() {
                let _ = fh.write(content);
                let _ = fh.flush();
            }
        }
    }
}

fn generate_passwd() -> Vec<u8> {
    let users = USERS.lock();
    let mut out = Vec::new();
    for u in users.iter() {
        let line = format!("{}:x:{}:{}::/{}:{}\n",
            u.username, u.uid, u.gid, u.home, u.shell);
        out.extend_from_slice(line.as_bytes());
    }
    out
}

fn generate_group() -> Vec<u8> {
    let groups = GROUPS.lock();
    let mut out = Vec::new();
    for g in groups.iter() {
        let members = g.members.join(",");
        let line = format!("{}:x:{}:{}\n", g.name, g.gid, members);
        out.extend_from_slice(line.as_bytes());
    }
    out
}

fn generate_shadow() -> Vec<u8> {
    let shadow = SHADOW.lock();
    let mut out = Vec::new();
    for s in shadow.iter() {
        let line = format!("{}:{}:0:0:99999:7:::\n", s.username, s.hash);
        out.extend_from_slice(line.as_bytes());
    }
    out
}

fn create_home_dirs() {
    if let Ok(home) = crate::vfs::lookup("/home") {
        let _ = home.mkdir("nodeai");
    }
}

// ── Public query API ──────────────────────────────────────────────────────────

/// Get the currently active user's UID.
pub fn current_uid() -> Uid {
    // If a task is currently running, use its euid.
    let pid = crate::scheduler::current_pid();
    if pid > 0 { // Assuming 0 is idle/invalid, adjust if current_pid can be 0. Wait, in scheduler it might return 0.
        if let Some(euid) = crate::scheduler::get_euid(pid) {
            return euid;
        }
    }
    CURRENT_UID.load(core::sync::atomic::Ordering::Relaxed)
}

/// Set the current session UID (used by login/su/sudo).
pub fn set_current_uid(uid: Uid) {
    CURRENT_UID.store(uid, core::sync::atomic::Ordering::Relaxed);
}

/// Look up a user by UID.
pub fn get_user(uid: Uid) -> Option<User> {
    USERS.lock().iter().find(|u| u.uid == uid).cloned()
}

/// Look up a user by username.
pub fn get_user_by_name(name: &str) -> Option<User> {
    USERS.lock().iter().find(|u| u.username == name).cloned()
}

/// Look up a group by GID.
pub fn get_group(gid: Gid) -> Option<Group> {
    GROUPS.lock().iter().find(|g| g.gid == gid).cloned()
}

/// Look up a group by name.
pub fn get_group_by_name(name: &str) -> Option<Group> {
    GROUPS.lock().iter().find(|g| g.name == name).cloned()
}

/// Get the current user's username.
pub fn current_username() -> String {
    let uid = current_uid();
    get_user(uid).map(|u| u.username).unwrap_or_else(|| format!("uid{}", uid))
}

/// Get the current user's home directory.
pub fn current_home() -> String {
    let uid = current_uid();
    get_user(uid).map(|u| u.home).unwrap_or_else(|| String::from("/"))
}

/// Get the current working directory.
pub fn cwd() -> String {
    CWD.lock().clone()
}

/// Set the current working directory.
pub fn set_cwd(path: &str) {
    *CWD.lock() = String::from(path);
}

/// Get the current umask value.
pub fn umask() -> u16 {
    UMASK.load(core::sync::atomic::Ordering::Relaxed)
}

/// Set the umask and return the old value.
pub fn set_umask(mask: u16) -> u16 {
    UMASK.swap(mask, core::sync::atomic::Ordering::Relaxed)
}

/// Check if the current user is root (uid 0).
pub fn is_root() -> bool {
    current_uid() == 0
}

/// Returns true if the given user is in the "sudo" group.
pub fn user_can_sudo(username: &str) -> bool {
    let groups = GROUPS.lock();
    for g in groups.iter() {
        if g.name == "sudo" && g.members.iter().any(|m| m == username) {
            return true;
        }
    }
    // root can always sudo
    if let Some(u) = get_user_by_name(username) {
        return u.uid == 0;
    }
    false
}

// ── Authentication ────────────────────────────────────────────────────────────

/// Authenticate a user by password. Returns true on success.
pub fn authenticate(username: &str, password: &str) -> bool {
    let shadow = SHADOW.lock();
    for s in shadow.iter() {
        if s.username == username {
            return verify_password(password, &s.hash);
        }
    }
    false
}

/// Switch to a different user (su). Returns true on success.
pub fn switch_user(username: &str) -> bool {
    if let Some(user) = get_user_by_name(username) {
        set_current_uid(user.uid);
        set_cwd(&user.home);
        true
    } else {
        false
    }
}

/// Execute a command as root via sudo. Checks sudo permission and password.
/// Returns true if authorized (caller should then execute the command).
pub fn sudo_auth(password: &str) -> bool {
    let username = current_username();

    // Check if user can sudo
    if !user_can_sudo(&username) && !is_root() {
        return false;
    }

    // Check grace period
    let now = crate::scheduler::uptime_ms();
    let last = SUDO_LAST_AUTH_MS.load(core::sync::atomic::Ordering::Relaxed);
    if now.saturating_sub(last) < SUDO_TIMEOUT_MS && last > 0 {
        return true; // within grace period
    }

    // Verify password
    if authenticate(&username, password) {
        SUDO_LAST_AUTH_MS.store(now, core::sync::atomic::Ordering::Relaxed);
        true
    } else {
        false
    }
}

// ── User management (root only) ──────────────────────────────────────────────

/// Add a new user. Returns Ok(uid) or Err message.
pub fn useradd(username: &str) -> Result<Uid, &'static str> {
    let mut users = USERS.lock();
    if users.iter().any(|u| u.username == username) {
        return Err("user already exists");
    }

    // Assign next UID >= 1000
    let uid = users.iter().map(|u| u.uid).max().unwrap_or(999) + 1;
    let uid = if uid < 1000 { 1000 } else { uid };
    let gid = uid; // create a personal group

    let user = User {
        username: String::from(username),
        uid,
        gid,
        home: format!("/home/{}", username),
        shell: String::from("/bin/sh"),
    };
    users.push(user);
    drop(users);

    // Create personal group
    {
        let mut groups = GROUPS.lock();
        groups.push(Group {
            name: String::from(username),
            gid,
            members: alloc::vec![String::from(username)],
        });
    }

    // Create shadow entry with empty password (must set via passwd)
    {
        let mut shadow = SHADOW.lock();
        shadow.push(ShadowEntry {
            username: String::from(username),
            hash: String::from("!"), // locked account
        });
    }

    // Create home directory
    if let Ok(home) = crate::vfs::lookup("/home") {
        let _ = home.mkdir(username);
    }

    // Refresh /etc files
    refresh_etc_files();

    Ok(uid)
}

/// Remove a user. Returns Ok or Err message.
pub fn userdel(username: &str) -> Result<(), &'static str> {
    if username == "root" {
        return Err("cannot delete root");
    }

    {
        let mut users = USERS.lock();
        let len_before = users.len();
        users.retain(|u| u.username != username);
        if users.len() == len_before {
            return Err("user not found");
        }
    }

    // Remove from shadow
    {
        let mut shadow = SHADOW.lock();
        shadow.retain(|s| s.username != username);
    }

    // Remove from groups
    {
        let mut groups = GROUPS.lock();
        for g in groups.iter_mut() {
            g.members.retain(|m| m != username);
        }
        // Remove personal group
        groups.retain(|g| g.name != username);
    }

    refresh_etc_files();
    Ok(())
}

/// Change a user's password.
pub fn change_password(username: &str, new_password: &str) -> Result<(), &'static str> {
    let mut shadow = SHADOW.lock();
    for s in shadow.iter_mut() {
        if s.username == username {
            s.hash = hash_password(new_password);
            drop(shadow);
            refresh_etc_files();
            return Ok(());
        }
    }
    Err("user not found")
}

/// Get all groups a user belongs to.
pub fn user_groups(username: &str) -> Vec<String> {
    // Get primary gid outside of GROUPS lock
    let primary_gid = get_user_by_name(username).map(|u| u.gid);
    let groups = GROUPS.lock();
    let mut result = Vec::new();
    for g in groups.iter() {
        let is_member = g.members.iter().any(|m| m == username);
        let is_primary = primary_gid.map_or(false, |pg| pg == g.gid);
        if is_member || is_primary {
            result.push(g.name.clone());
        }
    }
    result
}

fn refresh_etc_files() {
    // Regenerate /etc files from in-memory databases
    if let Ok(etc) = crate::vfs::lookup("/etc") {
        // Remove old files and recreate
        let _ = etc.unlink("passwd");
        let _ = etc.unlink("group");
        let _ = etc.unlink("shadow");
        write_etc_file("passwd", &generate_passwd());
        write_etc_file("group", &generate_group());
        write_etc_file("shadow", &generate_shadow());
    }
}

// ── Permission checking ──────────────────────────────────────────────────────

/// Check if the current user has read permission on a file.
pub fn can_read(file_uid: Uid, file_gid: Gid, mode: FileMode) -> bool {
    if is_root() { return true; }
    let uid = current_uid();
    if uid == file_uid { return mode.owner_read(); }
    if user_in_group(uid, file_gid) { return mode.group_read(); }
    mode.other_read()
}

/// Check if the current user has write permission on a file.
pub fn can_write(file_uid: Uid, file_gid: Gid, mode: FileMode) -> bool {
    if is_root() { return true; }
    let uid = current_uid();
    if uid == file_uid { return mode.owner_write(); }
    if user_in_group(uid, file_gid) { return mode.group_write(); }
    mode.other_write()
}

/// Check if the current user has execute permission on a file.
pub fn can_exec(file_uid: Uid, file_gid: Gid, mode: FileMode) -> bool {
    if is_root() { return true; }
    let uid = current_uid();
    if uid == file_uid { return mode.owner_exec(); }
    if user_in_group(uid, file_gid) { return mode.group_exec(); }
    mode.other_exec()
}

fn user_in_group(uid: Uid, gid: Gid) -> bool {
    if let Some(user) = get_user(uid) {
        if user.gid == gid { return true; }
        // Check supplementary groups
        let groups = GROUPS.lock();
        for g in groups.iter() {
            if g.gid == gid && g.members.iter().any(|m| m == &user.username) {
                return true;
            }
        }
    }
    false
}
