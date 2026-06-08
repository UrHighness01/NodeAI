//! WiFi subsystem — scan, connect, WPA2 supplicant over iwlwifi.
//!
//! Wraps the iwlwifi PCIe driver and provides a simple 802.11 management API:
//!   - `init()` — probe drivers on PCI
//!   - `scan() -> Vec<ApInfo>` — list visible access points (stub)
//!   - `connect(ssid, passphrase) -> bool` — associate (stub)
//!   - `disconnect()`
//!   - `is_connected() -> bool`
//!   - `ssid() -> Option<String>` — current SSID

pub mod iwlwifi;

use alloc::{vec::Vec, string::String};
use spin::Mutex;
use core::sync::atomic::{AtomicBool, Ordering};

// ── AP information ─────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct ApInfo {
    pub ssid:    String,
    pub bssid:   [u8; 6],
    pub channel: u8,
    pub rssi:    i8,
    pub secured: bool,
}

// ── Connection state ───────────────────────────────────────────────────────────

struct WifiState {
    connected:  bool,
    current_ap: Option<ApInfo>,
    scan_cache: Vec<ApInfo>,
}

static WIFI_STATE: Mutex<WifiState> = Mutex::new(WifiState {
    connected:  false,
    current_ap: None,
    scan_cache: Vec::new(),
});

static WIFI_READY: AtomicBool = AtomicBool::new(false);

// ── Public API ─────────────────────────────────────────────────────────────────

/// Initialise WiFi by probing hardware drivers.
pub fn init(phys_offset: u64) {
    let found = iwlwifi::probe(phys_offset);
    // Future: probe Realtek, Atheros, MediaTek drivers here

    if found {
        WIFI_READY.store(true, Ordering::Relaxed);
        crate::klog!(INFO, "WiFi: {} device(s) ready", iwlwifi::device_count());
    } else {
        crate::klog!(WARN, "WiFi: no supported adapter found");
    }
}

/// Returns true if a WiFi adapter is available.
pub fn is_available() -> bool { WIFI_READY.load(Ordering::Relaxed) }

/// Returns true if currently associated to an AP.
pub fn is_connected() -> bool { WIFI_STATE.lock().connected }

/// SSID of the associated network, if connected.
pub fn ssid() -> Option<String> {
    WIFI_STATE.lock().current_ap.as_ref().map(|ap| ap.ssid.clone())
}

/// Passive scan — returns cached AP list.
/// Real implementation would issue 802.11 probe requests through iwlwifi DMA.
pub fn scan() -> Vec<ApInfo> {
    // Return a synthetic list since real scan results require full DMA rings.
    let cache = WIFI_STATE.lock().scan_cache.clone();
    if cache.is_empty() {
        crate::klog!(DEBUG, "WiFi: scan — no APs in cache (firmware not associated)");
    }
    cache
}

/// Associate to `ssid` using WPA2 `passphrase`.
/// Stub: marks as connected if hardware is present.
pub fn connect(ssid: &str, _passphrase: &str) -> bool {
    if !is_available() {
        crate::klog!(WARN, "WiFi: connect — no adapter");
        return false;
    }
    // Full implementation: build 802.11 AUTH + ASSOC frames, 4-way EAPOL handshake.
    let ap = ApInfo {
        ssid:    String::from(ssid),
        bssid:   [0; 6],
        channel: 6,
        rssi:    -65,
        secured: true,
    };
    let mut state = WIFI_STATE.lock();
    state.connected  = true;
    state.current_ap = Some(ap);
    crate::klog!(INFO, "WiFi: connected to \"{}\" (stub)", ssid);
    true
}

/// Disassociate from current AP.
pub fn disconnect() {
    let mut state = WIFI_STATE.lock();
    if let Some(ref ap) = state.current_ap {
        crate::klog!(INFO, "WiFi: disconnecting from \"{}\"", ap.ssid);
    }
    state.connected  = false;
    state.current_ap = None;
}
