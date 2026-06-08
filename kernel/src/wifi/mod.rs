//! WiFi subsystem — AR9271 USB driver + 802.11 scan + WPA2.
//!
//! Phase 0: xHCI transfer ring (usb/mod.rs) ✅
//! Phase 1: AR9271 firmware loading via USB control transfers
//! Phase 2: 802.11 probe request/response — real scan()
//! Phase 3: Open association (no crypto)
//! Phase 4: WPA2-PSK (PBKDF2-SHA1 + 4-way EAPOL + CCMP/AES)
//!
//! Reference: Linux drivers/net/wireless/ath/ath9k/htc_drv_*.c

pub mod ar9271;

use alloc::{vec::Vec, string::String};
use spin::Mutex;
use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

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

static WIFI_READY:   AtomicBool = AtomicBool::new(false);
static AR9271_SLOT:  AtomicU8   = AtomicU8::new(0xFF); // 0xFF = not present

// ── Public API ─────────────────────────────────────────────────────────────────

/// Called by USB driver when AR9271 dongle is detected. slot = xHCI slot number.
pub fn ar9271_attach(slot: u8) {
    crate::klog!(INFO, "WiFi: AR9271 on USB slot {}, loading firmware...", slot);
    AR9271_SLOT.store(slot, Ordering::Relaxed);
    if ar9271::load_firmware(slot) {
        WIFI_READY.store(true, Ordering::Relaxed);
        crate::klog!(INFO, "WiFi: AR9271 ready — call wifi::scan() to find networks");
    } else {
        crate::klog!(WARN, "WiFi: AR9271 firmware load failed");
    }
}

/// Probe for PCI WiFi adapters (iwlwifi etc.) — AR9271 is USB, attached via ar9271_attach.
pub fn init(_phys_offset: u64) {
    // AR9271 USB detection happens in usb::probe_port() → ar9271_attach()
    // PCI WiFi (Intel iwlwifi, Realtek) would be probed here in future phases
    if !WIFI_READY.load(Ordering::Relaxed) {
        crate::klog!(WARN, "WiFi: no supported adapter found");
    }
}

/// Returns true if a WiFi adapter is available and firmware loaded.
pub fn is_available() -> bool { WIFI_READY.load(Ordering::Relaxed) }

/// Returns true if currently associated to an AP.
pub fn is_connected() -> bool { WIFI_STATE.lock().connected }

/// SSID of the associated network, if connected.
pub fn ssid() -> Option<String> {
    WIFI_STATE.lock().current_ap.as_ref().map(|ap| ap.ssid.clone())
}

/// Active scan: send 802.11 probe request, collect probe responses.
/// Returns real ApInfo list populated from over-the-air frames.
pub fn scan() -> Vec<ApInfo> {
    let slot = AR9271_SLOT.load(Ordering::Relaxed);
    if slot == 0xFF {
        crate::klog!(DEBUG, "WiFi: scan — no adapter");
        return WIFI_STATE.lock().scan_cache.clone();
    }
    let aps = ar9271::scan_networks(slot);
    if !aps.is_empty() {
        crate::klog!(INFO, "WiFi: scan found {} AP(s)", aps.len());
        WIFI_STATE.lock().scan_cache = aps.clone();
    }
    aps
}

/// Associate to `ssid`. Open networks only for now; WPA2 in Phase 4.
pub fn connect(ssid: &str, passphrase: &str) -> bool {
    if !is_available() {
        crate::klog!(WARN, "WiFi: connect — no adapter");
        return false;
    }
    let slot = AR9271_SLOT.load(Ordering::Relaxed);
    // Find the AP in scan cache
    let ap = {
        let state = WIFI_STATE.lock();
        state.scan_cache.iter().find(|a| a.ssid == ssid).cloned()
    };
    let ap = match ap {
        Some(a) => a,
        None => {
            crate::klog!(WARN, "WiFi: connect — SSID '{}' not in scan cache", ssid);
            return false;
        }
    };
    let ok = if ap.secured {
        ar9271::connect_wpa2(slot, &ap, passphrase)
    } else {
        ar9271::connect_open(slot, &ap)
    };
    if ok {
        let mut state = WIFI_STATE.lock();
        state.connected  = true;
        state.current_ap = Some(ap.clone());
        crate::klog!(INFO, "WiFi: associated to \"{}\"", ssid);
    }
    ok
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
