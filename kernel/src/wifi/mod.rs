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
pub mod crypto;

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

// ── WiFi data plane globals ────────────────────────────────────────────────────
use core::sync::atomic::AtomicU64;
static WIFI_PN:      AtomicU64  = AtomicU64::new(1); // TX packet number (48-bit)
static WIFI_IP:      Mutex<[u8; 4]> = Mutex::new([0u8; 4]);
static WIFI_BSSID:   Mutex<[u8; 6]> = Mutex::new([0u8; 6]);
static WIFI_TK:      Mutex<[u8; 16]> = Mutex::new([0u8; 16]); // CCMP TK from PTK[32..48]
static WIFI_MAC:     Mutex<[u8; 6]>  = Mutex::new([0u8; 6]);  // dongle MAC

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
        {
            let mut state = WIFI_STATE.lock();
            state.connected  = true;
            state.current_ap = Some(ap.clone());
        }
        crate::klog!(INFO, "WiFi: associated to \"{}\" — running DHCP", ssid);
        if !crate::net::dhcp_request_wifi(slot) {
            crate::klog!(WARN, "WiFi: DHCP failed — interface has no IP");
        }
    }
    ok
}

/// Return the WiFi interface IP (0.0.0.0 if not connected or no DHCP yet).
pub fn get_ip()    -> [u8; 4]    { *WIFI_IP.lock() }
pub fn wifi_mac()  -> [u8; 6]    { *WIFI_MAC.lock() }
pub fn scan_cache() -> Vec<ApInfo> { WIFI_STATE.lock().scan_cache.clone() }

/// Set WiFi IP after DHCP completes.
pub fn set_ip(ip: [u8; 4]) { *WIFI_IP.lock() = ip; }

/// Store CCMP TK (TK = PTK[32..48]) after successful 4-way handshake.
pub fn set_tk(tk: &[u8; 16], bssid: &[u8; 6], our_mac: &[u8; 6]) {
    *WIFI_TK.lock()   = *tk;
    *WIFI_BSSID.lock() = *bssid;
    *WIFI_MAC.lock()   = *our_mac;
}

/// Poll WiFi RX: receive encrypted frames, CCMP decrypt, inject into net stack.
/// Also handles DHCP responses on the WiFi interface.
/// Called from idle_loop every iteration alongside net::poll().
pub fn poll() {
    let slot = AR9271_SLOT.load(Ordering::Relaxed);
    if slot == 0xFF || !WIFI_STATE.lock().connected { return; }

    let tk   = *WIFI_TK.lock();
    let bssid = *WIFI_BSSID.lock();
    let our_mac = *WIFI_MAC.lock();

    // Read one frame from bulk IN (non-blocking: returns 0 if no data)
    let mut buf = [0u8; 2346];
    let n = crate::usb::bulk_in(slot, 0x81, &mut buf);
    if n < 32 { return; }

    // AR9271 RX: 4-byte descriptor prefix
    let frame = &buf[4..n];
    if frame.len() < 28 { return; }

    let fc = u16::from_le_bytes([frame[0], frame[1]]);
    let frame_type = (fc >> 2) & 0x3;
    let frame_subtype = (fc >> 4) & 0xF;

    // Data frame (type=2) with Protected bit set → CCMP decrypt
    if frame_type == 2 && (fc & 0x4000 != 0) {
        if frame.len() < 24 { return; }
        let mac_hdr: [u8; 24] = match frame[0..24].try_into() {
            Ok(h) => h, Err(_) => return,
        };
        if let Some(plaintext) = crypto::ccmp_decrypt(&tk, frame) {
            // plaintext = LLC/SNAP(8) + payload
            if plaintext.len() < 8 { return; }
            let ethertype = u16::from_be_bytes([plaintext[6], plaintext[7]]);
            let payload = &plaintext[8..];

            // Build Ethernet frame and inject into net stack
            // FromDS: Addr1=DA(our MAC), Addr2=BSSID(AP), Addr3=SA(original sender)
            let da: [u8; 6] = mac_hdr[4..10].try_into().unwrap_or(our_mac);
            let sa: [u8; 6] = mac_hdr[16..22].try_into().unwrap_or(bssid);

            let mut eth = alloc::vec::Vec::with_capacity(14 + payload.len());
            eth.extend_from_slice(&da);
            eth.extend_from_slice(&sa);
            eth.extend_from_slice(&ethertype.to_be_bytes());
            eth.extend_from_slice(payload);

            if let Some(reply) = crate::net::handle_frame(&eth) {
                // Send reply back over WiFi (CCMP encrypt)
                wifi_tx(slot, reply, &tk, &bssid, &our_mac);
            }
        }
    }
    // Unprotected data frames (e.g., DHCP before keys installed)
    else if frame_type == 2 && frame_subtype == 0 {
        if frame.len() < 32 { return; }
        // Skip 802.11 header + LLC/SNAP → raw payload
        let ethertype = u16::from_be_bytes([frame[30], frame[31]]);
        if frame.len() < 32 { return; }
        let payload = &frame[32..];
        let da: [u8; 6] = frame[4..10].try_into().unwrap_or(our_mac);
        let sa: [u8; 6] = frame[16..22].try_into().unwrap_or(bssid);

        let mut eth = alloc::vec::Vec::with_capacity(14 + payload.len());
        eth.extend_from_slice(&da);
        eth.extend_from_slice(&sa);
        eth.extend_from_slice(&ethertype.to_be_bytes());
        eth.extend_from_slice(payload);

        if let Some(reply) = crate::net::handle_frame(&eth) {
            wifi_tx_open(slot, reply, &bssid, &our_mac);
        }
    }
}

/// Transmit an Ethernet frame over WiFi with CCMP encryption.
fn wifi_tx(slot: u8, eth_frame: alloc::vec::Vec<u8>, tk: &[u8; 16],
           bssid: &[u8; 6], our_mac: &[u8; 6]) {
    if eth_frame.len() < 14 { return; }
    let pn = WIFI_PN.fetch_add(1, Ordering::Relaxed);

    // Build 802.11 data header (24 bytes): ToDS, Protected
    let mut mac_hdr = [0u8; 24];
    mac_hdr[0] = 0x08; // FC byte 0: Data
    mac_hdr[1] = 0x41; // FC byte 1: ToDS=1, Protected=1
    mac_hdr[4..10].copy_from_slice(bssid); // Addr1 = BSSID
    mac_hdr[10..16].copy_from_slice(our_mac); // Addr2 = STA (us)
    mac_hdr[16..22].copy_from_slice(&eth_frame[0..6]); // Addr3 = DA

    // Build LLC/SNAP + payload (plaintext)
    let ethertype = u16::from_be_bytes([eth_frame[12], eth_frame[13]]);
    let mut plaintext = alloc::vec![0xAA, 0xAA, 0x03, 0x00, 0x00, 0x00];
    plaintext.extend_from_slice(&ethertype.to_be_bytes());
    plaintext.extend_from_slice(&eth_frame[14..]);

    // CCMP encrypt
    let ciphertext = crypto::ccmp_encrypt(tk, &mac_hdr, our_mac, pn, &plaintext);

    // CCMP header (8 bytes): PN0, PN1, 0x00, KeyID|ExtIV, PN2, PN3, PN4, PN5
    let ccmp_hdr = [
        (pn & 0xFF) as u8,       // PN0
        ((pn >> 8) & 0xFF) as u8, // PN1
        0x00,
        0x20, // ExtIV=1, KeyID=0
        ((pn >> 16) & 0xFF) as u8, // PN2
        ((pn >> 24) & 0xFF) as u8, // PN3
        ((pn >> 32) & 0xFF) as u8, // PN4
        ((pn >> 40) & 0xFF) as u8, // PN5
    ];

    let mut tx = alloc::vec::Vec::with_capacity(32 + ciphertext.len());
    tx.extend_from_slice(&mac_hdr);
    tx.extend_from_slice(&ccmp_hdr);
    tx.extend_from_slice(&ciphertext);
    let _ = crate::usb::bulk_out(slot, 0x01, &tx);
}

/// Public thin wrapper for DHCP-over-WiFi (called from net::dhcp_request_wifi).
pub fn wifi_tx_open_pub(slot: u8, eth_frame: alloc::vec::Vec<u8>) {
    let bssid   = *WIFI_BSSID.lock();
    let our_mac = *WIFI_MAC.lock();
    wifi_tx_open(slot, eth_frame, &bssid, &our_mac);
}

/// Transmit an Ethernet frame over WiFi without encryption (pre-EAPOL / open network).
fn wifi_tx_open(slot: u8, eth_frame: alloc::vec::Vec<u8>, bssid: &[u8; 6], our_mac: &[u8; 6]) {
    if eth_frame.len() < 14 { return; }
    use ar9271::build_data_frame;
    let ethertype = u16::from_be_bytes([eth_frame[12], eth_frame[13]]);
    let mut llc_snap = alloc::vec![0xAA, 0xAA, 0x03, 0x00, 0x00, 0x00];
    llc_snap.extend_from_slice(&ethertype.to_be_bytes());
    llc_snap.extend_from_slice(&eth_frame[14..]);
    let frame = build_data_frame(*our_mac, *bssid, ethertype, &eth_frame[14..]);
    let _ = crate::usb::bulk_out(slot, 0x01, &frame);
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
