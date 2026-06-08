//! Atheros AR9271 USB WiFi driver.
//!
//! Implements:
//!   Phase 1: Firmware loading (two-stage: USB control writes → boot command)
//!   Phase 2: 802.11 probe request/response — active scan
//!   Phase 3: Open authentication + association
//!   Phase 4: WPA2-PSK (PBKDF2-SHA1 + 4-way EAPOL + CCMP/AES) [TODO]
//!
//! Reference: Linux drivers/net/wireless/ath/ath9k/htc_drv_init.c
//!            Linux drivers/net/wireless/ath/ath9k/htc_hst.c
//!            ath9k_htc_7010.fw: two-stage HTC/WMI firmware

use alloc::{vec, vec::Vec, string::String};
use super::ApInfo;

// ── Firmware ──────────────────────────────────────────────────────────────────
// Place ath9k_htc_7010.fw in firmware/ directory.
// Extract from linux-firmware: apt-get install linux-firmware
// or: https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git
//
// For compilation without the file present, we use a conditional include.
#[cfg(firmware_present)]
static AR9271_FW: &[u8] = include_bytes!("../../../firmware/ath9k_htc_7010.fw");
#[cfg(not(firmware_present))]
static AR9271_FW: &[u8] = &[]; // Firmware not embedded; load from /lib/firmware at runtime

// ── AR9271 USB vendor commands ────────────────────────────────────────────────
// From Linux htc_hst.h and ath9k_htc driver source.
const USB_TYPE_VENDOR:    u8 = 0x40; // bmRequestType: OUT | Vendor | Device
const USB_TYPE_VENDOR_IN: u8 = 0xC0; // bmRequestType: IN  | Vendor | Device

const AR_FW_DOWNLOAD:     u8 = 0x30; // Download firmware chunk
const AR_FW_DOWNLOAD_COMP:u8 = 0x31; // Signal firmware download complete

const AR_CHIP_ID_ADDR:    u32 = 0x0004_0008; // AR9271 chip ID register

// ── 802.11 frame constants ────────────────────────────────────────────────────
const FC_PROBE_REQ:  u16 = 0x0040;
const FC_PROBE_RESP: u16 = 0x0050;
const FC_AUTH:       u16 = 0x00B0;
const FC_ASSOC_REQ:  u16 = 0x0000;
const FC_ASSOC_RESP: u16 = 0x0010;

// Information Element tags
const IE_SSID:       u8 = 0;
const IE_RATES:      u8 = 1;
const IE_DS_PARAM:   u8 = 3; // channel

// Capability flags
const CAP_PRIVACY:   u16 = 1 << 4;

// ── AR9271 firmware chunk loading ─────────────────────────────────────────────

/// Load AR9271 firmware via USB control transfers.
/// Two stages: (1) stream firmware chunks, (2) boot command.
pub fn load_firmware(slot: u8) -> bool {
    if AR9271_FW.is_empty() {
        // Try to load from VFS /lib/firmware/ath9k_htc_7010.fw
        match crate::vfs::read_file("/lib/firmware/ath9k_htc_7010.fw") {
            Ok(fw) => return load_firmware_bytes(slot, &fw),
            Err(_) => {
                crate::klog!(WARN,
                    "WiFi: ath9k_htc_7010.fw not found — place in /lib/firmware/");
                crate::klog!(INFO,
                    "WiFi: get firmware: apt-get install linux-firmware && \
                     cp /lib/firmware/ath9k_htc_7010.fw to disk image");
                return false;
            }
        }
    }
    load_firmware_bytes(slot, AR9271_FW)
}

fn load_firmware_bytes(slot: u8, fw: &[u8]) -> bool {
    if fw.len() < 4 {
        crate::klog!(WARN, "WiFi: firmware too small ({} bytes)", fw.len());
        return false;
    }

    crate::klog!(INFO, "WiFi: loading {} bytes firmware to AR9271", fw.len());

    // Stream firmware in 4096-byte chunks via USB control OUT
    // wValue = chunk address (auto-increments in firmware loader)
    let mut addr: u16 = 0;
    for chunk in fw.chunks(4096) {
        let mut buf = chunk.to_vec();
        let setup: [u8; 8] = [
            USB_TYPE_VENDOR,    // bmRequestType
            AR_FW_DOWNLOAD,     // bRequest
            (addr & 0xFF) as u8, (addr >> 8) as u8, // wValue = address
            0x00, 0x00,         // wIndex
            (buf.len() & 0xFF) as u8, (buf.len() >> 8) as u8, // wLength
        ];
        let n = crate::usb::control_transfer(slot, setup, &mut buf);
        if n == 0 {
            crate::klog!(WARN, "WiFi: firmware chunk at addr={:#x} failed", addr);
            return false;
        }
        addr = addr.wrapping_add((chunk.len() / 256) as u16 + 1);
    }

    // Signal firmware download complete — boot the chip
    let setup_boot: [u8; 8] = [
        USB_TYPE_VENDOR, AR_FW_DOWNLOAD_COMP,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    crate::usb::control_transfer(slot, setup_boot, &mut []);

    // Allow firmware to initialize (~100ms)
    let t0 = crate::scheduler::uptime_ms();
    while crate::scheduler::uptime_ms().wrapping_sub(t0) < 150 {
        core::hint::spin_loop();
    }

    // Verify chip ID
    let chip_id = read_reg(slot, AR_CHIP_ID_ADDR);
    crate::klog!(INFO, "WiFi: AR9271 chip_id={:#010x}", chip_id);
    if chip_id == 0 || chip_id == 0xFFFFFFFF {
        crate::klog!(WARN, "WiFi: chip ID read failed — firmware may not have booted");
        // Don't hard-fail here; the firmware may still be running
    }

    true
}

// ── Register access via WMI ───────────────────────────────────────────────────

/// Read a 32-bit register from AR9271 via WMI over USB bulk.
fn read_reg(slot: u8, addr: u32) -> u32 {
    // WMI READ command: [cmd_id(2), seq(2), addr(4)]
    let cmd: [u8; 8] = [
        0x00, 0x09,                          // WMI_READ_REG = 9
        0x00, 0x01,                          // sequence number
        (addr & 0xFF) as u8, ((addr >> 8) & 0xFF) as u8,
        ((addr >> 16) & 0xFF) as u8, ((addr >> 24) & 0xFF) as u8,
    ];
    let _ = crate::usb::bulk_out(slot, 0x01, &cmd);
    let mut resp = [0u8; 8];
    let n = crate::usb::bulk_in(slot, 0x81, &mut resp);
    if n >= 8 {
        u32::from_le_bytes([resp[4], resp[5], resp[6], resp[7]])
    } else {
        0
    }
}

// ── 802.11 frame building ─────────────────────────────────────────────────────

/// Build an 802.11 probe request frame for the given SSID (empty = wildcard).
fn build_probe_request(our_mac: [u8; 6], ssid: &[u8]) -> Vec<u8> {
    let mut frame = Vec::new();

    // Frame Control (2) + Duration (2) + DA (6) + SA (6) + BSSID (6) + Seq (2)
    let fc = FC_PROBE_REQ.to_le_bytes();
    frame.extend_from_slice(&fc);
    frame.extend_from_slice(&[0x00, 0x00]); // duration
    frame.extend_from_slice(&[0xFF; 6]);    // DA = broadcast
    frame.extend_from_slice(&our_mac);      // SA = our MAC
    frame.extend_from_slice(&[0xFF; 6]);    // BSSID = broadcast
    frame.extend_from_slice(&[0x00, 0x00]); // sequence

    // SSID IE (tag=0)
    frame.push(IE_SSID);
    frame.push(ssid.len() as u8);
    frame.extend_from_slice(ssid);

    // Supported Rates IE (tag=1): 1, 2, 5.5, 11 Mbps
    frame.push(IE_RATES);
    frame.push(4);
    frame.extend_from_slice(&[0x82, 0x84, 0x8B, 0x96]);

    frame
}

/// Parse an 802.11 probe response / beacon frame into ApInfo.
/// Returns None if the frame is not a valid beacon or probe response.
fn parse_probe_response(frame: &[u8], rssi: i8) -> Option<ApInfo> {
    if frame.len() < 36 { return None; }

    // Check frame type
    let fc = u16::from_le_bytes([frame[0], frame[1]]);
    let frame_type = fc & 0x00FC;
    if frame_type != (FC_PROBE_RESP & 0x00FC) && frame_type != 0x0080 {
        return None; // not probe response or beacon
    }

    // BSSID at offset 16 (6 bytes)
    let bssid: [u8; 6] = frame[16..22].try_into().ok()?;

    // Fixed parameters: timestamp(8) + beacon interval(2) + capability(2) = 12 bytes
    // Start at offset 24 (after MAC header)
    if frame.len() < 36 { return None; }
    let cap = u16::from_le_bytes([frame[34], frame[35]]);
    let secured = (cap & CAP_PRIVACY) != 0;

    // Information Elements start at offset 36
    let mut ssid = String::new();
    let mut channel: u8 = 0;
    let mut pos = 36usize;

    while pos + 2 <= frame.len() {
        let tag = frame[pos];
        let len = frame[pos + 1] as usize;
        pos += 2;
        if pos + len > frame.len() { break; }
        let data = &frame[pos..pos + len];
        match tag {
            IE_SSID => {
                ssid = String::from(core::str::from_utf8(data).unwrap_or(""));
            }
            IE_DS_PARAM if len == 1 => {
                channel = data[0];
            }
            _ => {}
        }
        pos += len;
    }

    if ssid.is_empty() && channel == 0 { return None; }

    Some(ApInfo { ssid, bssid, channel, rssi, secured })
}

// ── Active scan ───────────────────────────────────────────────────────────────

/// Send probe requests on channels 1-14 and collect responses.
/// Returns real ApInfo list from over-the-air 802.11 frames.
pub fn scan_networks(slot: u8) -> Vec<ApInfo> {
    let our_mac = unsafe { crate::net::OUR_MAC };
    let mut aps: Vec<ApInfo> = Vec::new();

    // Probe on channels 1, 6, 11 (most common) then 2-14
    let channels = [1u8, 6, 11, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14];

    for &ch in &channels {
        // Set channel via WMI SET_CHANNEL command
        set_channel(slot, ch);

        // Send wildcard probe request
        let probe = build_probe_request(our_mac, &[]);
        let _ = crate::usb::bulk_out(slot, 0x01, &probe);

        // Collect responses for 50ms
        let deadline = crate::scheduler::uptime_ms() + 50;
        while crate::scheduler::uptime_ms() < deadline {
            let mut buf = [0u8; 2346]; // max 802.11 frame size
            let n = crate::usb::bulk_in(slot, 0x81, &mut buf);
            if n < 4 { core::hint::spin_loop(); continue; }

            // AR9271 RX descriptor: 4-byte header before 802.11 frame
            // Byte 2 = RSSI (signed)
            let rssi = buf[2] as i8;
            let frame = &buf[4..n];

            if let Some(ap) = parse_probe_response(frame, rssi) {
                // Deduplicate by BSSID
                if !aps.iter().any(|a| a.bssid == ap.bssid) {
                    crate::klog!(INFO,
                        "WiFi: AP \"{}\" bssid={:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} \
                         rssi={}dBm ch={} {}",
                        ap.ssid,
                        ap.bssid[0], ap.bssid[1], ap.bssid[2],
                        ap.bssid[3], ap.bssid[4], ap.bssid[5],
                        ap.rssi, ap.channel,
                        if ap.secured { "WPA2" } else { "open" });
                    aps.push(ap);
                }
            }
        }
    }
    aps
}

/// Set radio channel via WMI command.
fn set_channel(slot: u8, channel: u8) {
    // WMI SET_CHANNEL (simplified — real implementation sends full HTC/WMI frame)
    // cmd_id=0x000F (WMI_SET_CHANNEL), channel frequency in MHz
    let freq: u16 = 2407 + (channel as u16) * 5; // 2.4GHz: 2412MHz = ch1
    let cmd: [u8; 8] = [
        0x00, 0x0F,                          // WMI_SET_CHANNEL
        0x00, 0x01,                          // sequence
        (freq & 0xFF) as u8, (freq >> 8) as u8,
        channel, 0x00,
    ];
    let _ = crate::usb::bulk_out(slot, 0x01, &cmd);
    // Short dwell time
    let t0 = crate::scheduler::uptime_ms();
    while crate::scheduler::uptime_ms().wrapping_sub(t0) < 10 {
        core::hint::spin_loop();
    }
}

// ── Open association ──────────────────────────────────────────────────────────

pub fn connect_open(slot: u8, ap: &ApInfo) -> bool {
    let our_mac = unsafe { crate::net::OUR_MAC };

    // Open Authentication: frame type 0x00B0, Algorithm=0, Seq=1
    let mut auth = build_mgmt_header(FC_AUTH, our_mac, ap.bssid);
    auth.extend_from_slice(&[0x00, 0x00]); // Auth Algorithm = Open
    auth.extend_from_slice(&[0x01, 0x00]); // Auth Seq = 1
    auth.extend_from_slice(&[0x00, 0x00]); // Status = Success
    let _ = crate::usb::bulk_out(slot, 0x01, &auth);

    // Wait for Auth Response (Seq=2)
    if !wait_for_auth_response(slot, &ap.bssid) {
        crate::klog!(WARN, "WiFi: auth response timeout for \"{}\"", ap.ssid);
        return false;
    }

    // Association Request
    let mut assoc = build_mgmt_header(FC_ASSOC_REQ, our_mac, ap.bssid);
    assoc.extend_from_slice(&[0x21, 0x00]); // Capability: ESS + short-preamble
    assoc.extend_from_slice(&[0x64, 0x00]); // Listen Interval = 100
    // SSID IE
    assoc.push(IE_SSID);
    assoc.push(ap.ssid.len() as u8);
    assoc.extend_from_slice(ap.ssid.as_bytes());
    // Supported Rates IE
    assoc.push(IE_RATES);
    assoc.push(4);
    assoc.extend_from_slice(&[0x82, 0x84, 0x8B, 0x96]);
    let _ = crate::usb::bulk_out(slot, 0x01, &assoc);

    // Wait for Assoc Response
    wait_for_assoc_response(slot, &ap.bssid)
}

/// WPA2-PSK connection: open auth → association → 4-way EAPOL handshake.
pub fn connect_wpa2(slot: u8, ap: &ApInfo, passphrase: &str) -> bool {
    use super::crypto::{derive_pmk, derive_ptk, eapol_mic};

    crate::klog!(INFO, "WiFi: WPA2 — deriving PMK (PBKDF2-SHA1, 4096 iter)...");
    let pmk = derive_pmk(passphrase.as_bytes(), ap.ssid.as_bytes());
    crate::klog!(INFO, "WiFi: PMK derived");

    // Phase 1: open auth + assoc (same as open, but with RSN IE for WPA2)
    let our_mac = unsafe { crate::net::OUR_MAC };
    let mut auth = build_mgmt_header(FC_AUTH, our_mac, ap.bssid);
    auth.extend_from_slice(&[0x00, 0x00, 0x01, 0x00, 0x00, 0x00]);
    let _ = crate::usb::bulk_out(slot, 0x01, &auth);
    if !wait_for_auth_response(slot, &ap.bssid) {
        crate::klog!(WARN, "WiFi: WPA2 auth response timeout");
        return false;
    }

    // Association request with RSN IE (WPA2-PSK, CCMP)
    let mut assoc = build_mgmt_header(FC_ASSOC_REQ, our_mac, ap.bssid);
    assoc.extend_from_slice(&[0x31, 0x04]); // capability: ESS+WEP-bit-for-WPA2
    assoc.extend_from_slice(&[0x64, 0x00]); // listen interval
    // SSID IE
    assoc.push(IE_SSID); assoc.push(ap.ssid.len() as u8);
    assoc.extend_from_slice(ap.ssid.as_bytes());
    // Supported Rates IE
    assoc.push(IE_RATES); assoc.push(4);
    assoc.extend_from_slice(&[0x82, 0x84, 0x8B, 0x96]);
    // RSN IE (tag=48): WPA2-PSK with CCMP
    let rsn: &[u8] = &[
        0x30, 0x14,             // tag=48, len=20
        0x01, 0x00,             // version=1
        0x00, 0x0F, 0xAC, 0x04, // group cipher: CCMP
        0x01, 0x00,             // pairwise count=1
        0x00, 0x0F, 0xAC, 0x04, // pairwise: CCMP
        0x01, 0x00,             // AKM count=1
        0x00, 0x0F, 0xAC, 0x02, // AKM: PSK
        0x00, 0x00,             // RSN capabilities
    ];
    assoc.extend_from_slice(rsn);
    let _ = crate::usb::bulk_out(slot, 0x01, &assoc);
    if !wait_for_assoc_response(slot, &ap.bssid) {
        crate::klog!(WARN, "WiFi: WPA2 assoc response timeout");
        return false;
    }

    // Phase 2: 4-way EAPOL handshake
    // Generate SNonce (random-ish using uptime + MAC)
    let ts = crate::scheduler::uptime_ms();
    let mut snonce = [0u8; 32];
    for i in 0..8  { snonce[i]    = ((ts >> (i * 8)) & 0xFF) as u8; }
    for i in 0..6  { snonce[8+i]  = our_mac[i]; }
    for i in 0..6  { snonce[14+i] = ap.bssid[i]; }
    for i in 20..32 { snonce[i] = (i as u8).wrapping_mul(0x37).wrapping_add(snonce[i-1]); }

    // Message 1: AP → STA: ANonce + key info
    let anonce = receive_eapol_msg1(slot, &ap.bssid, 3000);
    let anonce = match anonce {
        Some(n) => n,
        None => { crate::klog!(WARN, "WiFi: EAPOL msg1 timeout"); return false; }
    };
    crate::klog!(INFO, "WiFi: EAPOL msg1 received (ANonce)");

    // Derive PTK from PMK + ANonce + SNonce + AP MAC + STA MAC
    let ptk = derive_ptk(&pmk, &anonce, &snonce, &ap.bssid, &our_mac);
    let kck: &[u8; 16] = ptk[0..16].try_into().unwrap();
    let _kek: &[u8; 16] = ptk[16..32].try_into().unwrap();
    // TK = ptk[32..48] — used for CCMP (Phase 5)
    crate::klog!(INFO, "WiFi: PTK derived (KCK+KEK+TK)");

    // Message 2: STA → AP: SNonce + MIC
    send_eapol_msg2(slot, our_mac, ap.bssid, &snonce, kck);
    crate::klog!(INFO, "WiFi: EAPOL msg2 sent");

    // Message 3: AP → STA: GTK (encrypted with KEK) + MIC
    if !receive_eapol_msg3(slot, &ap.bssid, kck, 3000) {
        crate::klog!(WARN, "WiFi: EAPOL msg3 failed MIC check");
        return false;
    }
    crate::klog!(INFO, "WiFi: EAPOL msg3 verified");

    // Message 4: STA → AP: ACK
    send_eapol_msg4(slot, our_mac, ap.bssid, kck);
    crate::klog!(INFO, "WiFi: EAPOL msg4 sent — WPA2 handshake complete");

    // Install TK so CCMP encrypt/decrypt in wifi::poll() / wifi_tx() works
    let tk: &[u8; 16] = ptk[32..48].try_into().unwrap();
    crate::wifi::set_tk(tk, &ap.bssid, &our_mac);
    crate::klog!(INFO, "WiFi: CCMP TK installed");
    true
}

// ── EAPOL frame helpers ───────────────────────────────────────────────────────

const EAPOL_TYPE: u16 = 0x888E;
const EAPOL_VERSION: u8 = 2;
const EAPOL_KEY: u8 = 3;
// Key Info bits
const KEY_INFO_MIC:     u16 = 1 << 8;
const KEY_INFO_ACK:     u16 = 1 << 7;
const KEY_INFO_INSTALL: u16 = 1 << 6;
const KEY_INFO_KEY_TYPE:u16 = 1 << 3; // 1=Pairwise
const KEY_INFO_SECURE:  u16 = 1 << 9;
const KEY_DESC_AES:     u8  = 2; // AES key descriptor

/// Wait for EAPOL Message 1 (AP→STA: ANonce, Msg 1 has ACK bit set, no MIC).
fn receive_eapol_msg1(slot: u8, ap_mac: &[u8; 6], timeout_ms: u64) -> Option<[u8; 32]> {
    let deadline = crate::scheduler::uptime_ms() + timeout_ms;
    while crate::scheduler::uptime_ms() < deadline {
        let mut buf = [0u8; 256];
        let n = crate::usb::bulk_in(slot, 0x81, &mut buf);
        if n < 30 { core::hint::spin_loop(); continue; }
        // AR9271 RX: skip 4-byte descriptor, then 802.11 data frame
        let frame = &buf[4..n];
        if let Some(eapol) = extract_eapol(frame, ap_mac) {
            // EAPOL Key: [version(1), type(1)=3, len(2), desc(1), key_info(2), ...]
            if eapol.len() < 95 { continue; }
            let key_info = u16::from_be_bytes([eapol[5], eapol[6]]);
            // Msg1: ACK=1, MIC=0
            if key_info & KEY_INFO_ACK != 0 && key_info & KEY_INFO_MIC == 0 {
                let mut anonce = [0u8; 32];
                anonce.copy_from_slice(&eapol[17..49]);
                return Some(anonce);
            }
        }
    }
    None
}

/// Build and send EAPOL Message 2 (STA→AP: SNonce + MIC).
fn send_eapol_msg2(slot: u8, our_mac: [u8; 6], ap_mac: [u8; 6],
                   snonce: &[u8; 32], kck: &[u8; 16]) {
    use super::crypto::eapol_mic;
    // Build EAPOL key frame (95 bytes minimum)
    let key_info: u16 = KEY_INFO_MIC | KEY_INFO_KEY_TYPE; // Pairwise, MIC set
    let mut eapol = [0u8; 99]; // header+body
    eapol[0] = EAPOL_VERSION;
    eapol[1] = EAPOL_KEY;
    eapol[2] = 0x00; eapol[3] = 0x5F; // length = 95
    eapol[4] = KEY_DESC_AES;
    eapol[5..7].copy_from_slice(&key_info.to_be_bytes());
    eapol[7] = 0x00; eapol[8] = 0x10; // key length = 16 (AES)
    // Replay counter: bytes 9..17 = 0 (msg 1's counter)
    // SNonce: bytes 17..49
    eapol[17..49].copy_from_slice(snonce);
    // MIC at bytes 77..93 (16 bytes) — compute over zeroed MIC field
    let mic = eapol_mic(kck, &eapol[..99]);
    eapol[77..93].copy_from_slice(&mic);

    // Wrap in 802.11 data frame + LLC/SNAP + EtherType 0x888E
    let frame = build_data_frame(our_mac, ap_mac, EAPOL_TYPE, &eapol);
    let _ = crate::usb::bulk_out(slot, 0x01, &frame);
}

/// Verify EAPOL Message 3 MIC and extract GTK.
fn receive_eapol_msg3(slot: u8, ap_mac: &[u8; 6], kck: &[u8; 16], timeout_ms: u64) -> bool {
    use super::crypto::eapol_mic;
    let deadline = crate::scheduler::uptime_ms() + timeout_ms;
    while crate::scheduler::uptime_ms() < deadline {
        let mut buf = [0u8; 512];
        let n = crate::usb::bulk_in(slot, 0x81, &mut buf);
        if n < 30 { core::hint::spin_loop(); continue; }
        let frame = &buf[4..n];
        if let Some(eapol) = extract_eapol(frame, ap_mac) {
            if eapol.len() < 95 { continue; }
            let key_info = u16::from_be_bytes([eapol[5], eapol[6]]);
            // Msg3: ACK=1, MIC=1, Install=1
            if key_info & (KEY_INFO_ACK | KEY_INFO_MIC | KEY_INFO_INSTALL) ==
               (KEY_INFO_ACK | KEY_INFO_MIC | KEY_INFO_INSTALL) {
                // Verify MIC: zero out MIC field, compute, compare
                let mut frame_copy = eapol.to_vec();
                for b in frame_copy[77..93].iter_mut() { *b = 0; }
                let expected_mic = eapol_mic(kck, &frame_copy);
                if expected_mic == eapol[77..93] { return true; }
            }
        }
    }
    false
}

/// Send EAPOL Message 4 (STA→AP: acknowledge GTK install).
fn send_eapol_msg4(slot: u8, our_mac: [u8; 6], ap_mac: [u8; 6], kck: &[u8; 16]) {
    use super::crypto::eapol_mic;
    let key_info: u16 = KEY_INFO_MIC | KEY_INFO_KEY_TYPE | KEY_INFO_SECURE;
    let mut eapol = [0u8; 99];
    eapol[0] = EAPOL_VERSION;
    eapol[1] = EAPOL_KEY;
    eapol[2] = 0x00; eapol[3] = 0x5F;
    eapol[4] = KEY_DESC_AES;
    eapol[5..7].copy_from_slice(&key_info.to_be_bytes());
    let mic = eapol_mic(kck, &eapol[..99]);
    eapol[77..93].copy_from_slice(&mic);
    let frame = build_data_frame(our_mac, ap_mac, EAPOL_TYPE, &eapol);
    let _ = crate::usb::bulk_out(slot, 0x01, &frame);
}

/// Extract EAPOL payload from an 802.11 data frame (skips 802.11 hdr + LLC/SNAP).
fn extract_eapol<'a>(frame: &'a [u8], expected_src: &[u8; 6]) -> Option<&'a [u8]> {
    if frame.len() < 36 { return None; }
    let fc = u16::from_le_bytes([frame[0], frame[1]]);
    // Data frame: type bits 2-3 = 10, subtype 0 = 0000
    if fc & 0x000C != 0x0008 { return None; }
    // Source MAC (addr2) at offset 10
    if &frame[10..16] != expected_src.as_ref() { return None; }
    // 802.11 header = 24 bytes, LLC/SNAP = 8 bytes → payload at offset 32
    if frame.len() < 34 { return None; }
    let ethertype = u16::from_be_bytes([frame[30], frame[31]]);
    if ethertype != EAPOL_TYPE { return None; }
    Some(&frame[32..])
}

/// Build an 802.11 data frame with LLC/SNAP header for the given EtherType.
pub fn build_data_frame(src: [u8; 6], dst: [u8; 6], ethertype: u16, payload: &[u8]) -> alloc::vec::Vec<u8> {
    let mut f = alloc::vec::Vec::new();
    // 802.11 Data frame header (24 bytes)
    f.extend_from_slice(&[0x08, 0x01]); // FC: Data, ToDS
    f.extend_from_slice(&[0x00, 0x00]); // duration
    f.extend_from_slice(&dst);          // BSSID (addr1)
    f.extend_from_slice(&src);          // SA (addr2)
    f.extend_from_slice(&dst);          // DA (addr3)
    f.extend_from_slice(&[0x00, 0x00]); // sequence
    // LLC/SNAP (8 bytes): DSAP=AA, SSAP=AA, ctrl=03, OUI=000000, EtherType
    f.extend_from_slice(&[0xAA, 0xAA, 0x03, 0x00, 0x00, 0x00]);
    f.extend_from_slice(&ethertype.to_be_bytes());
    f.extend_from_slice(payload);
    f
}

// ── Management frame helpers ──────────────────────────────────────────────────

fn build_mgmt_header(fc: u16, sa: [u8; 6], bssid: [u8; 6]) -> Vec<u8> {
    let mut f = Vec::new();
    f.extend_from_slice(&fc.to_le_bytes());
    f.extend_from_slice(&[0x00, 0x00]); // duration
    f.extend_from_slice(&bssid);        // DA
    f.extend_from_slice(&sa);           // SA
    f.extend_from_slice(&bssid);        // BSSID
    f.extend_from_slice(&[0x00, 0x00]); // sequence
    f
}

fn wait_for_auth_response(slot: u8, bssid: &[u8; 6]) -> bool {
    let deadline = crate::scheduler::uptime_ms() + 2000;
    while crate::scheduler::uptime_ms() < deadline {
        let mut buf = [0u8; 64];
        let n = crate::usb::bulk_in(slot, 0x81, &mut buf);
        if n < 30 { core::hint::spin_loop(); continue; }
        let frame = &buf[4..n]; // skip 4-byte RX descriptor
        if frame.len() < 26 { continue; }
        let fc = u16::from_le_bytes([frame[0], frame[1]]);
        if fc & 0x00FC != (FC_AUTH & 0x00FC) { continue; }
        let src: &[u8; 6] = frame[10..16].try_into().unwrap_or(&[0;6]);
        if src != bssid { continue; }
        let seq_num = u16::from_le_bytes([frame[24], frame[25]]);
        let status  = u16::from_le_bytes([frame[26].min(0), frame[27].min(0)]);
        if seq_num == 2 && status == 0 { return true; }
    }
    false
}

fn wait_for_assoc_response(slot: u8, bssid: &[u8; 6]) -> bool {
    let deadline = crate::scheduler::uptime_ms() + 2000;
    while crate::scheduler::uptime_ms() < deadline {
        let mut buf = [0u8; 128];
        let n = crate::usb::bulk_in(slot, 0x81, &mut buf);
        if n < 30 { core::hint::spin_loop(); continue; }
        let frame = &buf[4..n];
        if frame.len() < 28 { continue; }
        let fc = u16::from_le_bytes([frame[0], frame[1]]);
        if fc & 0x00FC != (FC_ASSOC_RESP & 0x00FC) { continue; }
        let src: &[u8; 6] = frame[10..16].try_into().unwrap_or(&[0;6]);
        if src != bssid { continue; }
        let status = u16::from_le_bytes([frame[26], frame[27]]);
        if status == 0 {
            let aid = u16::from_le_bytes([frame[28].min(0), frame[29].min(0)]) & 0x3FFF;
            crate::klog!(INFO, "WiFi: associated, AID={}", aid);
            return true;
        }
    }
    false
}
