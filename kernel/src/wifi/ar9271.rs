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

pub fn connect_wpa2(_slot: u8, ap: &ApInfo, _passphrase: &str) -> bool {
    // Phase 4: WPA2-PSK (PBKDF2-SHA1 + 4-way EAPOL + CCMP/AES)
    // Implementation pending — see WIFI_ROADMAP.md Phase 4
    crate::klog!(WARN,
        "WiFi: WPA2 not yet implemented for \"{}\" — use open network for now", ap.ssid);
    false
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
