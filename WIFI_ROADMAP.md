# NodeAI WiFi Implementation Roadmap
*Grounded by John. No mocks. Every line real.*

---

## Target Chip: Atheros AR9271 (USB)

**Why AR9271:**
- Only WiFi chip with fully open firmware (ath9k_htc_7010.fw, no NDA)
- USB-connected → uses our existing xHCI controller
- PCI Vendor:Device = 0x0CF3:0x9271
- Reference: Linux `drivers/net/wireless/ath/ath9k/htc_drv_*.c`
- Common hardware: TP-Link TL-WN722N v1, ALFA AWUS036NHA

**QEMU testing:**
```bash
# Plug in AR9271 dongle on host, then:
./scripts/run_qemu.sh --gui --wifi
# run_qemu.sh adds: -device usb-host,vendorid=0x0cf3,productid=0x9271
```

---

## Prerequisites (what must exist before WiFi works)

### Already done ✅
- PCI enumeration (`drivers/src/pci.rs`): read_config_u32, bar_mmio_base, enable_bus_master
- xHCI controller probe (bus + port reset, DCBAA, event ring)
- USB HID (keyboard/mouse) — uses interrupt transfers
- VirtIO-net TCP/IP stack — DHCP logic reusable for WiFi interface
- map_mmio(), alloc_frames(), IDT with free vectors

### Phase 0: xHCI Control + Bulk Transfer Ring — PREREQUISITE
**Gap identified by John:** xHCI driver has command/event rings but zero data TRB submission.
`usb/msc.rs:129` calls `xhci_bulk_transfer()` which doesn't exist.

Implement:
- `TransferRing` struct: 256-entry TRB ring, enqueue pointer, cycle bit
- `setup_trb()` / `data_trb()` / `status_trb()` for control transfers
- `normal_trb()` for bulk transfers
- Poll event ring for Transfer Event TRB completion
- `pub fn control_transfer(slot, endpoint, setup: [u8;8], data: &mut [u8], dir: Dir) -> usize`
- `pub fn bulk_out(slot, endpoint, data: &[u8]) -> usize`
- `pub fn bulk_in(slot, endpoint, buf: &mut [u8]) -> usize`

**~200 lines. Unblocks AR9271 AND USB mass storage.**

---

## Phase 1: AR9271 Firmware Loading

### Firmware
Embed as `include_bytes!` in kernel binary — no VFS dependency:
```rust
static FW: &[u8] = include_bytes!("../../firmware/ath9k_htc_7010.fw");
```
Firmware file: `firmware/ath9k_htc_7010.fw` (from Linux kernel, MIT-licensed)

### Two-stage loader (simplified)
Stage 1: Download firmware to chip RAM via USB control writes (64-byte chunks)
Stage 2: One vendor-specific control command to boot from RAM

```
Control OUT: bmRequestType=0x40, bRequest=0x30, wValue=chunk_addr, data=64 bytes
Final:        bmRequestType=0x40, bRequest=0x31, wValue=0, wLength=0
```

**~100 lines including firmware array handling.**

### Verification
Read chip ID via control IN: expect 0x9271 or 0x7010

---

## Phase 2: 802.11 Scan (Probe Request/Response)

### Probe Request frame (broadcast)
```
Frame Control: 0x0040 (probe request, managed mode)
Duration: 0
DA: FF:FF:FF:FF:FF:FF (broadcast)
SA: our MAC
BSSID: FF:FF:FF:FF:FF:FF
Seq: 0
SSID IE: tag=0, len=0 (wildcard)
Supported Rates IE: tag=1, len=4, rates=[0x82, 0x84, 0x8B, 0x96]
```
Send via bulk OUT on endpoint 1.

### Probe Response parsing
Poll bulk IN on endpoint 2. Parse:
- Frame Control bytes 0-1: check type=probe response (0x0050)
- BSSID at offset 16 (6 bytes)
- Sequence + timestamp: skip
- Capability info: check bit 4 (privacy) → secured
- SSID IE: tag=0, len, data → ssid string
- RSSI: from AR9271 RX descriptor prefix (4 bytes before 802.11 header)
- Channel: DS Parameter Set IE (tag=3, len=1, data=channel)

### scan() returns real data
```rust
ApInfo { ssid: "MyNetwork", bssid: [0xAA,...], rssi: -65, channel: 6, secured: true }
```

**~150 lines.**

---

## Phase 3: Open Association (No Encryption)

For open networks (secured=false):

1. **Auth Request**: Frame Control=0x00B0, Auth Algo=0, Seq=1, Status=0
2. **Auth Response**: expect Seq=2, Status=0
3. **Assoc Request**: include SSID IE + Supported Rates IE + capability
4. **Assoc Response**: expect Status=0, AID assigned
5. Mark interface as associated → `wifi::is_connected() = true`

**~120 lines.**

---

## Phase 4: DHCP on WiFi Interface

Reuse existing DHCP/UDP from net.rs.
WiFi interface gets its own IP (separate from VirtIO-net 10.0.2.15).
Add `net::add_interface()` to support multiple NICs.

**~80 lines.**

---

## Phase 5: WPA2-PSK

### Crypto needed (all implement from scratch, no external crate):
1. **HMAC-SHA1** (~130 lines): block-based, standard FIPS 198
2. **PBKDF2-SHA1** (~40 lines): 4096 iterations, 32-byte PMK
   *Note: 4096 iterations takes ~500ms on single-core bare metal — acceptable*
3. **AES-128** (~120 lines): 10-round SPN, 4×4 state, pre-computed S-box
4. **CCMP** (~80 lines): AES-CCM mode, 8-byte MIC, PN counter

### 4-way handshake (EAPOL frames):
- Message 1: AP sends ANonce
- Message 2: STA sends SNonce + MIC (PTK derived from PMK + ANonce + SNonce + MACs)
- Message 3: AP sends GTK (encrypted) + MIC
- Message 4: STA ACKs

All data frames then encrypted with CCMP (PTK for unicast, GTK for multicast).

**~370 lines crypto + ~100 lines EAPOL state machine.**

---

## Phase 6: GUI Integration

### Network Manager WiFi panel (extends existing netmgr):
- Tab: "WiFi" | "Ethernet"
- WiFi tab: `[Scan]` button → calls wifi::scan() → lists SSIDs
- Click network → password dialog (if secured)
- Connect → Phase 3 (open) or Phase 5 (WPA2)
- Status bar: SSID + signal strength bars + IP address

---

## Implementation order (John's recommendation)

| Phase | Work | Lines | Status |
|-------|------|-------|--------|
| 0 | xHCI transfer ring | ~200 | ✅ done |
| 1 | AR9271 firmware load | ~130 | ✅ done |
| 2 | 802.11 scan | ~150 | ✅ done |
| 3 | Open association | ~120 | ✅ done |
| 4 | DHCP on WiFi | ~80 | ✅ done (`net::dhcp_request_wifi()`) |
| 5 | WPA2-PSK | ~370 | ✅ done (CCMP TK bug fixed in 884de01+6d7af8b) |
| 6 | GUI | ~200 | ✅ done (netmgr WiFi section: W/D keys, AP list, status) |
| — | Keepalive | ~20 | ✅ done (null-data frame every 30s — 96d6fba) |

**Total: ~1250 lines of Rust. All genuine, all grounded. All done.**

### Post-roadmap hardening (future work)
- GTK rekey handling (AP sends new GTK in Group Key Handshake)
- Reconnect on deauth (monitor for deauth/disassoc frame, re-run connect())
- EPOLLET on WiFi RX socket (currently level-triggered fallback)
- SMP WiFi (WiFi poll currently single-core)

---

## Testing

```bash
# With AR9271 dongle plugged into host USB:
./scripts/run_qemu.sh --gui --wifi

# Expected Phase 2 serial output:
# [ INFO] WiFi: AR9271 firmware loaded (87040 bytes)
# [ INFO] WiFi: chip ID 0x9271 confirmed
# [ INFO] WiFi: scan found 3 AP(s)
# [ INFO] WiFi: AP "HomeNetwork" bssid=aa:bb:cc:dd:ee:ff rssi=-62 ch=6 WPA2
# [ INFO] WiFi: AP "OpenCafe" bssid=... rssi=-74 ch=11 open
```

---
*John's grounding criteria: No mocks. Every number from an actual radio frame.*
*Last updated: 2026-06-08. Roadmap complete as of commits 884de01, 6d7af8b, 96d6fba.*
