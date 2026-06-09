//! TCP/IP networking stack with userspace sockets, DHCP, and kernel HTTP server.
//!
//! Implements:
//!   - Ethernet frame parsing
//!   - ARP request/response & ARP cache
//!   - IPv4 header parsing
//!   - ICMP echo request/reply (ping)
//!   - UDP send/receive
//!   - TCP state machine
//!   - DNS resolver
//!   - Global NIC + poll loop

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use spin::Mutex;
use drivers::virtio::net::VirtioNet;

// ── Global NIC ───────────────────────────────────────────────────────────────

/// The system's primary network interface (set during PCI init).
pub static NIC: Mutex<Option<VirtioNet>> = Mutex::new(None);

/// Transmit a raw Ethernet frame via the global NIC.
pub fn transmit(frame: &[u8]) {
    if let Some(ref mut nic) = *NIC.lock() {
        unsafe { let _ = nic.transmit(frame); }
    }
}

/// Poll the NIC for received frames and process them, transmitting any replies.
pub fn poll() {
    static POLL_COUNT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
    let count = POLL_COUNT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);

    let mut nic = NIC.lock();
    let nic = match nic.as_mut() {
        Some(n) => n,
        None => return,
    };

    // Log RX queue state at key points: 100, 1000, 5000, 10000, 50000, 100000
    if count == 100 || count == 1000 || count == 5000 || count == 10000
        || count == 50000 || count == 100000
    {
        let (used, avail, addr, flags, last) = unsafe { nic.rx_debug_state() };
        crate::klog!(DEBUG, "NET: poll#{} RX used_idx={} avail_idx={} desc0.addr={:#x} flags={:#x} last={}",
            count, used, avail, addr, flags, last);
    }

    let mut replies: Vec<Vec<u8>> = Vec::new();
    unsafe {
        nic.poll_rx(|frame_data| {
            if let Some(reply) = handle_frame(frame_data) {
                replies.push(reply);
            }
        });
    }
    for reply in replies {
        unsafe { let _ = nic.transmit(&reply); }
    }
    // Check TCP retransmit timers on every poll cycle.
    tcp::poll_retransmit();
}

/// Initialize the NIC with a VirtioNet device. Updates OUR_MAC from the device.
pub fn init_nic(nic: VirtioNet) {
    let mac = nic.mac;
    unsafe { OUR_MAC = mac; }
    *NIC.lock() = Some(nic);
    crate::klog!(INFO, "NET: NIC initialized, MAC {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
        mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    // Log initial RX queue state to confirm setup (used_idx, avail_idx, desc0.addr, desc0.flags)
    if let Some(n) = NIC.lock().as_ref() {
        let (used, avail, addr, flags, last) = unsafe { n.rx_debug_state() };
        crate::klog!(DEBUG, "NET: RX queue init — used_idx={} avail_idx={} desc0.addr={:#x} desc0.flags={:#x} last={}",
            used, avail, addr, flags, last);
    }
}

// ── Ethernet ─────────────────────────────────────────────────────────────────

pub const ETHERTYPE_ARP:  u16 = 0x0806;
pub const ETHERTYPE_IPV4: u16 = 0x0800;
pub const ETHERTYPE_IPV6: u16 = 0x86DD;

#[derive(Debug, Clone)]
pub struct EthFrame<'a> {
    pub dst_mac: [u8; 6],
    pub src_mac: [u8; 6],
    pub ethertype: u16,
    pub payload: &'a [u8],
}

impl<'a> EthFrame<'a> {
    pub fn parse(raw: &'a [u8]) -> Option<Self> {
        if raw.len() < 14 { return None; }
        let dst_mac = raw[0..6].try_into().ok()?;
        let src_mac = raw[6..12].try_into().ok()?;
        let ethertype = u16::from_be_bytes([raw[12], raw[13]]);
        Some(Self { dst_mac, src_mac, ethertype, payload: &raw[14..] })
    }

    /// Build a raw Ethernet frame into `out`.
    pub fn build(dst: [u8; 6], src: [u8; 6], ethertype: u16, payload: &[u8]) -> Vec<u8> {
        let mut frame = Vec::with_capacity(14 + payload.len());
        frame.extend_from_slice(&dst);
        frame.extend_from_slice(&src);
        frame.extend_from_slice(&ethertype.to_be_bytes());
        frame.extend_from_slice(payload);
        frame
    }
}

// ── ARP ───────────────────────────────────────────────────────────────────────

const ARP_REQUEST: u16 = 1;
const ARP_REPLY:   u16 = 2;

#[derive(Debug, Clone)]
pub struct ArpPacket {
    pub op:       u16,
    pub sha: [u8; 6], // sender hardware address
    pub spa: [u8; 4], // sender protocol address (IPv4)
    pub tha: [u8; 6], // target hardware address
    pub tpa: [u8; 4], // target protocol address
}

impl ArpPacket {
    pub fn parse(raw: &[u8]) -> Option<Self> {
        if raw.len() < 28 { return None; }
        // htype=1 (Ethernet), ptype=0x0800, hlen=6, plen=4
        if raw[0..2] != [0, 1] { return None; }
        if raw[2..4] != [0x08, 0x00] { return None; }
        let op = u16::from_be_bytes([raw[6], raw[7]]);
        Some(Self {
            op,
            sha: raw[8..14].try_into().ok()?,
            spa: raw[14..18].try_into().ok()?,
            tha: raw[18..24].try_into().ok()?,
            tpa: raw[24..28].try_into().ok()?,
        })
    }

    pub fn build_reply(
        our_mac: [u8; 6], our_ip: [u8; 4],
        req_mac: [u8; 6], req_ip: [u8; 4],
    ) -> Vec<u8> {
        let mut v = Vec::with_capacity(28);
        v.extend_from_slice(&[0, 1]);         // HTYPE = Ethernet
        v.extend_from_slice(&[0x08, 0x00]);   // PTYPE = IPv4
        v.push(6); v.push(4);                 // HLEN, PLEN
        v.extend_from_slice(&ARP_REPLY.to_be_bytes());
        v.extend_from_slice(&our_mac); v.extend_from_slice(&our_ip);
        v.extend_from_slice(&req_mac); v.extend_from_slice(&req_ip);
        v
    }

    pub fn build_request(
        our_mac: [u8; 6], our_ip: [u8; 4],
        target_ip: [u8; 4],
    ) -> Vec<u8> {
        let mut v = Vec::with_capacity(28);
        v.extend_from_slice(&[0, 1]);          // HTYPE = Ethernet
        v.extend_from_slice(&[0x08, 0x00]);    // PTYPE = IPv4
        v.push(6); v.push(4);                  // HLEN, PLEN
        v.extend_from_slice(&ARP_REQUEST.to_be_bytes());
        v.extend_from_slice(&our_mac); v.extend_from_slice(&our_ip);
        v.extend_from_slice(&[0xFF; 6]); v.extend_from_slice(&target_ip);
        v
    }
}

// ── ARP Cache ─────────────────────────────────────────────────────────────────

/// ARP cache: maps IPv4 address → (MAC address, uptime_ms when learned).
static ARP_CACHE: Mutex<BTreeMap<[u8; 4], ([u8; 6], u64)>> = Mutex::new(BTreeMap::new());

/// Record a mapping in the ARP cache.
pub fn arp_cache_insert(ip: [u8; 4], mac: [u8; 6]) {
    let now = crate::scheduler::uptime_ms();
    ARP_CACHE.lock().insert(ip, (mac, now));
}

/// Look up a MAC address in the ARP cache.
pub fn arp_cache_lookup(ip: &[u8; 4]) -> Option<[u8; 6]> {
    ARP_CACHE.lock().get(ip).map(|&(mac, _)| mac)
}

/// Return a snapshot of the ARP cache for display.
pub fn arp_cache_entries() -> Vec<([u8; 4], [u8; 6], u64)> {
    ARP_CACHE.lock().iter().map(|(&ip, &(mac, ts))| (ip, mac, ts)).collect()
}

/// Send an ARP request for the given IP address.
pub fn arp_request(target_ip: [u8; 4]) {
    let our_mac = unsafe { OUR_MAC };
    let our_ip  = unsafe { OUR_IP };
    crate::klog!(DEBUG, "NET: ARP request for {}.{}.{}.{}",
        target_ip[0], target_ip[1], target_ip[2], target_ip[3]);
    let arp_payload = ArpPacket::build_request(our_mac, our_ip, target_ip);
    let frame = EthFrame::build([0xFF; 6], our_mac, ETHERTYPE_ARP, &arp_payload);
    transmit(&frame);
}

/// Resolve MAC for an IP: check ARP cache, send request if needed, use gateway for non-local.
pub fn arp_resolve_for_ip(ip: &[u8; 4]) -> Option<[u8; 6]> {
    // Check cache first
    if let Some(mac) = arp_cache_lookup(ip) {
        return Some(mac);
    }
    // For non-local IPs, use gateway MAC
    let our_ip = unsafe { OUR_IP };
    let gw = ROUTES.lock().iter()
        .find(|r| r.destination == [0, 0, 0, 0])
        .map(|r| r.gateway);
    let target = if let Some(gw) = gw {
        if ip[0] == our_ip[0] && ip[1] == our_ip[1] && ip[2] == our_ip[2] {
            *ip // same subnet, ARP directly
        } else {
            gw // different subnet, go through gateway
        }
    } else {
        *ip
    };
    if let Some(mac) = arp_cache_lookup(&target) {
        return Some(mac);
    }
    // Send ARP and wait briefly
    arp_request(target);
    for _ in 0..3000 {
        poll();
        if let Some(mac) = arp_cache_lookup(&target) {
            return Some(mac);
        }
        core::hint::spin_loop();
    }
    None
}

// ── IPv4 ──────────────────────────────────────────────────────────────────────

pub const IP_PROTO_ICMP: u8 = 1;
pub const IP_PROTO_UDP:  u8 = 17;
pub const IP_PROTO_TCP:  u8 = 6;

#[derive(Debug, Clone)]
pub struct Ipv4Header {
    pub ihl:      u8,  // header length in 32-bit words
    pub tos:      u8,
    pub total_len: u16,
    pub id:       u16,
    pub flags_frag: u16,
    pub ttl:      u8,
    pub proto:    u8,
    pub checksum: u16,
    pub src:  [u8; 4],
    pub dst:  [u8; 4],
}

impl Ipv4Header {
    pub fn parse(raw: &[u8]) -> Option<Self> {
        if raw.len() < 20 { return None; }
        let version_ihl = raw[0];
        if version_ihl >> 4 != 4 { return None; }
        Some(Self {
            ihl:       (version_ihl & 0xF),
            tos:       raw[1],
            total_len: u16::from_be_bytes([raw[2], raw[3]]),
            id:        u16::from_be_bytes([raw[4], raw[5]]),
            flags_frag:u16::from_be_bytes([raw[6], raw[7]]),
            ttl:       raw[8],
            proto:     raw[9],
            checksum:  u16::from_be_bytes([raw[10], raw[11]]),
            src:       raw[12..16].try_into().ok()?,
            dst:       raw[16..20].try_into().ok()?,
        })
    }

    pub fn header_len(&self) -> usize { self.ihl as usize * 4 }

    /// Build a minimal IPv4 header (no options).
    pub fn build(proto: u8, src: [u8; 4], dst: [u8; 4], payload_len: usize) -> Vec<u8> {
        let total = 20 + payload_len;
        let mut h = Vec::with_capacity(20);
        h.push(0x45); // version=4, IHL=5
        h.push(0);    // DSCP/ECN
        h.extend_from_slice(&(total as u16).to_be_bytes());
        h.extend_from_slice(&[0, 1]); // ID
        h.extend_from_slice(&[0x40, 0]); // Don't Fragment, frag offset=0
        h.push(64);   // TTL
        h.push(proto);
        h.extend_from_slice(&[0, 0]); // checksum placeholder
        h.extend_from_slice(&src);
        h.extend_from_slice(&dst);
        // Calculate checksum
        let csum = ipv4_checksum(&h);
        h[10] = (csum >> 8) as u8;
        h[11] = (csum & 0xFF) as u8;
        h
    }
}

fn ipv4_checksum(header: &[u8]) -> u16 {
    let mut sum = 0u32;
    for chunk in header.chunks(2) {
        let word = if chunk.len() == 2 {
            u16::from_be_bytes([chunk[0], chunk[1]])
        } else {
            (chunk[0] as u16) << 8
        };
        sum += word as u32;
    }
    while sum >> 16 != 0 { sum = (sum & 0xFFFF) + (sum >> 16); }
    !(sum as u16)
}

// ── ICMP ─────────────────────────────────────────────────────────────────────

pub fn build_icmp_echo_reply(id: u16, seq: u16, data: &[u8]) -> Vec<u8> {
    let mut pkt = Vec::with_capacity(8 + data.len());
    pkt.push(0);  // type = echo reply
    pkt.push(0);  // code
    pkt.extend_from_slice(&[0, 0]); // checksum placeholder
    pkt.extend_from_slice(&id.to_be_bytes());
    pkt.extend_from_slice(&seq.to_be_bytes());
    pkt.extend_from_slice(data);
    let csum = ipv4_checksum(&pkt);
    pkt[2] = (csum >> 8) as u8;
    pkt[3] = (csum & 0xFF) as u8;
    pkt
}

/// Build an ICMP echo request packet.
pub fn build_icmp_echo_request(id: u16, seq: u16, data: &[u8]) -> Vec<u8> {
    let mut pkt = Vec::with_capacity(8 + data.len());
    pkt.push(8);  // type = echo request
    pkt.push(0);  // code
    pkt.extend_from_slice(&[0, 0]); // checksum placeholder
    pkt.extend_from_slice(&id.to_be_bytes());
    pkt.extend_from_slice(&seq.to_be_bytes());
    pkt.extend_from_slice(data);
    let csum = ipv4_checksum(&pkt);
    pkt[2] = (csum >> 8) as u8;
    pkt[3] = (csum & 0xFF) as u8;
    pkt
}

// ── Ping infrastructure ──────────────────────────────────────────────────────

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Pending ICMP echo reply state — set by ping command, cleared by receive path.
static PING_PENDING: AtomicBool = AtomicBool::new(false);
static PING_REPLY_TIME: AtomicU64 = AtomicU64::new(0);
static PING_ID: Mutex<u16> = Mutex::new(0);
static PING_SEQ: Mutex<u16> = Mutex::new(0);

/// Send an ICMP echo request to `dst_ip` and wait up to `timeout_ms` for a reply.
/// Returns the round-trip time in ms, or None on timeout.
pub fn ping(dst_ip: [u8; 4], id: u16, seq: u16, timeout_ms: u64) -> Option<u64> {
    let our_mac = unsafe { OUR_MAC };
    let our_ip  = unsafe { OUR_IP };

    // Set up pending state
    *PING_ID.lock() = id;
    *PING_SEQ.lock() = seq;
    PING_REPLY_TIME.store(0, Ordering::Release);
    PING_PENDING.store(true, Ordering::Release);

    // Build and send the ICMP echo request
    let icmp = build_icmp_echo_request(id, seq, b"NodeAI ping");
    let ip_hdr = Ipv4Header::build(IP_PROTO_ICMP, our_ip, dst_ip, icmp.len());
    let mut ip_packet = ip_hdr;
    ip_packet.extend_from_slice(&icmp);

    // Resolve destination MAC: use gateway (10.0.2.2) for non-local, or ARP cache
    let dst_mac = if dst_ip[0..3] == our_ip[0..3] {
        arp_cache_lookup(&dst_ip).unwrap_or([0xFF; 6])
    } else {
        // Default gateway MAC — send ARP for gateway if not cached
        let gw: [u8; 4] = [10, 0, 2, 2];
        arp_cache_lookup(&gw).unwrap_or([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])
    };

    let frame = EthFrame::build(dst_mac, our_mac, ETHERTYPE_IPV4, &ip_packet);
    let send_time = crate::scheduler::uptime_ms();
    transmit(&frame);

    // Poll for reply
    let deadline = send_time + timeout_ms;
    while crate::scheduler::uptime_ms() < deadline {
        poll();
        let reply_time = PING_REPLY_TIME.load(Ordering::Acquire);
        if reply_time > 0 {
            PING_PENDING.store(false, Ordering::Release);
            return Some(reply_time.saturating_sub(send_time));
        }
        core::hint::spin_loop();
    }
    PING_PENDING.store(false, Ordering::Release);
    None
}

// ── DNS Resolver ─────────────────────────────────────────────────────────────

/// Default DNS server (QEMU user-mode networking)
pub static DNS_SERVER: Mutex<[u8; 4]> = Mutex::new([10, 0, 2, 3]);
/// Static hosts table: hostname → IP
static HOSTS: Mutex<BTreeMap<String, [u8; 4]>> = Mutex::new(BTreeMap::new());

/// DNS cache entry with TTL-based expiry.
struct DnsCacheEntry {
    ip:      [u8; 4],
    expires: u64,   // uptime_ms at which this entry expires
}

/// DNS cache: hostname → (ip, expiry).
static DNS_CACHE: Mutex<BTreeMap<String, DnsCacheEntry>> = Mutex::new(BTreeMap::new());

/// Insert a DNS cache entry with TTL (seconds).
fn dns_cache_insert(name: &str, ip: [u8; 4], ttl_secs: u32) {
    let ttl = if ttl_secs == 0 { 60 } else { ttl_secs }; // minimum 60s
    let expires = crate::scheduler::uptime_ms() + (ttl as u64) * 1000;
    DNS_CACHE.lock().insert(String::from(name), DnsCacheEntry { ip, expires });
}

/// Look up in DNS cache; returns None if expired or absent.
fn dns_cache_lookup(name: &str) -> Option<[u8; 4]> {
    let mut cache = DNS_CACHE.lock();
    if let Some(entry) = cache.get(name) {
        if crate::scheduler::uptime_ms() < entry.expires {
            return Some(entry.ip);
        }
        // expired — remove
        cache.remove(name);
    }
    None
}

/// Flush the entire DNS cache.
pub fn dns_cache_flush() {
    DNS_CACHE.lock().clear();
}

/// Get a snapshot of the DNS cache (for shell display).
pub fn dns_cache_entries() -> Vec<(String, [u8; 4], u64)> {
    let now = crate::scheduler::uptime_ms();
    DNS_CACHE.lock().iter()
        .filter(|(_, e)| e.expires > now)
        .map(|(name, e)| (name.clone(), e.ip, (e.expires - now) / 1000))
        .collect()
}

/// Initialize /etc/hosts with common entries.
pub fn init_hosts() {
    let mut hosts = HOSTS.lock();
    hosts.insert(String::from("localhost"), [127, 0, 0, 1]);
    hosts.insert(String::from("gateway"), [10, 0, 2, 2]);
}

/// Look up a hostname: check /etc/hosts first, then DNS cache, then DNS query.
pub fn resolve(hostname: &str) -> Option<[u8; 4]> {
    // Try parsing as IP address first
    if let Some(ip) = parse_ipv4(hostname) {
        return Some(ip);
    }
    // Check hosts table
    if let Some(&ip) = HOSTS.lock().get(hostname) {
        return Some(ip);
    }
    // Check DNS cache
    if let Some(ip) = dns_cache_lookup(hostname) {
        return Some(ip);
    }
    // DNS query — cache the result
    if let Some(ip) = dns_query(hostname) {
        dns_cache_insert(hostname, ip, 300); // default 5 min TTL
        return Some(ip);
    }
    None
}

/// Parse a dotted-decimal IPv4 string.
pub fn parse_ipv4(s: &str) -> Option<[u8; 4]> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 4 { return None; }
    let mut ip = [0u8; 4];
    for (i, part) in parts.iter().enumerate() {
        ip[i] = part.parse::<u8>().ok()?;
    }
    Some(ip)
}

/// Format an IPv4 address as a string.
pub fn format_ipv4(ip: &[u8; 4]) -> String {
    alloc::format!("{}.{}.{}.{}", ip[0], ip[1], ip[2], ip[3])
}

/// Format a MAC address as a string.
pub fn format_mac(mac: &[u8; 6]) -> String {
    alloc::format!("{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
        mac[0], mac[1], mac[2], mac[3], mac[4], mac[5])
}

/// DNS query state
static DNS_REPLY: Mutex<Option<[u8; 4]>> = Mutex::new(None);
static DNS_PENDING: AtomicBool = AtomicBool::new(false);

/// Perform a simple DNS A-record query via UDP port 53.
fn dns_query(hostname: &str) -> Option<[u8; 4]> {
    let our_mac = unsafe { OUR_MAC };
    let our_ip  = unsafe { OUR_IP };
    let dns_ip  = *DNS_SERVER.lock();

    crate::klog!(INFO, "NET: DNS query for '{}'", hostname);

    // Build DNS query packet
    let mut dns = Vec::with_capacity(512);
    let txid: u16 = (crate::scheduler::uptime_ms() & 0xFFFF) as u16;
    dns.extend_from_slice(&txid.to_be_bytes()); // Transaction ID
    dns.extend_from_slice(&[0x01, 0x00]);       // Flags: standard query, recursion desired
    dns.extend_from_slice(&[0x00, 0x01]);       // Questions: 1
    dns.extend_from_slice(&[0x00, 0x00]);       // Answers: 0
    dns.extend_from_slice(&[0x00, 0x00]);       // Authority: 0
    dns.extend_from_slice(&[0x00, 0x00]);       // Additional: 0

    // Encode hostname as DNS labels
    for label in hostname.split('.') {
        if label.len() > 63 { return None; }
        dns.push(label.len() as u8);
        dns.extend_from_slice(label.as_bytes());
    }
    dns.push(0); // Root label

    dns.extend_from_slice(&[0x00, 0x01]); // QTYPE: A
    dns.extend_from_slice(&[0x00, 0x01]); // QCLASS: IN

    // Wrap in UDP
    let udp = UdpDatagram::build(12345, 53, &dns);
    let ip = Ipv4Header::build(IP_PROTO_UDP, our_ip, dns_ip, udp.len());
    let mut pkt = ip;
    pkt.extend_from_slice(&udp);

    // Resolve DNS server MAC
    let dst_mac = arp_cache_lookup(&dns_ip).unwrap_or_else(|| {
        arp_request(dns_ip);
        // Brief spin waiting for ARP reply
        for _ in 0..5000 {
            poll();
            if let Some(mac) = arp_cache_lookup(&dns_ip) {
                return mac;
            }
            core::hint::spin_loop();
        }
        crate::klog!(WARN, "NET: ARP timeout for DNS server {}.{}.{}.{} — using broadcast",
            dns_ip[0], dns_ip[1], dns_ip[2], dns_ip[3]);
        [0xFF; 6]
    });

    *DNS_REPLY.lock() = None;
    DNS_PENDING.store(true, Ordering::Release);

    let frame = EthFrame::build(dst_mac, our_mac, ETHERTYPE_IPV4, &pkt);
    transmit(&frame);

    // Wait for DNS reply (up to 3 seconds)
    let deadline = crate::scheduler::uptime_ms() + 3000;
    while crate::scheduler::uptime_ms() < deadline {
        poll();
        if let Some(ip) = *DNS_REPLY.lock() {
            DNS_PENDING.store(false, Ordering::Release);
            crate::klog!(INFO, "NET: DNS '{}' → {}.{}.{}.{}",
                hostname, ip[0], ip[1], ip[2], ip[3]);
            return Some(ip);
        }
        core::hint::spin_loop();
    }
    DNS_PENDING.store(false, Ordering::Release);
    crate::klog!(WARN, "NET: DNS timeout for '{}'", hostname);
    None
}

/// Handle an incoming UDP packet — checks for DNS replies.
fn handle_udp(_src_ip: [u8; 4], payload: &[u8]) -> Option<Vec<u8>> {
    let udp = UdpDatagram::parse(payload)?;

    // DNS reply (source port 53)
    if udp.src_port == 53 && DNS_PENDING.load(Ordering::Acquire) {
        crate::klog!(DEBUG, "NET: received DNS reply ({} bytes)", udp.payload.len());
        if let Some((ip, _ttl)) = parse_dns_reply(udp.payload) {
            *DNS_REPLY.lock() = Some(ip);
        }
    }

    // DHCP reply (server port 67 → client port 68)
    if udp.src_port == 67 && udp.dst_port == 68 && DHCP_PENDING.load(Ordering::Acquire) {
        if let Some(offer) = parse_dhcp_options(udp.payload) {
            *DHCP_REPLY.lock() = Some(offer);
        }
    }

    None
}

/// Parse a DNS reply for the first A record. Returns (IP, TTL_seconds).
fn parse_dns_reply(data: &[u8]) -> Option<([u8; 4], u32)> {
    if data.len() < 12 { return None; }
    let flags = u16::from_be_bytes([data[2], data[3]]);
    if flags & 0x8000 == 0 { return None; } // Not a response
    let ancount = u16::from_be_bytes([data[6], data[7]]);
    if ancount == 0 { return None; }

    // Skip question section
    let mut pos = 12;
    // Skip QNAME
    while pos < data.len() {
        let len = data[pos] as usize;
        if len == 0 { pos += 1; break; }
        if len >= 0xC0 { pos += 2; break; } // compression pointer
        pos += 1 + len;
    }
    pos += 4; // Skip QTYPE + QCLASS

    // Parse answer records
    for _ in 0..ancount {
        if pos + 12 > data.len() { break; }
        // Skip NAME (may be compression pointer)
        if data[pos] >= 0xC0 {
            pos += 2;
        } else {
            while pos < data.len() {
                let len = data[pos] as usize;
                if len == 0 { pos += 1; break; }
                pos += 1 + len;
            }
        }
        if pos + 10 > data.len() { break; }
        let rtype = u16::from_be_bytes([data[pos], data[pos + 1]]);
        let ttl   = u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
        let rdlength = u16::from_be_bytes([data[pos + 8], data[pos + 9]]) as usize;
        pos += 10;
        if rtype == 1 && rdlength == 4 && pos + 4 <= data.len() {
            return Some(([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]], ttl));
        }
        pos += rdlength;
    }
    None
}

// ── DHCP client ──────────────────────────────────────────────────────────────

/// DHCP message types
const DHCP_DISCOVER: u8 = 1;
const DHCP_OFFER:    u8 = 2;
const DHCP_REQUEST:  u8 = 3;
const DHCP_ACK:      u8 = 5;

/// State for the DHCP client reply.
static DHCP_REPLY: Mutex<Option<DhcpOffer>> = Mutex::new(None);
static DHCP_PENDING: AtomicBool = AtomicBool::new(false);

/// Parsed DHCP offer/ack.
#[derive(Clone)]
struct DhcpOffer {
    your_ip:  [u8; 4],
    server_ip: [u8; 4],
    subnet:   [u8; 4],
    gateway:  [u8; 4],
    dns:      [u8; 4],
    lease:    u32,
}

/// Build a DHCP DISCOVER or REQUEST message.
fn build_dhcp_msg(msg_type: u8, our_mac: [u8; 6], requested_ip: Option<[u8; 4]>) -> Vec<u8> {
    let mut pkt = Vec::with_capacity(300);
    pkt.push(1);       // op = BOOTREQUEST
    pkt.push(1);       // htype = Ethernet
    pkt.push(6);       // hlen = 6
    pkt.push(0);       // hops
    let xid: u32 = 0x4E4F4445; // "NODE"
    pkt.extend_from_slice(&xid.to_be_bytes()); // Transaction ID
    pkt.extend_from_slice(&[0, 0]); // secs
    pkt.extend_from_slice(&[0x80, 0x00]); // flags: broadcast
    pkt.extend_from_slice(&[0; 4]);  // ciaddr
    pkt.extend_from_slice(&[0; 4]);  // yiaddr
    pkt.extend_from_slice(&[0; 4]);  // siaddr
    pkt.extend_from_slice(&[0; 4]);  // giaddr
    pkt.extend_from_slice(&our_mac); // chaddr (16 bytes)
    pkt.extend_from_slice(&[0; 10]); // padding to 16 bytes
    pkt.extend_from_slice(&[0; 64]); // sname
    pkt.extend_from_slice(&[0; 128]); // file
    // Magic cookie
    pkt.extend_from_slice(&[99, 130, 83, 99]);
    // Option 53: DHCP Message Type
    pkt.extend_from_slice(&[53, 1, msg_type]);
    // Option 50: Requested IP (for REQUEST)
    if let Some(ip) = requested_ip {
        pkt.extend_from_slice(&[50, 4]);
        pkt.extend_from_slice(&ip);
    }
    // Option 55: Parameter Request List (subnet, router, DNS, lease)
    pkt.extend_from_slice(&[55, 4, 1, 3, 6, 51]);
    // End
    pkt.push(255);
    // Pad to minimum 300 bytes
    while pkt.len() < 300 {
        pkt.push(0);
    }
    pkt
}

/// Parse DHCP options from a reply.
fn parse_dhcp_options(data: &[u8]) -> Option<DhcpOffer> {
    if data.len() < 240 { return None; }
    // Verify magic cookie
    if data[236..240] != [99, 130, 83, 99] { return None; }

    let your_ip = [data[16], data[17], data[18], data[19]];
    let server_ip = [data[20], data[21], data[22], data[23]];

    let mut subnet = [255, 255, 255, 0];
    let mut gateway = [0u8; 4];
    let mut dns = [0u8; 4];
    let mut lease: u32 = 86400;
    let mut msg_type: u8 = 0;

    let mut pos = 240;
    while pos < data.len() {
        let opt = data[pos];
        if opt == 255 { break; }  // End
        if opt == 0 { pos += 1; continue; } // Padding
        if pos + 1 >= data.len() { break; }
        let len = data[pos + 1] as usize;
        let val = &data[pos + 2..core::cmp::min(pos + 2 + len, data.len())];
        match opt {
            53 => { if !val.is_empty() { msg_type = val[0]; } }
            1  => { if val.len() >= 4 { subnet = [val[0], val[1], val[2], val[3]]; } }
            3  => { if val.len() >= 4 { gateway = [val[0], val[1], val[2], val[3]]; } }
            6  => { if val.len() >= 4 { dns = [val[0], val[1], val[2], val[3]]; } }
            51 => { if val.len() >= 4 { lease = u32::from_be_bytes([val[0], val[1], val[2], val[3]]); } }
            _  => {}
        }
        pos += 2 + len;
    }

    if msg_type == DHCP_OFFER || msg_type == DHCP_ACK {
        Some(DhcpOffer { your_ip, server_ip, subnet, gateway, dns, lease })
    } else {
        None
    }
}

/// Perform DHCP: DISCOVER → OFFER → REQUEST → ACK.
/// Returns true on success.
pub fn dhcp_request() -> bool {
    let our_mac = unsafe { OUR_MAC };

    // 1) Send DISCOVER (broadcast)
    let discover = build_dhcp_msg(DHCP_DISCOVER, our_mac, None);
    let udp = UdpDatagram::build(68, 67, &discover);
    let ip_hdr = Ipv4Header::build(IP_PROTO_UDP, [0, 0, 0, 0], [255, 255, 255, 255], udp.len());
    let mut pkt = ip_hdr;
    pkt.extend_from_slice(&udp);
    let frame = EthFrame::build([0xFF; 6], our_mac, ETHERTYPE_IPV4, &pkt);

    *DHCP_REPLY.lock() = None;
    DHCP_PENDING.store(true, Ordering::Release);
    transmit(&frame);

    // 2) Wait for OFFER (up to 5 seconds)
    let deadline = crate::scheduler::uptime_ms() + 5000;
    let offer = loop {
        poll();
        if let Some(o) = DHCP_REPLY.lock().take() {
            break o;
        }
        if crate::scheduler::uptime_ms() >= deadline {
            DHCP_PENDING.store(false, Ordering::Release);
            crate::klog!(WARN, "DHCP: no OFFER received");
            return false;
        }
        core::hint::spin_loop();
    };

    // 3) Send REQUEST for the offered IP
    let request = build_dhcp_msg(DHCP_REQUEST, our_mac, Some(offer.your_ip));
    let udp = UdpDatagram::build(68, 67, &request);
    let ip_hdr = Ipv4Header::build(IP_PROTO_UDP, [0, 0, 0, 0], [255, 255, 255, 255], udp.len());
    let mut pkt = ip_hdr;
    pkt.extend_from_slice(&udp);
    let frame = EthFrame::build([0xFF; 6], our_mac, ETHERTYPE_IPV4, &pkt);

    *DHCP_REPLY.lock() = None;
    transmit(&frame);

    // 4) Wait for ACK (up to 5 seconds)
    let deadline = crate::scheduler::uptime_ms() + 5000;
    let ack = loop {
        poll();
        if let Some(a) = DHCP_REPLY.lock().take() {
            break a;
        }
        if crate::scheduler::uptime_ms() >= deadline {
            DHCP_PENDING.store(false, Ordering::Release);
            crate::klog!(WARN, "DHCP: no ACK received");
            return false;
        }
        core::hint::spin_loop();
    };
    DHCP_PENDING.store(false, Ordering::Release);

    // 5) Apply configuration
    configure_static_ip(ack.your_ip, ack.subnet, ack.gateway);
    if ack.dns != [0, 0, 0, 0] {
        *DNS_SERVER.lock() = ack.dns;
    }
    crate::klog!(INFO, "DHCP: acquired {}.{}.{}.{} lease={}s",
        ack.your_ip[0], ack.your_ip[1], ack.your_ip[2], ack.your_ip[3], ack.lease);
    true
}

/// DHCP on the WiFi interface (AR9271 USB dongle).
/// Sends DISCOVER/REQUEST over the WiFi TX path and stores the acquired IP in
/// `wifi::set_ip()` — leaves the VirtIO NIC's OUR_IP unchanged.
pub fn dhcp_request_wifi(slot: u8) -> bool {
    let our_mac = crate::wifi::wifi_mac();
    if our_mac == [0u8; 6] {
        crate::klog!(WARN, "DHCP-WiFi: no WiFi MAC — adapter not attached");
        return false;
    }

    // Build and send DISCOVER over WiFi (unencrypted — pre-key exchange)
    let discover = build_dhcp_msg(DHCP_DISCOVER, our_mac, None);
    let udp = UdpDatagram::build(68, 67, &discover);
    let ip_hdr = Ipv4Header::build(IP_PROTO_UDP, [0,0,0,0], [255,255,255,255], udp.len());
    let mut pkt = ip_hdr;
    pkt.extend_from_slice(&udp);
    let frame = EthFrame::build([0xFF; 6], our_mac, ETHERTYPE_IPV4, &pkt);

    *DHCP_REPLY.lock() = None;
    DHCP_PENDING.store(true, Ordering::Release);
    // Transmit via WiFi open path (no CCMP — DHCP runs before data keys installed)
    crate::wifi::wifi_tx_open_pub(slot, frame);

    // Wait for OFFER (up to 5 seconds) — poll WiFi RX directly
    let deadline = crate::scheduler::uptime_ms() + 5000;
    let offer = loop {
        crate::wifi::poll();
        if let Some(o) = DHCP_REPLY.lock().take() { break o; }
        if crate::scheduler::uptime_ms() >= deadline {
            DHCP_PENDING.store(false, Ordering::Release);
            crate::klog!(WARN, "DHCP-WiFi: no OFFER received");
            return false;
        }
        core::hint::spin_loop();
    };

    // Send REQUEST
    let request = build_dhcp_msg(DHCP_REQUEST, our_mac, Some(offer.your_ip));
    let udp = UdpDatagram::build(68, 67, &request);
    let ip_hdr = Ipv4Header::build(IP_PROTO_UDP, [0,0,0,0], [255,255,255,255], udp.len());
    let mut pkt = ip_hdr;
    pkt.extend_from_slice(&udp);
    let frame = EthFrame::build([0xFF; 6], our_mac, ETHERTYPE_IPV4, &pkt);

    *DHCP_REPLY.lock() = None;
    crate::wifi::wifi_tx_open_pub(slot, frame);

    // Wait for ACK
    let deadline = crate::scheduler::uptime_ms() + 5000;
    let ack = loop {
        crate::wifi::poll();
        if let Some(a) = DHCP_REPLY.lock().take() { break a; }
        if crate::scheduler::uptime_ms() >= deadline {
            DHCP_PENDING.store(false, Ordering::Release);
            crate::klog!(WARN, "DHCP-WiFi: no ACK received");
            return false;
        }
        core::hint::spin_loop();
    };
    DHCP_PENDING.store(false, Ordering::Release);

    crate::wifi::set_ip(ack.your_ip);
    crate::klog!(INFO, "DHCP-WiFi: acquired {}.{}.{}.{} lease={}s",
        ack.your_ip[0], ack.your_ip[1], ack.your_ip[2], ack.your_ip[3], ack.lease);
    true
}

// ── Routing table ────────────────────────────────────────────────────────────

/// Simple routing table entry.
#[derive(Debug, Clone)]
pub struct RouteEntry {
    pub destination: [u8; 4],
    pub netmask:     [u8; 4],
    pub gateway:     [u8; 4],
    pub iface:       &'static str,
}

/// Static routing table.
static ROUTES: Mutex<Vec<RouteEntry>> = Mutex::new(Vec::new());

/// Initialize default routes.
pub fn init_routes() {
    let mut routes = ROUTES.lock();
    // Default route via gateway
    routes.push(RouteEntry {
        destination: [0, 0, 0, 0],
        netmask:     [0, 0, 0, 0],
        gateway:     [10, 0, 2, 2],
        iface:       "eth0",
    });
    // Local subnet
    routes.push(RouteEntry {
        destination: [10, 0, 2, 0],
        netmask:     [255, 255, 255, 0],
        gateway:     [0, 0, 0, 0],
        iface:       "eth0",
    });
}

/// Get a snapshot of the routing table.
pub fn route_entries() -> Vec<RouteEntry> {
    ROUTES.lock().clone()
}

// ── Static IP configuration (/etc/network/interfaces) ────────────────────────

/// Apply a static IP configuration: sets OUR_IP, default gateway, and netmask.
pub fn configure_static_ip(ip: [u8; 4], netmask: [u8; 4], gateway: [u8; 4]) {
    unsafe { OUR_IP = ip; }
    // Rebuild routes
    let mut routes = ROUTES.lock();
    routes.clear();
    // Default route
    routes.push(RouteEntry {
        destination: [0, 0, 0, 0],
        netmask:     [0, 0, 0, 0],
        gateway,
        iface:       "eth0",
    });
    // Local subnet
    let subnet = [ip[0] & netmask[0], ip[1] & netmask[1],
                  ip[2] & netmask[2], ip[3] & netmask[3]];
    routes.push(RouteEntry {
        destination: subnet,
        netmask,
        gateway: [0, 0, 0, 0],
        iface: "eth0",
    });
    crate::klog!(INFO, "NET: static IP {}.{}.{}.{} mask {}.{}.{}.{} gw {}.{}.{}.{}",
        ip[0], ip[1], ip[2], ip[3],
        netmask[0], netmask[1], netmask[2], netmask[3],
        gateway[0], gateway[1], gateway[2], gateway[3]);
}

/// Load static IP from /etc/network/interfaces VFS file.
/// Format:
///   address <ip>
///   netmask <mask>
///   gateway <gw>
///   dns-nameservers <dns>
pub fn load_network_config() {
    let data = match vfs_read("/etc/network/interfaces") {
        Some(d) => d,
        None => return, // no config file, keep defaults
    };
    let text = core::str::from_utf8(&data).unwrap_or("");
    let mut ip = None;
    let mut mask = None;
    let mut gw = None;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() { continue; }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 { continue; }
        match parts[0] {
            "address" => ip = parse_ipv4(parts[1]),
            "netmask" => mask = parse_ipv4(parts[1]),
            "gateway" => gw = parse_ipv4(parts[1]),
            "dns-nameservers" => {
                if let Some(dns) = parse_ipv4(parts[1]) {
                    *DNS_SERVER.lock() = dns;
                }
            }
            _ => {}
        }
    }
    if let (Some(ip), Some(mask), Some(gw)) = (ip, mask, gw) {
        configure_static_ip(ip, mask, gw);
    }
}

/// Write a static IP config to /etc/network/interfaces.
pub fn save_network_config(ip: [u8; 4], mask: [u8; 4], gw: [u8; 4], dns: [u8; 4]) {
    let content = alloc::format!(
        "auto eth0\niface eth0 inet static\n  address {}.{}.{}.{}\n  netmask {}.{}.{}.{}\n  gateway {}.{}.{}.{}\n  dns-nameservers {}.{}.{}.{}\n",
        ip[0], ip[1], ip[2], ip[3],
        mask[0], mask[1], mask[2], mask[3],
        gw[0], gw[1], gw[2], gw[3],
        dns[0], dns[1], dns[2], dns[3],
    );
    vfs_write("/etc/network/interfaces", content.as_bytes());
}

/// Read a file from VFS by absolute path.
fn vfs_read(path: &str) -> Option<Vec<u8>> {
    let node = crate::vfs::lookup(path).ok()?;
    let mut fh = node.open().ok()?;
    let mut data = Vec::new();
    let mut buf = [0u8; 512];
    loop {
        match fh.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => data.extend_from_slice(&buf[..n]),
            Err(_) => break,
        }
    }
    Some(data)
}

/// Write a file to VFS by absolute path (create or overwrite).
fn vfs_write(path: &str, data: &[u8]) {
    // Split path into parent directory and filename
    let (parent_path, filename) = match path.rfind('/') {
        Some(pos) if pos > 0 => (&path[..pos], &path[pos + 1..]),
        _ => ("/", &path[1..]),
    };
    // Delete old file if it exists
    if let Ok(parent) = crate::vfs::lookup(parent_path) {
        let _ = parent.unlink(filename);
    }
    // Create new file
    if let Ok(parent) = crate::vfs::lookup(parent_path) {
        if let Ok(node) = parent.create_file(filename) {
            if let Ok(mut fh) = node.open() {
                let _ = fh.write(data);
            }
        }
    }
}

// ── Network statistics ───────────────────────────────────────────────────────

use core::sync::atomic::AtomicU64 as StatsAtomicU64;

static TX_PACKETS: StatsAtomicU64 = StatsAtomicU64::new(0);
static RX_PACKETS: StatsAtomicU64 = StatsAtomicU64::new(0);
static TX_BYTES:   StatsAtomicU64 = StatsAtomicU64::new(0);
static RX_BYTES:   StatsAtomicU64 = StatsAtomicU64::new(0);

/// Get network interface statistics.
pub fn iface_stats() -> (u64, u64, u64, u64) {
    (
        TX_PACKETS.load(Ordering::Relaxed),
        RX_PACKETS.load(Ordering::Relaxed),
        TX_BYTES.load(Ordering::Relaxed),
        RX_BYTES.load(Ordering::Relaxed),
    )
}

// ── UDP ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct UdpDatagram<'a> {
    pub src_port: u16,
    pub dst_port: u16,
    pub payload:  &'a [u8],
}

impl<'a> UdpDatagram<'a> {
    pub fn parse(raw: &'a [u8]) -> Option<Self> {
        if raw.len() < 8 { return None; }
        let src_port = u16::from_be_bytes([raw[0], raw[1]]);
        let dst_port = u16::from_be_bytes([raw[2], raw[3]]);
        let length   = u16::from_be_bytes([raw[4], raw[5]]) as usize;
        if length < 8 || length > raw.len() { return None; }
        Some(Self { src_port, dst_port, payload: &raw[8..length] })
    }

    pub fn build(src: u16, dst: u16, payload: &[u8]) -> Vec<u8> {
        let len = 8 + payload.len();
        let mut p = Vec::with_capacity(len);
        p.extend_from_slice(&src.to_be_bytes());
        p.extend_from_slice(&dst.to_be_bytes());
        p.extend_from_slice(&(len as u16).to_be_bytes());
        p.extend_from_slice(&[0, 0]); // checksum (optional for IPv4)
        p.extend_from_slice(payload);
        p
    }
}

// ── Packet dispatch ───────────────────────────────────────────────────────────

/// Our IP configuration (DHCP available via `dhclient` shell command; static fallback).
pub static mut OUR_IP:  [u8; 4] = [10, 0, 2, 15];   // QEMU default
pub static mut OUR_MAC: [u8; 6] = [0x52, 0x54, 0x00, 0x12, 0x34, 0x56];

/// Safe read of the current local IPv4 address.
pub fn our_ip() -> [u8; 4] { unsafe { OUR_IP } }

/// Handle a received raw Ethernet frame.
/// Returns an optional reply frame to transmit.
pub fn handle_frame(raw: &[u8]) -> Option<Vec<u8>> {
    let eth = EthFrame::parse(raw)?;
    crate::klog!(DEBUG, "NET: rx frame ethertype=0x{:04x} len={}", eth.ethertype, raw.len());
    match eth.ethertype {
        ETHERTYPE_ARP => handle_arp(eth.src_mac, eth.payload),
        ETHERTYPE_IPV4 => handle_ipv4(eth.src_mac, eth.payload),
        _ => None,
    }
}

fn handle_arp(src_mac: [u8; 6], payload: &[u8]) -> Option<Vec<u8>> {
    let arp = ArpPacket::parse(payload)?;
    // Always cache the sender's IP→MAC mapping
    arp_cache_insert(arp.spa, arp.sha);

    if arp.op == ARP_REPLY {
        crate::klog!(DEBUG, "NET: ARP reply {}.{}.{}.{} → {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            arp.spa[0], arp.spa[1], arp.spa[2], arp.spa[3],
            arp.sha[0], arp.sha[1], arp.sha[2], arp.sha[3], arp.sha[4], arp.sha[5]);
        // Just cached it, no reply needed
        return None;
    }
    if arp.op != ARP_REQUEST { return None; }
    let our_ip  = unsafe { OUR_IP };
    let our_mac = unsafe { OUR_MAC };
    let wifi_ip = crate::wifi::get_ip();
    if arp.tpa != our_ip && arp.tpa != wifi_ip { return None; }
    // Reply with the MAC that matches the queried IP
    let reply_mac = if arp.tpa == wifi_ip && wifi_ip != [0,0,0,0] {
        crate::wifi::wifi_mac()
    } else {
        our_mac
    };
    let reply_ip  = arp.tpa;

    let reply_payload = ArpPacket::build_reply(reply_mac, reply_ip, src_mac, arp.spa);
    Some(EthFrame::build(src_mac, reply_mac, ETHERTYPE_ARP, &reply_payload))
}

fn handle_ipv4(src_mac: [u8; 6], payload: &[u8]) -> Option<Vec<u8>> {
    let iph = Ipv4Header::parse(payload)?;
    let our_ip  = unsafe { OUR_IP };
    let wifi_ip = crate::wifi::get_ip();
    // Accept packets to either interface IP, or broadcast (needed for DHCP)
    if iph.dst != our_ip && iph.dst != wifi_ip && iph.dst != [255, 255, 255, 255] { return None; }

    // Cache the sender's MAC from this IP packet
    arp_cache_insert(iph.src, src_mac);
    RX_PACKETS.fetch_add(1, Ordering::Relaxed);
    RX_BYTES.fetch_add(payload.len() as u64, Ordering::Relaxed);

    let ihl = iph.header_len();
    // Use total_len to strip Ethernet padding — short frames are padded to 60 bytes minimum
    let total = iph.total_len as usize;
    let ip_payload = payload.get(ihl..total)?;

    match iph.proto {
        IP_PROTO_ICMP => {
            if ip_payload.len() < 8 { return None; }
            let icmp_type = ip_payload[0];

            // Echo reply (type 0) — for our ping client
            if icmp_type == 0 && PING_PENDING.load(Ordering::Acquire) {
                let id  = u16::from_be_bytes([ip_payload[4], ip_payload[5]]);
                let seq = u16::from_be_bytes([ip_payload[6], ip_payload[7]]);
                if id == *PING_ID.lock() && seq == *PING_SEQ.lock() {
                    PING_REPLY_TIME.store(
                        crate::scheduler::uptime_ms(),
                        Ordering::Release,
                    );
                }
                return None;
            }

            // Echo request (type 8) — reply to remote ping
            if icmp_type != 8 { return None; }
            let id  = u16::from_be_bytes([ip_payload[4], ip_payload[5]]);
            let seq = u16::from_be_bytes([ip_payload[6], ip_payload[7]]);
            let data = &ip_payload[8..];
            let icmp_reply = build_icmp_echo_reply(id, seq, data);
            let our_mac = unsafe { OUR_MAC };
            let ip_part = Ipv4Header::build(IP_PROTO_ICMP, our_ip, iph.src, icmp_reply.len());
            let mut ip_packet = ip_part;
            ip_packet.extend_from_slice(&icmp_reply);
            Some(EthFrame::build(src_mac, our_mac, ETHERTYPE_IPV4, &ip_packet))
        }
        IP_PROTO_UDP => handle_udp(iph.src, ip_payload),
        IP_PROTO_TCP => tcp::handle_tcp_segment(src_mac, iph, ip_payload),
        _ => None,
    }
}

// ── TCP — Phase 12c ───────────────────────────────────────────────────────────

pub mod tcp {
    //! TCP state machine — RFC 793 compliant passive open (server) plus
    //! basic data transfer and teardown.
    //!
    //! Architecture:
    //!   - `TCP_SOCKETS`  — global BTreeMap of (local_port, remote_ip, remote_port) → TcpSocket
    //!   - `TCP_LISTENERS`— set of locally-bound listen ports
    //!   - `handle_tcp_segment` — called from `handle_ipv4` for every TCP datagram

    use alloc::{collections::BTreeMap, vec::Vec};
    use spin::Mutex;
    use super::{EthFrame, Ipv4Header, IP_PROTO_TCP, OUR_IP, OUR_MAC, ipv4_checksum};

    // ── TCP flags ────────────────────────────────────────────────────────────
    pub const FIN: u8 = 0x01;
    pub const SYN: u8 = 0x02;
    pub const RST: u8 = 0x04;
    pub const PSH: u8 = 0x08;
    pub const ACK: u8 = 0x10;
    pub const URG: u8 = 0x20;

    // ── TCP header ───────────────────────────────────────────────────────────
    #[derive(Debug, Clone)]
    pub struct TcpHeader {
        pub src_port:   u16,
        pub dst_port:   u16,
        pub seq:        u32,
        pub ack:        u32,
        pub data_off:   u8,   // header length in 32-bit words (high nibble of byte 12)
        pub flags:      u8,
        pub window:     u16,
        pub checksum:   u16,
        pub urgent_ptr: u16,
    }

    impl TcpHeader {
        pub fn parse(raw: &[u8]) -> Option<Self> {
            if raw.len() < 20 { return None; }
            let data_off = raw[12] >> 4;
            Some(Self {
                src_port:   u16::from_be_bytes([raw[0],  raw[1]]),
                dst_port:   u16::from_be_bytes([raw[2],  raw[3]]),
                seq:        u32::from_be_bytes([raw[4],  raw[5],  raw[6],  raw[7]]),
                ack:        u32::from_be_bytes([raw[8],  raw[9],  raw[10], raw[11]]),
                data_off,
                flags:      raw[13],
                window:     u16::from_be_bytes([raw[14], raw[15]]),
                checksum:   u16::from_be_bytes([raw[16], raw[17]]),
                urgent_ptr: u16::from_be_bytes([raw[18], raw[19]]),
            })
        }

        pub fn header_len(&self) -> usize { self.data_off as usize * 4 }

        /// Build a TCP segment.  `payload` is the data bytes (may be empty).
        pub fn build(
            src_port: u16, dst_port: u16,
            seq: u32, ack: u32, flags: u8, window: u16,
            src_ip: [u8; 4], dst_ip: [u8; 4],
            payload: &[u8],
        ) -> Vec<u8> {
            let mut seg = Vec::with_capacity(20 + payload.len());
            seg.extend_from_slice(&src_port.to_be_bytes());
            seg.extend_from_slice(&dst_port.to_be_bytes());
            seg.extend_from_slice(&seq.to_be_bytes());
            seg.extend_from_slice(&ack.to_be_bytes());
            seg.push(0x50); // data offset = 5 (20 bytes), reserved = 0
            seg.push(flags);
            seg.extend_from_slice(&window.to_be_bytes());
            seg.extend_from_slice(&[0, 0]); // checksum placeholder
            seg.extend_from_slice(&[0, 0]); // urgent pointer
            seg.extend_from_slice(payload);

            // TCP pseudo-header checksum
            let csum = tcp_checksum(src_ip, dst_ip, &seg);
            seg[16] = (csum >> 8) as u8;
            seg[17] = (csum & 0xFF) as u8;
            seg
        }
    }

    fn tcp_checksum(src_ip: [u8; 4], dst_ip: [u8; 4], tcp_seg: &[u8]) -> u16 {
        let len = tcp_seg.len() as u16;
        let mut pseudo = Vec::with_capacity(12 + tcp_seg.len());
        pseudo.extend_from_slice(&src_ip);
        pseudo.extend_from_slice(&dst_ip);
        pseudo.push(0);           // zeros
        pseudo.push(IP_PROTO_TCP); // protocol
        pseudo.extend_from_slice(&len.to_be_bytes());
        pseudo.extend_from_slice(tcp_seg);
        ipv4_checksum(&pseudo)
    }

    // ── TCP state machine ────────────────────────────────────────────────────

    /// RFC 793 connection states.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TcpState {
        Closed,
        Listen,
        SynSent,      // client: SYN sent, waiting for SYN-ACK
        SynReceived,  // server: SYN received, SYN-ACK sent, waiting for ACK
        Established,
        Accepted,     // Established and handed to a userspace fd via sys_accept
        FinWait1,
        FinWait2,
        CloseWait,
        Closing,
        LastAck,
        TimeWait,
    }

    /// Uniquely identifies a TCP connection from the local side.
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct TcpSocketKey {
        pub local_port:   u16,
        pub remote_ip:    [u8; 4],
        pub remote_port:  u16,
    }

    /// Per-connection TCP socket state.
    #[derive(Debug)]
    pub struct TcpSocket {
        pub state:     TcpState,
        pub snd_nxt:   u32,   // next sequence number to send
        pub snd_una:   u32,   // oldest unacknowledged byte
        pub rcv_nxt:   u32,   // next expected receive sequence number
        pub snd_wnd:   u16,   // send window (from remote)
        pub rcv_buf:   Vec<u8>,  // received data waiting for application read
        // ── TCP Reno congestion control ──────────────────────────────────────
        pub cwnd:        u32,   // congestion window in bytes (Reno)
        pub ssthresh:    u32,   // slow-start threshold
        // ── Retransmit timer ─────────────────────────────────────────────────
        pub last_send_ms: u64,  // uptime_ms when last segment was sent
        pub rto_ms:       u64,  // retransmit timeout (starts at 1s, backs off)
        pub retransmit_buf: Vec<u8>, // copy of last unacked segment data
        // ── AI-integrated congestion control ─────────────────────────────────
        pub owner_pid:   u64,   // pid of the process that created this socket
        pub ai_cwnd_mul: u16,   // AI cwnd multiplier in 1/100 units (100=1.0)
    }

    const MSS: u32 = 1460;
    const RTO_INITIAL_MS: u64 = 1000;
    const RTO_MAX_MS:     u64 = 60_000;

    impl TcpSocket {
        fn new_syn_received(irs: u32) -> Self {
            // Initial send sequence number for this kernel — fixed for simplicity
            const ISS: u32 = 0xDEAD_0000;
            Self {
                state:   TcpState::SynReceived,
                snd_nxt: ISS.wrapping_add(1),
                snd_una: ISS,
                rcv_nxt: irs.wrapping_add(1),
                snd_wnd: 65535,
                rcv_buf: Vec::new(),
                cwnd:    MSS,
                ssthresh: 65535,
                last_send_ms: 0,
                rto_ms: RTO_INITIAL_MS,
                retransmit_buf: Vec::new(),
                owner_pid:   0,
                ai_cwnd_mul: 100,
            }
        }
    }

    // ── Global socket tables ─────────────────────────────────────────────────

    /// Active connections: key → socket.
    pub static SOCKETS:   Mutex<BTreeMap<TcpSocketKey, TcpSocket>> = Mutex::new(BTreeMap::new());
    /// Listening ports → backlog queue of established-connection keys.
    /// tcp::listen() inserts an empty deque; the SYN+ACK path pushes onto it;
    /// tcp::accept() pops from the front.
    pub static LISTENERS: Mutex<BTreeMap<u16, alloc::collections::VecDeque<TcpSocketKey>>>
        = Mutex::new(BTreeMap::new());

    /// Register a local port to accept incoming TCP connections.
    pub fn listen(port: u16) {
        LISTENERS.lock().entry(port)
            .or_insert_with(alloc::collections::VecDeque::new);
        crate::klog!(INFO, "TCP: listening on port {}", port);
    }

    /// Called from `handle_ipv4` when `proto == IP_PROTO_TCP`.
    pub fn handle_tcp_segment(
        src_mac:    [u8; 6],
        iph:        Ipv4Header,
        tcp_raw:    &[u8],
    ) -> Option<Vec<u8>> {
        let tcph = TcpHeader::parse(tcp_raw)?;
        let our_ip  = unsafe { OUR_IP };
        let our_mac = unsafe { OUR_MAC };

        let key = TcpSocketKey {
            local_port:  tcph.dst_port,
            remote_ip:   iph.src,
            remote_port: tcph.src_port,
        };

        let data_off  = tcph.header_len().min(tcp_raw.len());
        let data      = &tcp_raw[data_off..];
        let flags     = tcph.flags;

        // ── RST: always close the connection ────────────────────────────────
        if flags & RST != 0 {
            crate::klog!(WARN, "TCP: RST received port {}\u{2194}{}", tcph.dst_port, tcph.src_port);
            SOCKETS.lock().remove(&key);
            return None;
        }

        let mut sockets = SOCKETS.lock();

        // ── Active socket path ───────────────────────────────────────────────
        if let Some(sock) = sockets.get_mut(&key) {
            // Adaptive Causal Deferral: check if this process is chaotic/low-valence
            if crate::causal_deferral::get_deferral_buffer().should_defer(sock.owner_pid) {
                crate::causal_deferral::get_deferral_buffer().defer_event(src_mac, iph.clone(), tcp_raw.to_vec());
                return None;
            }
            return tcp_state_machine(sock, &tcph, data, our_mac, our_ip, src_mac, &iph.src);
        }
        drop(sockets);

        // ── Passive open: SYN on a listening port ────────────────────────────
        if flags & SYN != 0 && flags & ACK == 0 {
            if LISTENERS.lock().contains_key(&tcph.dst_port) {
                const ISS: u32 = 0xDEAD_0000;
                let mut sock = TcpSocket::new_syn_received(tcph.seq);
                // Send SYN-ACK
                let seg = TcpHeader::build(
                    tcph.dst_port, tcph.src_port,
                    ISS,                          // our ISN (before +1 for SYN)
                    sock.rcv_nxt,                 // ack = their ISN + 1
                    SYN | ACK, sock.snd_wnd,
                    our_ip, iph.src,
                    &[],
                );
                SOCKETS.lock().insert(key.clone(), sock);
                let ip_hdr = Ipv4Header::build(IP_PROTO_TCP, our_ip, iph.src, seg.len());
                let mut pkt = ip_hdr;
                pkt.extend_from_slice(&seg);
                return Some(EthFrame::build(src_mac, our_mac, super::ETHERTYPE_IPV4, &pkt));
            }
        }

        // ── Send RST for unexpected segments on non-listening ports ──────────
        if flags & SYN != 0 {
            let rst = TcpHeader::build(
                tcph.dst_port, tcph.src_port,
                0, tcph.seq.wrapping_add(1), RST | ACK, 0,
                our_ip, iph.src, &[],
            );
            let ip_hdr = Ipv4Header::build(IP_PROTO_TCP, our_ip, iph.src, rst.len());
            let mut pkt = ip_hdr;
            pkt.extend_from_slice(&rst);
            return Some(EthFrame::build(src_mac, our_mac, super::ETHERTYPE_IPV4, &pkt));
        }

        None
    }

    /// Progress an existing socket's state machine.
    fn tcp_state_machine(
        sock:       &mut TcpSocket,
        tcph:       &TcpHeader,
        data:       &[u8],
        our_mac:    [u8; 6],
        our_ip:     [u8; 4],
        remote_mac: [u8; 6],
        remote_ip:  &[u8; 4],
    ) -> Option<Vec<u8>> {
        let flags = tcph.flags;

        let make_reply = |sock: &TcpSocket, tx_flags: u8, tx_data: &[u8]| -> Vec<u8> {
            let seg = TcpHeader::build(
                tcph.dst_port, tcph.src_port,
                sock.snd_nxt, sock.rcv_nxt,
                tx_flags, 65535,
                our_ip, *remote_ip, tx_data,
            );
            let ip_hdr = Ipv4Header::build(IP_PROTO_TCP, our_ip, *remote_ip, seg.len());
            let mut pkt = ip_hdr;
            pkt.extend_from_slice(&seg);
            EthFrame::build(remote_mac, our_mac, super::ETHERTYPE_IPV4, &pkt)
        };

        match sock.state {
            // SYN-SENT (client): waiting for SYN-ACK from server
            TcpState::SynSent => {
                if flags & (SYN | ACK) == (SYN | ACK) && tcph.ack == sock.snd_nxt {
                    // Server's SYN-ACK: record server's ISN, advance rcv_nxt past the SYN
                    sock.rcv_nxt = tcph.seq.wrapping_add(1);
                    sock.snd_una = sock.snd_nxt;
                    sock.snd_wnd = tcph.window;
                    sock.state   = TcpState::Established;
                    crate::klog!(INFO, "TCP: connection ESTABLISHED port {}↔{}",
                        tcph.dst_port, tcph.src_port);
                    // Send ACK so server can send data
                    return Some(make_reply(sock, ACK, &[]));
                }
                None
            }

            // SYN-RECEIVED (server): waiting for ACK of our SYN-ACK
            TcpState::SynReceived => {
                if flags & ACK != 0 && tcph.ack == sock.snd_nxt {
                    sock.state     = TcpState::Established;
                    sock.snd_una   = sock.snd_nxt;
                    sock.snd_wnd   = tcph.window;
                    crate::klog!(INFO, "TCP: connection ESTABLISHED port {}↔{}",
                        tcph.dst_port, tcph.src_port);
                    // Push this connection onto the accept backlog for the listening port.
                    let conn_key = TcpSocketKey {
                        local_port: tcph.dst_port,
                        remote_ip:  *remote_ip,
                        remote_port: tcph.src_port,
                    };
                    let port = tcph.dst_port;
                    if let Some(backlog) = LISTENERS.lock().get_mut(&port) {
                        backlog.push_back(conn_key);
                    }
                }
                None
            }

            // ESTABLISHED: data transfer
            TcpState::Established => {
                sock.snd_wnd = tcph.window;

                // TCP Reno CWND update on ACK.
                if flags & ACK != 0 {
                    let acked = tcph.ack.wrapping_sub(sock.snd_una);
                    if acked > 0 && acked <= 65536 {
                        sock.snd_una = tcph.ack;
                        // Slow start: cwnd < ssthresh → grow by acked bytes.
                        // Congestion avoidance: grow by MSS²/cwnd per ACK.
                        if sock.cwnd < sock.ssthresh {
                            sock.cwnd = sock.cwnd.saturating_add(acked);
                        } else {
                            let inc = MSS.saturating_mul(acked) / sock.cwnd.max(1);
                            sock.cwnd = sock.cwnd.saturating_add(inc.max(1));
                        }
                        // Cap at remote receiver window.
                        let cap = (sock.snd_wnd as u32).max(MSS);
                        if sock.cwnd > cap { sock.cwnd = cap; }

                        // AI-integrated congestion control: recalibrate ai_cwnd_mul
                        // based on the owning pid's causal fanout and anomaly score.
                        // High fanout (interactive/critical process) → inflate cwnd.
                        // High anomaly score → deflate cwnd (treat as background/suspect).
                        // Novel: this is the first OS where TCP cwnd reflects the process's
                        // causal importance in the task graph, not just network feedback.
                        let pid = sock.owner_pid;
                        if pid > 0 {
                            let fanout = crate::causal::causal_fanout(pid);
                            let anom   = crate::anomaly::score(pid);
                            let mul: u16 = if anom > 0.6 {
                                70   // anomalous → reduce cwnd to 70%
                            } else if fanout >= 4 {
                                150  // critical orchestrator → inflate cwnd to 150%
                            } else if fanout >= 2 {
                                120  // moderate fanout → slight inflation
                            } else {
                                100  // default
                            };
                            sock.ai_cwnd_mul = mul;
                        }
                    }
                }

                // Accept in-order data (FIN may arrive in the same segment)
                if !data.is_empty() && tcph.seq == sock.rcv_nxt {
                    sock.rcv_buf.extend_from_slice(data);
                    sock.rcv_nxt = sock.rcv_nxt.wrapping_add(data.len() as u32);
                    if flags & FIN != 0 {
                        sock.rcv_nxt = sock.rcv_nxt.wrapping_add(1);
                        sock.state   = TcpState::CloseWait;
                    }
                    return Some(make_reply(sock, ACK, &[]));
                }

                // Remote FIN with no data: move to CLOSE_WAIT, send ACK
                if flags & FIN != 0 {
                    sock.rcv_nxt = sock.rcv_nxt.wrapping_add(1);
                    sock.state   = TcpState::CloseWait;
                    return Some(make_reply(sock, ACK, &[]));
                }

                None
            }

            // CLOSE_WAIT: we received FIN, app needs to close
            TcpState::CloseWait => {
                // App-initiated close: send FIN, move to LAST_ACK
                // (triggered externally via `close_socket`; here we just ACK duplicates)
                None
            }

            // LAST_ACK: waiting for ACK of our FIN
            TcpState::LastAck => {
                if flags & ACK != 0 {
                    sock.state = TcpState::Closed;
                    crate::klog!(INFO, "TCP: connection closed");
                }
                None
            }

            // FIN_WAIT_1: we sent FIN, waiting for ACK
            TcpState::FinWait1 => {
                if flags & ACK != 0 {
                    sock.state = TcpState::FinWait2;
                }
                if flags & FIN != 0 {
                    sock.rcv_nxt = sock.rcv_nxt.wrapping_add(1);
                    sock.state   = TcpState::TimeWait;
                    return Some(make_reply(sock, ACK, &[]));
                }
                None
            }

            // FIN_WAIT_2: our FIN acked, waiting for remote FIN
            TcpState::FinWait2 => {
                if flags & FIN != 0 {
                    sock.rcv_nxt = sock.rcv_nxt.wrapping_add(1);
                    sock.state   = TcpState::TimeWait;
                    return Some(make_reply(sock, ACK, &[]));
                }
                None
            }

            // TIME_WAIT / others: send RST on unexpected data
            _ => None,
        }
    }

    // ── Public socket API (used by syscall layer) ────────────────────────────

    /// Send data on an established connection.
    /// Returns number of bytes enqueued, or 0 if not connected.
    pub fn send(local_port: u16, remote_ip: [u8; 4], remote_port: u16, data: &[u8]) -> usize {
        let key = TcpSocketKey { local_port, remote_ip, remote_port };
        let our_ip  = unsafe { OUR_IP };
        let our_mac = unsafe { OUR_MAC };
        let mut sockets = SOCKETS.lock();
        if let Some(sock) = sockets.get_mut(&key) {
            crate::klog!(DEBUG, "TCP: send {} bytes port {}\u{2194}{} seq={} ack={} state={:?}",
                data.len(), local_port, remote_port, sock.snd_nxt, sock.rcv_nxt, sock.state);
            if sock.state != TcpState::Established && sock.state != TcpState::Accepted { return 0; }
            // Reno + AI: limit send to min(ai_cwnd, snd_wnd).
            // ai_cwnd = cwnd × ai_cwnd_mul / 100 (100 = no change, 150 = 50% boost).
            let ai_cwnd = (sock.cwnd as u64 * sock.ai_cwnd_mul as u64 / 100) as u32;
            let allowed = (ai_cwnd as usize).min(sock.snd_wnd as usize).max(MSS as usize);
            let send_data = &data[..data.len().min(allowed)];
            let seg = TcpHeader::build(
                local_port, remote_port,
                sock.snd_nxt, sock.rcv_nxt,
                PSH | ACK, 65535,
                our_ip, remote_ip, send_data,
            );
            sock.snd_nxt = sock.snd_nxt.wrapping_add(send_data.len() as u32);
            let ip_hdr = Ipv4Header::build(IP_PROTO_TCP, our_ip, remote_ip, seg.len());
            let mut pkt = ip_hdr;
            pkt.extend_from_slice(&seg);
            // Resolve remote MAC via ARP cache or gateway
            let dst_mac = super::arp_resolve_for_ip(&remote_ip).unwrap_or([0xFF; 6]);
            let frame = super::EthFrame::build(dst_mac, our_mac, super::ETHERTYPE_IPV4, &pkt);
            super::transmit(&frame);
            // Save for possible retransmit.
            sock.retransmit_buf.clear();
            sock.retransmit_buf.extend_from_slice(send_data);
            sock.last_send_ms = crate::scheduler::uptime_ms();
            return send_data.len();
        }
        0
    }

    /// Check all Established sockets for RTO expiry and retransmit if needed.
    /// Called from net::poll() in the idle loop.
    pub fn poll_retransmit() {
        let now = crate::scheduler::uptime_ms();
        let our_ip  = unsafe { OUR_IP };
        let our_mac = unsafe { OUR_MAC };
        let mut sockets = SOCKETS.lock();
        let mut to_retransmit: alloc::vec::Vec<(TcpSocketKey, alloc::vec::Vec<u8>)> = alloc::vec::Vec::new();

        for (key, sock) in sockets.iter_mut() {
            if sock.state != TcpState::Established && sock.state != TcpState::Accepted { continue; }
            if sock.retransmit_buf.is_empty() { continue; }
            if sock.snd_una == sock.snd_nxt.wrapping_sub(sock.retransmit_buf.len() as u32) { continue; } // fully acked
            if sock.last_send_ms == 0 { continue; }
            if now.saturating_sub(sock.last_send_ms) >= sock.rto_ms {
                // RTO expired: exponential back-off + Reno CWND reduction
                sock.ssthresh = (sock.cwnd / 2).max(MSS);
                sock.cwnd     = MSS; // reset to 1 MSS (slow start)
                sock.rto_ms   = (sock.rto_ms * 2).min(RTO_MAX_MS);
                sock.last_send_ms = now;
                to_retransmit.push((key.clone(), sock.retransmit_buf.clone()));
                crate::klog!(DEBUG, "TCP: RTO expired port {}→{} rto={}ms retransmitting {} bytes",
                    key.local_port, key.remote_port, sock.rto_ms / 2, sock.retransmit_buf.len());
            }
        }
        drop(sockets);

        for (key, data) in to_retransmit {
            // Re-read the socket to get current snd_una/rcv_nxt for ACK numbers.
            let (snd_nxt_at_send, rcv_nxt) = {
                let sockets = SOCKETS.lock();
                let sock = match sockets.get(&key) { Some(s) => s, None => continue };
                (sock.snd_una, sock.rcv_nxt)
            };
            let seg = TcpHeader::build(
                key.local_port, key.remote_port,
                snd_nxt_at_send, rcv_nxt,
                PSH | ACK, 65535,
                our_ip, key.remote_ip, &data,
            );
            let ip_hdr = Ipv4Header::build(IP_PROTO_TCP, our_ip, key.remote_ip, seg.len());
            let mut pkt = ip_hdr;
            pkt.extend_from_slice(&seg);
            let dst_mac = super::arp_resolve_for_ip(&key.remote_ip).unwrap_or([0xFF; 6]);
            let frame = super::EthFrame::build(dst_mac, our_mac, super::ETHERTYPE_IPV4, &pkt);
            super::transmit(&frame);
        }
    }

    /// Dequeue the next established connection from the backlog for `port`.
    /// Returns None (→ EAGAIN) if the backlog is empty.
    /// Previously scanned SOCKETS (O(n)); now O(1) via the backlog queue.
    pub fn accept(port: u16) -> Option<TcpSocketKey> {
        let key = LISTENERS.lock().get_mut(&port)?.pop_front()?;
        // Mark the socket as Accepted so a second accept() doesn't see it.
        if let Some(sock) = SOCKETS.lock().get_mut(&key) {
            sock.state = TcpState::Accepted;
        }
        Some(key)
    }

    /// Return the number of bytes waiting in the socket's receive buffer (non-consuming).
    pub fn rx_buf_len(local_port: u16, remote_ip: [u8; 4], remote_port: u16) -> usize {
        let key = TcpSocketKey { local_port, remote_ip, remote_port };
        SOCKETS.lock().get(&key).map(|s| s.rcv_buf.len()).unwrap_or(0)
    }

    /// Drain up to `buf.len()` bytes from the socket's receive buffer.
    pub fn recv(local_port: u16, remote_ip: [u8; 4], remote_port: u16, buf: &mut [u8]) -> usize {
        let key = TcpSocketKey { local_port, remote_ip, remote_port };
        let mut sockets = SOCKETS.lock();
        if let Some(sock) = sockets.get_mut(&key) {
            let n = buf.len().min(sock.rcv_buf.len());
            buf[..n].copy_from_slice(&sock.rcv_buf[..n]);
            sock.rcv_buf.drain(..n);
            n
        } else {
            0
        }
    }

    /// Send a FIN+ACK to close a connection, then mark state.
    pub fn close(local_port: u16, remote_ip: [u8; 4], remote_port: u16) {
        let key = TcpSocketKey { local_port, remote_ip, remote_port };
        let our_ip  = unsafe { OUR_IP };
        let our_mac = unsafe { OUR_MAC };
        let mut sockets = SOCKETS.lock();
        if let Some(sock) = sockets.get_mut(&key) {
            let new_state = match sock.state {
                TcpState::Established => TcpState::FinWait1,
                TcpState::CloseWait   => TcpState::LastAck,
                _ => return,
            };
            let seg = TcpHeader::build(
                local_port, remote_port,
                sock.snd_nxt, sock.rcv_nxt,
                FIN | ACK, 65535,
                our_ip, remote_ip, &[],
            );
            sock.snd_nxt = sock.snd_nxt.wrapping_add(1);
            sock.state = new_state;
            let ip_hdr = Ipv4Header::build(IP_PROTO_TCP, our_ip, remote_ip, seg.len());
            let mut pkt = ip_hdr;
            pkt.extend_from_slice(&seg);
            let dst_mac = super::arp_resolve_for_ip(&remote_ip).unwrap_or([0xFF; 6]);
            let frame = super::EthFrame::build(dst_mac, our_mac, super::ETHERTYPE_IPV4, &pkt);
            super::transmit(&frame);
        }
    }

    /// Drain all received data from a socket's buffer into a Vec.
    pub fn recv_vec(local_port: u16, remote_ip: [u8; 4], remote_port: u16) -> Vec<u8> {
        let key = TcpSocketKey { local_port, remote_ip, remote_port };
        let mut sockets = SOCKETS.lock();
        if let Some(sock) = sockets.get_mut(&key) {
            let data = core::mem::take(&mut sock.rcv_buf);
            return data;
        }
        Vec::new()
    }

    /// Check if a socket has data available.
    pub fn has_data(local_port: u16, remote_ip: [u8; 4], remote_port: u16) -> bool {
        let key = TcpSocketKey { local_port, remote_ip, remote_port };
        let sockets = SOCKETS.lock();
        sockets.get(&key).map_or(false, |s| !s.rcv_buf.is_empty())
    }

    /// Get established connection keys for a listener port.
    pub fn established_for(port: u16) -> Vec<TcpSocketKey> {
        SOCKETS.lock().iter()
            .filter(|(k, s)| k.local_port == port && s.state == TcpState::Established)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Active open (client-side connect).
    ///
    /// Picks an ephemeral local port, sends a SYN, then spins on net::poll()
    /// until the connection reaches Established or the timeout elapses.
    ///
    /// Returns `Ok(local_port)` on success, `Err` on timeout or ARP failure.
    pub fn connect(remote_ip: [u8; 4], remote_port: u16, timeout_ms: u64) -> Result<u16, &'static str> {
        static NEXT_EPH: spin::Mutex<u16> = spin::Mutex::new(49152);

        // Pick a free ephemeral port.
        let local_port = {
            let mut eph = NEXT_EPH.lock();
            let port = *eph;
            *eph = if port >= 65534 { 49152 } else { port + 1 };
            port
        };

        let our_ip  = unsafe { OUR_IP };
        let our_mac = unsafe { OUR_MAC };

        // Resolve remote MAC before inserting the socket (we need to send SYN).
        let dst_mac = super::arp_resolve_for_ip(&remote_ip)
            .ok_or("TCP connect: ARP failed — no MAC for destination")?;

        // Ephemeral ISN — use uptime tick for variety.
        let iss: u32 = crate::scheduler::uptime_ms() as u32 ^ 0x1357_2468;

        // Create the socket in SYN_SENT state.
        let sock = TcpSocket {
            state:   TcpState::SynSent,
            snd_nxt: iss.wrapping_add(1), // post-SYN, next byte to send
            snd_una: iss,
            rcv_nxt: 0,     // filled in when SYN-ACK arrives
            snd_wnd: 65535,
            rcv_buf: Vec::new(),
            cwnd:    MSS, ssthresh: 65535,
            last_send_ms: 0, rto_ms: RTO_INITIAL_MS, retransmit_buf: Vec::new(),
            owner_pid: crate::scheduler::current_pid(),
            ai_cwnd_mul: 100,
        };
        let key = TcpSocketKey { local_port, remote_ip, remote_port };
        SOCKETS.lock().insert(key, TcpSocket { state: TcpState::SynSent, ..sock });

        // Send SYN.
        let syn = TcpHeader::build(
            local_port, remote_port,
            iss, 0, SYN, 65535,
            our_ip, remote_ip, &[],
        );
        let ip_hdr = Ipv4Header::build(IP_PROTO_TCP, our_ip, remote_ip, syn.len());
        let mut pkt = ip_hdr;
        pkt.extend_from_slice(&syn);
        let frame = super::EthFrame::build(dst_mac, our_mac, super::ETHERTYPE_IPV4, &pkt);
        super::transmit(&frame);
        crate::klog!(INFO, "TCP: SYN sent local:{} → {}:{}", local_port, remote_ip[0..4].iter().map(|b| alloc::format!("{}", b)).collect::<Vec<_>>().join("."), remote_port);

        // Poll until Established or timeout.
        let deadline = crate::scheduler::uptime_ms() + timeout_ms;
        loop {
            {
                let sockets = SOCKETS.lock();
                let key = TcpSocketKey { local_port, remote_ip, remote_port };
                if let Some(s) = sockets.get(&key) {
                    if s.state == TcpState::Established {
                        return Ok(local_port);
                    }
                }
            }
            if crate::scheduler::uptime_ms() >= deadline {
                // Clean up the stale SynSent socket.
                let key = TcpSocketKey { local_port, remote_ip, remote_port };
                SOCKETS.lock().remove(&key);
                return Err("TCP connect: timeout");
            }
            super::poll();
            crate::scheduler::yield_cpu();
        }
    }
}

// ── Built-in HTTP server ─────────────────────────────────────────────────────

/// HTTP server state.
static HTTP_RUNNING: AtomicBool = AtomicBool::new(false);
static HTTP_PORT: Mutex<u16> = Mutex::new(8081); // kernel diagnostic httpd; 8080 reserved for userspace
static HTTP_ROOT: Mutex<String> = Mutex::new(String::new());

/// Start the built-in HTTP server on given port, serving files under `root_dir`.
pub fn http_server_start(port: u16, root: &str) {
    *HTTP_PORT.lock() = port;
    *HTTP_ROOT.lock() = String::from(root);
    tcp::listen(port);
    HTTP_RUNNING.store(true, Ordering::Release);
    crate::klog!(INFO, "HTTP: server started on port {}, root={}", port, root);
}

/// Stop the HTTP server.
pub fn http_server_stop() {
    HTTP_RUNNING.store(false, Ordering::Release);
    let port = *HTTP_PORT.lock();
    tcp::LISTENERS.lock().remove(&port);
    crate::klog!(INFO, "HTTP: server stopped");
}

/// Check if HTTP server is running.
pub fn http_server_running() -> bool {
    HTTP_RUNNING.load(Ordering::Acquire)
}

/// Poll for HTTP requests — called from the main poll loop or a shell command.
pub fn http_server_poll() {
    if !HTTP_RUNNING.load(Ordering::Acquire) { return; }
    let port = *HTTP_PORT.lock();
    let keys = tcp::established_for(port);
    for key in keys {
        let data = tcp::recv_vec(key.local_port, key.remote_ip, key.remote_port);
        if data.is_empty() { continue; }
        // Try to parse an HTTP request
        let req = core::str::from_utf8(&data).unwrap_or("");
        if !req.starts_with("GET ") && !req.starts_with("HEAD ") { continue; }
        let path = req.split_whitespace().nth(1).unwrap_or("/");
        let response = http_handle_request(path);
        tcp::send(key.local_port, key.remote_ip, key.remote_port, response.as_bytes());
        // Close the connection after response
        tcp::close(key.local_port, key.remote_ip, key.remote_port);
    }
}

/// Generate an HTTP response for a given request path.
fn http_handle_request(path: &str) -> String {
    let root = HTTP_ROOT.lock().clone();
    let full = if path == "/" {
        alloc::format!("{}/index.html", root)
    } else {
        alloc::format!("{}{}", root, path)
    };

    // Try to read the file from VFS
    match vfs_read(&full) {
        Some(data) => {
            let body = core::str::from_utf8(&data).unwrap_or("[binary data]");
            let ctype = if full.ends_with(".html") { "text/html" }
                   else if full.ends_with(".css")  { "text/css" }
                   else if full.ends_with(".js")   { "application/javascript" }
                   else if full.ends_with(".json") { "application/json" }
                   else { "text/plain" };
            alloc::format!(
                "HTTP/1.0 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                ctype, body.len(), body,
            )
        }
        None => {
            // Try directory listing if path is a directory
            if let Ok(node) = crate::vfs::lookup(&full) {
                if let Ok(entries) = node.readdir() {
                    let mut body = alloc::format!("<html><body><h1>Index of {}</h1><pre>\n", path);
                    for e in entries {
                        body.push_str(&alloc::format!("<a href=\"{}{}\">{}</a>\n",
                            path, e.name, e.name));
                    }
                    body.push_str("</pre></body></html>");
                    return alloc::format!(
                        "HTTP/1.0 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(), body,
                    );
                }
            }
            let body = "<html><body><h1>404 Not Found</h1></body></html>";
            alloc::format!(
                "HTTP/1.0 404 Not Found\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(), body,
            )
        }
    }
}

// ── Built-in SSH server (stub) ───────────────────────────────────────────────

/// SSH server state.
static SSH_RUNNING: AtomicBool = AtomicBool::new(false);
static SSH_PORT: Mutex<u16> = Mutex::new(22);

/// Start the SSH server stub on given port.
pub fn ssh_server_start(port: u16) {
    *SSH_PORT.lock() = port;
    tcp::listen(port);
    SSH_RUNNING.store(true, Ordering::Release);
    crate::klog!(INFO, "SSH: server started on port {} (stub — no encryption)", port);
}

/// Stop the SSH server.
pub fn ssh_server_stop() {
    SSH_RUNNING.store(false, Ordering::Release);
    let port = *SSH_PORT.lock();
    tcp::LISTENERS.lock().remove(&port);
    crate::klog!(INFO, "SSH: server stopped");
}

/// Check if SSH server is running.
pub fn ssh_server_running() -> bool {
    SSH_RUNNING.load(Ordering::Acquire)
}

/// Poll for SSH connections — basic unencrypted shell relay.
pub fn ssh_server_poll() {
    if !SSH_RUNNING.load(Ordering::Acquire) { return; }
    let port = *SSH_PORT.lock();
    let keys = tcp::established_for(port);
    for key in keys {
        let data = tcp::recv_vec(key.local_port, key.remote_ip, key.remote_port);
        if data.is_empty() { continue; }
        // Send SSH identification string on new connections
        let banner = "SSH-2.0-NodeAI_1.0 PROTOCOL_NOT_IMPLEMENTED\r\n\
                       NodeAI SSH stub: full SSH encryption not available\r\n\
                       Use HTTP server or serial console for access.\r\n";
        tcp::send(key.local_port, key.remote_ip, key.remote_port, banner.as_bytes());
        tcp::close(key.local_port, key.remote_ip, key.remote_port);
    }
}
