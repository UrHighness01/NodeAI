//! WPA2-PSK cryptographic primitives — all implemented from scratch, no external crates.
//!
//! Implements:
//!   - SHA-1 (FIPS 180-4)
//!   - HMAC-SHA1 (RFC 2104)
//!   - PBKDF2-HMAC-SHA1 (RFC 2898) — PMK derivation
//!   - PRF-384 (IEEE 802.11-2020 §12.7.1.2) — PTK derivation
//!   - AES-128 software (FIPS 197) — for KCK/KEK/MIC
//!   - CCMP (IEEE 802.11i §8.3.3) — data frame encryption/decryption
//!   - AES-128-CMAC (RFC 4493) — EAPOL MIC computation
//!
//! Not implemented here (Phase 5): CCMP for data frame encryption.

// ═══════════════════════════════════════════════════════════════════════════════
// SHA-1 — FIPS 180-4
// ═══════════════════════════════════════════════════════════════════════════════

const H0: u32 = 0x67452301;
const H1: u32 = 0xEFCDAB89;
const H2: u32 = 0x98BADCFE;
const H3: u32 = 0x10325476;
const H4: u32 = 0xC3D2E1F0;

pub struct Sha1 {
    state:  [u32; 5],
    buf:    [u8; 64],
    buflen: usize,
    total:  u64,
}

impl Sha1 {
    pub fn new() -> Self {
        Self { state: [H0, H1, H2, H3, H4], buf: [0; 64], buflen: 0, total: 0 }
    }

    pub fn update(&mut self, data: &[u8]) {
        self.total += data.len() as u64;
        let mut off = 0;
        while off < data.len() {
            let space = 64 - self.buflen;
            let take  = space.min(data.len() - off);
            self.buf[self.buflen..self.buflen + take].copy_from_slice(&data[off..off + take]);
            self.buflen += take;
            off += take;
            if self.buflen == 64 {
                self.compress();
                self.buflen = 0;
            }
        }
    }

    pub fn finalize(mut self) -> [u8; 20] {
        let bit_len = self.total * 8;
        self.buf[self.buflen] = 0x80;
        self.buflen += 1;
        if self.buflen > 56 {
            for i in self.buflen..64 { self.buf[i] = 0; }
            self.compress();
            self.buflen = 0;
        }
        for i in self.buflen..56 { self.buf[i] = 0; }
        let bl = bit_len.to_be_bytes();
        self.buf[56..64].copy_from_slice(&bl);
        self.compress();
        let mut out = [0u8; 20];
        for (i, &s) in self.state.iter().enumerate() {
            out[i*4..i*4+4].copy_from_slice(&s.to_be_bytes());
        }
        out
    }

    fn compress(&mut self) {
        let w = &mut [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                self.buf[i*4], self.buf[i*4+1], self.buf[i*4+2], self.buf[i*4+3]
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16]).rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = self.state;
        for i in 0..80 {
            let (f, k) = match i {
                0..=19  => ((b & c) | (!b & d),         0x5A827999u32),
                20..=39 => (b ^ c ^ d,                   0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _       => (b ^ c ^ d,                   0xCA62C1D6),
            };
            let temp = a.rotate_left(5).wrapping_add(f).wrapping_add(e)
                        .wrapping_add(k).wrapping_add(w[i]);
            e = d; d = c; c = b.rotate_left(30); b = a; a = temp;
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
    }
}

pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h = Sha1::new();
    h.update(data);
    h.finalize()
}

// ═══════════════════════════════════════════════════════════════════════════════
// HMAC-SHA1 — RFC 2104
// ═══════════════════════════════════════════════════════════════════════════════

pub fn hmac_sha1(key: &[u8], data: &[u8]) -> [u8; 20] {
    // Key normalization: hash if > 64 bytes, zero-pad otherwise
    let mut k = [0u8; 64];
    if key.len() > 64 {
        let h = sha1(key);
        k[..20].copy_from_slice(&h);
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    // ipad / opad
    let mut ipad = [0u8; 64];
    let mut opad = [0u8; 64];
    for i in 0..64 { ipad[i] = k[i] ^ 0x36; opad[i] = k[i] ^ 0x5C; }

    let mut inner = Sha1::new();
    inner.update(&ipad);
    inner.update(data);
    let inner_hash = inner.finalize();

    let mut outer = Sha1::new();
    outer.update(&opad);
    outer.update(&inner_hash);
    outer.finalize()
}

/// HMAC-SHA1 over multiple data slices (avoids allocation).
pub fn hmac_sha1_multi(key: &[u8], parts: &[&[u8]]) -> [u8; 20] {
    let mut k = [0u8; 64];
    if key.len() > 64 {
        let h = sha1(key);
        k[..20].copy_from_slice(&h);
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0u8; 64];
    let mut opad = [0u8; 64];
    for i in 0..64 { ipad[i] = k[i] ^ 0x36; opad[i] = k[i] ^ 0x5C; }

    let mut inner = Sha1::new();
    inner.update(&ipad);
    for part in parts { inner.update(part); }
    let inner_hash = inner.finalize();

    let mut outer = Sha1::new();
    outer.update(&opad);
    outer.update(&inner_hash);
    outer.finalize()
}

// ═══════════════════════════════════════════════════════════════════════════════
// PBKDF2-HMAC-SHA1 — RFC 2898
// WPA2 PMK: PBKDF2(passphrase, ssid, 4096, 32)
// ═══════════════════════════════════════════════════════════════════════════════

pub fn pbkdf2_sha1(password: &[u8], salt: &[u8], iterations: u32, out: &mut [u8]) {
    let hlen = 20usize; // SHA-1 output length
    let blocks = (out.len() + hlen - 1) / hlen;
    for b in 1..=blocks {
        // U1 = HMAC(password, salt || INT(b))
        let bi = (b as u32).to_be_bytes();
        let mut u = hmac_sha1_multi(password, &[salt, &bi]);
        let mut t = u;
        for _ in 1..iterations {
            u = hmac_sha1(password, &u);
            for j in 0..hlen { t[j] ^= u[j]; }
        }
        let start = (b - 1) * hlen;
        let end   = out.len().min(start + hlen);
        out[start..end].copy_from_slice(&t[..end - start]);
    }
}

/// Derive WPA2 PMK from passphrase and SSID.
pub fn derive_pmk(passphrase: &[u8], ssid: &[u8]) -> [u8; 32] {
    let mut pmk = [0u8; 32];
    pbkdf2_sha1(passphrase, ssid, 4096, &mut pmk);
    pmk
}

// ═══════════════════════════════════════════════════════════════════════════════
// PRF-384 — IEEE 802.11-2020 §12.7.1.2
// PTK derivation: PRF-384(PMK, "Pairwise key expansion", A || B || ANonce || SNonce)
// ═══════════════════════════════════════════════════════════════════════════════

/// IEEE 802.11 pseudo-random function.
/// key=PMK, label="Pairwise key expansion", data=sorted_MACs||ANonce||SNonce
/// Returns 48 bytes (384 bits): KCK(16) + KEK(16) + TK(16).
pub fn prf_384(key: &[u8], label: &[u8], data: &[u8]) -> [u8; 48] {
    let mut out = [0u8; 48];
    let needed_bytes = 48usize;
    let iterations = (needed_bytes + 19) / 20; // ceil(48/20) = 3
    for i in 0..iterations {
        let counter = [i as u8];
        let zero    = [0u8];
        let mac = hmac_sha1_multi(key, &[label, &zero, data, &counter]);
        let start = i * 20;
        let end   = out.len().min(start + 20);
        out[start..end].copy_from_slice(&mac[..end - start]);
    }
    out
}

/// Derive PTK from PMK, ANonce, SNonce, AP MAC, STA MAC.
/// Returns [KCK(16) | KEK(16) | TK(16)].
pub fn derive_ptk(
    pmk:    &[u8; 32],
    anonce: &[u8; 32],
    snonce: &[u8; 32],
    ap_mac: &[u8; 6],
    sta_mac:&[u8; 6],
) -> [u8; 48] {
    // Data = min(AP,STA) || max(AP,STA) || min(ANonce,SNonce) || max(ANonce,SNonce)
    let (mac_min, mac_max) = if ap_mac < sta_mac { (ap_mac, sta_mac) } else { (sta_mac, ap_mac) };
    let (nn_min, nn_max)   = if anonce < snonce   { (anonce, snonce) } else { (snonce, anonce) };
    let label = b"Pairwise key expansion";
    let data: alloc::vec::Vec<u8> = mac_min.iter()
        .chain(mac_max.iter())
        .chain(nn_min.iter())
        .chain(nn_max.iter())
        .copied().collect();
    prf_384(pmk, label, &data)
}

// ═══════════════════════════════════════════════════════════════════════════════
// AES-128 — FIPS 197 (software implementation, no AES-NI)
// ═══════════════════════════════════════════════════════════════════════════════

// AES S-box (forward substitution)
#[rustfmt::skip]
const SBOX: [u8; 256] = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
];

// AES round constants
const RCON: [u8; 11] = [0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36];

fn xtime(x: u8) -> u8 {
    if x & 0x80 != 0 { (x << 1) ^ 0x1B } else { x << 1 }
}

fn gmul(a: u8, b: u8) -> u8 {
    let mut r = 0u8; let mut a = a; let mut b = b;
    for _ in 0..8 {
        if b & 1 != 0 { r ^= a; }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 { a ^= 0x1B; }
        b >>= 1;
    }
    r
}

fn sub_bytes(s: &mut [u8; 16]) {
    for b in s.iter_mut() { *b = SBOX[*b as usize]; }
}

fn shift_rows(s: &mut [u8; 16]) {
    // Row 1: shift 1
    let t = s[1]; s[1] = s[5]; s[5] = s[9]; s[9] = s[13]; s[13] = t;
    // Row 2: shift 2
    s.swap(2, 10); s.swap(6, 14);
    // Row 3: shift 3 (= shift 1 backwards)
    let t = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = s[3]; s[3] = t;
}

fn mix_columns(s: &mut [u8; 16]) {
    for c in 0..4 {
        let i = c * 4;
        let (s0, s1, s2, s3) = (s[i], s[i+1], s[i+2], s[i+3]);
        s[i]   = gmul(2,s0)^gmul(3,s1)^s2^s3;
        s[i+1] = s0^gmul(2,s1)^gmul(3,s2)^s3;
        s[i+2] = s0^s1^gmul(2,s2)^gmul(3,s3);
        s[i+3] = gmul(3,s0)^s1^s2^gmul(2,s3);
    }
}

fn add_round_key(state: &mut [u8; 16], rk: &[u8]) {
    for i in 0..16 { state[i] ^= rk[i]; }
}

/// Expand 16-byte key into 11 round keys (176 bytes).
pub fn aes128_key_expand(key: &[u8; 16]) -> [u8; 176] {
    let mut rk = [0u8; 176];
    rk[..16].copy_from_slice(key);
    for i in 1..11 {
        let prev = &rk[(i-1)*16..i*16];
        let mut temp = [prev[12], prev[13], prev[14], prev[15]];
        // RotWord + SubWord + RCON
        temp.rotate_left(1);
        for b in temp.iter_mut() { *b = SBOX[*b as usize]; }
        temp[0] ^= RCON[i];
        let mut new = [0u8; 16];
        for j in 0..4 { new[j]    = prev[j]    ^ temp[j]; }
        for j in 0..4 { new[j+4]  = prev[j+4]  ^ new[j]; }
        for j in 0..4 { new[j+8]  = prev[j+8]  ^ new[j+4]; }
        for j in 0..4 { new[j+12] = prev[j+12] ^ new[j+8]; }
        rk[i*16..(i+1)*16].copy_from_slice(&new);
    }
    rk
}

/// AES-128 block encryption (in-place, column-major state).
pub fn aes128_encrypt(block: &mut [u8; 16], rk: &[u8; 176]) {
    add_round_key(block, &rk[..16]);
    for r in 1..10 {
        sub_bytes(block);
        shift_rows(block);
        mix_columns(block);
        add_round_key(block, &rk[r*16..(r+1)*16]);
    }
    sub_bytes(block);
    shift_rows(block);
    add_round_key(block, &rk[160..]);
}

// ═══════════════════════════════════════════════════════════════════════════════
// AES-128-CMAC — RFC 4493
// Used for EAPOL MIC computation with KCK (first 16 bytes of PTK).
// ═══════════════════════════════════════════════════════════════════════════════

fn xor16(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for i in 0..16 { out[i] = a[i] ^ b[i]; }
    out
}

fn generate_subkeys(rk: &[u8; 176]) -> ([u8; 16], [u8; 16]) {
    let mut l = [0u8; 16];
    aes128_encrypt(&mut l, rk);
    let k1 = if l[0] & 0x80 == 0 {
        let mut k = l;
        for i in 0..15 { k[i] = (k[i] << 1) | (k[i+1] >> 7); }
        k[15] <<= 1;
        k
    } else {
        let mut k = l;
        for i in 0..15 { k[i] = (k[i] << 1) | (k[i+1] >> 7); }
        k[15] <<= 1;
        k[15] ^= 0x87;
        k
    };
    let k2 = if k1[0] & 0x80 == 0 {
        let mut k = k1;
        for i in 0..15 { k[i] = (k[i] << 1) | (k[i+1] >> 7); }
        k[15] <<= 1;
        k
    } else {
        let mut k = k1;
        for i in 0..15 { k[i] = (k[i] << 1) | (k[i+1] >> 7); }
        k[15] <<= 1;
        k[15] ^= 0x87;
        k
    };
    (k1, k2)
}

/// Compute AES-128-CMAC over `msg` using `key` (16 bytes).
/// Returns 16-byte MAC.
pub fn aes_cmac(key: &[u8; 16], msg: &[u8]) -> [u8; 16] {
    let rk = aes128_key_expand(key);
    let (k1, k2) = generate_subkeys(&rk);

    let n = (msg.len() + 15) / 16;
    let (n, flag) = if n == 0 { (1, false) } else { (n, msg.len() % 16 == 0) };

    let mut x = [0u8; 16];
    for i in 0..n - 1 {
        let block: [u8; 16] = msg[i*16..i*16+16].try_into().unwrap_or([0;16]);
        x = xor16(&x, &block);
        aes128_encrypt(&mut x, &rk);
    }

    // Last block (with padding if needed)
    let mut last = [0u8; 16];
    let last_start = (n - 1) * 16;
    let last_len   = msg.len() - last_start;
    last[..last_len].copy_from_slice(&msg[last_start..]);
    let last = if flag {
        xor16(&last, &k1)
    } else {
        last[last_len] = 0x80; // padding
        xor16(&last, &k2)
    };
    x = xor16(&x, &last);
    aes128_encrypt(&mut x, &rk);
    x
}

/// Compute EAPOL MIC using KCK (first 16 bytes of PTK) and AES-128-CMAC.
pub fn eapol_mic(kck: &[u8; 16], eapol_frame: &[u8]) -> [u8; 16] {
    aes_cmac(kck, eapol_frame)
}

// ═══════════════════════════════════════════════════════════════════════════════
// CCMP — IEEE 802.11i §8.3.3
// AES-128-CTR for encryption + AES-128-CBC-MAC (CCM) for integrity.
// 8-byte MIC, 8-byte CCMP header, 48-bit Packet Number (PN).
// ═══════════════════════════════════════════════════════════════════════════════

/// Build the 13-byte CCM nonce for CCMP.
/// priority(1) || A2(6) || PN(6)   — A2 = transmitter MAC
fn ccmp_nonce(a2: &[u8; 6], pn: u64) -> [u8; 13] {
    let mut n = [0u8; 13];
    n[0] = 0; // priority = 0 (QoS not implemented)
    n[1..7].copy_from_slice(a2);
    n[7]  = ((pn >> 40) & 0xFF) as u8;
    n[8]  = ((pn >> 32) & 0xFF) as u8;
    n[9]  = ((pn >> 24) & 0xFF) as u8;
    n[10] = ((pn >> 16) & 0xFF) as u8;
    n[11] = ((pn >>  8) & 0xFF) as u8;
    n[12] = ( pn        & 0xFF) as u8;
    n
}

/// Build CCMP AAD from 802.11 MAC header (IEEE 802.11i §8.3.3.3.2).
/// Clears WEP bit, zeroes Retry+PwrMgmt+MoreData, zeroes sequence number low bits.
fn ccmp_aad(mac_hdr: &[u8; 24]) -> [u8; 24] {
    let mut aad = *mac_hdr;
    aad[1] &= !0x40; // clear WEP/Protected bit
    aad[1] &= !0x38; // clear Retry, PwrMgmt, MoreData
    aad[22] &= 0x0F; // clear sequence number bits [15:4], keep fragment bits
    aad[23] = 0;
    aad
}

/// AES-128-CCM MIC computation over AAD and plaintext.
fn ccmp_cbcmac(tk: &[u8; 16], nonce: &[u8; 13], aad: &[u8; 24], plaintext: &[u8]) -> [u8; 8] {
    let rk = aes128_key_expand(tk);
    // Flags: Adata=1, M=(8-2)/2=3, L=len(q)-1=1 → 0x59
    let mut b0 = [0u8; 16];
    b0[0] = 0x59;
    b0[1..14].copy_from_slice(nonce);
    let plen = plaintext.len() as u16;
    b0[14] = (plen >> 8) as u8;
    b0[15] = (plen & 0xFF) as u8;

    // CBC-MAC over B0
    let mut x = b0;
    aes128_encrypt(&mut x, &rk);

    // AAD: length(2) || aad(24) → 26 bytes, padded to 32 (two blocks)
    let mut aad_block = [0u8; 32];
    aad_block[0] = 0; aad_block[1] = 24; // len = 24
    aad_block[2..26].copy_from_slice(aad);
    for block in aad_block.chunks(16) {
        let b: [u8; 16] = block.try_into().unwrap_or([0;16]);
        for i in 0..16 { x[i] ^= b[i]; }
        aes128_encrypt(&mut x, &rk);
    }

    // Plaintext blocks
    let padded_len = (plaintext.len() + 15) & !15;
    let mut i = 0;
    while i < padded_len {
        let mut b = [0u8; 16];
        let end = (i + 16).min(plaintext.len());
        b[..end - i].copy_from_slice(&plaintext[i..end]);
        for j in 0..16 { x[j] ^= b[j]; }
        aes128_encrypt(&mut x, &rk);
        i += 16;
    }

    // Encrypt T (CBC-MAC output) with S_0 (counter=0)
    let mut s0 = [0u8; 16];
    s0[0] = 0x01; // L-1 = 1
    s0[1..14].copy_from_slice(nonce);
    s0[14] = 0; s0[15] = 0; // counter = 0
    aes128_encrypt(&mut s0, &rk);

    let mut mic = [0u8; 8];
    for i in 0..8 { mic[i] = x[i] ^ s0[i]; }
    mic
}

/// AES-128-CTR keystream starting at counter=2 (counter=1 is used for MIC).
fn ccmp_ctr(tk: &[u8; 16], nonce: &[u8; 13], data: &mut [u8]) {
    let rk = aes128_key_expand(tk);
    let mut counter: u16 = 1;
    let mut i = 0;
    while i < data.len() {
        counter += 1;
        let mut ctr_blk = [0u8; 16];
        ctr_blk[0] = 0x01;
        ctr_blk[1..14].copy_from_slice(nonce);
        ctr_blk[14] = (counter >> 8) as u8;
        ctr_blk[15] = (counter & 0xFF) as u8;
        aes128_encrypt(&mut ctr_blk, &rk);
        let end = data.len().min(i + 16);
        for j in i..end { data[j] ^= ctr_blk[j - i]; }
        i += 16;
    }
}

/// CCMP encrypt: takes plaintext payload, returns ciphertext || MIC(8).
/// `mac_hdr`: 24-byte 802.11 MAC header. `a2`: transmitter MAC. `pn`: packet number.
pub fn ccmp_encrypt(tk: &[u8; 16], mac_hdr: &[u8; 24], a2: &[u8; 6],
                    pn: u64, plaintext: &[u8]) -> alloc::vec::Vec<u8> {
    let nonce = ccmp_nonce(a2, pn);
    let aad   = ccmp_aad(mac_hdr);
    let mic   = ccmp_cbcmac(tk, &nonce, &aad, plaintext);
    let mut ct = plaintext.to_vec();
    ccmp_ctr(tk, &nonce, &mut ct);
    ct.extend_from_slice(&mic);
    ct
}

/// CCMP decrypt and verify: strips CCMP header, decrypts, checks 8-byte MIC.
/// Returns Some(plaintext) on success, None on MIC failure or malformed input.
/// `frame`: full 802.11 data frame starting from FC. Must be ≥ 32+8=40 bytes.
pub fn ccmp_decrypt(tk: &[u8; 16], frame: &[u8]) -> Option<alloc::vec::Vec<u8>> {
    if frame.len() < 40 { return None; }

    // Extract MAC header (24 bytes) and CCMP header (8 bytes)
    let mac_hdr: &[u8; 24] = frame[0..24].try_into().ok()?;
    let ccmp_hdr = &frame[24..32];

    // Reconstruct PN from CCMP header (IEEE 802.11i §8.3.3.2 Figure 46):
    // PN0=hdr[0], PN1=hdr[1], hdr[2]=0, hdr[3]=KeyID(bit5)|ExtIV(bit5),
    // PN2=hdr[4], PN3=hdr[5], PN4=hdr[6], PN5=hdr[7]
    let pn = (ccmp_hdr[7] as u64) << 40 | (ccmp_hdr[6] as u64) << 32
           | (ccmp_hdr[5] as u64) << 24 | (ccmp_hdr[4] as u64) << 16
           | (ccmp_hdr[1] as u64) <<  8 | (ccmp_hdr[0] as u64);

    // Ciphertext = frame[32..len-8], MIC = frame[len-8..len]
    if frame.len() < 40 { return None; }
    let ct_end = frame.len() - 8;
    let ciphertext = &frame[32..ct_end];
    let rx_mic = &frame[ct_end..];

    // Decrypt with CTR
    let mut plaintext = ciphertext.to_vec();
    let nonce = ccmp_nonce(&mac_hdr[10..16].try_into().ok()?, pn); // A2 = Addr2
    ccmp_ctr(tk, &nonce, &mut plaintext);

    // Verify MIC
    let aad       = ccmp_aad(mac_hdr);
    let nonce_arr = ccmp_nonce(&mac_hdr[10..16].try_into().ok()?, pn);
    let expected  = ccmp_cbcmac(tk, &nonce_arr, &aad, &plaintext);
    if expected != rx_mic { return None; }

    Some(plaintext)
}
