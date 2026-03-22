//! ELF-64 binary loader — Phase 11 + Phase 21 (dynamic linker support).
//!
//! Loads static and dynamic ELF64 executables.
//! Phase 21 adds PT_INTERP support: the kernel loads the dynamic linker
//! (`/lib/ld-musl-x86_64.so.1`) and passes control to it, mirroring Linux.

use alloc::vec::Vec;
use alloc::string::String;

// ── ELF constants ─────────────────────────────────────────────────────────────

const ELFMAG:        [u8; 4] = [0x7f, b'E', b'L', b'F'];
const ELFCLASS64:    u8      = 2;
const ELFDATA2LSB:   u8      = 1;   // little-endian
const ET_EXEC:       u16     = 2;   // executable
const ET_DYN:        u16     = 3;   // shared object / PIE
const EM_X86_64:     u16     = 62;
const PT_LOAD:       u32     = 1;
const PT_INTERP:     u32     = 3;
const PT_PHDR:       u32     = 6;
const PT_GNU_STACK:  u32     = 0x6474_e551;

// ELF segment flags
const PF_X: u32 = 1;   // execute
const PF_W: u32 = 2;   // write
const PF_R: u32 = 4;   // read

// ── ELF-64 structures ─────────────────────────────────────────────────────────

/// ELF-64 file header (64 bytes).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Elf64Ehdr {
    pub e_ident:     [u8; 16],
    pub e_type:      u16,
    pub e_machine:   u16,
    pub e_version:   u32,
    pub e_entry:     u64,
    pub e_phoff:     u64,
    pub e_shoff:     u64,
    pub e_flags:     u32,
    pub e_ehsize:    u16,
    pub e_phentsize: u16,
    pub e_phnum:     u16,
    pub e_shentsize: u16,
    pub e_shnum:     u16,
    pub e_shstrndx:  u16,
}

/// ELF-64 program header (56 bytes).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Elf64Phdr {
    pub p_type:   u32,
    pub p_flags:  u32,
    pub p_offset: u64,
    pub p_vaddr:  u64,
    pub p_paddr:  u64,
    pub p_filesz: u64,
    pub p_memsz:  u64,
    pub p_align:  u64,
}

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElfError {
    /// Not a valid ELF binary (bad magic or class).
    NotElf,
    /// Wrong architecture or endianness.
    BadArch,
    /// Not an executable (shared objects, relocatables not supported yet).
    NotExecutable,
    /// Program header table is out of bounds.
    BadPhdrs,
    /// A PT_LOAD segment references data outside the file.
    SegmentOutOfBounds,
    /// Binary requires dynamic linking (PT_INTERP present).
    NeedsDynLinker,
    /// Input slice is too small.
    Truncated,
}

/// A mapped segment to be loaded into the address space.
#[derive(Debug, Clone)]
pub struct LoadedSegment {
    /// Virtual address where the segment should begin.
    pub vaddr:  u64,
    /// Page-aligned size in memory (may be larger than file data due to BSS).
    pub memsz:  u64,
    /// File data to copy to the beginning of the region.
    pub data:   Vec<u8>,
    /// Segment flags (PF_R / PF_W / PF_X).
    pub flags:  u32,
}

/// Result of parsing an ELF binary.
pub struct ElfImage {
    /// Virtual address of the program entry point.
    pub entry:    u64,
    /// Segments to load, in order.
    pub segments: Vec<LoadedSegment>,
    /// PT_INTERP path (dynamic linker), present for dynamically-linked binaries.
    pub interp:   Option<String>,
    /// Program header virtual address (for AT_PHDR auxv entry).
    pub phdr_vaddr: Option<u64>,
    /// Number of program headers.
    pub phnum:    u16,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse a static ELF64 binary and return an `ElfImage` describing what to map.
///
/// This does **not** create any page-table entries; the caller (process creation
/// path) is responsible for calling the VMM to map each `LoadedSegment`.
///
/// # Errors
/// Returns `ElfError` if the binary is invalid, wrong architecture, or dynamic.
pub fn parse(data: &[u8]) -> Result<ElfImage, ElfError> {
    // Minimum size: ELF header is 64 bytes.
    if data.len() < core::mem::size_of::<Elf64Ehdr>() {
        return Err(ElfError::Truncated);
    }

    // SAFETY: We just verified the slice is at least sizeof(Elf64Ehdr) bytes.
    let ehdr: &Elf64Ehdr = unsafe {
        &*(data.as_ptr() as *const Elf64Ehdr)
    };

    // Magic / class / endianness / version
    if ehdr.e_ident[0..4] != ELFMAG {
        return Err(ElfError::NotElf);
    }
    if ehdr.e_ident[4] != ELFCLASS64 {
        return Err(ElfError::NotElf);
    }
    if ehdr.e_ident[5] != ELFDATA2LSB {
        return Err(ElfError::BadArch);  // big-endian not supported
    }
    if ehdr.e_ident[6] != 1 {
        return Err(ElfError::NotElf);   // EV_CURRENT
    }

    // Architecture
    if ehdr.e_machine != EM_X86_64 {
        return Err(ElfError::BadArch);
    }
    if ehdr.e_type != ET_EXEC && ehdr.e_type != ET_DYN {
        return Err(ElfError::NotExecutable);
    }

    // Validate program header table bounds
    let phdr_size = core::mem::size_of::<Elf64Phdr>();
    if ehdr.e_phentsize as usize != phdr_size {
        return Err(ElfError::BadPhdrs);
    }
    let phdrs_start = ehdr.e_phoff as usize;
    let phdrs_end   = phdrs_start
        .checked_add((ehdr.e_phnum as usize).checked_mul(phdr_size).ok_or(ElfError::BadPhdrs)?)
        .ok_or(ElfError::BadPhdrs)?;
    if phdrs_end > data.len() {
        return Err(ElfError::BadPhdrs);
    }

    let mut segments = Vec::new();
    let mut interp: Option<String> = None;
    let mut phdr_vaddr: Option<u64> = None;

    for i in 0..ehdr.e_phnum as usize {
        let off = phdrs_start + i * phdr_size;
        // SAFETY: bounds verified above.
        let phdr: &Elf64Phdr = unsafe {
            &*(data.as_ptr().add(off) as *const Elf64Phdr)
        };

        match phdr.p_type {
            PT_INTERP => {
                // Phase 21: extract interpreter path instead of failing
                let start = phdr.p_offset as usize;
                let end   = start.saturating_add(phdr.p_filesz as usize);
                if end <= data.len() {
                    let raw = &data[start..end];
                    // Strip NUL terminator
                    let path_bytes = raw.split(|&b| b == 0).next().unwrap_or(raw);
                    if let Ok(s) = core::str::from_utf8(path_bytes) {
                        interp = Some(String::from(s));
                    }
                }
            }
            PT_PHDR => {
                phdr_vaddr = Some(phdr.p_vaddr);
            }
            PT_LOAD => {
                // Validate file range
                let file_start = phdr.p_offset as usize;
                let file_end   = file_start
                    .checked_add(phdr.p_filesz as usize)
                    .ok_or(ElfError::SegmentOutOfBounds)?;
                if file_end > data.len() {
                    return Err(ElfError::SegmentOutOfBounds);
                }

                // memsz must be >= filesz
                if phdr.p_memsz < phdr.p_filesz {
                    return Err(ElfError::SegmentOutOfBounds);
                }

                let seg_data = data[file_start..file_end].to_vec();

                segments.push(LoadedSegment {
                    vaddr:  phdr.p_vaddr,
                    memsz:  phdr.p_memsz,
                    data:   seg_data,
                    flags:  phdr.p_flags,
                });
            }
            PT_GNU_STACK => {
                // Honour NX stack flag in the future; for now just note it.
                if phdr.p_flags & PF_X != 0 {
                    crate::klog!(WARN, "ELF: executable stack requested — ignored (NX enforced)");
                }
            }
            _ => {} // other segment types ignored (PT_NOTE, PT_PHDR, etc.)
        }
    }

    Ok(ElfImage { entry: ehdr.e_entry, segments, interp, phdr_vaddr, phnum: ehdr.e_phnum })
}

/// Map the loaded segments described by `image` into the currently-active
/// page table using the kernel's VMM.
///
/// Each PT_LOAD segment is:
///  1. Mapped as user-accessible pages (permissions applied per ELF flags).
///  2. File data copied to the virtual address.
///  3. BSS region (memsz > filesz) zeroed.
///
/// For dynamically-linked binaries (interp != None), the dynamic linker is
/// also loaded and its entry point is returned instead of the app entry.
///
/// Returns the entry point virtual address on success.
///
/// # Safety
/// `image` must have been produced by `parse()`.  The caller must ensure no
/// conflicting mappings exist at the segment virtual addresses.
pub unsafe fn load_image(image: &ElfImage) -> Result<u64, ElfError> {

    for seg in &image.segments {
        // Align virtual address down to page boundary.
        let page_size:  u64 = 0x1000;
        let vaddr_aligned = seg.vaddr & !(page_size - 1);
        let page_offset   = seg.vaddr - vaddr_aligned;
        let total_size    = (page_offset + seg.memsz + page_size - 1) & !(page_size - 1);

        let writable   = seg.flags & PF_W != 0;
        let executable = seg.flags & PF_X != 0;

        // Map fresh zeroed physical frames into the address space.
        crate::memory::map_user_range(vaddr_aligned, total_size, writable, executable)
            .map_err(|_| ElfError::SegmentOutOfBounds)?;

        // Copy file data to the virtual address.
        let dst = seg.vaddr as *mut u8;
        core::ptr::copy_nonoverlapping(seg.data.as_ptr(), dst, seg.data.len());

        // Zero BSS (memsz - filesz bytes after the file data).
        let bss_start  = (seg.vaddr + seg.data.len() as u64) as *mut u8;
        let bss_len    = (seg.memsz - seg.data.len() as u64) as usize;
        if bss_len > 0 {
            core::ptr::write_bytes(bss_start, 0, bss_len);
        }
    }

    // Phase 21: if this binary needs a dynamic linker, load it and hand control
    // to its entry point.  The linker will map shared libs and then jump to the
    // app's entry.
    if let Some(ref interp_path) = image.interp {
        crate::klog!(INFO, "ELF: loading dynamic linker '{}'", interp_path);
        if let Some(entry) = load_interp(interp_path) {
            return Ok(entry);
        }
        crate::klog!(WARN, "ELF: dynamic linker '{}' not found, falling back to static entry", interp_path);
    }

    Ok(image.entry)
}

/// Load the dynamic linker ELF from VFS at `path` and return its entry point.
/// The linker is mapped at a fixed base address (0x0000_0040_0000_0000 — above
/// the application but below the stack) to keep it away from app segments.
const LDSO_LOAD_BIAS: u64 = 0x0000_7F00_0000_0000;

unsafe fn load_interp(path: &str) -> Option<u64> {
    use crate::vfs;
    let node = vfs::lookup(path).ok()?;
    let mut fh = node.open().ok()?;
    let mut buf = alloc::vec![0u8; 8 * 1024 * 1024]; // 8 MiB
    let n = fh.read(&mut buf).ok()?;
    let ldso_data = &buf[..n];

    // Parse the dynamic linker ELF (it's an ET_DYN shared object)
    let ldso = parse_dyn_so(ldso_data)?;

    // Map each segment at LDSO_LOAD_BIAS + p_vaddr
    for seg in &ldso.segments {
        let page_size: u64 = 0x1000;
        let vaddr     = LDSO_LOAD_BIAS + seg.vaddr;
        let vaddr_al  = vaddr & !(page_size - 1);
        let off       = vaddr - vaddr_al;
        let total     = (off + seg.memsz + page_size - 1) & !(page_size - 1);

        let writable   = seg.flags & PF_W != 0;
        let executable = seg.flags & PF_X != 0;

        crate::memory::map_user_range(vaddr_al, total, writable, executable).ok()?;
        let dst = vaddr as *mut u8;
        core::ptr::copy_nonoverlapping(seg.data.as_ptr(), dst, seg.data.len());
        let bss_start = (vaddr + seg.data.len() as u64) as *mut u8;
        let bss_len   = (seg.memsz - seg.data.len() as u64) as usize;
        if bss_len > 0 {
            core::ptr::write_bytes(bss_start, 0, bss_len);
        }
    }

    Some(LDSO_LOAD_BIAS + ldso.entry)
}

/// Parse an ET_DYN ELF (shared object / dynamic linker).
/// Same as `parse()` but accepts ET_DYN and skips PT_INTERP.
fn parse_dyn_so(data: &[u8]) -> Option<ElfImage> {
    if data.len() < core::mem::size_of::<Elf64Ehdr>() { return None; }
    let ehdr: &Elf64Ehdr = unsafe { &*(data.as_ptr() as *const Elf64Ehdr) };
    if ehdr.e_ident[0..4] != ELFMAG { return None; }
    if ehdr.e_ident[4] != ELFCLASS64 { return None; }
    if ehdr.e_machine != EM_X86_64 { return None; }

    let phdr_sz   = core::mem::size_of::<Elf64Phdr>();
    let ph_start  = ehdr.e_phoff as usize;
    let ph_end    = ph_start + ehdr.e_phnum as usize * phdr_sz;
    if ph_end > data.len() { return None; }

    let mut segments = Vec::new();
    for i in 0..ehdr.e_phnum as usize {
        let off = ph_start + i * phdr_sz;
        let phdr: &Elf64Phdr = unsafe { &*(data.as_ptr().add(off) as *const Elf64Phdr) };
        if phdr.p_type != PT_LOAD { continue; }
        let fs = phdr.p_offset as usize;
        let fe = fs + phdr.p_filesz as usize;
        if fe > data.len() { return None; }
        segments.push(LoadedSegment {
            vaddr: phdr.p_vaddr,
            memsz: phdr.p_memsz,
            data:  data[fs..fe].to_vec(),
            flags: phdr.p_flags,
        });
    }
    Some(ElfImage { entry: ehdr.e_entry, segments, interp: None, phdr_vaddr: None, phnum: ehdr.e_phnum })
}
