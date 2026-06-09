//! ELF-64 binary loader with static + dynamic (PT_INTERP) linking support.
//!
//! Loads static and dynamic ELF64 executables.
//! PT_INTERP support: the kernel loads the dynamic linker
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
                // PT_INTERP: extract interpreter path instead of failing
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
                // NX stack is enforced by page table flags (stack segment mapped without X bit).
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

    // If this binary needs a dynamic linker (PT_INTERP), load it and hand control
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

// ── ELF relocation engine for kernel modules ──────────────────────────────────
//
// Supports relocatable ET_REL ELF objects (the output of `rustc --crate-type=cdylib`
// or `cc -r`).  Processes SHT_RELA sections applying:
//   R_X86_64_64    — absolute 64-bit address
//   R_X86_64_PC32  — 32-bit PC-relative (call / jmp)
//   R_X86_64_PLT32 — same as PC32 for our purposes (no PLT needed in-kernel)
//   R_X86_64_32    — 32-bit absolute (zero-extended)
//   R_X86_64_32S   — 32-bit absolute (sign-extended)
//
// The linker walks every SHT_RELA section, resolves the symbol against either
// the module's own section table or the kernel's exported symbol table, and
// patches the relocated bytes in place.

const ET_REL:    u16 = 1;   // relocatable object
const SHT_NULL:  u32 = 0;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB:   u32 = 2;
const SHT_STRTAB:   u32 = 3;
const SHT_RELA:     u32 = 4;
const SHT_NOBITS:   u32 = 8;  // BSS

const R_X86_64_NONE:  u32 = 0;
const R_X86_64_64:    u32 = 1;
const R_X86_64_PC32:  u32 = 2;
const R_X86_64_32:    u32 = 10;
const R_X86_64_32S:   u32 = 11;
const R_X86_64_PLT32: u32 = 4;

const STB_GLOBAL: u8 = 1;
const STB_WEAK:   u8 = 2;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Elf64Shdr {
    pub sh_name:      u32,
    pub sh_type:      u32,
    pub sh_flags:     u64,
    pub sh_addr:      u64,
    pub sh_offset:    u64,
    pub sh_size:      u64,
    pub sh_link:      u32,
    pub sh_info:      u32,
    pub sh_addralign: u64,
    pub sh_entsize:   u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Elf64Sym {
    st_name:  u32,
    st_info:  u8,
    st_other: u8,
    st_shndx: u16,
    st_value: u64,
    st_size:  u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Elf64Rela {
    r_offset: u64,
    r_info:   u64,
    r_addend: i64,
}

impl Elf64Rela {
    fn sym_idx(&self)  -> usize { (self.r_info >> 32) as usize }
    fn rel_type(&self) -> u32   { (self.r_info & 0xFFFF_FFFF) as u32 }
}

/// Error type for the relocation engine.
#[derive(Debug)]
pub enum RelocError {
    NotRelocatable,
    BadSectionTable,
    MissingSymbol(String),
    Overflow32,
    UnsupportedReloc(u32),
    BadStrtab,
}

/// Kernel-exported symbol table entry.  Modules can call any symbol listed here.
pub struct KernelExport {
    pub name: &'static str,
    pub addr: u64,
}

/// Resolve a kernel symbol name to its address.
/// Function pointer → integer casts must happen at runtime, not const-eval time.
pub fn resolve_kernel_symbol(name: &str) -> Option<u64> {
    // These casts are evaluated at call time (runtime), not during const-init.
    let exports: &[(&str, u64)] = &[
        ("scheduler_uptime_ms",   crate::scheduler::uptime_ms   as *const () as u64),
        ("scheduler_current_pid", crate::scheduler::current_pid as *const () as u64),
        ("scheduler_send_signal", crate::scheduler::send_signal as *const () as u64),
        ("vfs_write_file",        crate::vfs::write_file        as *const () as u64),
        ("vfs_read_file",         crate::vfs::read_file         as *const () as u64),
        ("entropy_fill",          crate::entropy::fill          as *const () as u64),
        ("entropy_stir",          crate::entropy::stir          as *const () as u64),
    ];
    exports.iter().find(|(n, _)| *n == name).map(|(_, a)| *a)
}

/// Relocate a loaded module image in `buf` (mutable byte slice at `load_addr`).
///
/// `data` is the original ELF file bytes.  `buf` must be the writable mapping
/// of the module already allocated at `load_addr`.
///
/// Returns `Ok(entry_addr)` — the virtual address of `module_init` if found, or
/// `load_addr` as a fallback.
pub fn relocate_module(data: &[u8], buf: &mut [u8], load_addr: u64) -> Result<u64, RelocError> {
    if data.len() < core::mem::size_of::<Elf64Ehdr>() {
        return Err(RelocError::NotRelocatable);
    }
    let ehdr: &Elf64Ehdr = unsafe { &*(data.as_ptr() as *const Elf64Ehdr) };
    if ehdr.e_ident[0..4] != ELFMAG || ehdr.e_type != ET_REL {
        return Err(RelocError::NotRelocatable);
    }

    let shdr_size = core::mem::size_of::<Elf64Shdr>();
    let sh_off    = ehdr.e_shoff as usize;
    let sh_count  = ehdr.e_shnum as usize;
    if sh_off + sh_count * shdr_size > data.len() {
        return Err(RelocError::BadSectionTable);
    }

    // ── Build per-section load addresses ─────────────────────────────────────
    // For ET_REL, sections have no pre-assigned VA.  We lay them out linearly
    // in the module buffer starting from load_addr.
    let mut section_addrs: Vec<u64> = Vec::with_capacity(sh_count);
    let mut cursor: u64 = load_addr;
    for i in 0..sh_count {
        let shdr = shdr_at(data, sh_off, i, shdr_size)?;
        if shdr.sh_type == SHT_NOBITS || shdr.sh_type == SHT_NULL {
            section_addrs.push(cursor);
        } else if shdr.sh_size > 0 {
            let align = shdr.sh_addralign.max(1);
            cursor = (cursor + align - 1) & !(align - 1);
            section_addrs.push(cursor);
            // Copy section data into module buffer.
            let off = shdr.sh_offset as usize;
            let sz  = shdr.sh_size as usize;
            if off + sz <= data.len() {
                let buf_off = (cursor - load_addr) as usize;
                if buf_off + sz <= buf.len() {
                    buf[buf_off..buf_off + sz].copy_from_slice(&data[off..off + sz]);
                }
            }
            cursor += shdr.sh_size;
        } else {
            section_addrs.push(cursor);
        }
    }

    // ── Find .symtab and .strtab ──────────────────────────────────────────────
    let (symtab_shdr, strtab_shdr) = find_symtab(data, sh_off, sh_count, shdr_size)?;
    let syms = read_symbols(data, &symtab_shdr);
    let strtab_off = strtab_shdr.sh_offset as usize;
    let strtab_end = strtab_off + strtab_shdr.sh_size as usize;
    let strtab = if strtab_end <= data.len() { &data[strtab_off..strtab_end] } else { &[] };

    // ── Apply RELA relocations ────────────────────────────────────────────────
    for i in 0..sh_count {
        let shdr = shdr_at(data, sh_off, i, shdr_size)?;
        if shdr.sh_type != SHT_RELA { continue; }

        let target_sec = shdr.sh_info as usize; // section being relocated
        if target_sec >= sh_count { continue; }
        let target_addr = section_addrs[target_sec];

        let rela_off  = shdr.sh_offset as usize;
        let rela_size = core::mem::size_of::<Elf64Rela>();
        let rela_count = (shdr.sh_size as usize) / rela_size;

        for r in 0..rela_count {
            let rela: Elf64Rela = unsafe {
                core::ptr::read_unaligned(
                    data.as_ptr().add(rela_off + r * rela_size) as *const Elf64Rela
                )
            };

            let sym_idx = rela.sym_idx();
            let sym = if sym_idx < syms.len() { syms[sym_idx] } else { continue };

            // Resolve symbol value.
            let sym_val: u64 = if sym.st_shndx == 0 {
                // External symbol — look up in kernel exports or module sections.
                let sym_name = read_str(strtab, sym.st_name as usize);
                resolve_kernel_symbol(sym_name)
                    .ok_or_else(|| RelocError::MissingSymbol(
                        alloc::format!("{}", sym_name)
                    ))?
            } else if (sym.st_shndx as usize) < sh_count {
                section_addrs[sym.st_shndx as usize] + sym.st_value
            } else {
                continue;
            };

            // Patch location in module buffer.
            let patch_va  = target_addr + rela.r_offset;
            let patch_off = (patch_va - load_addr) as usize;
            let addend    = rela.r_addend;

            apply_reloc(buf, patch_off, rela.rel_type(), sym_val, patch_va, addend)?;
        }
    }

    // ── Find module_init entry point ──────────────────────────────────────────
    let entry = find_symbol(strtab, &syms, "module_init", &section_addrs)
        .unwrap_or(load_addr);

    Ok(entry)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn shdr_at(data: &[u8], sh_off: usize, idx: usize, shdr_size: usize)
    -> Result<Elf64Shdr, RelocError>
{
    let off = sh_off + idx * shdr_size;
    if off + shdr_size > data.len() { return Err(RelocError::BadSectionTable); }
    Ok(unsafe { core::ptr::read_unaligned(data.as_ptr().add(off) as *const Elf64Shdr) })
}

fn find_symtab(data: &[u8], sh_off: usize, sh_count: usize, shdr_size: usize)
    -> Result<(Elf64Shdr, Elf64Shdr), RelocError>
{
    let mut symtab = None;
    for i in 0..sh_count {
        let sh = shdr_at(data, sh_off, i, shdr_size)?;
        if sh.sh_type == SHT_SYMTAB { symtab = Some((i, sh)); break; }
    }
    let (symtab_idx, symtab_sh) = symtab.ok_or(RelocError::BadSectionTable)?;
    // sh_link for SYMTAB points to the associated STRTAB.
    let strtab_idx = symtab_sh.sh_link as usize;
    let strtab_sh  = shdr_at(data, sh_off, strtab_idx, shdr_size)
        .map_err(|_| RelocError::BadStrtab)?;
    Ok((symtab_sh, strtab_sh))
}

fn read_symbols(data: &[u8], symtab: &Elf64Shdr) -> Vec<Elf64Sym> {
    let sym_size  = core::mem::size_of::<Elf64Sym>();
    let off       = symtab.sh_offset as usize;
    let count     = (symtab.sh_size as usize) / sym_size;
    let mut syms  = Vec::with_capacity(count);
    for i in 0..count {
        let p = off + i * sym_size;
        if p + sym_size > data.len() { break; }
        let s: Elf64Sym = unsafe { core::ptr::read_unaligned(data.as_ptr().add(p) as *const Elf64Sym) };
        syms.push(s);
    }
    syms
}

fn read_str(strtab: &[u8], off: usize) -> &str {
    if off >= strtab.len() { return ""; }
    let slice = &strtab[off..];
    let end   = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
    core::str::from_utf8(&slice[..end]).unwrap_or("")
}

fn find_symbol(strtab: &[u8], syms: &[Elf64Sym], name: &str,
               section_addrs: &[u64]) -> Option<u64>
{
    for s in syms {
        if read_str(strtab, s.st_name as usize) == name
            && (s.st_shndx as usize) < section_addrs.len()
        {
            return Some(section_addrs[s.st_shndx as usize] + s.st_value);
        }
    }
    None
}

fn apply_reloc(buf: &mut [u8], off: usize, rel_type: u32,
               sym: u64, pc: u64, addend: i64) -> Result<(), RelocError>
{
    match rel_type {
        R_X86_64_NONE => {}

        R_X86_64_64 => {
            // S + A
            let val = sym.wrapping_add(addend as u64);
            if off + 8 > buf.len() { return Ok(()); }
            buf[off..off+8].copy_from_slice(&val.to_le_bytes());
        }

        R_X86_64_PC32 | R_X86_64_PLT32 => {
            // S + A - P  (32-bit PC-relative)
            let val = (sym as i64).wrapping_add(addend).wrapping_sub(pc as i64);
            if val > i32::MAX as i64 || val < i32::MIN as i64 {
                return Err(RelocError::Overflow32);
            }
            if off + 4 > buf.len() { return Ok(()); }
            buf[off..off+4].copy_from_slice(&(val as i32).to_le_bytes());
        }

        R_X86_64_32 => {
            // S + A, zero-extended into 32 bits
            let val = sym.wrapping_add(addend as u64);
            if val > u32::MAX as u64 { return Err(RelocError::Overflow32); }
            if off + 4 > buf.len() { return Ok(()); }
            buf[off..off+4].copy_from_slice(&(val as u32).to_le_bytes());
        }

        R_X86_64_32S => {
            // S + A, sign-extended into 32 bits
            let val = (sym as i64).wrapping_add(addend);
            if val > i32::MAX as i64 || val < i32::MIN as i64 {
                return Err(RelocError::Overflow32);
            }
            if off + 4 > buf.len() { return Ok(()); }
            buf[off..off+4].copy_from_slice(&(val as i32).to_le_bytes());
        }

        other => return Err(RelocError::UnsupportedReloc(other)),
    }
    Ok(())
}
