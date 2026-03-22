/*
 * NodeAI Kernel Linker Script
 *
 * Memory layout for x86_64 bare-metal kernel:
 *   - Kernel loaded at virtual 0xFFFFFFFF80000000 (higher half, -2 GiB)
 *   - Physical load address: 0x100000 (1 MiB) — above BIOS area
 *
 * When using bootloader-api the bootloader sets up the page tables and
 * loads the kernel ELF at the addresses specified here.
 */

OUTPUT_FORMAT(elf64-x86-64)
OUTPUT_ARCH(i386:x86-64)
ENTRY(_start)

/* Kernel virtual base (higher half) */
KERNEL_VMA = 0xFFFFFFFF80000000;
/* Physical load address */
KERNEL_LMA = 0x00100000;

SECTIONS {
    /* ── text (executable) ──────────────────────────────────────────── */
    .text KERNEL_VMA : AT(KERNEL_LMA) {
        KEEP(*(.boot))          /* boot header must be first */
        *(.text .text.*)
        *(.gnu.linkonce.t.*)

        . = ALIGN(4096);
    }

    /* ── read-only data ─────────────────────────────────────────────── */
    .rodata ALIGN(4096) : {
        *(.rodata .rodata.*)
        *(.gnu.linkonce.r.*)
        . = ALIGN(4096);
    }

    /* ── read-write data (initialized) ─────────────────────────────── */
    .data ALIGN(4096) : {
        *(.data .data.*)
        *(.gnu.linkonce.d.*)
        . = ALIGN(4096);
    }

    /* ── BSS (zero-initialized) ─────────────────────────────────────── */
    .bss ALIGN(4096) : {
        _bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(4096);
        _bss_end = .;
    }

    /* ── Stack (16 KiB, page aligned) ──────────────────────────────── */
    .stack ALIGN(4096) : {
        _stack_bottom = .;
        . += 0x4000;    /* 16 KiB initial kernel stack */
        _stack_top = .;
    }

    /* Provide total kernel size for boot assertions */
    _kernel_end = .;
    _kernel_size = _kernel_end - KERNEL_VMA;

    /* ── Discard debug / metadata not needed at runtime ────────────── */
    /DISCARD/ : {
        *(.comment)
        *(.eh_frame)
        *(.note.GNU-stack)
    }
}
