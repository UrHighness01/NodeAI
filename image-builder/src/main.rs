//! NodeAI image-builder
//!
//! Host-side tool that wraps a compiled `nodeai-kernel` ELF binary in a
//! bootable BIOS (MBR) and UEFI disk image using the `bootloader` crate.
//!
//! Usage:
//!   image-builder <kernel-elf-path> <output-dir>
//!
//! Outputs:
//!   <output-dir>/nodeai-bios.img  — legacy BIOS bootable raw disk image
//!   <output-dir>/nodeai-uefi.img  — UEFI bootable fat32 disk image

use std::{env, path::PathBuf};

fn main() {
    let mut args = env::args().skip(1);
    let kernel_path: PathBuf = args
        .next()
        .expect("Usage: image-builder <kernel-elf> <output-dir>")
        .into();
    let out_dir: PathBuf = args
        .next()
        .expect("Usage: image-builder <kernel-elf> <output-dir>")
        .into();

    if !kernel_path.exists() {
        eprintln!("Error: kernel binary not found at {}", kernel_path.display());
        std::process::exit(1);
    }

    std::fs::create_dir_all(&out_dir).expect("failed to create output directory");

    // ── BIOS image ────────────────────────────────────────────────────────────
    let bios_path = out_dir.join("nodeai-bios.img");
    println!("Building BIOS image → {}", bios_path.display());
    bootloader::BiosBoot::new(&kernel_path)
        .create_disk_image(&bios_path)
        .unwrap_or_else(|e| {
            eprintln!("BIOS image error: {e}");
            std::process::exit(1);
        });
    println!("  ✓ BIOS: {}", bios_path.display());

    // ── UEFI image ────────────────────────────────────────────────────────────
    let uefi_path = out_dir.join("nodeai-uefi.img");
    println!("Building UEFI image → {}", uefi_path.display());
    bootloader::UefiBoot::new(&kernel_path)
        .create_disk_image(&uefi_path)
        .unwrap_or_else(|e| {
            eprintln!("UEFI image error: {e}");
            std::process::exit(1);
        });
    println!("  ✓ UEFI: {}", uefi_path.display());
}
