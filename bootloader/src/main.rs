//! NodeAI Bootloader entry — delegates to the `bootloader` crate.
//! The bootloader crate handles: real mode → protected mode → long mode,
//! page table setup, physical memory mapping, then calls kernel_main.

#![no_std]
#![no_main]

use bootloader::BootInfo;

/// Boot entry — called by the bootloader crate after CPU is in 64-bit mode.
/// Immediately hands off to the kernel crate.
#[no_mangle]
pub extern "C" fn _start(boot_info: &'static mut BootInfo) -> ! {
    // The bootloader crate adds this via its own linker magic.
    // We just need this stub to link correctly.
    // In practice, `bootloader` v0.11 uses `entry_point!` macro in the kernel crate.
    loop {}
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
