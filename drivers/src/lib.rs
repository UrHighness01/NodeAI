//! Drivers crate — device driver implementations.
//! VirtIO drivers are the primary target for VirtualBox testing.

#![no_std]
extern crate alloc;

pub mod virtio;
pub mod pci;
pub mod input;
pub mod serial;
