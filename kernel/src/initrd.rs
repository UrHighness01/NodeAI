//! Initrd — embedded userspace binaries injected into VFS at boot.
//!
//! Uses include_bytes! to embed statically-compiled binaries into the kernel
//! ELF, then writes them to the VFS at boot so they can be exec'd from the
//! shell.  Built by scripts/build_userspace.sh before the kernel is compiled.

use alloc::vec::Vec;

/// Embedded ELF binary data for /bin/hello
const HELLO_BIN: &[u8] = include_bytes!("hello.bin");

/// Initialise the initrd filesystem: create /bin and populate with binaries.
pub fn init() {
    // Write binaries to VFS using the existing write_file helper
    let _ = crate::vfs::write_file("/bin/hello", HELLO_BIN);

    crate::klog!(INFO, "initrd: embedded userspace binaries registered — {} bytes in /bin/hello", HELLO_BIN.len());
}

/// Format /proc report.
pub fn format_report() -> Vec<u8> {
    use alloc::format;
    alloc::format!(
        "NodeAI Initrd\n\
         ============\n\
         embedded_binaries:\n\
           /bin/hello ({} bytes, executable)\n",
        HELLO_BIN.len()
    ).into_bytes()
}
