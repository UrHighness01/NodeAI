//! Initrd — embedded userspace binaries injected into VFS at boot.
//!
//! Uses include_bytes! to embed statically-compiled binaries into the kernel
//! ELF, then writes them to the VFS at boot so they can be exec'd from the
//! shell.  Built by scripts/build_userspace.sh before the kernel is compiled.

use alloc::vec::Vec;
use alloc::format;

/// Embedded ELF binary data for /bin/hello
const HELLO_BIN: &[u8] = include_bytes!("hello.bin");

/// Initialise the initrd filesystem: create /bin and populate with binaries.
pub fn init() {
    // Create /bin directory in the root ramfs
    let root = crate::vfs::root();
    let _ = root.mkdir("bin");

    // Write binary to VFS
    match crate::vfs::write_file("/bin/hello", HELLO_BIN) {
        Ok(_) => {
            crate::klog!(INFO, "initrd: /bin/hello registered — {} bytes, VFS accessible", HELLO_BIN.len());
        }
        Err(e) => {
            crate::klog!(WARN, "initrd: /bin/hello write failed: {:?}", e);
        }
    }
}

/// Format /proc report.
pub fn format_report() -> Vec<u8> {
    alloc::format!(
        "NodeAI Initrd\n\
         ============\n\
         embedded_binaries:\n\
           /bin/hello ({} bytes, executable)\n",
        HELLO_BIN.len()
    ).into_bytes()
}
