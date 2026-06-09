use core::sync::atomic::{AtomicBool, Ordering};

#[repr(align(64))]
pub struct XSaveArea {
    data: [u8; 4096], // 4KB is enough for AVX-512 state, well beyond the 832 bytes needed for AVX2.
}

impl XSaveArea {
    const fn new() -> Self {
        Self { data: [0; 4096] }
    }
}

static IN_SIMD: AtomicBool = AtomicBool::new(false);
static mut XSAVE_BUFFER: XSaveArea = XSaveArea::new();

/// Initialize SIMD support by enabling OSXSAVE in CR4 and configuring XCR0.
pub fn init() {
    unsafe {
        // Enable OSXSAVE (bit 18) in CR4
        let mut cr4: u64;
        core::arch::asm!("mov {}, cr4", out(reg) cr4);
        cr4 |= 1 << 18;
        core::arch::asm!("mov cr4, {}", in(reg) cr4);

        // Configure XCR0 to enable x87 (bit 0), SSE (bit 1), and AVX (bit 2)
        // XCR0 is accessed via xsetbv with ecx = 0
        let xcr0: u64 = 7; 
        let eax = (xcr0 & 0xFFFF_FFFF) as u32;
        let edx = (xcr0 >> 32) as u32;
        core::arch::asm!("xsetbv", in("ecx") 0, in("eax") eax, in("edx") edx);
    }
    crate::klog!(INFO, "SIMD: OSXSAVE and XCR0 configured for AVX2 state management.");
}

/// Execute a closure with SIMD registers available.
/// Disables preemption (interrupts), saves SIMD state, runs the closure, restores state, and restores preemption.
#[inline(always)]
pub fn with_simd<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    // Disable interrupts to prevent preemption
    let rflags: u64;
    unsafe {
        core::arch::asm!(
            "pushfq",
            "pop {}",
            "cli",
            out(reg) rflags
        );
    }
    let irqs_were_enabled = (rflags & (1 << 9)) != 0;

    // Ensure we don't nest SIMD usage on the same CPU, as we use a single static buffer
    assert!(
        !IN_SIMD.swap(true, Ordering::Acquire),
        "Nested with_simd calls are not supported"
    );

    unsafe {
        // Save full AVX state (x87 + SSE + AVX = bits 0, 1, 2 = 7)
        let mask_eax: u32 = 7;
        let mask_edx: u32 = 0;
        core::arch::asm!(
            "xsave64 [{}]",
            in(reg) &mut XSAVE_BUFFER,
            in("eax") mask_eax,
            in("edx") mask_edx
        );
    }

    // Run the closure
    let result = f();

    unsafe {
        // Restore AVX state
        let mask_eax: u32 = 7;
        let mask_edx: u32 = 0;
        core::arch::asm!(
            "xrstor64 [{}]",
            in(reg) &XSAVE_BUFFER,
            in("eax") mask_eax,
            in("edx") mask_edx
        );
    }

    IN_SIMD.store(false, Ordering::Release);

    // Restore interrupts if they were enabled
    if irqs_were_enabled {
        unsafe { core::arch::asm!("sti"); }
    }

    result
}
