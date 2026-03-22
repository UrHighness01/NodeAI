//! Global Descriptor Table (GDT) and Task State Segment (TSS).
//!
//! The GDT is required in 64-bit mode and defines code/data segments and the TSS.
//! The TSS provides the Interrupt Stack Table (IST) entries so that critical
//! exceptions (double-fault, NMI) can switch to a known-good stack.
//!
//! Phase 1 of the NodeAI kernel roadmap.

use x86_64::{
    structures::{
        gdt::{Descriptor, GlobalDescriptorTable, SegmentSelector},
        tss::TaskStateSegment,
    },
    VirtAddr,
};
use spin::Once;

// ── IST stack indices (0-based) ───────────────────────────────────────────────

/// IST slot 0: double-fault handler stack.
pub const DOUBLE_FAULT_IST_INDEX: u16 = 0;
/// IST slot 1: NMI handler stack.
pub const NMI_IST_INDEX: u16 = 1;

// ── Static TSS and GDT ───────────────────────────────────────────────────────

/// Size of each IST stack (8 KiB — large enough for exception handling, small enough
/// to fit in the low physical region before paging is fully set up).
const IST_STACK_SIZE: usize = 8 * 1024;

/// IST stacks as static byte arrays (placed in BSS → zero-initialized).
static mut DOUBLE_FAULT_STACK: [u8; IST_STACK_SIZE] = [0u8; IST_STACK_SIZE];
static mut NMI_STACK: [u8; IST_STACK_SIZE] = [0u8; IST_STACK_SIZE];

static TSS: Once<TaskStateSegment> = Once::new();
static GDT: Once<(GlobalDescriptorTable, Selectors)> = Once::new();

/// Segment selectors used by the kernel at runtime.
pub struct Selectors {
    pub kernel_code_segment: SegmentSelector,
    pub kernel_data_segment: SegmentSelector,
    pub user_code_segment: SegmentSelector,
    pub user_data_segment: SegmentSelector,
    pub tss_selector: SegmentSelector,
}

// ── Initialisation ────────────────────────────────────────────────────────────

/// Initialise the TSS and GDT, then load them.
/// Must be called once, early in boot, with interrupts disabled.
pub fn init() {
    let tss = TSS.call_once(|| {
        let mut tss = TaskStateSegment::new();

        // Set up IST stacks (stacks grow downward → use the high end of the array).
        tss.interrupt_stack_table[DOUBLE_FAULT_IST_INDEX as usize] = {
            let stack_end = unsafe { DOUBLE_FAULT_STACK.as_ptr().add(IST_STACK_SIZE) };
            VirtAddr::from_ptr(stack_end)
        };
        tss.interrupt_stack_table[NMI_IST_INDEX as usize] = {
            let stack_end = unsafe { NMI_STACK.as_ptr().add(IST_STACK_SIZE) };
            VirtAddr::from_ptr(stack_end)
        };

        tss
    });

    let (gdt, selectors) = GDT.call_once(|| {
        let mut gdt = GlobalDescriptorTable::new();

        let kernel_code = gdt.append(Descriptor::kernel_code_segment());
        let kernel_data = gdt.append(Descriptor::kernel_data_segment());
        let user_data   = gdt.append(Descriptor::user_data_segment());
        let user_code   = gdt.append(Descriptor::user_code_segment());
        let tss_sel     = gdt.append(Descriptor::tss_segment(tss));

        (gdt, Selectors {
            kernel_code_segment: kernel_code,
            kernel_data_segment: kernel_data,
            user_code_segment:   user_code,
            user_data_segment:   user_data,
            tss_selector:        tss_sel,
        })
    });

    // Load the GDT and update segment registers.
    gdt.load();

    unsafe {
        use x86_64::instructions::segmentation::{CS, DS, ES, Segment};
        use x86_64::instructions::tables::load_tss;

        CS::set_reg(selectors.kernel_code_segment);
        // In 64-bit mode, DS/ES/FS/GS are not used for code but need valid descriptors.
        DS::set_reg(selectors.kernel_data_segment);
        ES::set_reg(selectors.kernel_data_segment);

        load_tss(selectors.tss_selector);
    }

    crate::klog!(INFO, "GDT loaded: CS={:#x} DS={:#x} TSS={:#x}",
        selectors.kernel_code_segment.0,
        selectors.kernel_data_segment.0,
        selectors.tss_selector.0,
    );
}

/// Returns the kernel code segment selector (used by the IDT for interrupt gates).
pub fn kernel_cs() -> SegmentSelector {
    GDT.get().expect("GDT not initialised").1.kernel_code_segment
}

/// Returns the kernel data segment selector.
pub fn kernel_ds() -> SegmentSelector {
    GDT.get().expect("GDT not initialised").1.kernel_data_segment
}

/// Returns the user code segment selector (ring 3).
pub fn user_cs() -> SegmentSelector {
    GDT.get().expect("GDT not initialised").1.user_code_segment
}

/// Returns the user data segment selector (ring 3).
pub fn user_ds() -> SegmentSelector {
    GDT.get().expect("GDT not initialised").1.user_data_segment
}
