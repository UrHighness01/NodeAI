//! Interrupt subsystem — IDT, exception handlers, APIC, IRQ routing.

use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame, PageFaultErrorCode};
use spin::Once;

pub(crate) mod apic;
pub(crate) mod io_apic;
pub use apic::LOCAL_APIC_BASE;

static IDT: Once<InterruptDescriptorTable> = Once::new();

pub fn init() {
    let idt = IDT.call_once(build_idt);
    idt.load();
    crate::klog!(INFO, "IDT loaded");

    // Initialise the APIC (replaces legacy 8259 PIC).
    // Interrupts are NOT enabled here — call
    // `x86_64::instructions::interrupts::enable()` from main once all
    // subsystems are ready, so the timer handler can safely run.
    unsafe { apic::init_apic(); }
}

fn build_idt() -> InterruptDescriptorTable {
    let mut idt = InterruptDescriptorTable::new();

    idt.breakpoint.set_handler_fn(breakpoint_handler);
    idt.invalid_opcode.set_handler_fn(invalid_opcode_handler);
    idt.divide_error.set_handler_fn(divide_error_handler);
    idt.stack_segment_fault.set_handler_fn(stack_segment_handler);
    idt.segment_not_present.set_handler_fn(segment_not_present_handler);
    idt.general_protection_fault.set_handler_fn(gpf_handler);
    idt.page_fault.set_handler_fn(page_fault_handler);

    // Double fault MUST run on a separate IST stack (stack pointer may be corrupted).
    unsafe {
        idt.double_fault
            .set_handler_fn(double_fault_handler)
            .set_stack_index(crate::gdt::DOUBLE_FAULT_IST_INDEX);

        idt[apic::TIMER_VECTOR]
            .set_handler_addr(x86_64::VirtAddr::new(timer_handler as *const () as u64));
        idt[apic::KEYBOARD_VECTOR]
            .set_handler_fn(keyboard_handler);
        idt[apic::MOUSE_VECTOR]
            .set_handler_fn(mouse_handler);
        idt[apic::SPURIOUS_VECTOR]
            .set_handler_fn(spurious_handler);
    }

    idt
}

// ── Exception Handlers ────────────────────────────────────────────────────────

extern "x86-interrupt" fn breakpoint_handler(frame: InterruptStackFrame) {
    crate::klog!(WARN, "#BP breakpoint at {:#x}", frame.instruction_pointer.as_u64());
}

extern "x86-interrupt" fn double_fault_handler(
    frame: InterruptStackFrame,
    _error_code: u64,
) -> ! {
    crate::klog!(ERROR, "#DF double fault — ip={:#x} sp={:#x}",
        frame.instruction_pointer.as_u64(),
        frame.stack_pointer.as_u64(),
    );
    loop { x86_64::instructions::hlt(); }
}

extern "x86-interrupt" fn page_fault_handler(
    frame: InterruptStackFrame,
    error_code: PageFaultErrorCode,
) {
    let cr2 = x86_64::registers::control::Cr2::read()
        .map(|a| a.as_u64())
        .unwrap_or(0xdeadbeef);

    // Demand-paging: attempt to map the faulting page if it falls in a
    // valid user region (heap or stack).  Kernel faults are always fatal.
    let ip  = frame.instruction_pointer.as_u64();
    let is_user = error_code.contains(PageFaultErrorCode::USER_MODE);
    let is_present = error_code.contains(PageFaultErrorCode::PROTECTION_VIOLATION);

    if is_user && !is_present {
        // Not-present fault in user mode — try demand allocation.
        let page_sz = crate::memory::PAGE_SIZE;
        let page_virt = cr2 & !(page_sz - 1);
        let pid = crate::scheduler::current_pid();
        let brk = crate::scheduler::get_user_brk(pid);

        // Valid regions: heap (0x40_0000_0000 .. brk) and stack (7FFF_F800_0000 .. 7FFF_FFFF_F000)
        const HEAP_BASE:    u64 = 0x0000_0040_0000_0000;
        const STACK_BOTTOM: u64 = 0x0000_7FFF_F000_0000;
        const STACK_TOP:    u64 = 0x0000_7FFF_FFFF_F000;

        let in_heap  = brk > HEAP_BASE && page_virt >= HEAP_BASE && page_virt < brk;
        let in_stack = page_virt >= STACK_BOTTOM && page_virt < STACK_TOP;

        if in_heap || in_stack {
            match crate::memory::map_user_range(page_virt, page_sz, true, false) {
                Ok(()) => {
                    crate::klog!(DEBUG, "#PF demand-map {:#x} ok (pid={})", page_virt, pid);
                    return; // handled — resume faulting instruction
                }
                Err(e) => {
                    crate::klog!(ERROR, "#PF demand-map {:#x} failed: {}", page_virt, e);
                }
            }
        }
        // Address is out of valid range — send SIGSEGV; task will die at next syscall return.
        crate::klog!(ERROR, "#PF SIGSEGV pid={} addr={:#x} ip={:#x}", pid, cr2, ip);
        crate::scheduler::send_signal(pid, 11); // SIGSEGV
        crate::scheduler::yield_cpu();
        return;
    }

    // Kernel fault or protection violation — unrecoverable.
    crate::klog!(ERROR, "#PF FATAL {:#x} (code {:?}) ip={:#x}", cr2, error_code, ip);
    loop { x86_64::instructions::hlt(); }
}

extern "x86-interrupt" fn gpf_handler(frame: InterruptStackFrame, error_code: u64) {
    crate::klog!(ERROR, "#GP code={:#x} ip={:#x}",
        error_code, frame.instruction_pointer.as_u64(),
    );
    loop { x86_64::instructions::hlt(); }
}

extern "x86-interrupt" fn invalid_opcode_handler(frame: InterruptStackFrame) {
    crate::klog!(ERROR, "#UD invalid opcode at {:#x}", frame.instruction_pointer.as_u64());
    loop { x86_64::instructions::hlt(); }
}

extern "x86-interrupt" fn divide_error_handler(frame: InterruptStackFrame) {
    crate::klog!(ERROR, "#DE divide-by-zero at {:#x}", frame.instruction_pointer.as_u64());
    loop { x86_64::instructions::hlt(); }
}

extern "x86-interrupt" fn stack_segment_handler(frame: InterruptStackFrame, ec: u64) {
    crate::klog!(ERROR, "#SS stack segment fault (ec={:#x}) at {:#x}", ec,
        frame.instruction_pointer.as_u64());
    loop { x86_64::instructions::hlt(); }
}

extern "x86-interrupt" fn segment_not_present_handler(frame: InterruptStackFrame, ec: u64) {
    crate::klog!(ERROR, "#NP segment not present (ec={:#x}) at {:#x}", ec,
        frame.instruction_pointer.as_u64());
    loop { x86_64::instructions::hlt(); }
}

// ── Hardware IRQ Handlers ──────────────────────────────────────────────────────

/// Naked APIC timer handler — performs a full preemptive context switch.
///
/// On entry the CPU has already pushed the 5-word IRET frame:
///   [rsp+0]=RIP [rsp+8]=CS [rsp+16]=RFLAGS [rsp+24]=RSP [rsp+32]=SS
///
/// We push all 15 GPRs, call `schedule_from_interrupt(rsp)` which saves the
/// current task, picks the next, and returns the next task's kernel RSP.
/// We then point RSP at the new stack and pop registers before IRETQ.
///
/// Push order (first push = highest addr, last push = lowest addr):
///   rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11, rbx, rbp, r12, r13, r14, r15
/// So after all pushes: rsp → r15 slot (offset 0), rax slot is at offset +112.
/// Offset of fpu_ptr within PercpuData (gs-relative).
/// gs:0=self_ptr, gs:8=cpu_id+pad, gs:16=kernel_rsp, gs:24=user_rsp,
/// gs:32=ticks_per_ms+pad, gs:40=signal_new_rip, gs:48=signal_new_rsp,
/// gs:56=signal_new_rflags, gs:64=signal_signum, gs:72=fpu_ptr.
const GS_FPU_PTR: usize = 72;

#[unsafe(naked)]
unsafe extern "C" fn timer_handler() {
    core::arch::naked_asm!(
        // ── Save GPRs ─────────────────────────────────────────────────────
        "push rax",
        "push rcx",
        "push rdx",
        "push rsi",
        "push rdi",
        "push r8",
        "push r9",
        "push r10",
        "push r11",
        "push rbx",
        "push rbp",
        "push r12",
        "push r13",
        "push r14",
        "push r15",

        // ── Save FPU/SSE state (fxsave64) ─────────────────────────────────
        // r11 is already saved on stack; safe to use as scratch here.
        "mov r11, qword ptr gs:[{fpu_off}]", // load fpu_ptr (0 if not yet set)
        "test r11, r11",
        "jz 1f",
        "fxsave64 [r11]",
        "1:",

        // ── Schedule ─────────────────────────────────────────────────────
        "mov rdi, rsp",
        "call {schedule}",         // rax = new_rsp; also updates gs:fpu_ptr
        "mov rsp, rax",

        // ── Restore FPU/SSE state (fxrstor64) ─────────────────────────────
        "mov r11, qword ptr gs:[{fpu_off}]", // new task's fpu_ptr
        "test r11, r11",
        "jz 2f",
        "fxrstor64 [r11]",
        "2:",

        // ── Restore GPRs and return ────────────────────────────────────────
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbp",
        "pop rbx",
        "pop r11",
        "pop r10",
        "pop r9",
        "pop r8",
        "pop rdi",
        "pop rsi",
        "pop rdx",
        "pop rcx",
        "pop rax",
        "iretq",

        schedule = sym crate::scheduler::schedule_from_interrupt,
        fpu_off  = const GS_FPU_PTR,
    );
}

extern "x86-interrupt" fn keyboard_handler(_frame: InterruptStackFrame) {
    // Read scancode + decode into event queue.
    drivers::input::keyboard_irq_handler();
    unsafe { apic::eoi(); }
}

extern "x86-interrupt" fn spurious_handler(_frame: InterruptStackFrame) {
    // Spurious APIC interrupt — no EOI needed
}

extern "x86-interrupt" fn mouse_handler(_frame: InterruptStackFrame) {
    drivers::input::mouse_irq_handler();
    unsafe { apic::eoi(); }
}

