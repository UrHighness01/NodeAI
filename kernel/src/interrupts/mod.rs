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
            .set_handler_addr(x86_64::VirtAddr::new(timer_handler as u64));
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
    crate::klog!(ERROR, "#PF {:#x} (code {:?}) ip={:#x}",
        cr2, error_code, frame.instruction_pointer.as_u64(),
    );
    // TODO Phase 3: hand off to VMM demand-paging / CoW handler
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
#[naked]
unsafe extern "C" fn timer_handler() {
    core::arch::asm!(
        // Save all GPRs (in order matching FRAME_* offsets in task.rs).
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
        // Call schedule_from_interrupt(old_rsp=rsp).
        // System V: first arg in rdi, return in rax.
        "mov rdi, rsp",
        "call {schedule}",
        // rax = new_rsp (may equal old rsp if no switch).
        "mov rsp, rax",
        // Restore all GPRs from new stack.
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
        // Return to the task (new or same) via IRETQ.
        "iretq",
        schedule = sym crate::scheduler::schedule_from_interrupt,
        options(noreturn),
    );
}

extern "x86-interrupt" fn keyboard_handler(_frame: InterruptStackFrame) {
    // Read scancode + decode into event queue.
    drivers::input::keyboard_irq_handler();
    // Route keypresses to the launcher (when open) or to the shell.
    while let Some(ev) = drivers::input::poll_event() {
        if ev.pressed {
            if crate::desktop::launcher_is_open() {
                // Inside launcher: ESC closes, BS edits search, Enter launches, printable = search
                match ev.scancode {
                    0x01 => crate::desktop::launcher_key(0x1B), // ESC
                    0x0E => crate::desktop::launcher_key(0x08), // Backspace
                    0x1C => crate::desktop::launcher_key(b'\n'), // Enter
                    _ => {
                        if let Some(ch) = ev.ascii {
                            let b = ch as u8;
                            if b >= 0x20 && b < 0x7F {
                                crate::desktop::launcher_key(b);
                            }
                        }
                    }
                }
            } else if crate::desktop::app_is_open() {
                // GUI app window: route to app key handlers
                if let Some(special) = ev.special {
                    crate::desktop::app_special_key(special);
                } else {
                    match ev.scancode {
                        0x01 => crate::desktop::app_char_key(0x1B), // ESC
                        0x0E => crate::desktop::app_char_key(0x08), // Backspace
                        0x1C => crate::desktop::app_char_key(b'\n'), // Enter
                        _ => {
                            if let Some(ch) = ev.ascii {
                                let b = ch as u8;
                                if b >= 0x20 && b < 0x7F {
                                    crate::desktop::app_char_key(b);
                                }
                            }
                        }
                    }
                }
            } else {
                // Normal shell routing
                if let Some(special) = ev.special {
                    crate::shell::on_special_key(special);
                } else {
                    match ev.scancode {
                        // Backspace (scancode 0x0E) → send BS to shell
                        0x0E => crate::shell::on_char(0x08),
                        // Enter (scancode 0x1C) → send newline to shell
                        0x1C => crate::shell::on_char(b'\n'),
                        // Tab (scancode 0x0F) → send tab to shell
                        0x0F => crate::shell::on_char(b'\t'),
                        // All other printable keys via ascii lookup
                        _ => {
                            if let Some(ch) = ev.ascii {
                                let b = ch as u8;
                                if b >= 0x20 && b < 0x7F {
                                    crate::shell::on_char(b);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    unsafe { apic::eoi(); }
}

extern "x86-interrupt" fn spurious_handler(_frame: InterruptStackFrame) {
    // Spurious APIC interrupt — no EOI needed
}

extern "x86-interrupt" fn mouse_handler(_frame: InterruptStackFrame) {
    drivers::input::mouse_irq_handler();
    while let Some(ev) = drivers::input::poll_mouse_event() {
        crate::desktop::mouse_event(ev.dx, ev.dy, ev.left, ev.right);
    }
    unsafe { apic::eoi(); }
}

