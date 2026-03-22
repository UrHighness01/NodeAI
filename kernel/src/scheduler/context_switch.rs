//! Context switch — save/restore CPU state between tasks.
//!
//! `switch_context(old, new)` saves the current task's registers into `old`
//! and restores the new task's registers from `new`, then returns.
//!
//! The layout of `CpuContext` must match the push/pop order here.

use super::task::CpuContext;

/// Switch CPU context from `old` to `new`.
///
/// # Safety
/// - Both pointers must be valid and non-null.
/// - Must be called with interrupts disabled or in a guard-locked section.
/// - The new task must have a valid stack pointer (rsp) in its context.
#[inline(never)]
pub unsafe fn switch_context(old: *mut CpuContext, new: *const CpuContext) {
    // We use a "naked" inline-asm approach:
    //   1. Save callee-saved registers + rip (return address) into `old`.
    //   2. Restore registers from `new`.
    //   3. Return — we return into the new task (its saved rip becomes our ret addr).
    //
    // The C ABI guarantees rdi = old, rsi = new on entry.
    core::arch::asm!(
        // ── Save current context into *old ─────────────────────────────────
        "mov [{old} + 0x00], r15",
        "mov [{old} + 0x08], r14",
        "mov [{old} + 0x10], r13",
        "mov [{old} + 0x18], r12",
        "mov [{old} + 0x20], rbp",
        "mov [{old} + 0x28], rbx",
        // rip: save the resume address (label below) as the task's saved RIP.
        "lea rax, [rip + 99f]",
        "mov [{old} + 0x70], rax",   // rip offset in CpuContext
        "mov [{old} + 0x80], rsp",   // rsp offset

        // ── Restore new context from *new ───────────────────────────────────
        "mov r15, [{new} + 0x00]",
        "mov r14, [{new} + 0x08]",
        "mov r13, [{new} + 0x10]",
        "mov r12, [{new} + 0x18]",
        "mov rbp, [{new} + 0x20]",
        "mov rbx, [{new} + 0x28]",
        "mov rsp, [{new} + 0x80]",   // restore stack
        // Push new rip as return address, then ret to it
        "push qword ptr [{new} + 0x70]",
        "ret",

        // ── Resume label (we return here when this task is switched back in) ─
        "99:",
        old = in(reg) old,
        new = in(reg) new,
        // Tell the compiler all registers are clobbered so it doesn't cache values.
        out("rax") _,
        options(nostack),
    );
}
