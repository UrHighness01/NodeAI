//! Adversarial Critic (Scheduler GAN)
//!
//! Feeds pathological (stress-test) syscall sequences into the transformer
//! scheduler to train it to handle adversarial thrashing workloads.
//! This acts as a secondary training signal (GAN-style regularization).

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

static CRITIC_INVOCATIONS: AtomicU64 = AtomicU64::new(0);

const STRESS_SEQUENCES: &[[u16; 16]] = &[
    // 1. Fork bomb sequence (clone, clone, clone)
    [56; 16], // 56 = clone on x86_64
    // 2. Synchronous I/O thrash (read, write, sync)
    [0; 16],  // will be populated at runtime
    // 3. Memory pressure (mmap, madvise, mmap)
    [9; 16],  // 9 = mmap
];

pub fn tick() {
    let count = CRITIC_INVOCATIONS.fetch_add(1, Ordering::Relaxed);
    
    // Only run the critic every ~10 seconds
    if count % 100 != 0 { return; }

    // Provide an adversarial sequence to the scheduler.
    // We want the scheduler to learn that a fork-bomb (sequence 0)
    // leads to massive wait times (thrashing) and deserves a heavy nice penalty.
    let fork_bomb = &STRESS_SEQUENCES[0];

    // target outputs: [nice_delta, burst_ticks, prefault_pages, predicted_wait_us]
    // We penalize the fork bomb by teaching the model it causes 50,000us waits
    // and deserves a massive nice penalty.
    let adversarial_target = [19.0, 1.0, 0.0, 50000.0];

    // Inject this directly into the transformer's SGD loop
    crate::transformer_sched::train_adversarial(fork_bomb, adversarial_target);
}
