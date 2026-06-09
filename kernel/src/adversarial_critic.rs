use alloc::vec::Vec;
use crate::fuzzer::perturb_path;

pub fn init() {
    crate::scheduler::spawn_kernel_thread("adversarial_critic", critic_thread_main);
}

fn critic_thread_main() -> ! {
    loop {
        // Sleep for a while (yield loop)
        for _ in 0..500 {
            crate::scheduler::yield_cpu();
        }

        // Get all active PIDs
        let pids: Vec<u64> = x86_64::instructions::interrupts::without_interrupts(|| {
            let tasks = crate::scheduler::TASKS.lock();
            tasks.keys().copied().collect()
        });

        for pid in pids {
            let score = crate::anomaly::score(pid);
            // Target processes that are slightly suspicious but not fully quarantined
            if score > 0.1 && score < 0.5 {
                run_shadow_sequence(pid, score);
            }
        }
    }
}

fn run_shadow_sequence(pid: u64, current_score: f32) {
    let targets = ["/etc/passwd", "/etc/shadow", "/boot/vmlinuz", "/etc/hostname"];
    let mut shadow_success = false;

    for target in targets {
        let fuzzed = perturb_path(target);
        if crate::vfs::lookup(&fuzzed).is_ok() {
            shadow_success = true;
            break;
        }
    }

    if shadow_success {
        crate::klog!(WARN, "CRITIC: pid={} vulnerable to path perturbation! Bumping namespace confinement.", pid);
        // Preemptively increase anomaly score by +0.4 to bump namespace containment
        let new_score = (current_score + 0.4).min(1.0);
        crate::namespaces::update(pid, new_score);

        // Also simulate a shadow syscall anomaly (e.g. 2 = sys_open, 0 = sys_read)
        let _ = crate::anomaly::score_sequence(pid, &[2, 0, 2, 0]);
    }
}
