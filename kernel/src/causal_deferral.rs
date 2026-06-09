use alloc::vec::Vec;
use alloc::collections::VecDeque;
use spin::{Mutex, Once};
use crate::net::Ipv4Header;

#[derive(Clone)]
pub struct DeferredTcpEvent {
    pub src_mac: [u8; 6],
    pub iph: Ipv4Header,
    pub tcp_raw: Vec<u8>,
}

pub struct CausalDeferralBuffer {
    deferred_events: Mutex<VecDeque<DeferredTcpEvent>>,
}

pub static DEFERRAL_BUFFER: Once<CausalDeferralBuffer> = Once::new();

pub fn get_deferral_buffer() -> &'static CausalDeferralBuffer {
    DEFERRAL_BUFFER.call_once(|| CausalDeferralBuffer {
        deferred_events: Mutex::new(VecDeque::new()),
    })
}

impl CausalDeferralBuffer {
    /// Evaluates if a notification for a target PID should be deferred.
    pub fn should_defer(&self, target_pid: u64) -> bool {
        if target_pid == 0 {
            return false;
        }
        // High anomaly score implies chaotic or low-valence behavior.
        // We defer these events to batch them and reduce context-switching overhead.
        let anomaly_score = crate::anomaly::score(target_pid);
        anomaly_score > 0.5
    }

    /// Defer a TCP segment event.
    pub fn defer_event(&self, src_mac: [u8; 6], iph: Ipv4Header, tcp_raw: Vec<u8>) {
        self.deferred_events.lock().push_back(DeferredTcpEvent {
            src_mac,
            iph,
            tcp_raw,
        });
    }

    /// Flush all deferred events. Called from ai_engine::process_tick.
    pub fn flush(&self) {
        let mut queue = self.deferred_events.lock();
        while let Some(event) = queue.pop_front() {
            // Process the deferred packet. We drop the lock so net::tcp::handle_tcp_segment can acquire it.
            drop(queue);
            if let Some(reply) = crate::net::tcp::handle_tcp_segment(event.src_mac, event.iph, &event.tcp_raw) {
                crate::net::transmit(&reply);
            }
            queue = self.deferred_events.lock();
        }
    }
}
