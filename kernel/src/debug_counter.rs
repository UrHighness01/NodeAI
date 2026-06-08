use core::sync::atomic::{AtomicUsize, Ordering};
pub static KBD_INTS: AtomicUsize = AtomicUsize::new(0);
pub static MOUSE_INTS: AtomicUsize = AtomicUsize::new(0);
pub static TIMER_INTS: AtomicUsize = AtomicUsize::new(0);
