use spin::Mutex;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

const HISTORY_LEN: usize = 2;
const MAX_RESERVE_PAGES: usize = 256; // Max 1MB preallocated per pid

struct PidState {
    history: [u16; HISTORY_LEN],
    history_idx: usize,
    reserve: Vec<u64>,
}

struct BigramModel {
    /// Exponential moving average of allocated pages for each (prev, curr) syscall pair.
    /// Format: (prev, curr) -> ema_pages (stored as f32)
    ema: BTreeMap<(u16, u16), f32>,
}

static PID_STATES: Mutex<BTreeMap<u64, PidState>> = Mutex::new(BTreeMap::new());
static GLOBAL_MODEL: Mutex<BigramModel> = Mutex::new(BigramModel { ema: BTreeMap::new() });

pub fn record_syscall(pid: u64, nr: u16) {
    let mut states = PID_STATES.lock();
    let state = states.entry(pid).or_insert_with(|| PidState {
        history: [0; HISTORY_LEN],
        history_idx: 0,
        reserve: Vec::new(),
    });
    state.history[state.history_idx % HISTORY_LEN] = nr;
    state.history_idx += 1;

    if state.history_idx >= 2 {
        let prev = state.history[(state.history_idx - 2) % HISTORY_LEN];
        let curr = nr;
        let predicted_pages = {
            let model = GLOBAL_MODEL.lock();
            model.ema.get(&(prev, curr)).copied().unwrap_or(0.0) as usize
        };

        // If we predict upcoming memory pressure, pre-allocate frames
        if predicted_pages > 0 {
            let to_allocate = predicted_pages.min(MAX_RESERVE_PAGES);
            for _ in 0..to_allocate {
                if state.reserve.len() >= MAX_RESERVE_PAGES { break; }
                if let Some(frame) = super::pmm::alloc_frame() {
                    state.reserve.push(frame);
                } else {
                    break;
                }
            }
        }
    }
}

pub fn record_allocation(pid: u64, bytes: usize) {
    let pages = (bytes + super::pmm::PAGE_SIZE as usize - 1) / super::pmm::PAGE_SIZE as usize;
    if pages == 0 { return; }

    let states = PID_STATES.lock();
    if let Some(state) = states.get(&pid) {
        if state.history_idx >= 2 {
            let prev = state.history[(state.history_idx - 2) % HISTORY_LEN];
            let curr = state.history[(state.history_idx - 1) % HISTORY_LEN];
            
            let mut model = GLOBAL_MODEL.lock();
            let ema = model.ema.entry((prev, curr)).or_insert(0.0);
            // Alpha = 0.2
            *ema = *ema * 0.8 + (pages as f32) * 0.2;
        }
    }
}

pub fn alloc_frame_predictive(pid: u64) -> Option<u64> {
    {
        let mut states = PID_STATES.lock();
        if let Some(state) = states.get_mut(&pid) {
            if let Some(frame) = state.reserve.pop() {
                return Some(frame);
            }
        }
    }
    // Fallback
    super::pmm::alloc_frame()
}

pub fn remove_pid(pid: u64) {
    let mut states = PID_STATES.lock();
    if let Some(state) = states.remove(&pid) {
        for frame in state.reserve {
            unsafe { super::pmm::free_frame(frame); }
        }
    }
}
