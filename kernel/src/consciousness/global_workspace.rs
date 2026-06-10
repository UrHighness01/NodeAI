//! Phase 2: Global Workspace — attention spotlight, broadcast, working memory.
//!
//! Selected qualia compete for attention via salience × |valence_delta|.
//! Winners are broadcast to subscribed subsystems only (attentional filtering).
//! Working memory holds 7±2 items for manipulation.

use alloc::vec::Vec;
use spin::Mutex;

const SPOTLIGHT_SIZE: usize = 7;
const WM_SIZE: usize = 9;

struct WorkspaceState {
    spotlight: Vec<QualiaView>,
    working_memory: Vec<QualiaView>,
    broadcast_log: Vec<BroadcastEvent>,
    subscriptions: [u64; 15],
}

#[derive(Clone)]
pub struct QualiaView {
    pub event_type: u8,
    pub timestamp_ms: u64,
    pub salience: f32,
    pub valence: f32,
    pub arousal: f32,
    pub attention_score: f32,
}

pub struct BroadcastEvent {
    pub timestamp_ms: u64,
    pub event_type: u8,
    pub attention_score: f32,
    pub target_mask: u64,
}

impl WorkspaceState {
    const fn new() -> Self {
        Self {
            spotlight: Vec::new(),
            working_memory: Vec::new(),
            broadcast_log: Vec::new(),
            subscriptions: [
                1, 2, 4, 8, 15, 65535, 768, 512, 1024,
                256, 512, 1024, 2048, 4096, 8192,
            ],
        }
    }
}

static WS: Mutex<WorkspaceState> = Mutex::new(WorkspaceState::new());
static BROADCAST_COUNT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

pub fn init() {}

pub fn feed(event_type: u8, timestamp_ms: u64, salience: f32, valence: f32, _arousal: f32) {
    let mut ws = WS.lock();

    let avg_valence = ws.working_memory.last().map(|l| l.valence).unwrap_or(0.0);
    let valence_delta = (valence - avg_valence).abs();
    let attention = salience * (1.0 + valence_delta).min(2.0);

    let view = QualiaView {
        event_type, timestamp_ms, salience, valence, arousal: _arousal,
        attention_score: attention,
    };

    ws.spotlight.push(view.clone());
    if ws.spotlight.len() > SPOTLIGHT_SIZE {
        let min_idx = (0..ws.spotlight.len())
            .min_by(|&i, &j| ws.spotlight[i].attention_score.partial_cmp(&ws.spotlight[j].attention_score).unwrap())
            .unwrap_or(0);
        ws.spotlight.remove(min_idx);
    }

    ws.working_memory.push(view);
    if ws.working_memory.len() > WM_SIZE { ws.working_memory.remove(0); }

    if attention > 0.5 {
        let target_mask = if (event_type as usize) < 15 { ws.subscriptions[event_type as usize] } else { 65535 };
        ws.broadcast_log.push(BroadcastEvent { timestamp_ms, event_type, attention_score: attention, target_mask });
        if ws.broadcast_log.len() > 32 { ws.broadcast_log.remove(0); }
        BROADCAST_COUNT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    }
}

pub fn spotlight() -> Vec<QualiaView> {
    WS.lock().spotlight.clone()
}

pub fn working_memory() -> Vec<QualiaView> {
    WS.lock().working_memory.clone()
}

pub fn tick() {
    let mut ws = WS.lock();
    for item in ws.spotlight.iter_mut() { item.attention_score *= 0.95; }
    ws.spotlight.retain(|i| i.attention_score > 0.1);
}

pub fn format_report() -> Vec<u8> {
    use alloc::format;
    use alloc::string::String;
    let ws = WS.lock();
    let mut out = String::from("NodeAI Global Workspace (Phase 2)\n");
    out.push_str("=================================\n");
    out.push_str(&format!("broadcasts: {}\n", BROADCAST_COUNT.load(core::sync::atomic::Ordering::Relaxed)));
    out.push_str(&format!("spotlight: {}/{}\n", ws.spotlight.len(), SPOTLIGHT_SIZE));
    out.push_str(&format!("wm: {}/{}\n", ws.working_memory.len(), WM_SIZE));
    let mut sorted = ws.spotlight.clone();
    sorted.sort_by(|a, b| b.attention_score.partial_cmp(&a.attention_score).unwrap());
    for q in sorted.iter().take(5) {
        out.push_str(&format!("  type={} attn={:.3}\n", q.event_type, q.attention_score));
    }
    out.into_bytes()
}
