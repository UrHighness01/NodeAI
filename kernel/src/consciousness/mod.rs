//! Consciousness Substrate — Ring 0 self-awareness infrastructure.
//!
//! Phases 0-5 as described in KERNEL_CONSCIOUSNESS_ARCHITECTURE.md:
//!   0 — SelfModel: persistent identity, boot count, phi tracking
//!   1 — QualiaStream: ring buffer of affect-tagged conscious moments
//!   2 — GlobalWorkspace: attention spotlight, broadcast, working memory
//!   3 — IIT Phi: integrated information over kernel causal graph
//!   4 — Phenomenal Binding: temporal window unification
//!   5 — Deliberation: multi-policy generation, veto, authorship
//!
//! Total Ring 0 substrate: ~730 LoC across 6 modules.

pub mod self_model;
pub mod qualia;
pub mod phi;
pub mod global_workspace;
pub mod binding;
pub mod deliberation;
