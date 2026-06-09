//! AI Subsystem — Kernel-integrated AI inference engine.
//!
//! Architecture:
//!   - Inference runtime (no_std, SIMD-accelerated)
//!   - Decision domains (scheduler, memory, I/O, security, power)
//!   - Safety constraint engine (hard rules AI cannot override)
//!   - Kernel event bus (publish/subscribe between kernel subsystems and AI)
//!   - Audit log (every AI decision is recorded)
//!
//! Phase 8 of the NodeAI roadmap.

#![no_std]
extern crate alloc;

pub mod inference;
pub mod model;
pub mod domains;
pub mod safety;
pub mod event_bus;
pub mod audit;
pub mod vector_store;
pub mod aligned_vec;
