//! SLM Evaluation Harness on Infinity Agent Runtime - library
//!
//! Rust re-implementation of the verified Python evaluation logic
//! (SPEC_V3.md). Ports the pure, deterministic core: config, models, scoring,
//! gate, and store.

pub mod config;
pub mod eval;
pub mod models;
pub mod split;
pub mod store;
pub mod inference;

// NOTE: `orchestrator.rs`, `adapter_cord.rs`, and `tools/` are an experimental
// RAP-runtime integration and are intentionally NOT declared here — they target
// an older Infinity schema and do not compile against the current store/config.
// The supported, tested core is the deterministic engine above. See
// `slm-eval-infinity/EXPERIMENTAL.md`.
