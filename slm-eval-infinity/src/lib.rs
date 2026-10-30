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
