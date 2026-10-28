//! Evaluation statistics module: scoring, micro-F1, bootstrap CI, Cohen's κ.

pub mod gate;
pub mod scoring;

pub use gate::*;
pub use scoring::*;
pub use crate::models::{AggregateScore, Case, FieldOutcome, GateDecision, Metric, Prediction, StratumResult};
