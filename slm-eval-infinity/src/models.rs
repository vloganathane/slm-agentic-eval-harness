//! Core data types for the SLM Evaluation Harness (ported from Python).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single evaluation case (sealed holdout or synthetic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case {
    pub case_id: String,
    pub text: String,
    pub gold: serde_json::Value,
    pub strata: serde_json::Value,
    pub ocr_signal: Option<String>,
}

impl Case {
    pub fn new(case_id: String, text: String, gold: serde_json::Value, strata: serde_json::Value, ocr_signal: Option<String>) -> Self {
        Self { case_id, text, gold, strata, ocr_signal }
    }
}

/// A model prediction for a case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub case_id: String,
    pub fields: HashMap<String, serde_json::Value>,
}

/// Element-level outcome for one (case, field) pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldOutcome {
    pub case_id: String,
    pub tp: u32,
    pub fp: u32,
    pub fn_: u32,
}

impl FieldOutcome {
    pub fn new(case_id: String, tp: u32, fp: u32, fn_: u32) -> Self {
        Self { case_id, tp, fp, fn_ }
    }
    /// A field outcome is "correct" when tp > 0 and fp == 0 and fn == 0.
    pub fn correct(&self) -> bool {
        self.tp > 0 && self.fp == 0 && self.fn_ == 0
    }
}

/// Point estimate + bootstrap CI for a metric.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Metric {
    pub point: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub n: usize,
    pub resamples: usize,
}

/// A stratum result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumResult {
    pub key: String,
    pub n: usize,
    pub micro_f1: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub status: StratumStatus,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum StratumStatus {
    Ok,
    Warning,
    Blocker,
    InsufficientN,
}

impl StratumStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            StratumStatus::Ok => "ok",
            StratumStatus::Warning => "warning",
            StratumStatus::Blocker => "blocker",
            StratumStatus::InsufficientN => "insufficient_n",
        }
    }
}

/// The gate decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateDecision {
    pub status: String,
    pub promote: bool,
    pub reason: String,
    pub blockers: Vec<String>,
    pub warnings: Vec<String>,
}

/// Result of scoring a single case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseScore {
    pub case_id: String,
    pub outcomes: Vec<FieldOutcome>,
}

/// Aggregated score across cases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateScore {
    pub micro_f1: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub n: usize,
    pub resamples: usize,
}
