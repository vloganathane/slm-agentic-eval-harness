//! Core data models for slm-eval-infinity

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case {
    pub case_id: String,
    pub text: String,
    pub gold: GoldStandard,
    pub strata: StrataInfo,
    pub ocr_signal: Option<String>, // "ok" | "unextractable"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldStandard {
    pub merchant: Option<String>,
    pub date: Option<String>,
    pub total: Option<String>,
    pub subtotal: Option<String>,
    pub tax: Option<String>,
    pub items: Vec<LineItem>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineItem {
    pub name: String,
    pub qty: Option<u32>,
    pub unit_price: Option<String>,
    pub price: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrataInfo {
    pub quality: String, // "high" | "medium" | "low"
    pub layout: String,  // "receipt" | "invoice" | "mixed"
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub case_id: String,
    pub prediction: GoldStandard,
    pub metadata: Option<PredictionMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionMetadata {
    pub inference_time_ms: u64,
    pub model: String,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreResult {
    pub case_id: String,
    pub field_scores: HashMap<String, FieldScore>,
    pub strata: StrataInfo,
    pub is_quarantined: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldScore {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateScores {
    pub micro_f1: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub per_field: HashMap<String, FieldScore>,
    pub per_strata: HashMap<String, StrataAggregate>,
    pub dataset_defect: usize,
    pub total_cases: usize,
    pub evaluated_cases: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrataAggregate {
    pub n: usize,
    pub micro_f1: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub quality: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    pub kappa: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub n_pairs: usize,
    pub n_agree: usize,
    pub n_disagree: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateDecision {
    pub decision: GateDecisionType,
    pub promote: bool,
    pub micro_f1: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub kappa: f64,
    pub kappa_ci_lower: f64,
    pub strata_pass: bool,
    pub calibration_pass: bool,
    pub pending_reviews: usize,
    pub reasons: Vec<String>,
    pub exit_code: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GateDecisionType {
    Promote,
    Blocked,
    BlockedPendingReview,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticCase {
    pub case_id: String,
    pub text: String,
    pub gold: GoldStandard,
    pub strata: StrataInfo,
    pub source: String, // "template" | "web"
    pub merchant: String,
    pub minhash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewTask {
    pub task_id: String,
    pub run_id: String,
    pub case: SyntheticCase,
    pub reason: ReviewReason,
    pub status: ReviewStatus,
    pub reviewer: Option<String>,
    pub resolution: Option<ReviewResolution>,
    pub created_at: String,
    pub resolved_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReviewReason {
    SyntheticInjection,
    JudgeDisagreement,
    LowConfidence,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReviewStatus {
    Pending,
    Approved,
    Rejected,
    Modified,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewResolution {
    pub status: ReviewStatus,
    pub modified_case: Option<SyntheticCase>,
    pub comment: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEntry {
    pub stage: String,
    pub operation: String,
    pub cost_usd: f64,
    pub timestamp: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    pub run_id: String,
    pub config_hash: String,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub status: RunStatus,
    pub total_cost_usd: f64,
    pub stages_completed: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Completed,
    Failed,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateOutput {
    pub gate: GateDecision,
    pub scores: AggregateScores,
    pub calibration: CalibrationResult,
    pub run_metadata: RunMetadata,
}