//! Evaluation config (ported from slm_eval/config.py, SPEC_V3 §8.3).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegexValidator {
    #[serde(rename = "type")]
    pub r#type: String,
    pub pattern: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NumericValidator {
    #[serde(rename = "type")]
    pub r#type: String,
    pub tolerance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DateValidator {
    #[serde(rename = "type")]
    pub r#type: String,
    pub formats: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StringValidator {
    #[serde(rename = "type")]
    pub r#type: String,
    pub normalize: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ListValidator {
    #[serde(rename = "type")]
    pub r#type: String,
    pub item_match: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum FieldValidator {
    #[serde(rename = "regex")]
    Regex { pattern: String },
    #[serde(rename = "numeric")]
    Numeric { tolerance: f64 },
    #[serde(rename = "date")]
    Date { formats: Vec<String> },
    #[serde(rename = "string")]
    String { normalize: Option<Vec<String>> },
    #[serde(rename = "list")]
    List { item_match: Option<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FieldsConfig {
    #[serde(default)]
    pub subjective_fields: Vec<String>,
    #[serde(default)]
    pub exact_match_fields: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validators: Option<BTreeMap<String, FieldValidator>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StrataConfig {
    #[serde(default)]
    pub keys: Vec<String>,
    #[serde(default)]
    pub mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitConfig {
    pub fraction: f64,
    pub seed: i64,
    pub strategy: String,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self { fraction: 0.2, seed: 42, strategy: "stratified".to_string() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsConfig {
    pub seed: i64,
    pub resamples: usize,
}

impl Default for StatsConfig {
    fn default() -> Self {
        Self { seed: 42, resamples: 10_000 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdsConfig {
    pub target_f1: f64,
    pub failure_threshold: f64,
    pub min_per_stratum: usize,
}

impl Default for ThresholdsConfig {
    fn default() -> Self {
        Self { target_f1: 0.85, failure_threshold: 0.5, min_per_stratum: 50 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    pub path: String,
    pub min_pairs: usize,
    pub target_kappa: f64,
    pub recalibrate_every_rounds: usize,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self { path: "calibration/judge.jsonl".to_string(), min_pairs: 50, target_kappa: 0.85, recalibrate_every_rounds: 5 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JudgeConfig {
    pub endpoint: String,
    pub model: String,
    pub api_key_env: String,
    pub prompt_version: String,
    pub calibration: CalibrationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_attempts: usize,
    pub backoff_base: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self { max_attempts: 3, backoff_base: 2.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InferenceConfig {
    pub backend: String,
    pub base_url: String,
    pub timeout_seconds: usize,
    pub retry: RetryConfig,
}

impl InferenceConfig {
    fn default_backend() -> String { "llama_server".to_string() }
    fn default_base_url() -> String { "http://127.0.0.1:8080/v1".to_string() }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HumanReviewConfig {
    pub timeout_hours: i64,
    pub notify: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CostControlConfig {
    pub max_cost_per_round_usd: f64,
    pub alert_at_usd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReportingConfig {
    pub formats: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    pub task_description: String,
    #[serde(default = "default_task_type")]
    pub task_type: String,
    pub fields: FieldsConfig,
    pub strata: StrataConfig,
    pub split: SplitConfig,
    pub stats: StatsConfig,
    pub thresholds: ThresholdsConfig,
    pub judge: JudgeConfig,
    pub inference: InferenceConfig,
    pub human_review: HumanReviewConfig,
    pub cost_control: CostControlConfig,
    pub reporting: ReportingConfig,
}

fn default_task_type() -> String { "extraction".to_string() }

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            task_description: String::new(),
            task_type: default_task_type(),
            fields: FieldsConfig::default(),
            strata: StrataConfig::default(),
            split: SplitConfig::default(),
            stats: StatsConfig::default(),
            thresholds: ThresholdsConfig::default(),
            judge: JudgeConfig::default(),
            inference: InferenceConfig::default(),
            human_review: HumanReviewConfig::default(),
            cost_control: CostControlConfig::default(),
            reporting: ReportingConfig::default(),
        }
    }
}

impl EvalConfig {
    pub fn from_yaml(path: &std::path::Path) -> Result<Self, anyhow::Error> {
        let s = std::fs::read_to_string(path)?;
        let cfg: EvalConfig = serde_yaml::from_str(&s)?;
        Ok(cfg)
    }
}
