//! Configuration types for slm-eval-infinity

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvalConfig {
    pub run: RunConfig,
    pub dataset: DatasetConfig,
    pub model: ModelConfig,
    pub judge: JudgeConfig,
    pub split: SplitConfig,
    pub scoring: ScoringConfig,
    pub gate: GateConfig,
    pub synthetic: Option<SyntheticConfig>,
    pub review: Option<ReviewConfig>,
    pub idiot_index: Option<IdiotIndexConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RunConfig {
    pub run_id: String,
    pub output_dir: PathBuf,
    #[serde(default = "default_true")]
    pub seal_holdout: bool,
    #[serde(default)]
    pub resume: bool,
}

fn default_true() -> bool { true }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DatasetConfig {
    pub path: PathBuf,
    pub adapter: String, // "cord", "custom"
    pub strata_key: Option<String>,
    #[serde(default)]
    pub quarantine_unextractable: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    pub endpoint: String, // OpenAI-compatible endpoint
    pub model: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default = "default_temp")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub json_mode: bool,
}

fn default_temp() -> f32 { 0.0 }
fn default_max_tokens() -> u32 { 2048 }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JudgeConfig {
    pub endpoint: String,
    pub model: String,
    pub api_key_env: String,
    #[serde(default = "default_kappa_target")]
    pub kappa_target: f64,
    #[serde(default)]
    pub calibration_file: Option<PathBuf>,
}

fn default_kappa_target() -> f64 { 0.85 }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SplitConfig {
    #[serde(default = "default_train_ratio")]
    pub train_ratio: f64,
    #[serde(default = "default_strata_mode")]
    pub strata_mode: String, // "single_key"
    #[serde(default)]
    pub seed: u64,
}

fn default_train_ratio() -> f64 { 0.8 }
fn default_strata_mode() -> String { "single_key".to_string() }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScoringConfig {
    #[serde(default = "default_bootstrap")]
    pub bootstrap_resamples: usize,
    #[serde(default = "default_ci_level")]
    pub ci_level: f64,
}

fn default_bootstrap() -> usize { 1000 }
fn default_ci_level() -> f64 { 0.95 }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GateConfig {
    pub micro_f1_target: f64,
    pub kappa_target: f64,
    pub min_strata_n: usize,
    #[serde(default)]
    pub thresholds: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SyntheticConfig {
    #[serde(default = "default_synthetic_rounds")]
    pub max_rounds: usize,
    #[serde(default = "default_propose_per_round")]
    pub propose_per_round: usize,
    #[serde(default = "default_minhash_threshold")]
    pub minhash_threshold: f64,
    #[serde(default)]
    pub use_web_search: bool,
    #[serde(default)]
    pub web_provider: String, // "crw", "tavily", "searxng"
    #[serde(default)]
    pub web_api_key_env: Option<String>,
    #[serde(default = "default_max_cost")]
    pub max_cost_per_round_usd: f64,
    #[serde(default = "default_alert_cost")]
    pub alert_at_usd: f64,
}

fn default_synthetic_rounds() -> usize { 2 }
fn default_propose_per_round() -> usize { 20 }
fn default_minhash_threshold() -> f64 { 0.6 }
fn default_max_cost() -> f64 { 0.0 }
fn default_alert_cost() -> f64 { 10.0 }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReviewConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_auto_assign")]
    pub auto_assign: bool,
}

fn default_auto_assign() -> bool { false }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IdiotIndexConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub block_on_cuttable: bool,
    #[serde(default = "default_warn_threshold")]
    pub warn_threshold: usize,
}

fn default_warn_threshold() -> usize { 2 }

impl EvalConfig {
    pub fn load(path: &PathBuf) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: EvalConfig = toml::from_str(&content)?;
        Ok(config)
    }
}