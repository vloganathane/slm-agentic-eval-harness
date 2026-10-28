//! SQLite state store (ported from slm_eval/store.py, SPEC_V3 §4.2/§4.3).

use anyhow::Result;
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};
use sqlx::Row;
use std::str::FromStr;

pub const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS run (
  run_id TEXT PRIMARY KEY, model_path TEXT, model_sha256 TEXT,
  dataset_path TEXT, dataset_hash TEXT, holdout_hash TEXT,
  status TEXT CHECK (status IN ('created','running','completed','failed')),
  config_hash TEXT, code_version TEXT,
  created_at TEXT, updated_at TEXT, version INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS stage_checkpoint (
  run_id TEXT, stage TEXT, input_hash TEXT, status TEXT,
  output_ref TEXT, error TEXT, started_at TEXT, completed_at TEXT,
  PRIMARY KEY (run_id, stage)
);
CREATE TABLE IF NOT EXISTS score (
  run_id TEXT, round INTEGER, case_id TEXT, field TEXT,
  correct INTEGER, tp INTEGER, fp INTEGER, fn INTEGER,
  judge_model TEXT, prompt_hash TEXT, latency_ms INTEGER,
  UNIQUE (run_id, round, case_id, field, prompt_hash)
);
CREATE TABLE IF NOT EXISTS stratum_case (
  run_id TEXT, case_id TEXT, stratum_key TEXT, stratum_value TEXT
);
CREATE TABLE IF NOT EXISTS calibration (
  id INTEGER PRIMARY KEY, pair_hash TEXT UNIQUE, gold TEXT,
  judge TEXT, label TEXT, kappa REAL, pairs INTEGER, created_at TEXT
);
CREATE TABLE IF NOT EXISTS review_task (
  run_id TEXT, case_id TEXT, reason TEXT, state TEXT,
  assigned_at TEXT, due_at TEXT, decided_at TEXT, decision TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS cost_ledger (
  run_id TEXT, stage TEXT, usd REAL, tokens_in INTEGER, tokens_out INTEGER, ts TEXT
);
CREATE TABLE IF NOT EXISTS proposal (
  run_id TEXT, round INTEGER, agent TEXT, kind TEXT, payload TEXT,
  state TEXT, created_at TEXT,
  UNIQUE (run_id, round, agent, kind, payload)
);
CREATE TABLE IF NOT EXISTS error_category (
  id INTEGER PRIMARY KEY, run_id TEXT, case_id TEXT, field TEXT,
  category TEXT, subcategory TEXT, confidence REAL, reviewer TEXT, created_at TEXT,
  UNIQUE (run_id, case_id, field, category)
);
CREATE TABLE IF NOT EXISTS review_comment (
  id INTEGER PRIMARY KEY, run_id TEXT, case_id TEXT, reason TEXT,
  reviewer TEXT, comment TEXT, created_at TEXT
);
";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus { Created, Running, Completed, Failed }

impl FromStr for RunStatus {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "created" => Ok(RunStatus::Created),
            "running" => Ok(RunStatus::Running),
            "completed" => Ok(RunStatus::Completed),
            "failed" => Ok(RunStatus::Failed),
            _ => Err(format!("invalid run status: {}", s)),
        }
    }
}

pub struct EvalStore {
    pool: SqlitePool,
}

impl EvalStore {
    pub async fn new(database_url: &str) -> Result<Self> {
        let db_path = database_url.trim_start_matches("sqlite:");
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .acquire_timeout(std::time::Duration::from_secs(30))
            .connect(&format!("sqlite://{}", db_path))
            .await?;
        sqlx::query(SCHEMA).execute(&pool).await?;
        Ok(Self { pool })
    }

    pub async fn create_run(&self, run_id: &str, model_path: &str, model_sha256: &str, dataset_path: &str, dataset_hash: &str, config_hash: &str, code_version: &str) -> Result<()> {
        sqlx::query(
            "INSERT INTO run (run_id, model_path, model_sha256, dataset_path, dataset_hash, config_hash, code_version, status, created_at, updated_at, version) VALUES (?, ?, ?, ?, ?, ?, ?, 'created', datetime('now'), datetime('now'), 0)",
        )
        .bind(run_id).bind(model_path).bind(model_sha256).bind(dataset_path).bind(dataset_hash)
        .bind(config_hash).bind(code_version)
        .execute(&self.pool).await?;
        Ok(())
    }

    pub async fn get_run(&self, run_id: &str) -> Result<Option<RunStatus>> {
        let rows = sqlx::query("SELECT status FROM run WHERE run_id = ?")
            .bind(run_id)
            .fetch_all(&self.pool)
            .await?;
        if rows.is_empty() {
            return Ok(None);
        }
        let status: String = rows[0].try_get("status")?;
        Ok(RunStatus::from_str(&status).ok())
    }
}
