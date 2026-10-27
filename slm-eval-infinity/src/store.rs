//! SQLite-backed persistent store for slm-eval-infinity

use anyhow::Result;
use sqlx::{sqlite::SqlitePoolOptions, SqlitePool, Row};
use std::sync::Arc;
use chrono::Utc;

use crate::models::{
    Case, Prediction, ScoreResult, AggregateScores, CalibrationResult, GateDecision,
    GateDecisionType,
    SyntheticCase, ReviewTask, ReviewStatus, ReviewResolution,
    CostEntry, RunMetadata, RunStatus,
};

#[derive(Clone)]
pub struct EvalStore {
    pool: Arc<SqlitePool>,
}

impl EvalStore {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await?;
        
        let store = Self { pool: Arc::new(pool) };
        store.migrate().await?;
        Ok(store)
    }

    async fn migrate(&self) -> Result<()> {
        // Core tables
        sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                gold_json TEXT NOT NULL,
                strata_json TEXT NOT NULL,
                ocr_signal TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )"#
        ).execute(&*self.pool).await?;

        sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS predictions (
                case_id TEXT PRIMARY KEY,
                prediction_json TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (case_id) REFERENCES cases(case_id)
            )"#
        ).execute(&*self.pool).await?;

        sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS scores (
                case_id TEXT PRIMARY KEY,
                field_scores_json TEXT NOT NULL,
                strata_json TEXT NOT NULL,
                is_quarantined INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (case_id) REFERENCES cases(case_id)
            )"#
        ).execute(&*self.pool).await?;

        sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS aggregate_scores (
                run_id TEXT PRIMARY KEY,
                micro_f1 REAL NOT NULL,
                ci_lower REAL NOT NULL,
                ci_upper REAL NOT NULL,
                per_field_json TEXT NOT NULL,
                per_strata_json TEXT NOT NULL,
                dataset_defect INTEGER NOT NULL,
                total_cases INTEGER NOT NULL,
                evaluated_cases INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )"#
        ).execute(&*self.pool).await?;

        sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS calibrations (
                run_id TEXT PRIMARY KEY,
                kappa REAL NOT NULL,
                ci_lower REAL NOT NULL,
                ci_upper REAL NOT NULL,
                n_pairs INTEGER NOT NULL,
                n_agree INTEGER NOT NULL,
                n_disagree INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )"#
        ).execute(&*self.pool).await?;

        sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS gates (
                run_id TEXT PRIMARY KEY,
                decision TEXT NOT NULL,
                promote INTEGER NOT NULL,
                micro_f1 REAL NOT NULL,
                ci_lower REAL NOT NULL,
                ci_upper REAL NOT NULL,
                kappa REAL NOT NULL,
                kappa_ci_lower REAL NOT NULL,
                strata_pass INTEGER NOT NULL,
                calibration_pass INTEGER NOT NULL,
                pending_reviews INTEGER NOT NULL,
                reasons_json TEXT NOT NULL,
                exit_code INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )"#
        ).execute(&*self.pool).await?;

        sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS synthetic_cases (
                case_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                text TEXT NOT NULL,
                gold_json TEXT NOT NULL,
                strata_json TEXT NOT NULL,
                source TEXT NOT NULL,
                merchant TEXT NOT NULL,
                minhash TEXT NOT NULL,
                injected INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )"#
        ).execute(&*self.pool).await?;

        sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS review_tasks (
                task_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                case_json TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                reviewer TEXT,
                resolution_json TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                resolved_at TEXT
            )"#
        ).execute(&*self.pool).await?;

        sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS cost_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                operation TEXT NOT NULL,
                cost_usd REAL NOT NULL,
                timestamp TEXT NOT NULL,
                metadata_json TEXT
            )"#
        ).execute(&*self.pool).await?;

        sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS run_metadata (
                run_id TEXT PRIMARY KEY,
                config_hash TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                total_cost_usd REAL NOT NULL DEFAULT 0,
                stages_completed_json TEXT NOT NULL DEFAULT '[]',
                version INTEGER NOT NULL DEFAULT 1
            )"#
        ).execute(&*self.pool).await?;

        // Indexes
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_synthetic_run ON synthetic_cases(run_id)").execute(&*self.pool).await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_review_run ON review_tasks(run_id)").execute(&*self.pool).await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_cost_run ON cost_ledger(run_id)").execute(&*self.pool).await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_review_status ON review_tasks(status)").execute(&*self.pool).await?;

        Ok(())
    }

    // Cases
    pub async fn insert_cases(&self, cases: &[Case]) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        for case in cases {
            sqlx::query(
                r#"INSERT OR REPLACE INTO cases (case_id, text, gold_json, strata_json, ocr_signal)
                   VALUES (?, ?, ?, ?, ?)"#
            )
            .bind(&case.case_id)
            .bind(&case.text)
            .bind(serde_json::to_string(&case.gold)?)
            .bind(serde_json::to_string(&case.strata)?)
            .bind(&case.ocr_signal)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn get_cases(&self, case_ids: &[String]) -> Result<Vec<Case>> {
        if case_ids.is_empty() {
            return Ok(vec![]);
        }
        let placeholders = case_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let query = format!("SELECT case_id, text, gold_json, strata_json, ocr_signal FROM cases WHERE case_id IN ({})", placeholders);
        let mut q = sqlx::query(&query);
        for id in case_ids {
            q = q.bind(id);
        }
        let rows = q.fetch_all(&*self.pool).await?;
        let mut cases = Vec::new();
        for row in rows {
            cases.push(Case {
                case_id: row.get("case_id"),
                text: row.get("text"),
                gold: serde_json::from_str(row.get("gold_json"))?,
                strata: serde_json::from_str(row.get("strata_json"))?,
                ocr_signal: row.get("ocr_signal"),
            });
        }
        Ok(cases)
    }

    pub async fn get_all_cases(&self) -> Result<Vec<Case>> {
        let rows = sqlx::query("SELECT case_id, text, gold_json, strata_json, ocr_signal FROM cases")
            .fetch_all(&*self.pool).await?;
        let mut cases = Vec::new();
        for row in rows {
            cases.push(Case {
                case_id: row.get("case_id"),
                text: row.get("text"),
                gold: serde_json::from_str(row.get("gold_json"))?,
                strata: serde_json::from_str(row.get("strata_json"))?,
                ocr_signal: row.get("ocr_signal"),
            });
        }
        Ok(cases)
    }

    // Predictions
    pub async fn insert_predictions(&self, predictions: &[Prediction]) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        for pred in predictions {
            sqlx::query(
                r#"INSERT OR REPLACE INTO predictions (case_id, prediction_json, metadata_json)
                   VALUES (?, ?, ?)"#
            )
            .bind(&pred.case_id)
            .bind(serde_json::to_string(&pred.prediction)?)
            .bind(pred.metadata.as_ref().map(serde_json::to_string).transpose()?)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn get_predictions(&self, case_ids: &[String]) -> Result<Vec<Prediction>> {
        if case_ids.is_empty() {
            return Ok(vec![]);
        }
        let placeholders = case_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let query = format!("SELECT case_id, prediction_json, metadata_json FROM predictions WHERE case_id IN ({})", placeholders);
        let mut q = sqlx::query(&query);
        for id in case_ids {
            q = q.bind(id);
        }
        let rows = q.fetch_all(&*self.pool).await?;
        let mut preds = Vec::new();
        for row in rows {
            preds.push(Prediction {
                case_id: row.get("case_id"),
                prediction: serde_json::from_str(row.get("prediction_json"))?,
                metadata: row.get::<Option<String>, _>("metadata_json")
                    .and_then(|s| serde_json::from_str(&s).ok()),
            });
        }
        Ok(preds)
    }

    // Scores
    pub async fn insert_scores(&self, scores: &[ScoreResult]) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        for score in scores {
            sqlx::query(
                r#"INSERT OR REPLACE INTO scores (case_id, field_scores_json, strata_json, is_quarantined)
                   VALUES (?, ?, ?, ?)"#
            )
            .bind(&score.case_id)
            .bind(serde_json::to_string(&score.field_scores)?)
            .bind(serde_json::to_string(&score.strata)?)
            .bind(score.is_quarantined as i64)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    // Aggregate scores
    pub async fn insert_aggregate_scores(&self, run_id: &str, scores: &AggregateScores) -> Result<()> {
        sqlx::query(
            r#"INSERT OR REPLACE INTO aggregate_scores 
               (run_id, micro_f1, ci_lower, ci_upper, per_field_json, per_strata_json, dataset_defect, total_cases, evaluated_cases)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"#
        )
        .bind(run_id)
        .bind(scores.micro_f1)
        .bind(scores.ci_lower)
        .bind(scores.ci_upper)
        .bind(serde_json::to_string(&scores.per_field)?)
        .bind(serde_json::to_string(&scores.per_strata)?)
        .bind(scores.dataset_defect as i64)
        .bind(scores.total_cases as i64)
        .bind(scores.evaluated_cases as i64)
        .execute(&*self.pool)
        .await?;
        Ok(())
    }

    pub async fn get_aggregate_scores(&self, run_id: &str) -> Result<Option<AggregateScores>> {
        let row = sqlx::query(
            "SELECT * FROM aggregate_scores WHERE run_id = ?"
        ).bind(run_id).fetch_optional(&*self.pool).await?;
        Ok(row.map(|r| AggregateScores {
            micro_f1: r.get("micro_f1"),
            ci_lower: r.get("ci_lower"),
            ci_upper: r.get("ci_upper"),
            per_field: serde_json::from_str(r.get("per_field_json")).unwrap_or_default(),
            per_strata: serde_json::from_str(r.get("per_strata_json")).unwrap_or_default(),
            dataset_defect: r.get::<i64, _>("dataset_defect") as usize,
            total_cases: r.get::<i64, _>("total_cases") as usize,
            evaluated_cases: r.get::<i64, _>("evaluated_cases") as usize,
        }))
    }

    // Calibration
    pub async fn insert_calibration(&self, run_id: &str, cal: &CalibrationResult) -> Result<()> {
        sqlx::query(
            r#"INSERT OR REPLACE INTO calibrations 
               (run_id, kappa, ci_lower, ci_upper, n_pairs, n_agree, n_disagree)
               VALUES (?, ?, ?, ?, ?, ?, ?)"#
        )
        .bind(run_id)
        .bind(cal.kappa)
        .bind(cal.ci_lower)
        .bind(cal.ci_upper)
        .bind(cal.n_pairs as i64)
        .bind(cal.n_agree as i64)
        .bind(cal.n_disagree as i64)
        .execute(&*self.pool)
        .await?;
        Ok(())
    }

    pub async fn get_calibration(&self, run_id: &str) -> Result<Option<CalibrationResult>> {
        let row = sqlx::query("SELECT * FROM calibrations WHERE run_id = ?")
            .bind(run_id).fetch_optional(&*self.pool).await?;
        Ok(row.map(|r| CalibrationResult {
            kappa: r.get("kappa"),
            ci_lower: r.get("ci_lower"),
            ci_upper: r.get("ci_upper"),
            n_pairs: r.get::<i64, _>("n_pairs") as usize,
            n_agree: r.get::<i64, _>("n_agree") as usize,
            n_disagree: r.get::<i64, _>("n_disagree") as usize,
        }))
    }

    // Gate
    pub async fn insert_gate(&self, run_id: &str, gate: &GateDecision) -> Result<()> {
        sqlx::query(
            r#"INSERT OR REPLACE INTO gates 
               (run_id, decision, promote, micro_f1, ci_lower, ci_upper, kappa, kappa_ci_lower, 
                strata_pass, calibration_pass, pending_reviews, reasons_json, exit_code)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"#
        )
        .bind(run_id)
        .bind(serde_json::to_string(&gate.decision)?)
        .bind(gate.promote as i64)
        .bind(gate.micro_f1)
        .bind(gate.ci_lower)
        .bind(gate.ci_upper)
        .bind(gate.kappa)
        .bind(gate.kappa_ci_lower)
        .bind(gate.strata_pass as i64)
        .bind(gate.calibration_pass as i64)
        .bind(gate.pending_reviews as i64)
        .bind(serde_json::to_string(&gate.reasons)?)
        .bind(gate.exit_code)
        .execute(&*self.pool)
        .await?;
        Ok(())
    }

    pub async fn get_gate(&self, run_id: &str) -> Result<Option<GateDecision>> {
        let row = sqlx::query("SELECT * FROM gates WHERE run_id = ?")
            .bind(run_id).fetch_optional(&*self.pool).await?;
        Ok(row.map(|r| GateDecision {
            decision: serde_json::from_str(r.get("decision_json")).unwrap_or(GateDecisionType::Error),
            promote: r.get::<i64, _>("promote") != 0,
            micro_f1: r.get("micro_f1"),
            ci_lower: r.get("ci_lower"),
            ci_upper: r.get("ci_upper"),
            kappa: r.get("kappa"),
            kappa_ci_lower: r.get("kappa_ci_lower"),
            strata_pass: r.get::<i64, _>("strata_pass") != 0,
            calibration_pass: r.get::<i64, _>("calibration_pass") != 0,
            pending_reviews: r.get::<i64, _>("pending_reviews") as usize,
            reasons: serde_json::from_str(r.get("reasons_json")).unwrap_or_default(),
            exit_code: r.get("exit_code"),
        }))
    }

    // Synthetic cases
    pub async fn insert_synthetic_cases(&self, run_id: &str, cases: &[SyntheticCase]) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        for case in cases {
            sqlx::query(
                r#"INSERT OR REPLACE INTO synthetic_cases 
                   (case_id, run_id, text, gold_json, strata_json, source, merchant, minhash, injected)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"#
            )
            .bind(&case.case_id)
            .bind(run_id)
            .bind(&case.text)
            .bind(serde_json::to_string(&case.gold)?)
            .bind(serde_json::to_string(&case.strata)?)
            .bind(&case.source)
            .bind(&case.merchant)
            .bind(&case.minhash)
            .bind(0i64)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn get_synthetic_cases(&self, run_id: &str) -> Result<Vec<SyntheticCase>> {
        let rows = sqlx::query(
            "SELECT * FROM synthetic_cases WHERE run_id = ?"
        ).bind(run_id).fetch_all(&*self.pool).await?;
        let mut cases = Vec::new();
        for row in rows {
            cases.push(SyntheticCase {
                case_id: row.get("case_id"),
                text: row.get("text"),
                gold: serde_json::from_str(row.get("gold_json"))?,
                strata: serde_json::from_str(row.get("strata_json"))?,
                source: row.get("source"),
                merchant: row.get("merchant"),
                minhash: row.get("minhash"),
            });
        }
        Ok(cases)
    }

    pub async fn mark_synthetic_injected(&self, run_id: &str, case_ids: &[String]) -> Result<()> {
        if case_ids.is_empty() { return Ok(()); }
        let mut tx = self.pool.begin().await?;
        for id in case_ids {
            sqlx::query("UPDATE synthetic_cases SET injected = 1 WHERE run_id = ? AND case_id = ?")
                .bind(run_id).bind(id).execute(&mut *tx).await?;
        }
        tx.commit().await?;
        Ok(())
    }

    // Review tasks
    pub async fn insert_review_tasks(&self, tasks: &[ReviewTask]) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        for task in tasks {
            sqlx::query(
                r#"INSERT OR REPLACE INTO review_tasks 
                   (task_id, run_id, case_json, reason, status, reviewer, resolution_json, created_at, resolved_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"#
            )
            .bind(&task.task_id)
            .bind(&task.run_id)
            .bind(serde_json::to_string(&task.case)?)
            .bind(serde_json::to_string(&task.reason)?)
            .bind(serde_json::to_string(&task.status)?)
            .bind(&task.reviewer)
            .bind(task.resolution.as_ref().map(serde_json::to_string).transpose()?)
            .bind(&task.created_at)
            .bind(&task.resolved_at)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn get_review_tasks(&self, run_id: &str, status: Option<ReviewStatus>) -> Result<Vec<ReviewTask>> {
        let query = if let Some(s) = status {
            format!("SELECT * FROM review_tasks WHERE run_id = ? AND status = '{}'", serde_json::to_string(&s)?)
        } else {
            "SELECT * FROM review_tasks WHERE run_id = ?".to_string()
        };
        let rows = sqlx::query(&query).bind(run_id).fetch_all(&*self.pool).await?;
        let mut tasks = Vec::new();
        for row in rows {
            tasks.push(ReviewTask {
                task_id: row.get("task_id"),
                run_id: row.get("run_id"),
                case: serde_json::from_str(row.get("case_json"))?,
                reason: serde_json::from_str(row.get("reason_json"))?,
                status: serde_json::from_str(row.get("status_json"))?,
                reviewer: row.get("reviewer"),
                resolution: row.get::<Option<String>, _>("resolution_json")
                    .and_then(|s| serde_json::from_str(&s).ok()),
                created_at: row.get("created_at"),
                resolved_at: row.get("resolved_at"),
            });
        }
        Ok(tasks)
    }

    pub async fn update_review_task(&self, task_id: &str, status: ReviewStatus, resolution: Option<ReviewResolution>, reviewer: Option<String>) -> Result<()> {
        let resolved_at = if status != ReviewStatus::Pending {
            Some(Utc::now().to_rfc3339())
        } else { None };
        sqlx::query(
            r#"UPDATE review_tasks SET status = ?, reviewer = ?, resolution_json = ?, resolved_at = ?
               WHERE task_id = ?"#
        )
        .bind(serde_json::to_string(&status)?)
        .bind(reviewer)
        .bind(resolution.as_ref().map(serde_json::to_string).transpose()?)
        .bind(&resolved_at)
        .bind(task_id)
        .execute(&*self.pool)
        .await?;
        Ok(())
    }

    pub async fn pending_review_count(&self, run_id: &str) -> Result<usize> {
        let row = sqlx::query(
            "SELECT COUNT(*) as cnt FROM review_tasks WHERE run_id = ? AND status = 'pending'"
        ).bind(run_id).fetch_one(&*self.pool).await?;
        Ok(row.get::<i64, _>("cnt") as usize)
    }

    // Cost ledger
    pub async fn record_cost(&self, run_id: &str, entry: &CostEntry) -> Result<()> {
        sqlx::query(
            r#"INSERT INTO cost_ledger (run_id, stage, operation, cost_usd, timestamp, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?)"#
        )
        .bind(run_id)
        .bind(&entry.stage)
        .bind(&entry.operation)
        .bind(entry.cost_usd)
        .bind(&entry.timestamp)
        .bind(serde_json::to_string(&entry.metadata).unwrap_or_default())
        .execute(&*self.pool)
        .await?;
        Ok(())
    }

    pub async fn get_total_cost(&self, run_id: &str) -> Result<f64> {
        let row = sqlx::query("SELECT COALESCE(SUM(cost_usd), 0.0) as total FROM cost_ledger WHERE run_id = ?")
            .bind(run_id).fetch_one(&*self.pool).await?;
        Ok(row.get("total"))
    }

    // Run metadata
    pub async fn create_run(&self, run_id: &str, config_hash: &str) -> Result<()> {
        sqlx::query(
            r#"INSERT OR REPLACE INTO run_metadata 
               (run_id, config_hash, started_at, status, stages_completed_json, version)
               VALUES (?, ?, ?, 'running', '[]', 1)"#
        )
        .bind(run_id)
        .bind(config_hash)
        .bind(Utc::now().to_rfc3339())
        .execute(&*self.pool)
        .await?;
        Ok(())
    }

    pub async fn update_run_stage(&self, run_id: &str, stage: &str) -> Result<()> {
        // Optimistic locking via version
        let row = sqlx::query("SELECT stages_completed_json, version FROM run_metadata WHERE run_id = ?")
            .bind(run_id).fetch_optional(&*self.pool).await?;
        if let Some(r) = row {
            let mut stages: Vec<String> = serde_json::from_str(r.get("stages_completed_json")).unwrap_or_default();
            if !stages.contains(&stage.to_string()) {
                stages.push(stage.to_string());
            }
            let version: i64 = r.get("version");
            sqlx::query(
                "UPDATE run_metadata SET stages_completed_json = ?, version = version + 1 WHERE run_id = ? AND version = ?"
            )
            .bind(serde_json::to_string(&stages)?)
            .bind(run_id)
            .bind(version)
            .execute(&*self.pool)
            .await?;
        }
        Ok(())
    }

    pub async fn complete_run(&self, run_id: &str, status: RunStatus, total_cost: f64) -> Result<()> {
        sqlx::query(
            "UPDATE run_metadata SET completed_at = ?, status = ?, total_cost_usd = ? WHERE run_id = ?"
        )
        .bind(Utc::now().to_rfc3339())
        .bind(serde_json::to_string(&status)?)
        .bind(total_cost)
        .bind(run_id)
        .execute(&*self.pool)
        .await?;
        Ok(())
    }

    pub async fn get_run_metadata(&self, run_id: &str) -> Result<Option<RunMetadata>> {
        let row = sqlx::query("SELECT * FROM run_metadata WHERE run_id = ?")
            .bind(run_id).fetch_optional(&*self.pool).await?;
        Ok(row.map(|r| RunMetadata {
            run_id: r.get("run_id"),
            config_hash: r.get("config_hash"),
            started_at: r.get("started_at"),
            completed_at: r.get("completed_at"),
            status: serde_json::from_str(&r.get::<String, _>("status")).unwrap_or(RunStatus::Running),
            total_cost_usd: r.get("total_cost_usd"),
            stages_completed: serde_json::from_str(&r.get::<String, _>("stages_completed_json")).unwrap_or_default(),
        }))
    }
}