use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;

use slm_eval_infinity::config::EvalConfig;
use slm_eval_infinity::models::{Case, FieldOutcome};
use slm_eval_infinity::inference::run_inference;
use slm_eval_infinity::eval::scoring::{micro_f1, bootstrap_micro_f1};
use slm_eval_infinity::eval::gate::{decide_gate, GateInputs};

#[derive(Parser)]
#[command(name = "slm-eval-infinity", version, about = "SLM Evaluation Harness on Infinity Agent Runtime")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(long, global = true, default_value = "sqlite:runs/slm_eval.db")]
    database: String,

    #[arg(long, global = true)]
    config: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run full evaluation pipeline
    Run {
        #[arg(short, long)]
        run_id: String,
        #[arg(long)]
        dataset: Option<PathBuf>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        model_endpoint: Option<String>,
    },
    /// Run only the gate evaluation on existing scores
    Gate {
        #[arg(short, long)]
        run_id: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run { run_id, dataset, model, model_endpoint } => {
            println!("slm-eval-infinity run: run_id={}", run_id);
            println!("database={}", cli.database);
            if let Some(ref d) = dataset { println!("dataset={}", d.display()); }
            if let Some(ref m) = model { println!("model={}", m); }
            if let Some(ref m) = model_endpoint { println!("model_endpoint={}", m); }

            let start = Instant::now();

            // Load config
            let config = if let Some(ref p) = cli.config {
                EvalConfig::from_yaml(p)?
            } else {
                EvalConfig::default()
            };

            // Load dataset
            let dataset_path = dataset.unwrap_or_else(|| PathBuf::from("data/cord/cases.jsonl"));
            if !dataset_path.exists() {
                return Err(anyhow::anyhow!(format!("Dataset file not found: {}", dataset_path.display())));
            }
            
            let mut cases = Vec::new();
            let content = std::fs::read_to_string(&dataset_path)?;
            for line in content.lines() {
                if line.trim().is_empty() { continue; }
                let v: serde_json::Value = serde_json::from_str(line)?;
                let case_id = v.get("case_id").and_then(|x| x.as_str()).unwrap_or("unknown").to_string();
                let text = v.get("text").and_then(|x| x.as_str()).unwrap_or("").to_string();
                let gold = v.get("gold").cloned().unwrap_or(serde_json::Value::Null);
                let strata = v.get("strata").cloned().unwrap_or(serde_json::Value::Null);
                let ocr_signal = v.get("ocr_signal").and_then(|x| x.as_str()).map(|s| s.to_string());
                let case = Case::new(case_id, text, gold, strata, ocr_signal);
                cases.push(case);
            }
            println!("Loaded {} cases from dataset", cases.len());

            // For now, use all cases as holdout (full evaluation)
            let holdout: Vec<Case> = cases.clone();
            println!("Using {} cases for evaluation", holdout.len());

            // Prepare fields for inference
            let exact_fields: Vec<String> = config.fields.exact_match_fields.clone();
            let subjective_fields: Vec<String> = config.fields.subjective_fields.clone();

            // Run inference
            let model_name = model.unwrap_or_else(|| "qwen2.5-1.5b-instruct".to_string());
            let model_endpoint = model_endpoint.unwrap_or_else(|| config.inference.base_url.clone());
            
            println!("Running inference with model {} at {}", model_name, model_endpoint);
            
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .map_err(|e| anyhow::anyhow!("Failed to build HTTP client: {}", e))?;

            let predictions = run_inference(
                &client,
                &model_endpoint,
                &model_name,
                config.inference.timeout_seconds,
                config.inference.retry.max_attempts,
                config.inference.retry.backoff_base,
                &holdout,
                &exact_fields,
                &subjective_fields,
            ).await?;

            println!("Inference complete: {} predictions", predictions.len());

            // Score predictions
            let mut outcomes: Vec<FieldOutcome> = Vec::new();
            
            // Simple scoring: exact match on all fields
            for pred in &predictions {
                let case_id = &pred.case_id;
                let case = holdout.iter().find(|c| c.case_id == *case_id).unwrap();
                
                for fname in &exact_fields {
                    if let Some(pred_val) = pred.fields.get(fname) {
                        let gold_v = case.gold.get(fname);
                        let pred_str = pred_val.as_str().unwrap_or("");
                        let gold_str = gold_v.and_then(|v| v.as_str()).unwrap_or("");
                        
                        if pred_str == gold_str {
                            outcomes.push(FieldOutcome::new(case_id.clone(), 1, 0, 0));
                        } else {
                            outcomes.push(FieldOutcome::new(case_id.clone(), 0, 1, 1));
                        }
                    }
                }
                
                // Subjective fields - mark as incorrect for now (no judge)
                for _fname in &subjective_fields {
                    outcomes.push(FieldOutcome::new(case_id.clone(), 0, 1, 1));
                }
            }
            
            println!("Scored {} outcomes", outcomes.len());

            // Calculate micro-F1
            let point = micro_f1(&outcomes);
            println!("Micro-F1: {:.4}", point);

            // Calculate bootstrap CI
            let seed_u64: u64 = config.stats.seed as u64;
            let ci_metric = bootstrap_micro_f1(&outcomes, config.stats.resamples, seed_u64, 0.05);
            println!("Micro-F1 (bootstrap): {:.4} [{:.4}, {:.4}]", 
                ci_metric.point, ci_metric.ci_lower, ci_metric.ci_upper);

            // --- Stage 6: Gate evaluation ---
            let defect_rate = 0.0;
            let defect_n = 0;
            let scored_n = outcomes.len();
            let total_n = holdout.len();
            
            let gate_inputs = GateInputs {
                run_completed: true,
                overall_micro_f1: ci_metric.point,
                overall_ci_lower: ci_metric.ci_lower,
                overall_ci_upper: ci_metric.ci_upper,
                overall_n: ci_metric.n as usize,
                strata: Vec::new(),
                kappa: None,
                pending_reviews: 0,
                subjective_fields: config.fields.subjective_fields.clone(),
                defect_rate: Some(defect_rate),
                defect_n,
                total_n,
            };
            
            let decision = decide_gate(&gate_inputs, &config.thresholds);
            println!("Gate decision: status={}, promote={}", decision.status, decision.promote);
            println!("Blockers: {:?}", decision.blockers);

            // --- Stage 7: Write gate.json ---
            let gate_path = format!("runs/{}/gate.json", run_id);
            let gate_payload = serde_json::json!({
                "run_id": run_id,
                "status": decision.status,
                "promote": decision.promote,
                "reason": decision.reason,
                "blockers": decision.blockers,
                "warnings": decision.warnings,
                "metrics": {
                    "micro_f1": format!("{:.4}", ci_metric.point),
                    "ci_lower": format!("{:.4}", ci_metric.ci_lower),
                    "ci_upper": format!("{:.4}", ci_metric.ci_upper),
                    "resamples": ci_metric.resamples,
                    "seed": config.stats.seed,
                },
                "strata": [],
                "dataset_defect": {
                    "rate": format!("{:.4}", defect_rate),
                    "defect_n": defect_n,
                    "total_n": holdout.len(),
                    "scored_n": scored_n,
                    "quarantined": defect_n > 0,
                },
                "judge": {"kappa": serde_json::Value::Null, "pairs": 0, "model": config.judge.model},
                "pending_reviews": 0,
                "cost_usd": 0.0,
                "provenance": {"config_hash": config.config_hash, "code_version": "git:unknown"},
            });
            
            let gate_dir = std::path::Path::new(&gate_path).parent().unwrap_or(std::path::Path::new("."));
            std::fs::create_dir_all(gate_dir)?;
            std::fs::write(&gate_path, serde_json::to_string_pretty(&gate_payload)?)?;
            println!("Wrote gate.json to {}", gate_path);

            let elapsed = start.elapsed();
            println!("\n=== Evaluation Summary ===");
            println!("Runtime: {:.2}s", elapsed.as_secs_f64());
            println!("Micro-F1: {:.4}", ci_metric.point);
            println!("Gate status: {}", decision.status);
            if !decision.promote {
                for b in &decision.blockers {
                    println!("  Blocker: {}", b);
                }
            }
            println!("Dataset: {} total, {} scored, {} defect", holdout.len(), scored_n, defect_n);
            
            let exit_code = if decision.promote { 0 } else { 1 };
            std::process::exit(exit_code);
        }
        Commands::Gate { run_id } => {
            println!("slm-eval-infinity gate: run_id={}", run_id);
            // In full implementation, this would run gate evaluation on existing scores
            // For now, just exit with 0
            println!("Gate evaluation (simulation)");
            Ok(())
        }
    }
}
