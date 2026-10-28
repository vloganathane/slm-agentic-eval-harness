//! Main entry point for slm-eval-infinity (Rust port of the Python harness).

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use slm_eval_infinity::config::EvalConfig;
use slm_eval_infinity::models::Case;

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
            if let Some(m) = model { println!("model={}", m); }
            if let Some(m) = model_endpoint { println!("model_endpoint={}", m); }
            let _cfg = if let Some(ref p) = cli.config {
                EvalConfig::from_yaml(p)?
            } else {
                EvalConfig::default()
            };
            if let Some(ref d) = dataset {
                if d.exists() {
                    let content = std::fs::read_to_string(d)?;
                    for line in content.lines() {
                        if line.trim().is_empty() { continue; }
                        let v: serde_json::Value = serde_json::from_str(line)?;
                        let case_id = v.get("case_id").and_then(|x| x.as_str()).unwrap_or("unknown").to_string();
                        let text = v.get("text").and_then(|x| x.as_str()).unwrap_or("").to_string();
                        let gold = v.get("gold").cloned().unwrap_or(serde_json::Value::Null);
                        let strata = v.get("strata").cloned().unwrap_or(serde_json::Value::Null);
                        let ocr_signal = v.get("ocr_signal").and_then(|x| x.as_str()).map(|s| s.to_string());
                        let _case = Case::new(case_id, text, gold, strata, ocr_signal);
                    }
                }
            }
            Ok(())
        }
        Commands::Gate { run_id } => {
            println!("slm-eval-infinity gate: run_id={}", run_id);
            Ok(())
        }
    }
}
