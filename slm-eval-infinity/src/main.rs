//! Main entry point for slm-eval-infinity

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

mod config;
mod models;
mod store;
mod tools;
mod orchestrator;
mod eval {
    pub mod scoring;
    pub mod gate;
}

use crate::config::EvalConfig;
use crate::orchestrator::EvalOrchestrator;

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
    /// Generate synthetic cases
    Synthetic {
        #[arg(short, long)]
        run_id: String,
        #[arg(long, default_value = "20")]
        n: usize,
    },
    /// Human review queue management
    Review {
        #[command(subcommand)]
        action: ReviewAction,
    },
    /// Inspect run artifacts
    Inspect {
        #[arg(short, long)]
        run_id: String,
    },
    /// Calibrate judge
    Calibrate {
        #[arg(short, long)]
        run_id: String,
        #[arg(long, default_value = "100")]
        n_pairs: usize,
    },
    /// Start Infinity daemon with all tool servers
    Daemon,
}

#[derive(Subcommand)]
enum ReviewAction {
    List { run_id: String },
    Approve { task_id: String },
    Reject { task_id: String },
    Modify { task_id: String, case_json: String },
}

#[tokio::main]
async fn main() -> Result<()> {
    let local = tokio::task::LocalSet::new();
    local.run_until(async_main()).await
}

async fn async_main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run { run_id, dataset, model, model_endpoint } => {
            let mut config = if let Some(config_path) = cli.config {
                EvalConfig::load(&config_path)?
            } else {
                default_config(&run_id)?
            };

            // Override from CLI
            if let Some(d) = dataset { config.dataset.path = d; }
            if let Some(m) = model { config.model.model = m; }
            if let Some(e) = model_endpoint { config.model.endpoint = e; }
            config.run.run_id = run_id.clone();

            let mut orchestrator = EvalOrchestrator::new(config, &cli.database).await?;
            let output = orchestrator.run().await?;

            println!("=== GATE RESULT ===");
            println!("Decision: {:?}", output.gate.decision);
            println!("Promote: {}", output.gate.promote);
            println!("micro-F1: {:.3} (CI: [{:.3}, {:.3}])", output.gate.micro_f1, output.gate.ci_lower, output.gate.ci_upper);
            println!("κ: {:.3} (CI: [{:.3}, {:.3}])", output.gate.kappa, output.gate.kappa_ci_lower, output.gate.kappa_ci_lower);
            println!("Exit code: {}", output.gate.exit_code);
            
            std::process::exit(output.gate.exit_code);
        }
        Commands::Gate { run_id } => {
            let store = crate::store::EvalStore::new(&cli.database).await?;
            if let Some(gate) = store.get_gate(&run_id).await? {
                println!("{:?}", gate);
                std::process::exit(gate.exit_code);
            } else {
                eprintln!("No gate found for run: {}", run_id);
                std::process::exit(2);
            }
        }
        Commands::Synthetic { run_id, n } => {
            println!("Generating {} synthetic cases for run {}", n, run_id);
            // Would call synthetic tool
        }
        Commands::Review { action } => {
            let store = crate::store::EvalStore::new(&cli.database).await?;
            match action {
                ReviewAction::List { run_id } => {
                    let tasks = store.get_review_tasks(&run_id, None).await?;
                    for t in tasks {
                        println!("{} [{:?}] {:?}", t.task_id, t.status, t.reason);
                    }
                }
                ReviewAction::Approve { task_id } => {
                    store.update_review_task(&task_id, crate::models::ReviewStatus::Approved, None, None).await?;
                    println!("Approved {}", task_id);
                }
                ReviewAction::Reject { task_id } => {
                    store.update_review_task(&task_id, crate::models::ReviewStatus::Rejected, None, None).await?;
                    println!("Rejected {}", task_id);
                }
                ReviewAction::Modify { task_id, case_json } => {
                    let case: crate::models::SyntheticCase = serde_json::from_str(&case_json)?;
                    let resolution = crate::models::ReviewResolution {
                        status: crate::models::ReviewStatus::Modified,
                        modified_case: Some(case),
                        comment: None,
                    };
                    store.update_review_task(&task_id, crate::models::ReviewStatus::Modified, Some(resolution), None).await?;
                    println!("Modified {}", task_id);
                }
            }
        }
        Commands::Inspect { run_id } => {
            let store = crate::store::EvalStore::new(&cli.database).await?;
            if let Some(meta) = store.get_run_metadata(&run_id).await? {
                println!("Run: {}", meta.run_id);
                println!("Status: {:?}", meta.status);
                println!("Cost: ${:.4}", meta.total_cost_usd);
                println!("Stages: {:?}", meta.stages_completed);
            }
            if let Some(gate) = store.get_gate(&run_id).await? {
                println!("\nGate: {:?}", gate.decision);
                println!("micro-F1: {:.3}", gate.micro_f1);
            }
        }
        Commands::Calibrate { run_id, n_pairs } => {
            println!("Calibrating judge for run {} with {} pairs", run_id, n_pairs);
            // Would run calibration
        }
        Commands::Daemon => {
            println!("Starting Infinity daemon with tool servers...");
            println!("This would start the daemon and all 7 tool servers");
            println!("Use the launch_infinity.sh script instead");
        }
    }

    Ok(())
}

fn default_config(run_id: &str) -> Result<EvalConfig> {
    use crate::config::*;
    use std::collections::HashMap;

    Ok(EvalConfig {
        run: RunConfig {
            run_id: run_id.to_string(),
            output_dir: PathBuf::from(format!("runs/{}", run_id)),
            seal_holdout: true,
            resume: false,
        },
        dataset: DatasetConfig {
            path: PathBuf::from("data/cord"),
            adapter: "cord".to_string(),
            strata_key: Some("quality".to_string()),
            quarantine_unextractable: true,
        },
        model: ModelConfig {
            endpoint: "http://localhost:8080/v1".to_string(),
            model: "qwen2.5-1.5b-instruct".to_string(),
            api_key_env: None,
            temperature: 0.0,
            max_tokens: 2048,
            json_mode: true,
        },
        judge: JudgeConfig {
            endpoint: "https://integrate.api.nvidia.com/v1".to_string(),
            model: "nvidia/nemotron-3.5-bolt-e4b".to_string(),
            api_key_env: "NVIDIA_API_KEY".to_string(),
            kappa_target: 0.85,
            calibration_file: Some(PathBuf::from("calibration/judge.jsonl")),
        },
        split: SplitConfig {
            train_ratio: 0.8,
            strata_mode: "single_key".to_string(),
            seed: 42,
        },
        scoring: ScoringConfig {
            bootstrap_resamples: 1000,
            ci_level: 0.95,
        },
        gate: GateConfig {
            micro_f1_target: 0.85,
            kappa_target: 0.85,
            min_strata_n: 30,
            thresholds: HashMap::new(),
        },
        synthetic: Some(SyntheticConfig {
            max_rounds: 2,
            propose_per_round: 20,
            minhash_threshold: 0.6,
            use_web_search: true,
            web_provider: "crw".to_string(),
            web_api_key_env: None,
            max_cost_per_round_usd: 0.0,
            alert_at_usd: 10.0,
        }),
        review: Some(ReviewConfig {
            enabled: true,
            auto_assign: false,
        }),
        idiot_index: Some(IdiotIndexConfig {
            enabled: true,
            block_on_cuttable: true,
            warn_threshold: 2,
        }),
    })
}