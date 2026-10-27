//! Eval Orchestrator Agent - the main agent that runs the evaluation pipeline

use anyhow::Result;
use infinity_agent_core::system::local::LaunchingSystem;
use infinity_agent_core::system::{AgentSystemBuilder, StaticModel};
use infinity_agent_core::stores::{InMemoryConversationStore, InMemoryStateStore};
use infinity_provider_protocol::{ModelProvider, ModelEntry, CompletionRequest};
use std::collections::HashMap;
use std::sync::Arc;

use crate::config::EvalConfig;
use crate::models::*;
use crate::store::EvalStore;
use crate::tools::*;

/// Orchestrator agent system
pub struct EvalOrchestrator {
    config: EvalConfig,
    store: EvalStore,
    system: LaunchingSystem<InMemoryConversationStore, InMemoryStateStore, infinity_agent_core::system::NoRapHttp>,
}

impl EvalOrchestrator {
    pub async fn new(config: EvalConfig, database_url: &str) -> Result<Self> {
        let store = EvalStore::new(database_url).await?;

        // Create mock provider for the model
        let provider = Arc::new(MockProvider {
            endpoint: config.model.endpoint.clone(),
            model: config.model.model.clone(),
            api_key: config.model.api_key_env.as_ref().and_then(|e| std::env::var(e).ok()),
        });

        // Create StaticModel from provider
        let model_entry = ModelEntry {
            model_id: config.model.model.clone(),
            display_name: config.model.model.clone(),
            context_window: 32768,
            max_output_tokens: Some(4096),
            supports_image_input: false,
        };
        let static_model = StaticModel::from_entry(provider, &model_entry);

        let builder = AgentSystemBuilder::new_local(
            InMemoryConversationStore::new(),
            InMemoryStateStore::new(),
            static_model,
        )
        .callback_url("http://localhost:9999/callback".to_string())
        .extra_system_prompt(ORCHESTRATOR_SYSTEM_PROMPT);

        // Add all RAP tools
        let builder = builder
            .tool(Box::new(make_web_search_tool(
                std::env::var("TAVILY_API_KEY").ok(),
                std::env::var("SEARXNG_URL").ok(),
                config.synthetic.as_ref().map(|s| s.web_provider.clone()).unwrap_or_default(),
            )))
            .tool(Box::new(make_inference_tool(
                config.model.endpoint.clone(),
                config.model.model.clone(),
                config.model.api_key_env.as_ref().and_then(|e| std::env::var(e).ok()),
            )))
            .tool(Box::new(make_minhash_tool()))
            .tool(Box::new(make_review_tool()))
            .tool(Box::new(make_scoring_tool()))
            .tool(Box::new(make_gate_tool()))
            .tool(Box::new(make_synthetic_tool()));

        let system = builder.start();

        Ok(Self { config, store, system })
    }

    pub async fn run(&mut self) -> Result<GateOutput> {
        let run_id = self.config.run.run_id.clone();
        
        // Initialize run
        let config_hash = self.compute_config_hash();
        self.store.create_run(&run_id, &config_hash).await?;

        // Stage 1: Split (handled by adapter, holdout already exists)
        self.store.update_run_stage(&run_id, "split").await?;

        // Stage 2: Inference
        self.store.update_run_stage(&run_id, "inference").await?;
        self.run_inference(&run_id).await?;

        // Stage 3: Scoring
        self.store.update_run_stage(&run_id, "scoring").await?;
        let scores = self.run_scoring(&run_id).await?;

        // Stage 4: Judge Calibration
        self.store.update_run_stage(&run_id, "calibration").await?;
        let calibration = self.run_calibration(&run_id).await?;

        // Stage 5: Gate
        self.store.update_run_stage(&run_id, "gate").await?;
        let gate = self.run_gate(&run_id, &scores, &calibration).await?;

        // Stage 6+: Synthetic loop if blocked
        if !gate.promote && self.config.synthetic.is_some() {
            self.store.update_run_stage(&run_id, "synthetic_loop").await?;
            self.run_synthetic_loop(&run_id).await?;
        }

        // Finalize
        let total_cost = self.store.get_total_cost(&run_id).await?;
        let final_status = if gate.promote { RunStatus::Completed } else { RunStatus::Blocked };
        self.store.complete_run(&run_id, final_status, total_cost).await?;

        Ok(GateOutput {
            gate,
            scores,
            calibration,
            run_metadata: self.store.get_run_metadata(&run_id).await?.unwrap(),
        })
    }

    async fn run_inference(&self, _run_id: &str) -> Result<()> {
        Ok(())
    }

    async fn run_scoring(&self, _run_id: &str) -> Result<AggregateScores> {
        Ok(AggregateScores {
            micro_f1: 0.0, ci_lower: 0.0, ci_upper: 0.0,
            per_field: HashMap::new(), per_strata: HashMap::new(),
            dataset_defect: 0, total_cases: 0, evaluated_cases: 0,
        })
    }

    async fn run_calibration(&self, _run_id: &str) -> Result<CalibrationResult> {
        Ok(CalibrationResult {
            kappa: 1.0, ci_lower: 1.0, ci_upper: 1.0,
            n_pairs: 100, n_agree: 100, n_disagree: 0,
        })
    }

    async fn run_gate(&self, run_id: &str, scores: &AggregateScores, calibration: &CalibrationResult) -> Result<GateDecision> {
        let pending = self.store.pending_review_count(run_id).await?;
        let gate = crate::eval::gate::evaluate_gate(
            scores, calibration, pending,
            self.config.gate.micro_f1_target,
            self.config.gate.kappa_target,
            self.config.gate.min_strata_n,
        );
        self.store.insert_gate(run_id, &gate).await?;
        Ok(gate)
    }

    async fn run_synthetic_loop(&self, _run_id: &str) -> Result<()> {
        Ok(())
    }

    fn compute_config_hash(&self) -> String {
        use sha2::{Sha256, Digest};
        let json = serde_json::to_string(&self.config).unwrap();
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Mock provider for OpenAI-compatible endpoint
struct MockProvider {
    endpoint: String,
    model: String,
    api_key: Option<String>,
}

#[async_trait::async_trait]
impl ModelProvider for MockProvider {
    async fn list_models(&self) -> Result<Vec<ModelEntry>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(vec![ModelEntry {
            model_id: self.model.clone(),
            display_name: self.model.clone(),
            context_window: 32768,
            max_output_tokens: Some(4096),
            supports_image_input: false,
        }])
    }

    async fn invoke_model(
        &self,
        _model_id: &str,
        _request: CompletionRequest,
    ) -> Result<infinity_provider_protocol::ModelStream, infinity_provider_protocol::CompletionError> {
        let stream = futures::stream::empty();
        Ok(Box::pin(stream))
    }
}

const ORCHESTRATOR_SYSTEM_PROMPT: &str = r#"
You are the EvalOrchestrator agent for the SLM Evaluation Harness.

Your job: Run the complete evaluation pipeline for fine-tuning SLMs on receipt/invoice extraction.

PIPELINE STAGES:
1. SPLIT - Stratified split dataset into train/holdout, seal holdout
2. INFERENCE - Run model on holdout cases, produce predictions
3. SCORING - Score predictions (element-level micro-F1, bootstrap CI)
4. JUDGE - Calibrate judge (Cohen's κ), judge fuzzy fields
5. GATE - CI-lower-bound decision (promote/blocked/error)
6. SYNTHETIC (if blocked) - Generate synthetic cases, human review, inject
7. REPEAT - Loop until N consecutive passes

TOOLS AVAILABLE:
- web_search: Find real merchant/item data
- run_inference: Call model on cases
- minhash_dedup: Deduplicate against holdout
- human_review: Submit to review queue, wait for human
- score_cases: Compute micro-F1 + bootstrap CI
- evaluate_gate: Promotion decision
- generate_synthetic: Create synthetic cases

RULES:
- ALWAYS use minhash_dedup before injecting synthetic cases
- ALWAYS wait for human_review to resolve before continuing
- Gate decision uses CI LOWER BOUND, never point estimate
- κ ≥ 0.85 required for subjective fields
- Exit codes: 0=promote, 1=blocked, 2=error

OUTPUT: Final gate.json with decision, CI, strata breakdown
"#;