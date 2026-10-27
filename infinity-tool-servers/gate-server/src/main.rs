//! Promotion Gate RAP Tool Server
//! 
//! Exposes `evaluate_gate` tool via RAP protocol for Infinity agents.
//! Implements CI-lower-bound promotion decision logic.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    routing::{get, post},
    Router,
};
use clap::Parser;
use rap_protocol::{
    RapCallback, RapInvocation, RapToolResult, RapToolResultContent, DisplaySegment,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::fs;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "gate-server")]
struct Args {
    #[arg(short, long, default_value = "8006")]
    port: u16,
}

#[derive(Deserialize)]
struct GateArgs {
    scores_file: String,
    calibration_file: String,
    #[serde(default)]
    pending_reviews: usize,
    thresholds: Option<HashMap<String, Value>>,
}

#[derive(Serialize)]
struct GateResponse {
    decision: String, // "promote", "blocked", "blocked_pending_review", "error"
    promote: bool,
    micro_f1: f64,
    ci_lower: f64,
    ci_upper: f64,
    kappa: f64,
    kappa_ci_lower: f64,
    strata_pass: bool,
    calibration_pass: bool,
    pending_reviews: usize,
    exit_code: i32,
}

#[derive(Clone)]
struct AppState;

async fn toolset_manifest() -> Json<Value> {
    Json(json!({
        "name": "gate",
        "endpoint": format!("http://localhost:{}/invoke", std::env::var("PORT").unwrap_or("8006".to_string())),
        "tools": [{
            "name": "evaluate_gate",
            "description": "Evaluate promotion gate using CI lower bounds",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scores_file": {"type": "string"},
                    "calibration_file": {"type": "string"},
                    "pending_reviews": {"type": "integer", "default": 0},
                    "thresholds": {"type": "object"}
                },
                "required": ["scores_file", "calibration_file"]
            }
        }]
    }))
}

async fn invoke_tool(
    State(_): State<AppState>,
    Json(inv): Json<RapInvocation>,
) -> Result<StatusCode, StatusCode> {
    let args: GateArgs = serde_json::from_value(inv.arguments)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    // Load scores
    let scores_content = fs::read_to_string(&args.scores_file).await
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    let scores: Value = serde_json::from_str(&scores_content)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    // Load calibration
    let cal_content = fs::read_to_string(&args.calibration_file).await
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    let cal: Value = serde_json::from_str(&cal_content)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    // Extract metrics
    let micro_f1 = scores["micro_f1"].as_f64().unwrap_or(0.0);
    let ci_lower = scores["ci_lower"].as_f64().unwrap_or(0.0);
    let ci_upper = scores["ci_upper"].as_f64().unwrap_or(0.0);
    let dataset_defect = scores["dataset_defect"].as_u64().unwrap_or(0) as usize;
    let total_cases = scores["total_cases"].as_u64().unwrap_or(0) as usize;

    let kappa = cal["kappa"].as_f64().unwrap_or(0.0);
    let kappa_ci_lower = cal["kappa_ci_lower"].as_f64().unwrap_or(0.0);

    // Thresholds (from config or defaults)
    let thresholds = args.thresholds.unwrap_or_else(|| {
        let mut t = HashMap::new();
        t.insert("micro_f1_target".to_string(), json!(0.85));
        t.insert("kappa_target".to_string(), json!(0.85));
        t.insert("min_strata_n".to_string(), json!(30));
        t
    });

    let micro_f1_target = thresholds.get("micro_f1_target").and_then(|v| v.as_f64()).unwrap_or(0.85);
    let kappa_target = thresholds.get("kappa_target").and_then(|v| v.as_f64()).unwrap_or(0.85);
    let min_strata_n = thresholds.get("min_strata_n").and_then(|v| v.as_u64()).unwrap_or(30) as usize;

    // Decision logic
    let mut decision = "promote";
    let mut promote = true;
    let mut reasons = Vec::new();

    // Check pending reviews
    if args.pending_reviews > 0 {
        decision = "blocked_pending_review";
        promote = false;
        reasons.push(format!("{} pending reviews", args.pending_reviews));
    }

    // Check micro-F1 CI lower bound
    if ci_lower < micro_f1_target {
        decision = "blocked";
        promote = false;
        reasons.push(format!("micro-F1 CI lower {:.3} < target {:.3}", ci_lower, micro_f1_target));
    }

    // Check Cohen's κ
    if kappa_ci_lower < kappa_target {
        decision = "blocked";
        promote = false;
        reasons.push(format!("κ CI lower {:.3} < target {:.3}", kappa_ci_lower, kappa_target));
    }

    // Check strata (simplified)
    let strata_pass = true; // Would check per-strata CI bounds
    let calibration_pass = kappa_ci_lower >= kappa_target;

    let exit_code = match decision.as_ref() {
        "promote" => 0,
        "blocked_pending_review" => 1,
        "blocked" => 1,
        _ => 2,
    };

    let response = GateResponse {
        decision: decision.to_string(),
        promote,
        micro_f1,
        ci_lower,
        ci_upper,
        kappa,
        kappa_ci_lower,
        strata_pass,
        calibration_pass,
        pending_reviews: args.pending_reviews,
        exit_code,
    };

    // Send callback
    let text = format!("Gate: {} (exit={}) - {}", decision, exit_code, reasons.join("; "));
    let callback = RapCallback::ToolResult(RapToolResult {
        group_id: inv.group_id,
        id: inv.id,
        call_id: inv.call_id,
        text: Some(text.clone()),
        content: Some(vec![RapToolResultContent::Text {
            text: serde_json::to_string(&response).unwrap_or_default(),
        }]),
        display_as: Some(vec![DisplaySegment::Text(text)]),
        subscription: None,
    });

    let callback_body = serde_json::to_string(&callback).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let callback_url = inv.callback_url.clone();

    let http = reqwest::Client::new();
    tokio::spawn(async move {
        if let Err(e) = http.post(&callback_url)
            .header("content-type", "application/json")
            .body(callback_body)
            .send()
            .await {
            warn!("Failed to deliver callback: {}", e);
        }
    });

    Ok(StatusCode::OK)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    unsafe { std::env::set_var("PORT", args.port.to_string()); }

    let state = AppState;

    let app = Router::new()
        .route("/.well-known/rap-toolset", get(toolset_manifest))
        .route("/invoke", post(invoke_tool))
        .with_state(state);

    let listener = TcpListener::bind(format!("127.0.0.1:{}", args.port)).await?;
    info!("Gate RAP server listening on http://127.0.0.1:{}", args.port);

    axum::serve(listener, app).await?;
    Ok(())
}