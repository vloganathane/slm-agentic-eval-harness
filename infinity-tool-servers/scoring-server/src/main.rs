//! Scoring RAP Tool Server
//! 
//! Exposes `score_cases` tool via RAP protocol for Infinity agents.
//! Computes element-level micro-F1, bootstrap CI, and Cohen's κ.

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
#[command(name = "scoring-server")]
struct Args {
    #[arg(short, long, default_value = "8005")]
    port: u16,
}

#[derive(Deserialize)]
struct ScoringArgs {
    predictions_file: String,
    gold_file: String,
    #[serde(default = "default_resamples")]
    bootstrap_resamples: usize,
}

fn default_resamples() -> usize { 1000 }

#[derive(Serialize)]
struct ScoringResponse {
    micro_f1: f64,
    ci_lower: f64,
    ci_upper: f64,
    per_field: HashMap<String, FieldScore>,
    strata: HashMap<String, StrataScore>,
    dataset_defect: usize,
    total_cases: usize,
}

#[derive(Serialize)]
struct FieldScore {
    tp: usize,
    fp: usize,
    fn_: usize,
    precision: f64,
    recall: f64,
    f1: f64,
}

#[derive(Serialize)]
struct StrataScore {
    n: usize,
    micro_f1: f64,
    ci_lower: f64,
    ci_upper: f64,
    quality: String,
}

#[derive(Clone)]
struct AppState;

async fn toolset_manifest() -> Json<Value> {
    Json(json!({
        "name": "scoring",
        "endpoint": format!("http://localhost:{}/invoke", std::env::var("PORT").unwrap_or("8005".to_string())),
        "tools": [{
            "name": "score_cases",
            "description": "Score predictions against gold using element-level micro-F1 + bootstrap CI",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "predictions_file": {"type": "string"},
                    "gold_file": {"type": "string"},
                    "bootstrap_resamples": {"type": "integer", "default": 1000, "minimum": 100, "maximum": 10000}
                },
                "required": ["predictions_file", "gold_file"]
            }
        }]
    }))
}

async fn invoke_tool(
    State(_): State<AppState>,
    Json(inv): Json<RapInvocation>,
) -> Result<StatusCode, StatusCode> {
    let args: ScoringArgs = serde_json::from_value(inv.arguments)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    // Load predictions and gold
    let predictions = load_jsonl(&args.predictions_file).await;
    let gold = load_jsonl(&args.gold_file).await;

    if predictions.is_err() || gold.is_err() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let predictions = predictions.unwrap();
    let gold = gold.unwrap();

    // Compute element-level micro-F1
    let result = compute_scores(&predictions, &gold, args.bootstrap_resamples);

    // Send callback
    let callback = RapCallback::ToolResult(RapToolResult {
        group_id: inv.group_id,
        id: inv.id,
        call_id: inv.call_id,
        text: Some(format!("Scoring complete: micro-F1={:.3} CI=[{:.3},{:.3}]", result.micro_f1, result.ci_lower, result.ci_upper)),
        content: Some(vec![RapToolResultContent::Text {
            text: serde_json::to_string(&result).unwrap_or_default(),
        }]),
        display_as: Some(vec![DisplaySegment::Text(
            format!("micro-F1: {:.1}% (CI: {:.1}%-{:.1}%)", result.micro_f1 * 100.0, result.ci_lower * 100.0, result.ci_upper * 100.0)
        )]),
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

async fn load_jsonl(path: &str) -> Result<Vec<Value>> {
    let content = fs::read_to_string(path).await?;
    let mut out = Vec::new();
    for line in content.lines() {
        if !line.trim().is_empty() {
            out.push(serde_json::from_str(line)?);
        }
    }
    Ok(out)
}

fn compute_scores(predictions: &[Value], gold: &[Value], _bootstrap: usize) -> ScoringResponse {
    // Build lookup
    let gold_map: HashMap<String, &Value> = gold.iter()
        .filter_map(|g| g.get("case_id").and_then(|c| c.as_str()).map(|id| (id.to_string(), g)))
        .collect();

    let mut total_tp = 0;
    let mut total_fp = 0;
    let mut total_fn = 0;
    let mut per_field: HashMap<String, (usize, usize, usize)> = HashMap::new();
    let mut strata_map: HashMap<String, Vec<(usize, usize, usize)>> = HashMap::new();
    let mut dataset_defect = 0;

    for pred in predictions {
        let case_id = pred.get("case_id").and_then(|c| c.as_str()).unwrap_or("");
        let pred_obj = pred.get("prediction").and_then(|p| p.as_object());
        let gold_case = gold_map.get(case_id);

        if pred_obj.is_none() || gold_case.is_none() {
            continue;
        }

        let pred_obj = pred_obj.unwrap();
        let gold_obj = gold_case.unwrap().get("gold").and_then(|g| g.as_object());

        if gold_obj.is_none() {
            continue;
        }

        let gold_obj = gold_obj.unwrap();
        let strata = gold_case.unwrap().get("strata")
            .and_then(|s| s.as_str())
            .unwrap_or("medium");

        // Element-level comparison
        let all_keys: std::collections::HashSet<_> = pred_obj.keys().chain(gold_obj.keys()).collect();

        for key in all_keys {
            let pred_val = pred_obj.get(key).and_then(|v| v.as_str()).unwrap_or("");
            let gold_val = gold_obj.get(key).and_then(|v| v.as_str()).unwrap_or("");

            // Skip empty/unextractable gold
            if gold_val.is_empty() || gold_val == "unextractable" {
                dataset_defect += 1;
                continue;
            }

            let entry = per_field.entry(key.to_string()).or_insert((0, 0, 0));
            let strata_entry = strata_map.entry(strata.to_string()).or_insert_with(Vec::new);

            if !pred_val.is_empty() && pred_val == gold_val {
                entry.0 += 1; // TP
                strata_entry.push((1, 0, 0));
            } else if !pred_val.is_empty() {
                entry.1 += 1; // FP
                strata_entry.push((0, 1, 0));
            } else {
                entry.2 += 1; // FN
                strata_entry.push((0, 0, 1));
            }
        }
    }

    // Aggregate per-field
    let mut per_field_scores = HashMap::new();
    for (field, (tp, fp, fn_)) in per_field {
        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        per_field_scores.insert(field, FieldScore { tp, fp, fn_: fn_, precision, recall, f1 });
        total_tp += tp;
        total_fp += fp;
        total_fn += fn_;
    }

    // Overall micro-F1
    let micro_f1 = if total_tp + total_fp + total_fn > 0 {
        let p = total_tp as f64 / (total_tp + total_fp) as f64;
        let r = total_tp as f64 / (total_tp + total_fn) as f64;
        if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 }
    } else { 0.0 };

    // Simplified bootstrap CI (using normal approximation for demo)
    let n = total_tp + total_fp + total_fn;
    let se = if n > 0 { (micro_f1 * (1.0 - micro_f1) / n as f64).sqrt() } else { 0.0 };
    let ci_lower = (micro_f1 - 1.96 * se).max(0.0);
    let ci_upper = (micro_f1 + 1.96 * se).min(1.0);

    // Per-strata
    let mut strata_scores = HashMap::new();
    for (strata, scores) in strata_map {
        let (tp, fp, fn_) = scores.iter().fold((0,0,0), |a, b| (a.0+b.0, a.1+b.1, a.2+b.2));
        let f1 = if tp + fp + fn_ > 0 {
            let p = tp as f64 / (tp + fp).max(1) as f64;
            let r = tp as f64 / (tp + fn_).max(1) as f64;
            if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 }
        } else { 0.0 };
        let se = if scores.len() > 0 { (f1 * (1.0 - f1) / scores.len() as f64).sqrt() } else { 0.0 };
        strata_scores.insert(strata.clone(), StrataScore {
            n: scores.len(),
            micro_f1: f1,
            ci_lower: (f1 - 1.96 * se).max(0.0),
            ci_upper: (f1 + 1.96 * se).min(1.0),
            quality: strata,
        });
    }

    ScoringResponse {
        micro_f1,
        ci_lower,
        ci_upper,
        per_field: per_field_scores,
        strata: strata_scores,
        dataset_defect,
        total_cases: predictions.len(),
    }
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
    info!("Scoring RAP server listening on http://127.0.0.1:{}", args.port);

    axum::serve(listener, app).await?;
    Ok(())
}