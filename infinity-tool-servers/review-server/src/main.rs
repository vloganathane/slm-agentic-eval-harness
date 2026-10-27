//! Human Review Queue RAP Tool Server
//! 
//! Exposes `human_review` tool via RAP protocol for Infinity agents.
//! Submits cases to review queue, waits for human resolution.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

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
use sqlx::{sqlite::SqlitePoolOptions, SqlitePool};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "review-server")]
struct Args {
    #[arg(short, long, default_value = "8004")]
    port: u16,

    #[arg(long, default_value = "sqlite:runs/slm_eval.db")]
    database_url: String,
}

#[derive(Deserialize)]
struct ReviewArgs {
    run_id: String,
    cases: Vec<Value>,
    reason: String,
    #[serde(default)]
    reviewer: Option<String>,
}

#[derive(Serialize)]
struct ReviewResponse {
    task_ids: Vec<String>,
    status: String,
}

#[derive(Clone)]
struct AppState {
    pool: SqlitePool,
    wait_channels: Arc<tokio::sync::Mutex<HashMap<String, mpsc::Sender<ReviewResolution>>>>,
}

#[derive(Debug, Clone)]
struct ReviewResolution {
    task_id: String,
    status: String, // "approved", "rejected", "modified"
    resolution: Option<Value>,
}

async fn toolset_manifest() -> Json<Value> {
    Json(json!({
        "name": "review",
        "endpoint": format!("http://localhost:{}/invoke", std::env::var("PORT").unwrap_or("8004".to_string())),
        "tools": [{
            "name": "human_review",
            "description": "Submit cases to human review queue, wait for resolution",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "cases": {"type": "array", "items": {"type": "object"}},
                    "reason": {"type": "string", "enum": ["synthetic_injection", "judge_disagreement", "low_confidence"]},
                    "reviewer": {"type": "string"}
                },
                "required": ["run_id", "cases", "reason"]
            }
        }]
    }))
}

async fn invoke_tool(
    State(state): State<AppState>,
    Json(inv): Json<RapInvocation>,
) -> Result<StatusCode, StatusCode> {
    let args: ReviewArgs = serde_json::from_value(inv.arguments)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    // Insert review tasks
    let mut task_ids = Vec::new();
    for case in &args.cases {
        let task_id = uuid::Uuid::new_v4().to_string();
        let case_json = serde_json::to_string(case).unwrap();
        
        sqlx::query(
            r#"INSERT INTO review_task (task_id, run_id, case_json, reason, status, created_at)
               VALUES (?, ?, ?, ?, 'pending', datetime('now'))"#
        )
        .bind(task_id.clone())
        .bind(args.run_id.clone())
        .bind(case_json)
        .bind(args.reason.clone())
        .execute(&state.pool)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        task_ids.push(task_id);
    }

    // If no cases, return immediately
    if task_ids.is_empty() {
        return send_callback(&inv, "No cases to review", json!({"task_ids": [], "status": "no_cases"}), StatusCode::OK).await;
    }

    // Create a channel to wait for resolution
    let (tx, mut rx) = mpsc::channel::<ReviewResolution>(task_ids.len());
    
    // Register wait channel
    let wait_key = format!("{}:{}", args.run_id, task_ids[0]);
    state.wait_channels.lock().await.insert(wait_key.clone(), tx);

    // Send initial callback that we're waiting
    let _ = send_callback(&inv, 
        &format!("Submitted {} cases to review queue, waiting for human...", task_ids.len()),
        json!({"task_ids": &task_ids, "status": "waiting"}),
        StatusCode::OK
    ).await;

    // Wait for all tasks to resolve (with timeout)
    let mut resolved = 0;
    let mut final_status = "pending";
    let mut final_resolution: Option<Value> = None;

    while resolved < task_ids.len() {
        tokio::select! {
            Some(res) = rx.recv() => {
                resolved += 1;
                if res.status != "pending" {
                    final_status = &res.status;
                    final_resolution = res.resolution;
                }
                if resolved == task_ids.len() {
                    break;
                }
            }
            _ = tokio::time::sleep(Duration::from_secs(3600)) => {
                // 1 hour timeout
                final_status = "timeout";
                break;
            }
        }
    }

    // Clean up
    state.wait_channels.lock().await.remove(&wait_key);

    // Final callback
    let result = json!({
        "task_ids": task_ids,
        "status": final_status,
        "resolution": final_resolution
    });

    send_callback(&inv, &format!("Review complete: {}", final_status), result, StatusCode::OK).await
}

async fn send_callback(
    inv: &RapInvocation,
    text: &str,
    content: Value,
    status: StatusCode,
) -> Result<StatusCode, StatusCode> {
    let callback = RapCallback::ToolResult(RapToolResult {
        group_id: inv.group_id.clone(),
        id: inv.id.clone(),
        call_id: inv.call_id.clone(),
        text: Some(text.to_string()),
        content: Some(vec![RapToolResultContent::Text {
            text: content.to_string(),
        }]),
        display_as: Some(vec![DisplaySegment::Text(text.to_string())]),
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

    Ok(status)
}

// HTTP endpoint for human to resolve reviews
async fn resolve_review(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Result<StatusCode, StatusCode> {
    let task_id = payload["task_id"].as_str().ok_or(StatusCode::BAD_REQUEST)?;
    let status = payload["status"].as_str().ok_or(StatusCode::BAD_REQUEST)?;
    let resolution = payload.get("resolution").cloned();

    sqlx::query(
        "UPDATE review_task SET status = ?, resolved_at = datetime('now') WHERE task_id = ?"
    )
    .bind(status)
    .bind(task_id)
    .execute(&state.pool)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Notify waiting agent
    let key = format!("review:{}", task_id);
    if let Some(tx) = state.wait_channels.lock().await.get(&key) {
        let _ = tx.send(ReviewResolution {
            task_id: task_id.to_string(),
            status: status.to_string(),
            resolution,
        }).await;
    }

    Ok(StatusCode::OK)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    unsafe { std::env::set_var("PORT", args.port.to_string()); }

    let pool = SqlitePoolOptions::new()
        .max_connections(10)
        .connect(&args.database_url)
        .await?;

    // Ensure review_task table exists
    sqlx::query(
        r#"CREATE TABLE IF NOT EXISTS review_task (
            task_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            case_json TEXT NOT NULL,
            reason TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            reviewer TEXT,
            created_at TEXT NOT NULL,
            resolved_at TEXT
        )"#
    )
    .execute(&pool)
    .await?;

    let state = AppState {
        pool,
        wait_channels: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
    };

    let app = Router::new()
        .route("/.well-known/rap-toolset", get(toolset_manifest))
        .route("/invoke", post(invoke_tool))
        .route("/resolve", post(resolve_review))
        .with_state(state);

    let listener = TcpListener::bind(format!("127.0.0.1:{}", args.port)).await?;
    info!("Review RAP server listening on http://127.0.0.1:{}", args.port);

    axum::serve(listener, app).await?;
    Ok(())
}