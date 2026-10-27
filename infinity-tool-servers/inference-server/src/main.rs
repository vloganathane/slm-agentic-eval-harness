//! Inference RAP Tool Server
//! 
//! Exposes `run_inference` tool via RAP protocol for Infinity agents.
//! Wraps llama-server / OpenAI-compatible endpoints.

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
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "inference-server")]
struct Args {
    #[arg(short, long, default_value = "8002")]
    port: u16,

    /// Model endpoint (OpenAI-compatible)
    #[arg(long, default_value = "http://localhost:8080/v1")]
    model_endpoint: String,

    /// Default model
    #[arg(long, default_value = "qwen2.5-1.5b-instruct")]
    default_model: String,

    /// API key (optional)
    #[arg(long, env = "MODEL_API_KEY")]
    api_key: Option<String>,
}

#[derive(Deserialize)]
struct InferenceArgs {
    prompt: String,
    #[serde(default = "default_model")]
    model: String,
    #[serde(default)]
    json_mode: bool,
    #[serde(default = "default_temp")]
    temperature: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
}

fn default_model() -> String { "qwen2.5-1.5b-instruct".to_string() }
fn default_temp() -> f32 { 0.0 }
fn default_max_tokens() -> u32 { 2048 }

#[derive(Serialize)]
struct InferenceResponse {
    text: String,
    usage: Option<Value>,
    model: String,
}

#[derive(Clone)]
struct AppState {
    http: Client,
    model_endpoint: String,
    default_model: String,
    api_key: Option<String>,
}

async fn toolset_manifest(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "name": "inference",
        "endpoint": format!("http://localhost:{}/invoke", std::env::var("PORT").unwrap_or("8002".to_string())),
        "tools": [{
            "name": "run_inference",
            "description": "Run model inference on a prompt (llama-server / OpenAI-compatible)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "model": {"type": "string", "default": state.default_model},
                    "json_mode": {"type": "boolean", "default": false},
                    "temperature": {"type": "number", "default": 0.0},
                    "max_tokens": {"type": "integer", "default": 2048}
                },
                "required": ["prompt"]
            }
        }]
    }))
}

async fn invoke_tool(
    State(state): State<AppState>,
    Json(inv): Json<RapInvocation>,
) -> Result<StatusCode, StatusCode> {
    let args: InferenceArgs = serde_json::from_value(inv.arguments)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let mut body = json!({
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    });

    if args.json_mode {
        body["response_format"] = json!({"type": "json_object"});
    }

    let mut req = state.http
        .post(&format!("{}/chat/completions", state.model_endpoint))
        .json(&body);

    if let Some(key) = &state.api_key {
        req = req.bearer_auth(key);
    }

    let response = req.send().await
        .map_err(|e| {
            warn!("Inference request failed: {}", e);
            StatusCode::BAD_GATEWAY
        })?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        warn!("Inference error {}: {}", status, text);
        return Err(StatusCode::BAD_GATEWAY);
    }

    let data: Value = response.json().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    let text = data["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    let usage = data.get("usage").cloned();
    let total_tokens = usage.as_ref().map(|u| u["total_tokens"].as_u64().unwrap_or(0)).unwrap_or(0);

    let response_obj = InferenceResponse {
        text,
        usage,
        model: args.model.clone(),
    };

    // Send callback
    let callback = RapCallback::ToolResult(RapToolResult {
        group_id: inv.group_id,
        id: inv.id,
        call_id: inv.call_id,
        text: Some(format!("Inference complete ({} chars)", response_obj.text.len())),
        content: Some(vec![RapToolResultContent::Text {
            text: serde_json::to_string(&response_obj).unwrap_or_default(),
        }]),
        display_as: Some(vec![DisplaySegment::Text(
            format!("Inference: {} model, {} tokens", args.model, total_tokens)
        )]),
        subscription: None,
    });

    let callback_body = serde_json::to_string(&callback).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let callback_url = inv.callback_url.clone();

    let http = state.http.clone();
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

    let http = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;

    let state = AppState {
        http,
        model_endpoint: args.model_endpoint,
        default_model: args.default_model,
        api_key: args.api_key,
    };

    let app = Router::new()
        .route("/.well-known/rap-toolset", get(toolset_manifest))
        .route("/invoke", post(invoke_tool))
        .with_state(state);

    let listener = TcpListener::bind(format!("127.0.0.1:{}", args.port)).await?;
    info!("Inference RAP server listening on http://127.0.0.1:{}", args.port);

    axum::serve(listener, app).await?;
    Ok(())
}