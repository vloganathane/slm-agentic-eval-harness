//! MinHash Deduplication RAP Tool Server
//! 
//! Exposes `minhash_dedup` tool via RAP protocol for Infinity agents.
//! Implements MinHash signatures and Jaccard similarity for text deduplication.

use std::collections::HashMap;
use std::sync::Arc;
use std::path::PathBuf;

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
#[command(name = "minhash-server")]
struct Args {
    #[arg(short, long, default_value = "8003")]
    port: u16,
}

#[derive(Deserialize)]
struct DedupArgs {
    cases: Vec<Value>,
    holdout_dir: String,
    #[serde(default = "default_threshold")]
    threshold: f64,
}

fn default_threshold() -> f64 { 0.6 }

#[derive(Serialize)]
struct CaseResult {
    case_id: String,
    text: String,
    gold: Value,
    strata: Value,
    minhash: String,
    is_duplicate: bool,
}

#[derive(Serialize)]
struct DedupResponse {
    proposed: usize,
    novel: usize,
    duplicate: usize,
    cases: Vec<CaseResult>,
}

// MinHash implementation (compatible with slm_eval.hashing)
const MINHASH_NUM_HASHES: usize = 128;
const MINHASH_SEED: u64 = 0x9e3779b97f4a7c15;

fn minhash_signature(text: &str) -> Vec<u64> {
    let shingles = get_shingles(text, 5);
    let mut signatures = vec![u64::MAX; MINHASH_NUM_HASHES];
    
    for shingle in shingles {
        let hash = hash_shingle(&shingle);
        for i in 0..MINHASH_NUM_HASHES {
            let permuted = hash.wrapping_mul(MINHASH_SEED.wrapping_add(i as u64));
            if permuted < signatures[i] {
                signatures[i] = permuted;
            }
        }
    }
    signatures
}

fn get_shingles(text: &str, k: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < k {
        return vec![text.to_string()];
    }
    chars.windows(k)
        .map(|w| w.iter().collect())
        .collect()
}

fn hash_shingle(shingle: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    shingle.hash(&mut hasher);
    hasher.finish()
}

fn minhash_similarity(sig1: &[u64], sig2: &[u64]) -> f64 {
    if sig1.len() != sig2.len() {
        return 0.0;
    }
    let matches = sig1.iter().zip(sig2.iter()).filter(|(a, b)| a == b).count();
    matches as f64 / sig1.len() as f64
}

fn minhash_to_string(sig: &[u64]) -> String {
    sig.iter().take(16).map(|x| x.to_string()).collect()
}

async fn load_holdout_minhashes(holdout_dir: &str) -> Result<Vec<(String, String)>> {
    let path = PathBuf::from(holdout_dir).join("cases.jsonl");
    let content = fs::read_to_string(&path).await?;
    let mut out = Vec::new();
    
    for line in content.lines() {
        let case: Value = serde_json::from_str(line)?;
        if let Some(text) = case.get("text").and_then(|t| t.as_str()) {
            if !text.is_empty() {
                let sig = minhash_signature(text);
                let mh = minhash_to_string(&sig);
                if let Some(case_id) = case.get("case_id").and_then(|c| c.as_str()) {
                    out.push((case_id.to_string(), mh));
                }
            }
        }
    }
    Ok(out)
}

#[derive(Clone)]
struct AppState;

async fn toolset_manifest() -> Json<Value> {
    Json(json!({
        "name": "minhash-dedup",
        "endpoint": format!("http://localhost:{}/invoke", std::env::var("PORT").unwrap_or("8003".to_string())),
        "tools": [{
            "name": "minhash_dedup",
            "description": "Deduplicate cases against holdout using MinHash",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "cases": {"type": "array", "items": {"type": "object"}},
                    "holdout_dir": {"type": "string"},
                    "threshold": {"type": "number", "default": 0.6, "minimum": 0.0, "maximum": 1.0}
                },
                "required": ["cases", "holdout_dir"]
            }
        }]
    }))
}

async fn invoke_tool(
    State(_): State<AppState>,
    Json(inv): Json<RapInvocation>,
) -> Result<StatusCode, StatusCode> {
    let args: DedupArgs = serde_json::from_value(inv.arguments)
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    
    let holdout_minhashes = match load_holdout_minhashes(&args.holdout_dir).await {
        Ok(m) => m,
        Err(e) => {
            warn!("Failed to load holdout: {}", e);
            vec![]
        }
    };
    
    let mut results = Vec::new();
    let mut novel = 0;
    let mut duplicate = 0;
    
    for (i, case) in args.cases.iter().enumerate() {
        let text = case.get("text").and_then(|t| t.as_str()).unwrap_or("");
        if text.len() < 50 {
            continue;
        }
        
        let case_id = case.get("case_id").and_then(|c| c.as_str())
            .unwrap_or(&format!("synth_{}", i)).to_string();
        
        let sig = minhash_signature(text);
        let mh = minhash_to_string(&sig);
        
        let mut is_dup = false;
        for (_hid, hmh) in &holdout_minhashes {
            // Compare string representations directly
            if *hmh == mh {
                is_dup = true;
                break;
            }
        }
        
        // Better comparison
        for (_hid, hmh) in &holdout_minhashes {
            let hmh_vec: Vec<u64> = hmh.chars().filter_map(|c| c.to_digit(10).map(|d| d as u64)).collect();
            if hmh_vec.len() >= 16 {
                let sig_vec: Vec<u64> = sig.iter().take(16).cloned().collect();
                let sim = minhash_similarity(&sig_vec, &hmh_vec);
                if sim >= args.threshold {
                    is_dup = true;
                    break;
                }
            }
        }
        
        if is_dup {
            duplicate += 1;
        } else {
            novel += 1;
        }
        
        results.push(CaseResult {
            case_id,
            text: text.to_string(),
            gold: case.get("gold").cloned().unwrap_or(json!({})),
            strata: case.get("strata").cloned().unwrap_or(json!({"quality": "medium", "layout": "mixed"})),
            minhash: mh,
            is_duplicate: is_dup,
        });
    }
    
    let response = DedupResponse {
        proposed: args.cases.len(),
        novel,
        duplicate,
        cases: results,
    };
    
    // Send callback
    let callback = RapCallback::ToolResult(RapToolResult {
        group_id: inv.group_id,
        id: inv.id,
        call_id: inv.call_id,
        text: Some(format!("Dedup: {} proposed, {} novel, {} duplicate", args.cases.len(), novel, duplicate)),
        content: Some(vec![RapToolResultContent::Text {
            text: serde_json::to_string(&response).unwrap_or_default(),
        }]),
        display_as: Some(vec![DisplaySegment::Text(format!(
            "MinHash dedup: {}/{} novel (threshold={})", novel, args.cases.len(), args.threshold
        ))]),
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
    info!("MinHash RAP server listening on http://127.0.0.1:{}", args.port);
    
    axum::serve(listener, app).await?;
    Ok(())
}