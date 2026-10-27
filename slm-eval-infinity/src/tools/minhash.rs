//! MinHash Deduplication RAP Tool

use infinity_agent_core::tools::rap_tool::{RapTool, RapToolDescriptor};
use rap_client::http::SimpleHttpClient;
use serde_json::json;

const MINHASH_NUM_HASHES: usize = 128;
const MINHASH_SEED: u64 = 0x9e3779b97f4a7c15;

fn minhash_signature(text: &str) -> Vec<u64> {
    let shingles = get_shingles(text, 5);
    let mut signatures = vec![u64::MAX; MINHASH_NUM_HASHES];
    for shingle in shingles {
        let hash = hash_shingle(&shingle);
        for i in 0..MINHASH_NUM_HASHES {
            let permuted = hash.wrapping_mul(MINHASH_SEED.wrapping_add(i as u64));
            if permuted < signatures[i] { signatures[i] = permuted; }
        }
    }
    signatures
}

fn get_shingles(text: &str, k: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < k { return vec![text.to_string()]; }
    chars.windows(k).map(|w| w.iter().collect()).collect()
}

fn hash_shingle(shingle: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    shingle.hash(&mut hasher);
    hasher.finish()
}

fn minhash_similarity(sig1: &[u64], sig2: &[u64]) -> f64 {
    if sig1.len() != sig2.len() { return 0.0; }
    let matches = sig1.iter().zip(sig2.iter()).filter(|(a, b)| a == b).count();
    matches as f64 / sig1.len() as f64
}

fn minhash_to_string(sig: &[u64]) -> String {
    sig.iter().take(16).map(|x| x.to_string()).collect()
}

pub fn make_minhash_tool() -> RapTool<SimpleHttpClient> {
    let descriptor = RapToolDescriptor {
        name: "minhash_dedup".to_string(),
        description: "Deduplicate cases against holdout using MinHash".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "cases": {"type": "array", "items": {"type": "object"}},
                "holdout_dir": {"type": "string"},
                "threshold": {"type": "number", "default": 0.6}
            },
            "required": ["cases", "holdout_dir"]
        }),
        display_script: None,
    };

    RapTool {
        descriptor,
        endpoint: "http://localhost:8003/invoke".to_string(),
        http_client: SimpleHttpClient::new(),
        callback_url: None,
    }
}

// Local functions for use without RAP
pub async fn load_holdout_minhashes(holdout_dir: &str) -> anyhow::Result<Vec<(String, Vec<u64>)>> {
    use tokio::fs;
    let path = std::path::PathBuf::from(holdout_dir).join("cases.jsonl");
    let content = fs::read_to_string(&path).await?;
    let mut out = Vec::new();
    for line in content.lines() {
        let case: serde_json::Value = serde_json::from_str(line)?;
        if let Some(text) = case.get("text").and_then(|t| t.as_str()) {
            if !text.is_empty() {
                let sig = minhash_signature(text);
                if let Some(case_id) = case.get("case_id").and_then(|c| c.as_str()) {
                    out.push((case_id.to_string(), sig));
                }
            }
        }
    }
    Ok(out)
}

pub fn dedup_cases(
    cases: &[serde_json::Value],
    holdout_minhashes: &[(String, Vec<u64>)],
    threshold: f64,
) -> Vec<(serde_json::Value, bool, String)> { // (case, is_duplicate, minhash)
    let mut results = Vec::new();
    for case in cases {
        let text = case.get("text").and_then(|t| t.as_str()).unwrap_or("");
        if text.len() < 50 {
            results.push((case.clone(), false, String::new()));
            continue;
        }
        let sig = minhash_signature(text);
        let mh = minhash_to_string(&sig);
        
        let mut is_dup = false;
        for (_hid, hmh) in holdout_minhashes {
            let sim = minhash_similarity(&sig, hmh);
            if sim >= threshold {
                is_dup = true;
                break;
            }
        }
        results.push((case.clone(), is_dup, mh));
    }
    results
}