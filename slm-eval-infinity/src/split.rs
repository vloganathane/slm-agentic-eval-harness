//! Stratified train/holdout split (ported from slm_eval/split.py).

use serde_json::Value;

/// Compute strata key for a case from the strata JSON.
pub fn strata_key(strata: &Option<Value>, keys: &[String]) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(v) = strata {
        if let Some(obj) = v.as_object() {
            for k in keys {
                if let Some(val) = obj.get(k) {
                    parts.push(format!("{}={}", k, val.as_str().unwrap_or(&format!("{}", val))));
                }
            }
        }
    }
    parts.join(";")
}

/// Stratified split: deterministic hash-based assignment.
pub fn stratified_split(
    case_ids: &[String],
    _strata_keys: &[String],
    holdout_fraction: f64,
    seed: u64,
) -> (Vec<String>, Vec<String>) {
    // Deterministic pseudo-random assignment via hash of case_id + seed
    let mut holdout = Vec::new();
    let mut train = Vec::new();
    for id in case_ids {
        let hasher = fnv1a(id.as_bytes(), seed);
        let r = (hasher >> 33) as f64 / ((hasher & 0xFFFFFFFF) as f64 + 1.0);
        if r < holdout_fraction {
            holdout.push(id.clone());
        } else {
            train.push(id.clone());
        }
    }
    (train, holdout)
}

fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut h: u64 = seed;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
