//! Deterministic scoring: validators + element-level TP/FP/FN (SPEC_V3 §6.1, D-10).

pub use crate::models::Metric;
pub use crate::models::{FieldOutcome};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

pub const DEFAULT_NORMALIZE_OPS: &[&str] = &["trim", "collapse_ws"];

/// Normalize a string value by applying ops in order: trim, collapse_ws, casefold.
pub fn normalize(value: &Option<String>, ops: &[String]) -> String {
    let s = if let Some(v) = value { v.clone() } else { String::new() };
    let trimmed = s.trim().to_string();
    let mut result = trimmed;
    if ops.iter().any(|o| o == "collapse_ws") {
        let mut out = String::new();
        let mut prev_space = false;
        for c in result.chars() {
            if c.is_whitespace() {
                if !prev_space && !out.is_empty() { out.push(' '); }
                prev_space = true;
            } else { out.push(c); prev_space = false; }
        }
        result = out.trim().to_string();
    }
    let mut final_result = result;
    if ops.iter().any(|o| o == "casefold") {
        final_result = final_result.to_lowercase();
    }
    final_result
}

/// Parse a numeric value, tolerating currency formatting / separators.
pub fn to_float(value: &Option<String>) -> Option<f64> {
    let s = normalize(value, &[]);
    if s.is_empty() { return None; }
    let cleaned: String = s.chars().filter(|c| !",₹$€£ \t\r\n\u{a0}".contains(*c)).collect();
    let cleaned = cleaned.trim().to_string();
    if cleaned.starts_with('(') && cleaned.ends_with(')') {
        let inner = cleaned[1..cleaned.len() - 1].trim().to_string();
        let v = to_float(&Some(inner));
        return v.map(|f| -f);
    }
    cleaned.parse::<f64>().ok()
}

#[derive(Debug, Clone)]
pub enum Validator {
    Regex { pattern: String },
    Numeric { tolerance: f64 },
    Date { formats: Vec<String> },
    String { normalize: Vec<String> },
    List,
}

impl Validator {
    pub fn is_list(&self) -> bool { matches!(self, Validator::List) }
}

fn regex_full_match(pattern: &str, s: &str) -> bool {
    match regex::Regex::new(&format!("^{}$", pattern)) {
        Ok(re) => re.is_match(s),
        Err(_) => false,
    }
}

fn parse_date(_s: &str, formats: &[String]) -> Option<String> {
    for f in formats {
        let parts: Vec<&str> = f.split('-').collect();
        if parts.len() == 3 {
            if let (Ok(y), Ok(m), Ok(d)) = (parts[0].parse::<i32>(), parts[1].parse::<i32>(), parts[2].parse::<i32>()) {
                if m >= 1 && m <= 12 && d >= 1 && d <= 31 {
                    return Some(format!("{}-{:02}-{:02}", y, m, d));
                }
            }
        }
    }
    None
}

fn item_key(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let mut keys: Vec<String> = map.keys().cloned().collect();
            keys.sort();
            let mut parts = Vec::new();
            for k in &keys {
                if let Some(val) = map.get(k) {
                    parts.push(format!("{}={}", k, val.as_str().unwrap_or(&format!("{}", val))));
                }
            }
            parts.join("|")
        }
        _ => v.as_str().map(|s| s.to_string()).unwrap_or_else(|| format!("{}", v)),
    }
}

/// Score one field. Returns (tp, fp, fn).
pub fn score_field(
    pred: &Option<Value>,
    gold: &Option<Value>,
    validator: &Option<Validator>,
) -> (u32, u32, u32) {
    let gold_list = gold.as_ref().map(|g| g.is_array()).unwrap_or(false);
    if validator.as_ref().map(|v| v.is_list()).unwrap_or(false) || gold_list {
        return list_outcome(pred, gold);
    }
    let pred_s = pred.as_ref().and_then(|v| v.as_str()).map(|s| s.to_string());
    let gold_s = gold.as_ref().and_then(|v| v.as_str()).map(|s| s.to_string());
    match validator {
        Some(Validator::Regex { pattern }) => {
            if let (Some(p), Some(g)) = (&pred_s, &gold_s) {
                if p == g && regex_full_match(pattern, &p) {
                    return (1, 0, 0);
                }
            }
            return (0, 1, 1);
        }
        Some(Validator::Numeric { tolerance }) => {
            if let (Some(p), Some(g)) = (to_float(&pred_s), to_float(&gold_s)) {
                if (p - g).abs() <= *tolerance { return (1, 0, 0); }
            }
            return (0, 1, 1);
        }
        Some(Validator::Date { formats }) => {
            if let (Some(p), Some(g)) = (pred_s.as_deref().and_then(|s| parse_date(s, &formats)), gold_s.as_deref().and_then(|s| parse_date(s, &formats))) {
                if p == g { return (1, 0, 0); }
            }
            return (0, 1, 1);
        }
        Some(Validator::String { normalize: ops }) => {
            let p = pred_s.as_deref().map(|s| normalize(&Some(s.to_string()), ops));
            let g = gold_s.as_deref().map(|s| normalize(&Some(s.to_string()), ops));
            if p == g { return (1, 0, 0); }
            return (0, 1, 1);
        }
        _ => {
            let p = pred_s.as_deref().map(|s| normalize(&Some(s.to_string()), &[]));
            let g = gold_s.as_deref().map(|s| normalize(&Some(s.to_string()), &[]));
            if p == g { return (1, 0, 0); }
            return (0, 1, 1);
        }
    }
}

fn list_outcome(pred: &Option<Value>, gold: &Option<Value>) -> (u32, u32, u32) {
    let pred_items: Vec<String> = match pred {
        Some(Value::Array(arr)) => arr.iter().map(|v| item_key(v)).collect(),
        Some(v) => vec![item_key(v)],
        None => vec![],
    };
    let gold_items: Vec<String> = match gold {
        Some(Value::Array(arr)) => arr.iter().map(|v| item_key(v)).collect(),
        Some(v) => vec![item_key(v)],
        None => vec![],
    };
    let mut remaining = gold_items.clone();
    let mut tp = 0u32;
    for p in &pred_items {
        if let Some(pos) = remaining.iter().position(|x| *x == *p) {
            remaining.remove(pos);
            tp += 1;
        }
    }
    let fp = (pred_items.len() - tp as usize) as u32;
    let fn_ = (remaining.len()) as u32;
    (tp, fp, fn_)
}

/// Score one case across configured exact-match fields. Returns outcomes.
pub fn score_case(
    case_id: &str,
    prediction: &Option<Value>,
    gold: &Option<Value>,
    exact_fields: &[String],
    subjective_fields: &[String],
    judge_labels: &Option<HashMap<String, bool>>,
) -> Vec<FieldOutcome> {
    let mut outcomes = Vec::new();
    let mut fields = BTreeMap::new();
    if let Some(p) = prediction {
        if let Some(obj) = p.as_object() {
            for (k, v) in obj {
                fields.insert(k.clone(), v.clone());
            }
        }
    }
    for fname in exact_fields {
        let pred = fields.get(fname).cloned();
        let gold_v = gold.as_ref().and_then(|g| g.get(fname).cloned());
        let validator = None;
        let (tp, fp, fn_) = score_field(&pred, &gold_v, &validator);
        outcomes.push(FieldOutcome::new(case_id.to_string(), tp, fp, fn_));
    }
    for fname in subjective_fields {
        let ok = judge_labels.as_ref().and_then(|m| m.get(fname)).copied().unwrap_or(false);
        let (tp, fp, fn_) = if ok { (1, 0, 0) } else { (0, 1, 1) };
        outcomes.push(FieldOutcome::new(case_id.to_string(), tp, fp, fn_));
    }
    outcomes
}

/// Element-level micro-F1: ΣTP / (ΣTP + ΣFP + ΣFN). Undefined denominators score 0.0.
pub fn micro_f1(outcomes: &[FieldOutcome]) -> f64 {
    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut fn_ = 0u64;
    for o in outcomes {
        tp += o.tp as u64;
        fp += o.fp as u64;
        fn_ += o.fn_ as u64;
    }
    let denom = tp + fp + fn_;
    if denom == 0 { return 0.0; }
    tp as f64 / denom as f64
}

/// Deterministic PRNG (LCG) for bootstrap resampling.
struct Lcg { state: u64 }
impl Lcg {
    fn new(seed: u64) -> Self { Self { state: seed.max(1) } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state >> 33
    }
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n.max(1)
    }
}

/// Percentile bootstrap over cases (SPEC_V3 §6.5).
pub fn bootstrap_micro_f1(outcomes: &[FieldOutcome], resamples: usize, seed: u64, alpha: f64) -> Metric {
    if outcomes.is_empty() {
        return Metric { point: 0.0, ci_lower: 0.0, ci_upper: 0.0, n: 0, resamples };
    }
    let mut by_case: BTreeMap<String, (u64, u64, u64)> = BTreeMap::new();
    for o in outcomes {
        let e = by_case.entry(o.case_id.clone()).or_insert((0, 0, 0));
        e.0 += o.tp as u64;
        e.1 += o.fp as u64;
        e.2 += o.fn_ as u64;
    }
    let case_ids: Vec<String> = by_case.keys().cloned().collect();
    let n = case_ids.len();
    let point = micro_f1(outcomes);
    let mut rng = Lcg::new(seed);
    let mut samples: Vec<f64> = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let idx: Vec<usize> = (0..n).map(|_| rng.next_usize(n)).collect();
        let mut tp_s = 0u64; let mut fp_s = 0u64; let mut fn_s = 0u64;
        for &i in &idx { let e = by_case.get(&case_ids[i]).copied().unwrap_or((0,0,0)); tp_s += e.0; fp_s += e.1; fn_s += e.2; }
        let denom = tp_s + fp_s + fn_s;
        let v = if denom > 0 { tp_s as f64 / denom as f64 } else { 0.0 };
        samples.push(v);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut lo = 0.0; let mut hi = 1.0;
    if !samples.is_empty() {
        let pos_lo = ((samples.len() as f64) * alpha / 2.0).floor() as usize;
        let pos_hi = ((samples.len() as f64) * (1.0 - alpha / 2.0)).ceil() as usize;
        lo = samples[pos_lo.min(samples.len() - 1)];
        hi = samples[pos_hi.min(samples.len() - 1)];
    }
    Metric { point, ci_lower: lo, ci_upper: hi, n, resamples }
}

/// Cohen's κ between two label sequences (D-04).
pub fn cohen_kappa(a: &[String], b: &[String]) -> f64 {
    if a.len() != b.len() { return f64::NAN; }
    let n = a.len();
    if n == 0 { return f64::NAN; }
    let mut labels: Vec<String> = a.to_vec();
    labels.extend_from_slice(b);
    labels.sort();
    labels.dedup();
    let k = labels.len();
    if k < 2 { return f64::NAN; }
    let idx: HashMap<&str, usize> = labels.iter().enumerate().map(|(i, l)| (l.as_str(), i)).collect();
    let mut confusion = vec![vec![0i64; k]; k];
    for i in 0..n {
        let x = *idx.get(a[i].as_str()).unwrap_or(&0);
        let y = *idx.get(b[i].as_str()).unwrap_or(&0);
        confusion[x][y] += 1;
    }
    let p_o: f64 = (0..k).map(|i| (0..k).map(|j| confusion[i][j] as f64).sum::<f64>()).sum::<f64>() / n as f64;
    let mut row_marg = vec![0.0f64; k];
    let mut col_marg = vec![0.0f64; k];
    for i in 0..k { row_marg[i] = (0..k).map(|j| confusion[i][j] as f64).sum::<f64>() / n as f64; }
    for j in 0..k { col_marg[j] = (0..k).map(|i| confusion[i][j] as f64).sum::<f64>() / n as f64; }
    let p_e: f64 = row_marg.iter().zip(col_marg.iter()).map(|(r, c)| r * c).sum::<f64>();
    if p_e == 1.0 { return f64::NAN; }
    (p_o - p_e) / (1.0 - p_e)
}

/// Percentile bootstrap CI on κ (§6.4).
pub fn kappa_ci(a: &[String], b: &[String], resamples: usize, seed: u64, alpha: f64) -> (f64, f64) {
    if a.len() != b.len() || a.is_empty() { return (f64::NAN, f64::NAN); }
    let n = a.len();
    let mut vals: Vec<f64> = Vec::new();
    let mut rng = Lcg::new(seed);
    for _ in 0..resamples {
        let idx: Vec<usize> = (0..n).map(|_| rng.next_usize(n)).collect();
        let aa: Vec<String> = idx.iter().map(|&i| a[i].clone()).collect();
        let bb: Vec<String> = idx.iter().map(|&i| b[i].clone()).collect();
        let k = cohen_kappa(&aa, &bb);
        if !k.is_nan() { vals.push(k); }
    }
    if vals.is_empty() { return (f64::NAN, f64::NAN); }
    vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let pos_lo = ((vals.len() as f64) * alpha / 2.0).floor() as usize;
    let pos_hi = ((vals.len() as f64) * (1.0 - alpha / 2.0)).ceil() as usize;
    (vals[pos_lo.min(vals.len() - 1)], vals[pos_hi.min(vals.len() - 1)])
}
