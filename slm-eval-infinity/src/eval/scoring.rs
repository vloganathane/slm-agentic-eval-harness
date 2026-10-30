//! Deterministic scoring: validators + element-level TP/FP/FN (SPEC_V3 §6.1, D-10).

pub use crate::models::Metric;
pub use crate::models::{FieldOutcome};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

pub const DEFAULT_NORMALIZE_OPS: &[&str] = &["trim", "collapse_ws"];

/// Normalize a string value by applying ops in order: trim, collapse_ws, casefold.
/// An empty `ops` slice means the default ops (`trim`, `collapse_ws`) — matching
/// the Python reference (`slm_eval.scoring.normalize`).
pub fn normalize(value: &Option<String>, ops: &[String]) -> String {
    let default_ops = ops.is_empty();
    let s = if let Some(v) = value { v.clone() } else { String::new() };
    let trimmed = s.trim().to_string();
    let mut result = trimmed;
    if default_ops || ops.iter().any(|o| o == "collapse_ws") {
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

fn parse_date(s: &str, formats: &[String]) -> Option<String> {
    let s = s.trim();
    for f in formats {
        let parsed = match f.as_str() {
            "%Y-%m-%d" => parse_day_first(s, '-', true),
            "%Y/%m/%d" => parse_day_first(s, '/', true),
            "%d/%m/%Y" => parse_day_first(s, '/', false),
            "%d-%m-%Y" => parse_day_first(s, '-', false),
            _ => None,
        };
        if parsed.is_some() {
            return parsed;
        }
    }
    None
}

/// Parse `Y-M-D` (year first) or `D/M/Y` (year last) into canonical `YYYY-MM-DD`.
fn parse_day_first(s: &str, sep: char, year_first: bool) -> Option<String> {
    let parts: Vec<&str> = s.split(sep).collect();
    if parts.len() != 3 {
        return None;
    }
    let (y, m, d) = if year_first {
        (
            parts[0].parse::<i32>().ok()?,
            parts[1].parse::<u32>().ok()?,
            parts[2].parse::<u32>().ok()?,
        )
    } else {
        (
            parts[2].parse::<i32>().ok()?,
            parts[1].parse::<u32>().ok()?,
            parts[0].parse::<u32>().ok()?,
        )
    };
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    Some(format!("{y:04}-{m:02}-{d:02}"))
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn opt_str(v: &str) -> Option<Value> {
        Some(Value::String(v.to_string()))
    }

    fn opt_s(v: &str) -> Option<String> {
        Some(v.to_string())
    }

    #[test]
    fn normalize_collapse_ws_and_casefold() {
        assert_eq!(normalize(&opt_s("  Hello   World  "), &[]), "Hello World");
        let ops = vec!["trim".to_string(), "collapse_ws".to_string(), "casefold".to_string()];
        assert_eq!(normalize(&opt_s("  Hi There "), &ops), "hi there");
    }

    #[test]
    fn numeric_tolerance() {
        let v = Validator::Numeric { tolerance: 0.01 };
        assert_eq!(score_field(&opt_str("100.005"), &opt_str("100.0"), &Some(v.clone())), (1, 0, 0));
        assert_eq!(score_field(&opt_str("101.0"), &opt_str("100.0"), &Some(v)), (0, 1, 1));
    }

    #[test]
    fn regex_requires_format_and_value() {
        let v = Validator::Regex { pattern: r"^\d{3}$".to_string() };
        assert_eq!(score_field(&opt_str("123"), &opt_str("123"), &Some(v.clone())), (1, 0, 0));
        assert_eq!(score_field(&opt_str("12"), &opt_str("12"), &Some(v.clone())), (0, 1, 1));
        assert_eq!(score_field(&opt_str("999"), &opt_str("123"), &Some(v)), (0, 1, 1));
    }

    #[test]
    fn date_validator_normalizes_formats() {
        let v = Validator::Date {
            formats: vec!["%Y-%m-%d".to_string(), "%d/%m/%Y".to_string()],
        };
        assert_eq!(score_field(&opt_str("05/03/2024"), &opt_str("2024-03-05"), &Some(v.clone())), (1, 0, 0));
        assert_eq!(score_field(&opt_str("2024-03-06"), &opt_str("2024-03-05"), &Some(v)), (0, 1, 1));
    }

    #[test]
    fn string_normalize_trims_and_casefolds() {
        let v = Validator::String {
            normalize: vec!["trim".to_string(), "collapse_ws".to_string(), "casefold".to_string()],
        };
        assert_eq!(score_field(&opt_str(" INV  001 "), &opt_str("inv 001"), &Some(v)), (1, 0, 0));
    }

    #[test]
    fn list_item_matching_counts_elements() {
        let gold = json!([{"sku": "A", "qty": 1}, {"sku": "B", "qty": 2}]);
        let pred = json!([{"sku": "A", "qty": 1}, {"sku": "C", "qty": 9}]);
        assert_eq!(score_field(&Some(pred), &Some(gold), &Some(Validator::List)), (1, 1, 1));
    }

    #[test]
    fn list_exact_match_and_empty_prediction() {
        let gold = json!([{"sku": "A", "qty": 1}]);
        assert_eq!(score_field(&Some(gold.clone()), &Some(gold.clone()), &Some(Validator::List)), (1, 0, 0));
        assert_eq!(score_field(&Some(json!([])), &Some(gold), &Some(Validator::List)), (0, 0, 1));
    }

    #[test]
    fn scalar_wrong_value_is_fp_and_fn() {
        assert_eq!(score_field(&opt_str("INV-2"), &opt_str("INV-1"), &None), (0, 1, 1));
    }

    #[test]
    fn micro_f1_is_element_level() {
        let outcomes = vec![
            FieldOutcome::new("a".into(), 1, 0, 0),
            FieldOutcome::new("b".into(), 0, 1, 1),
        ];
        assert!((micro_f1(&outcomes) - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(micro_f1(&[]), 0.0);
    }

    #[test]
    fn bootstrap_is_deterministic_for_a_seed() {
        let outcomes = vec![
            FieldOutcome::new("a".into(), 1, 0, 0),
            FieldOutcome::new("b".into(), 0, 1, 1),
            FieldOutcome::new("c".into(), 1, 0, 0),
        ];
        let m1 = bootstrap_micro_f1(&outcomes, 500, 42, 0.05);
        let m2 = bootstrap_micro_f1(&outcomes, 500, 42, 0.05);
        assert_eq!(m1.ci_lower, m2.ci_lower);
        assert_eq!(m1.ci_upper, m2.ci_upper);
        assert!(m1.ci_lower <= m1.point && m1.point <= m1.ci_upper);
    }
}
