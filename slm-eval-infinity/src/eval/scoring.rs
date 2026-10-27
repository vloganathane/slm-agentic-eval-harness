//! Scoring logic: micro-F1, bootstrap CI, Cohen's κ

use anyhow::Result;
use rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet};

use crate::models::{Case, Prediction, AggregateScores, FieldScore, StrataAggregate, CalibrationResult};

pub fn score_predictions(cases: &[Case], predictions: &[Prediction], bootstrap_resamples: usize, ci_level: f64) -> Result<AggregateScores> {
    // Build lookups
    let case_map: HashMap<String, &Case> = cases.iter().map(|c| (c.case_id.clone(), c)).collect();
    let pred_map: HashMap<String, &Prediction> = predictions.iter().map(|p| (p.case_id.clone(), p)).collect();

    let mut all_field_counts: HashMap<String, (usize, usize, usize)> = HashMap::new(); // field -> (tp, fp, fn)
    let mut strata_counts: HashMap<String, Vec<(usize, usize, usize)>> = HashMap::new(); // strata -> vec of (tp, fp, fn)
    let mut dataset_defect = 0;
    let mut evaluated = 0;

    for (case_id, case) in &case_map {
        let pred = match pred_map.get(case_id) {
            Some(p) => p,
            None => continue, // no prediction for this case
        };

        // Check quarantine
        if case.ocr_signal.as_deref() == Some("unextractable") {
            dataset_defect += 1;
            continue;
        }

        let gold = &case.gold;
        let pred_gold = &pred.prediction;

        // Compare all fields
        let mut all_keys: HashSet<String> = HashSet::new();
        for k in gold.extra.keys() { all_keys.insert(k.clone()); }
        for k in pred_gold.extra.keys() { all_keys.insert(k.clone()); }
        for k in ["merchant", "date", "total", "subtotal", "tax"] { all_keys.insert(k.to_string()); }

        let strata_key = format!("{}:{}", case.strata.quality, case.strata.layout);
        let mut strata_tpfpfn = Vec::new();

        for key in all_keys {
            let gold_val = get_gold_value(gold, &key);
            let pred_val = get_pred_value(pred_gold, &key);

            // Skip empty gold (dataset defect)
            if gold_val.is_none() || gold_val.as_ref().map_or(true, |s| s.is_empty()) {
                dataset_defect += 1;
                continue;
            }

            let entry = all_field_counts.entry(key.clone()).or_insert((0, 0, 0));
            let (tp, fp, fn_) = compare_values(&gold_val, &pred_val);
            entry.0 += tp;
            entry.1 += fp;
            entry.2 += fn_;
            strata_tpfpfn.push((tp, fp, fn_));
        }

        if !strata_tpfpfn.is_empty() {
            strata_counts.entry(strata_key).or_default().extend(strata_tpfpfn);
            evaluated += 1;
        }
    }

    // Also score items (nested array comparison)
    // Simplified: treat each item as a field
    for (case_id, case) in &case_map {
        let pred = match pred_map.get(case_id) {
            Some(p) => p,
            None => continue,
        };

        let gold_items = &case.gold.items;
        let pred_items = &pred.prediction.items;

        // Simple matching by name (position-independent)
        for gitem in gold_items {
            let mut matched = false;
            for pitem in pred_items {
                if pitem.name == gitem.name {
                    // Compare price
                    let (tp, fp, fn_) = compare_values(&Some(gitem.price.clone()), &Some(pitem.price.clone()));
                    let entry = all_field_counts.entry("items.price".to_string()).or_insert((0, 0, 0));
                    entry.0 += tp; entry.1 += fp; entry.2 += fn_;
                    
                    // Compare qty if present
                    if gitem.qty.is_some() && pitem.qty.is_some() {
                        let (tp, fp, fn_) = compare_values(
                            &gitem.qty.map(|v| v.to_string()),
                            &pitem.qty.map(|v| v.to_string()),
                        );
                        let entry = all_field_counts.entry("items.qty".to_string()).or_insert((0, 0, 0));
                        entry.0 += tp; entry.1 += fp; entry.2 += fn_;
                    }
                    matched = true;
                    break;
                }
            }
            if !matched {
                // FN - gold item not predicted
                let entry = all_field_counts.entry("items.price".to_string()).or_insert((0, 0, 0));
                entry.2 += 1;
            }
        }

        // FP - predicted items not in gold
        for pitem in pred_items {
            if !gold_items.iter().any(|g| g.name == pitem.name) {
                let entry = all_field_counts.entry("items.price".to_string()).or_insert((0, 0, 0));
                entry.1 += 1;
            }
        }
    }

    // Compute per-field F1
    let mut per_field = HashMap::new();
    let mut total_tp = 0;
    let mut total_fp = 0;
    let mut total_fn = 0;

    for (field, (tp, fp, fn_)) in &all_field_counts {
        let precision = if tp + fp > 0 { *tp as f64 / (*tp + *fp) as f64 } else { 0.0 };
        let recall = if tp + fn_ > 0 { *tp as f64 / (*tp + *fn_) as f64 } else { 0.0 };
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        per_field.insert(field.clone(), FieldScore {
            tp: *tp, fp: *fp, fn_: *fn_, precision, recall, f1
        });
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

    // Bootstrap CI
    let (ci_lower, ci_upper) = bootstrap_ci(cases, predictions, bootstrap_resamples, ci_level)?;

    // Per-strata
    let mut per_strata = HashMap::new();
    for (strata_key, scores) in strata_counts {
        let (tp, fp, fn_) = scores.iter().fold((0,0,0), |a, b| (a.0+b.0, a.1+b.1, a.2+b.2));
        let f1 = if tp + fp + fn_ > 0 {
            let p = tp as f64 / (tp + fp).max(1) as f64;
            let r = tp as f64 / (tp + fn_).max(1) as f64;
            if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 }
        } else { 0.0 };
        
        let n = scores.len();
        let se = if n > 0 { (f1 * (1.0 - f1) / n as f64).sqrt() } else { 0.0 };
        let z = 1.96; // 95%
        let parts: Vec<_> = strata_key.split(':').collect();
        let quality = parts.get(0).copied().unwrap_or("medium").to_string();
        
        per_strata.insert(strata_key, StrataAggregate {
            n, micro_f1: f1,
            ci_lower: (f1 - z * se).max(0.0),
            ci_upper: (f1 + z * se).min(1.0),
            quality,
        });
    }

    Ok(AggregateScores {
        micro_f1,
        ci_lower,
        ci_upper,
        per_field,
        per_strata,
        dataset_defect,
        total_cases: cases.len(),
        evaluated_cases: evaluated,
    })
}

fn get_gold_value(gold: &crate::models::GoldStandard, key: &str) -> Option<String> {
    match key {
        "merchant" => gold.merchant.clone(),
        "date" => gold.date.clone(),
        "total" => gold.total.clone(),
        "subtotal" => gold.subtotal.clone(),
        "tax" => gold.tax.clone(),
        _ => gold.extra.get(key).and_then(|v| v.as_str().map(|s| s.to_string())),
    }
}

fn get_pred_value(gold: &crate::models::GoldStandard, key: &str) -> Option<String> {
    get_gold_value(gold, key) // same structure
}

fn compare_values(gold: &Option<String>, pred: &Option<String>) -> (usize, usize, usize) {
    match (gold, pred) {
        (Some(g), Some(p)) if g == p => (1, 0, 0), // TP
        (Some(_), Some(_)) => (0, 1, 0), // FP
        (Some(_), None) => (0, 0, 1), // FN
        (None, Some(_)) => (0, 1, 0), // FP
        (None, None) => (0, 0, 0), // TN (ignored)
    }
}

fn bootstrap_ci(cases: &[Case], predictions: &[Prediction], n_resamples: usize, ci_level: f64) -> Result<(f64, f64)> {
    if cases.is_empty() {
        return Ok((0.0, 0.0));
    }

    let case_ids: Vec<_> = cases.iter().map(|c| c.case_id.clone()).collect();
    let mut rng = rand::thread_rng();
    let mut f1_scores = Vec::with_capacity(n_resamples);

    for _ in 0..n_resamples {
        // Resample with replacement
        let sample: Vec<_> = (0..case_ids.len())
            .map(|_| case_ids.choose(&mut rng).unwrap())
            .collect();
        
        // Build sampled cases/predictions
        let sampled_cases: Vec<_> = sample.iter().filter_map(|id| cases.iter().find(|c| &c.case_id == *id)).cloned().collect();
        let sampled_preds: Vec<_> = sample.iter().filter_map(|id| predictions.iter().find(|p| &p.case_id == *id)).cloned().collect();
        
        let result = score_predictions(&sampled_cases, &sampled_preds, 0, 0.0)?; // no nested bootstrap
        f1_scores.push(result.micro_f1);
    }

    f1_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let alpha = (1.0 - ci_level) / 2.0;
    let lower_idx = (alpha * n_resamples as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha) * n_resamples as f64).ceil() as usize;
    
    Ok((
        f1_scores[lower_idx.min(f1_scores.len() - 1)],
        f1_scores[upper_idx.min(f1_scores.len() - 1)],
    ))
}

/// Cohen's κ for judge calibration
pub fn cohens_kappa(agreements: &[(bool, bool)], n_classes: usize) -> CalibrationResult {
    let n = agreements.len();
    if n == 0 {
        return CalibrationResult {
            kappa: 0.0, ci_lower: 0.0, ci_upper: 0.0,
            n_pairs: 0, n_agree: 0, n_disagree: 0,
        };
    }

    let n_agree = agreements.iter().filter(|(a, b)| a == b).count();
    let n_disagree = n - n_agree;
    let po = n_agree as f64 / n as f64;

    // Expected agreement by chance (assuming balanced)
    let pe = 1.0 / n_classes as f64;
    
    let kappa = if 1.0 - pe > 0.0 { (po - pe) / (1.0 - pe) } else { 0.0 };

    // CI using Fleiss-Cohen approximation
    let se = ((po * (1.0 - po)) / (n as f64 * (1.0 - pe).powi(2))).sqrt();
    let z = 1.96; // 95%
    let ci_lower = (kappa - z * se).max(-1.0);
    let ci_upper = (kappa + z * se).min(1.0);

    CalibrationResult {
        kappa,
        ci_lower,
        ci_upper,
        n_pairs: n,
        n_agree,
        n_disagree,
    }
}