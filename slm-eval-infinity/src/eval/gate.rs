//! Gate decision (SPEC_V3 §6.5): pure, CI-lower-bound-only, no point estimates.

use crate::config::ThresholdsConfig;
use crate::models::{GateDecision, StratumResult, StratumStatus};

#[derive(Debug, Clone)]
pub struct GateInputs {
    pub run_completed: bool,
    pub overall_micro_f1: f64,
    pub overall_ci_lower: f64,
    pub overall_ci_upper: f64,
    pub overall_n: usize,
    pub strata: Vec<StratumResult>,
    pub kappa: Option<f64>,
    pub pending_reviews: usize,
    pub subjective_fields: Vec<String>,
    pub defect_rate: Option<f64>,
    pub defect_n: usize,
    pub total_n: usize,
}

pub fn classify_strata(strata: &[StratumResult], thresholds: &ThresholdsConfig) -> Vec<StratumResult> {
    let mut out = Vec::new();
    for s in strata {
        let status = if s.n < thresholds.min_per_stratum as usize {
            StratumStatus::InsufficientN
        } else if s.ci_upper <= thresholds.failure_threshold {
            StratumStatus::Blocker
        } else if s.micro_f1 < thresholds.target_f1 {
            StratumStatus::Warning
        } else {
            StratumStatus::Ok
        };
        out.push(StratumResult {
            key: s.key.clone(),
            n: s.n,
            micro_f1: s.micro_f1,
            ci_lower: s.ci_lower,
            ci_upper: s.ci_upper,
            status,
        });
    }
    out
}

pub fn dataset_defect_rate(signal_values: &[String]) -> (f64, usize) {
    if signal_values.is_empty() { return (0.0, 0); }
    let defect_n = signal_values.iter().filter(|v| v.as_str() == "unextractable").count();
    (defect_n as f64 / signal_values.len() as f64, defect_n)
}

pub fn decide_gate(inputs: &GateInputs, thresholds: &ThresholdsConfig) -> GateDecision {
    if !inputs.run_completed {
        return GateDecision { status: "error".into(), promote: false, reason: "run_incomplete".into(), blockers: vec![], warnings: vec![] };
    }
    let mut blockers: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    let micro_f1 = inputs.overall_micro_f1;
    let ci_lower = inputs.overall_ci_lower;
    let _ci_upper = inputs.overall_ci_upper;
    if ci_lower < thresholds.target_f1 {
        blockers.push(format!("ci_lower_below_target: {:.3} < {} (point {:.3} cannot gate)", ci_lower, thresholds.target_f1, micro_f1));
    }

    let classified = classify_strata(&inputs.strata, thresholds);
    for s in classified {
        if s.status == StratumStatus::Blocker {
            blockers.push(format!("stratum_blocker: {} n={} ci_upper={:.3}", s.key, s.n, s.ci_upper));
        } else if s.status == StratumStatus::Warning {
            warnings.push(format!("stratum_below_target: {} micro_f1={:.3}", s.key, s.micro_f1));
        } else if s.status == StratumStatus::InsufficientN {
            warnings.push(format!("insufficient_n: {} n={} < {}", s.key, s.n, thresholds.min_per_stratum));
        }
    }

    if let Some(dr) = inputs.defect_rate {
        if dr > 0.0 {
            warnings.push(format!("dataset_defect: {}/{} cases ({:.2}) unextractable — excluded from micro-F1", inputs.defect_n, inputs.total_n, dr));
        }
    }

    if !inputs.subjective_fields.is_empty() {
        match inputs.kappa {
            None => blockers.push("kappa_missing: subjective fields present but no calibration".to_string()),
            Some(k) => { if k < 0.85 { blockers.push(format!("kappa_below_target: {:.3} < 0.85", k)); } }
        }
    }

    if inputs.pending_reviews > 0 {
        warnings.push(format!("pending_reviews: {}", inputs.pending_reviews));
        return GateDecision { status: "blocked_pending_review".into(), promote: false, reason: "pending_human_review".into(), blockers, warnings };
    }

    if !blockers.is_empty() {
        return GateDecision { status: "blocked".into(), promote: false, reason: blockers[0].split(':').next().unwrap_or("blocked").to_string(), blockers, warnings };
    }

    GateDecision { status: "promote".into(), promote: true, reason: "all_checks_passed".into(), blockers, warnings }
}
