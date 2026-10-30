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
            warnings.push(format!("dataset_defect: {}/{} cases ({:.0}%) unextractable — excluded from micro-F1", inputs.defect_n, inputs.total_n, dr * 100.0));
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// The same fixture is exercised by `tests/test_gate_parity.py`; Python is
    /// the reference implementation, so any drift in this port fails here.
    fn parity_fixture() -> Value {
        serde_json::from_str(include_str!("../../../tests/fixtures/gate_cases.json"))
            .expect("gate parity fixture must be valid JSON")
    }

    fn strings(v: &Value) -> Vec<String> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_string())
            .collect()
    }

    fn stratum(v: &Value) -> StratumResult {
        StratumResult {
            key: v["key"].as_str().unwrap().to_string(),
            n: v["n"].as_u64().unwrap() as usize,
            micro_f1: v["micro_f1"].as_f64().unwrap(),
            ci_lower: v["ci_lower"].as_f64().unwrap(),
            ci_upper: v["ci_upper"].as_f64().unwrap(),
            status: StratumStatus::Ok,
        }
    }

    #[test]
    fn gate_matches_python_parity_fixture() {
        let doc = parity_fixture();
        let t = &doc["thresholds"];
        let thresholds = ThresholdsConfig {
            target_f1: t["target_f1"].as_f64().unwrap(),
            failure_threshold: t["failure_threshold"].as_f64().unwrap(),
            min_per_stratum: t["min_per_stratum"].as_u64().unwrap() as usize,
        };

        for case in doc["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let i = &case["inputs"];
            let inputs = GateInputs {
                run_completed: i["run_completed"].as_bool().unwrap(),
                overall_micro_f1: i["micro_f1"].as_f64().unwrap(),
                overall_ci_lower: i["ci_lower"].as_f64().unwrap(),
                overall_ci_upper: i["ci_upper"].as_f64().unwrap(),
                overall_n: i["overall_n"].as_u64().unwrap() as usize,
                strata: i["strata"].as_array().unwrap().iter().map(stratum).collect(),
                kappa: i["kappa"].as_f64(),
                pending_reviews: i["pending_reviews"].as_u64().unwrap() as usize,
                subjective_fields: strings(&i["subjective_fields"]),
                defect_rate: i["defect_rate"].as_f64(),
                defect_n: i["defect_n"].as_u64().unwrap() as usize,
                total_n: i["total_n"].as_u64().unwrap() as usize,
            };

            let got = decide_gate(&inputs, &thresholds);
            let exp = &case["expected"];
            assert_eq!(got.status, exp["status"].as_str().unwrap(), "status: {name}");
            assert_eq!(got.promote, exp["promote"].as_bool().unwrap(), "promote: {name}");
            assert_eq!(got.reason, exp["reason"].as_str().unwrap(), "reason: {name}");
            assert_eq!(got.blockers, strings(&exp["blockers"]), "blockers: {name}");
            assert_eq!(got.warnings, strings(&exp["warnings"]), "warnings: {name}");
        }
    }

    #[test]
    fn classify_strata_matches_python_statuses() {
        let thresholds = ThresholdsConfig { target_f1: 0.85, failure_threshold: 0.5, min_per_stratum: 50 };
        let strata = vec![
            StratumResult { key: "ok".into(), n: 100, micro_f1: 0.9, ci_lower: 0.86, ci_upper: 0.94, status: StratumStatus::Ok },
            StratumResult { key: "warn".into(), n: 100, micro_f1: 0.8, ci_lower: 0.75, ci_upper: 0.85, status: StratumStatus::Ok },
            StratumResult { key: "block".into(), n: 100, micro_f1: 0.4, ci_lower: 0.3, ci_upper: 0.5, status: StratumStatus::Ok },
            StratumResult { key: "small".into(), n: 10, micro_f1: 0.9, ci_lower: 0.8, ci_upper: 1.0, status: StratumStatus::Ok },
        ];
        let got: Vec<&str> = classify_strata(&strata, &thresholds).iter().map(|s| s.status.as_str()).collect();
        assert_eq!(got, vec!["ok", "warning", "blocker", "insufficient_n"]);
    }
}
