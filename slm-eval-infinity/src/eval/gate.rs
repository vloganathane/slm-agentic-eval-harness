//! Gate evaluation logic for promotion decisions

use anyhow::Result;
use crate::models::{GateDecision, GateDecisionType, AggregateScores, CalibrationResult};

/// Evaluate the gate decision based on scores, calibration, and pending reviews
/// 
/// Uses CI lower bounds (never point estimates) per SPEC_V3.md requirements.
/// Exit codes: 0=promote, 1=blocked, 2=error
pub fn evaluate_gate(
    scores: &AggregateScores,
    calibration: &CalibrationResult,
    pending_reviews: usize,
    micro_f1_target: f64,
    kappa_target: f64,
    min_strata_n: usize,
) -> GateDecision {
    let mut reasons = Vec::new();
    let mut promote = true;

    // Check micro-F1 using CI LOWER BOUND (not point estimate)
    if scores.ci_lower < micro_f1_target {
        promote = false;
        reasons.push(format!(
            "micro-F1 CI lower bound {:.3} < target {:.3}",
            scores.ci_lower, micro_f1_target
        ));
    }

    // Check κ calibration using CI LOWER BOUND
    if calibration.ci_lower < kappa_target {
        promote = false;
        reasons.push(format!(
            "κ CI lower bound {:.3} < target {:.3}",
            calibration.ci_lower, kappa_target
        ));
    }

    // Check minimum strata sample size
    for (strata, agg) in &scores.per_strata {
        if agg.n < min_strata_n {
            promote = false;
            reasons.push(format!(
                "Strata '{}' has n={} < min_n={}",
                strata, agg.n, min_strata_n
            ));
        }
    }

    // Check pending reviews - if any pending, cannot promote
    if pending_reviews > 0 {
        promote = false;
        reasons.push(format!("{} review tasks pending resolution", pending_reviews));
    }

    // Determine decision type and exit code
    let (decision, exit_code) = if !promote {
        if pending_reviews > 0 {
            (GateDecisionType::BlockedPendingReview, 1)
        } else if !reasons.is_empty() {
            (GateDecisionType::Blocked, 1)
        } else {
            (GateDecisionType::Error, 2)
        }
    } else {
        (GateDecisionType::Promote, 0)
    };

    GateDecision {
        decision,
        promote,
        micro_f1: scores.micro_f1,
        ci_lower: scores.ci_lower,
        ci_upper: scores.ci_upper,
        kappa: calibration.kappa,
        kappa_ci_lower: calibration.ci_lower,
        strata_pass: reasons.iter().all(|r| !r.contains("Strata")),
        calibration_pass: calibration.ci_lower >= kappa_target,
        pending_reviews,
        reasons,
        exit_code,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{AggregateScores, FieldScore, StrataAggregate, CalibrationResult};
    use std::collections::HashMap;

    fn make_test_scores(micro_f1: f64, ci_lower: f64, ci_upper: f64) -> AggregateScores {
        AggregateScores {
            micro_f1,
            ci_lower,
            ci_upper,
            per_field: HashMap::new(),
            per_strata: HashMap::new(),
            dataset_defect: 0,
            total_cases: 100,
            evaluated_cases: 100,
        }
    }

    fn make_test_calibration(kappa: f64, ci_lower: f64, ci_upper: f64) -> CalibrationResult {
        CalibrationResult {
            kappa,
            ci_lower,
            ci_upper,
            n_pairs: 100,
            n_agree: 90,
            n_disagree: 10,
        }
    }

    #[test]
    fn test_gate_promote() {
        let scores = make_test_scores(0.90, 0.87, 0.93);
        let cal = make_test_calibration(0.95, 0.90, 1.0);
        let gate = evaluate_gate(&scores, &cal, 0, 0.85, 0.85, 5);
        assert!(gate.promote);
        assert_eq!(gate.decision, GateDecisionType::Promote);
        assert_eq!(gate.exit_code, 0);
    }

    #[test]
    fn test_gate_blocked_micro_f1() {
        let scores = make_test_scores(0.90, 0.80, 0.95); // CI lower < target
        let cal = make_test_calibration(0.95, 0.90, 1.0);
        let gate = evaluate_gate(&scores, &cal, 0, 0.85, 0.85, 5);
        assert!(!gate.promote);
        assert_eq!(gate.decision, GateDecisionType::Blocked);
        assert_eq!(gate.exit_code, 1);
        assert!(gate.reasons.iter().any(|r| r.contains("micro-F1")));
    }

    #[test]
    fn test_gate_blocked_kappa() {
        let scores = make_test_scores(0.90, 0.87, 0.93);
        let cal = make_test_calibration(0.90, 0.80, 0.95); // CI lower < target
        let gate = evaluate_gate(&scores, &cal, 0, 0.85, 0.85, 5);
        assert!(!gate.promote);
        assert_eq!(gate.decision, GateDecisionType::Blocked);
        assert_eq!(gate.exit_code, 1);
        assert!(gate.reasons.iter().any(|r| r.contains("κ")));
    }

    #[test]
    fn test_gate_blocked_pending_review() {
        let scores = make_test_scores(0.90, 0.87, 0.93);
        let cal = make_test_calibration(0.95, 0.90, 1.0);
        let gate = evaluate_gate(&scores, &cal, 3, 0.85, 0.85, 5);
        assert!(!gate.promote);
        assert_eq!(gate.decision, GateDecisionType::BlockedPendingReview);
        assert_eq!(gate.exit_code, 1);
        assert!(gate.reasons.iter().any(|r| r.contains("pending")));
    }

    #[test]
    fn test_gate_uses_ci_lower_not_point() {
        // Point estimate above target, but CI lower below - should BLOCK
        let scores = make_test_scores(0.95, 0.80, 0.99); // CI lower = 0.80 < 0.85
        let cal = make_test_calibration(0.95, 0.90, 1.0);
        let gate = evaluate_gate(&scores, &cal, 0, 0.85, 0.85, 5);
        assert!(!gate.promote);
        assert!(gate.reasons.iter().any(|r| r.contains("micro-F1")));
    }
}