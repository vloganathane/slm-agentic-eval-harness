"""Generate the shared Python gate parity fixture.

Run from the repo root::

    python scripts/gen_gate_parity_fixture.py

The fixture is read by ``tests/test_gate_parity.py`` to assert Python behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

from slm_eval.config import ThresholdsConfig
from slm_eval.gate import GateInputs, StratumResult, decide_gate

FIXTURE = Path(__file__).parent.parent / "tests" / "fixtures" / "gate_cases.json"

THRESHOLDS = {"target_f1": 0.85, "failure_threshold": 0.5, "min_per_stratum": 50}


def _inputs(**overrides):
    base = {
        "overall": (0.87, 0.855, 0.90),
        "overall_n": 200,
        "strata": [],
        "kappa": 0.9,
        "pending_reviews": 0,
        "run_completed": True,
        "subjective_fields": [],
        "defect_rate": None,
        "defect_n": 0,
        "total_n": 0,
    }
    base.update(overrides)
    return base


CASES = [
    ("promote", _inputs()),
    ("ci_lower_blocks_point_clears", _inputs(overall=(0.871, 0.842, 0.897))),
    ("point_alone_cannot_gate", _inputs(overall=(1.0, 0.6, 1.0))),
    ("ci_lower_equal_target_promotes", _inputs(overall=(0.86, 0.85, 0.87))),
    ("incomplete_run_is_error", _inputs(run_completed=False)),
    ("pending_review_blocks", _inputs(pending_reviews=3)),
    (
        "stratum_blocker_at_failure_threshold",
        _inputs(strata=[("layout=rotated", 60, 0.4, 0.3, 0.5)]),
    ),
    (
        "stratum_warning_below_target",
        _inputs(strata=[("lang=hi", 60, 0.8, 0.7, 0.9)]),
    ),
    (
        "small_stratum_is_insufficient_n",
        _inputs(strata=[("lang=mr", 10, 0.1, 0.0, 0.2)]),
    ),
    ("kappa_missing_blocks", _inputs(kappa=None, subjective_fields=["vendor_name"])),
    ("kappa_low_blocks", _inputs(kappa=0.7, subjective_fields=["vendor_name"])),
    ("kappa_not_required_without_subjective", _inputs(kappa=None, subjective_fields=[])),
    (
        "dataset_defect_is_warning_only",
        _inputs(defect_rate=0.25, defect_n=5, total_n=20),
    ),
]


def _serialize_inputs(raw: dict) -> dict:
    micro_f1, ci_lower, ci_upper = raw["overall"]
    strata = [
        {
            "key": key,
            "n": n,
            "micro_f1": micro_f1,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
        for key, n, micro_f1, ci_lower, ci_upper in raw["strata"]
    ]
    return {
        "run_completed": raw["run_completed"],
        "micro_f1": micro_f1,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "overall_n": raw["overall_n"],
        "strata": strata,
        "kappa": raw["kappa"],
        "pending_reviews": raw["pending_reviews"],
        "subjective_fields": raw["subjective_fields"],
        "defect_rate": raw["defect_rate"],
        "defect_n": raw["defect_n"],
        "total_n": raw["total_n"],
    }


def _to_gate_inputs(raw: dict) -> GateInputs:
    return GateInputs(
        overall=raw["overall"],
        overall_n=raw["overall_n"],
        strata=[StratumResult(*s) for s in raw["strata"]],
        kappa=raw["kappa"],
        pending_reviews=raw["pending_reviews"],
        run_completed=raw["run_completed"],
        subjective_fields=raw["subjective_fields"],
        defect_rate=raw["defect_rate"],
        defect_n=raw["defect_n"],
        total_n=raw["total_n"],
    )


def main() -> None:
    thresholds = ThresholdsConfig(**THRESHOLDS)
    cases = []
    for name, raw in CASES:
        decision = decide_gate(_to_gate_inputs(raw), thresholds)
        cases.append(
            {
                "name": name,
                "inputs": _serialize_inputs(raw),
                "expected": {
                    "status": decision.status,
                    "promote": decision.promote,
                    "reason": decision.reason,
                    "blockers": decision.blockers,
                    "warnings": decision.warnings,
                },
            }
        )

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(
        json.dumps({"thresholds": THRESHOLDS, "cases": cases}, indent=2) + "\n"
    )
    print(f"wrote {FIXTURE} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
