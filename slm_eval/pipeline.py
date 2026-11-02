"""Orchestrator: runs stages 1–7 with checkpoints, emits gate.json + events (SPEC_V3 §5).

The measurement path is fully deterministic (deterministic lane, §7.2). No LLM is
involved except the judge, which is cached by prompt_hash. Resume is stage-level:
a COMPLETED stage with a matching input_hash is skipped.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from slm_eval import fixture
from slm_eval.ai_data_integration import ai_data_integration_agent
from slm_eval.config import EvalConfig
from slm_eval.errors import ConfigError, LeakageError, SlmEvalError
from slm_eval.gate import GateInputs, StratumResult, dataset_defect_rate, decide_gate
from slm_eval.hashing import sha256_json
from slm_eval.judge import CalibrationStatus, Judge
from slm_eval.llm import LLMClient
from slm_eval.scoring import score_case
from slm_eval.split import seal_holdout, split_stratified
from slm_eval.stats import FieldOutcome, bootstrap_micro_f1, classification_report
from slm_eval.store import Store

STAGES = ["validate", "split", "infer", "score", "aggregate", "judge_calibrate", "gate"]


class Orchestrator:
    def __init__(
        self,
        cfg: EvalConfig,
        store: Store,
        run_dir: Path,
        judge_client: LLMClient,
        *,
        code_version: str = "git:unknown",
        resume: bool = False,
        config_path: str | Path | None = None,
        force_cost: bool = False,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.run_dir = Path(run_dir)
        self.judge_client = judge_client
        self.code_version = code_version
        self.resume = resume
        self.force_cost = force_cost
        # Source config path (provenance) — agents need the real file, not a
        # hardcoded default, to read thresholds/reporting settings.
        self.config_path = Path(config_path) if config_path is not None else None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = self.run_dir.name
        self._events = self.run_dir / "events.jsonl"
        self._started: dict[str, float] = {}

    # -- events -------------------------------------------------------------

    def _emit(self, stage: str, event: str, level: str = "info", **metrics: Any) -> None:
        rec = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "level": level, "stage": stage, "event": event,
            "run_id": self.run_id, "round": 1, **metrics,
        }
        with self._events.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    def _begin(self, stage: str, input_hash: str) -> bool:
        """Return True if stage should run; handles checkpoint + resume (§5.2)."""
        if self.resume and self.store.stage_is_fresh(self.run_id, stage, input_hash):
            self._emit(stage, "skipped_resume")
            return False
        self.store.upsert_checkpoint(
            self.run_id, stage, "running", input_hash=input_hash, started=True
        )
        self._emit(stage, "started")
        self._started[stage] = time.monotonic()
        return True

    def _finish(self, stage: str, input_hash: str, **metrics: Any) -> None:
        elapsed = int((time.monotonic() - self._started.get(stage, time.monotonic())) * 1000)
        self.store.upsert_checkpoint(
            self.run_id, stage, "completed", input_hash=input_hash, completed=True
        )
        self._emit(stage, "completed", latency_ms=elapsed, **metrics)

    def _fail(self, stage: str, input_hash: str, err: Exception) -> None:
        self.store.upsert_checkpoint(
            self.run_id, stage, "failed", input_hash=input_hash, error=str(err)[:500]
        )
        self._emit(stage, "failed", level="error", error=str(err)[:500])

    # -- pipeline -------------------------------------------------------------

    def run(
        self,
        model_path: str,
        model_sha256: str,
        dataset_path: str,
        predictions: dict[str, dict[str, Any]] | None = None,
        calibration: CalibrationStatus | None = None,
        predictor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run stages 1–7. Returns the gate.json payload.

        ``predictions`` maps case_id → predicted fields (test/fixture callers).
        ``predictor`` is called per holdout case during the infer stage when no
        prediction was supplied — this is how a live model under test is wired in
        without leaking train data or duplicating the split logic.
        """
        cfg_hash = self.cfg.config_hash
        if self.store.get_run_safe(self.run_id) is None:
            self.store.create_run(
                run_id=self.run_id, model_path=model_path, model_sha256=model_sha256,
                dataset_path=dataset_path, dataset_hash="", config_hash=cfg_hash,
                code_version=self.code_version,
            )
        run = self.store.get_run(self.run_id)
        version = run["version"]

        # --- validate (stage 1) ---
        h_validate = cfg_hash
        if self._begin("validate", h_validate):
            try:
                if self.cfg.task_type != "extraction":
                    raise ConfigError(f"unsupported task_type: {self.cfg.task_type}")
                self._finish("validate", h_validate)
            except SlmEvalError as e:
                self._fail("validate", h_validate, e)
                raise

        # --- split + seal (stage 2) ---
        cases = fixture.read_dataset(dataset_path)
        split = split_stratified(
            cases, self.cfg.strata.keys, self.cfg.split.fraction,
            self.cfg.split.seed, self.cfg.thresholds.min_per_stratum,
        )
        h_split = sha256_json(
            {"holdout": split.holdout_hash, "n": len(split.holdout), "cfg": cfg_hash}
        )
        if self._begin("split", h_split):
            try:
                seal_holdout(split, self.run_dir)
                version = self.store.seal_holdout(self.run_id, split.holdout_hash, version)
                self._finish(
                    "split", h_split, holdout_n=len(split.holdout),
                    train_n=len(split.train), holdout_hash=split.holdout_hash[:12],
                )
            except LeakageError as e:
                self._fail("split", h_split, e)
                raise

        holdout = split.holdout

        # --- dataset defect quarantine (§2) ---
        # Cases whose input carries no signal (blank/garbage OCR) are excluded
        # from the micro-F1 denominator — a broken input is not a weak model.
        # The gate reports them separately as `dataset_defect`.
        signal_values = [
            str(c.get("strata", {}).get("ocr_signal", "usable")) for c in holdout
        ]
        defect_rate, defect_n = dataset_defect_rate(signal_values)
        usable_holdout = [
            c for c in holdout
            if str(c.get("strata", {}).get("ocr_signal", "usable")) == "usable"
        ]

        # Real predictions only: explicit (tests/fixture callers), resumed from a
        # prior partial run's predictions.jsonl, or produced live by ``predictor``.
        # Gold-as-prediction is NEVER used implicitly — resuming a crashed run with
        # gold stand-ins would fabricate a perfect score.
        preds: dict[str, dict[str, Any]] = dict(predictions or {})
        preds_file = self.run_dir / "predictions.jsonl"
        if preds_file.exists():
            for line in preds_file.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "case_id" in rec:
                    preds[rec["case_id"]] = rec["prediction"]

        # --- infer (stage 3) ---
        h_infer = sha256_json({"holdout": split.holdout_hash, "model": model_sha256})
        if self._begin("infer", h_infer):
            missing = [c for c in usable_holdout if c["case_id"] not in preds]
            if missing and predictor is None:
                err = SlmEvalError(f"missing predictions for {len(missing)} cases")
                self._fail("infer", h_infer, err)
                raise err
            for c in missing:
                preds[c["case_id"]] = predictor(c)
            self._write_predictions(preds)
            self._finish("infer", h_infer, cases=len(usable_holdout),
                         quarantined=defect_n)

        # --- cost alert (after infer) ---
        # The cap is a ceiling, not a floor: reaching it exactly is allowed
        # (a $0 local-only budget must not fail a run that spent $0). `--force`
        # records the overrun and continues, per SPEC_V3 §5.4.
        cost_so_far = self.store.cost_total(self.run_id)
        if cost_so_far > self.cfg.cost_control.alert_at_usd:
            self._emit("cost_alert", "warning", level="warning",
                       cost_usd=cost_so_far, threshold=self.cfg.cost_control.alert_at_usd)
        if cost_so_far > self.cfg.cost_control.max_cost_per_round_usd:
            if self.force_cost:
                self._emit("cost_control", "warning", level="warning",
                           cost_usd=cost_so_far,
                           threshold=self.cfg.cost_control.max_cost_per_round_usd,
                           forced=True)
            else:
                err = SlmEvalError(f"cost limit exceeded: ${cost_so_far:.4f} > ${self.cfg.cost_control.max_cost_per_round_usd:.4f}")
                self._fail("cost_control", h_infer, err)
                raise err

        # --- score (stage 4): exact-match + judge verdicts folded in ---
        h_score = sha256_json({"infer": h_infer, "cfg": cfg_hash})
        if self._begin("score", h_score):
            self._run_score_stage(usable_holdout, preds)
        # Load outcomes from disk (not from the stage body) so that resume past a
        # completed score stage still feeds aggregate/gate correctly.
        outcomes = self._load_outcomes()
        if not outcomes:
            err = SlmEvalError("no scores found — score stage did not produce output")
            self._fail("score", h_score, err)
            raise err

        # --- aggregate (stage 5): bootstrap CI overall + per stratum ---
        h_agg = sha256_json({"score": h_score, "seed": self.cfg.stats.seed})
        overall_metric = bootstrap_micro_f1(
            outcomes, self.cfg.stats.resamples, self.cfg.stats.seed
        )
        if self._begin("aggregate", h_agg):
            self._finish(
                "aggregate", h_agg,
                micro_f1=round(overall_metric.point, 4),
                ci=[round(overall_metric.ci_lower, 4), round(overall_metric.ci_upper, 4)],
            )

        stratum_results = self._stratum_metrics(usable_holdout, outcomes)

        # --- judge_calibrate (stage 6): κ gate (§6.4) ---
        h_cal = sha256_json({"kappa": None if calibration is None else calibration.kappa})
        if self._begin("judge_calibrate", h_cal):
            self._finish("judge_calibrate", h_cal, has_calibration=calibration is not None)

        # --- gate (stage 7) ---
        h_gate = sha256_json({"agg": h_agg, "cal": h_cal, "strata": [asdict(s) for s in stratum_results]})
        inputs = GateInputs(
            overall=(overall_metric.point, overall_metric.ci_lower, overall_metric.ci_upper),
            overall_n=overall_metric.n,
            strata=stratum_results,
            kappa=None if calibration is None else calibration.kappa,
            kappa_required=bool(self.cfg.fields.subjective_fields),
            pending_reviews=self.store.pending_review_count(self.run_id),
            run_completed=True,
            subjective_fields=list(self.cfg.fields.subjective_fields),
            defect_rate=defect_rate,
            defect_n=defect_n,
            total_n=len(holdout),
        )
        decision = decide_gate(inputs, self.cfg.thresholds)

        gate_payload = self._write_gate(
            model_path, model_sha256, split, overall_metric, stratum_results,
            calibration, decision, defect_rate=defect_rate, defect_n=defect_n,
            classification=self._classification_reports(),
        )
        if self._begin("gate", h_gate):
            self._finish("gate", h_gate, status=decision.status, promote=decision.promote)

        # --- AI Data Science Team post-processing (optional stage 8) ---
        if self.config_path is not None and "json" in self.cfg.reporting.formats:
            self._post_process(dataset_path)

        return gate_payload

    # -- helpers ---------------------------------------------------------------

    def _write_predictions(self, preds: dict[str, dict[str, Any]]) -> None:
        """Persist predictions with their case_id so a resumed run can reload them."""
        lines = "".join(
            json.dumps({"case_id": cid, "prediction": p}, sort_keys=True) + "\n"
            for cid, p in sorted(preds.items())
        )
        (self.run_dir / "predictions.jsonl").write_text(lines)

    def _run_score_stage(
        self, holdout: list[dict], preds: dict[str, dict[str, Any]]
    ) -> None:
        judge = Judge(self.cfg.judge, self.judge_client, self.cfg.task_description)
        outcomes: list[FieldOutcome] = []
        label_rows: list[dict[str, Any]] = []
        label_fields = self.cfg.fields.label_fields
        for c in holdout:
            judge_labels: dict[str, bool] = {}
            for fname in self.cfg.fields.subjective_fields:
                v = judge.judge_field(
                    fname, c["gold"].get(fname), preds[c["case_id"]].get(fname)
                )
                judge_labels[fname] = v.correct
                self.store.add_cost(self.run_id, "score", v.usd, v.tokens_in, v.tokens_out)
            for fname in label_fields:
                label_rows.append(
                    {
                        "case_id": c["case_id"],
                        "field": fname,
                        "gold": c["gold"].get(fname),
                        "pred": preds[c["case_id"]].get(fname),
                    }
                )
            outcomes.extend(
                score_case(c["case_id"], preds[c["case_id"]], c["gold"], self.cfg, judge_labels)
            )
        self.store.insert_scores(
            [
                {"run_id": self.run_id, "round": 1, "case_id": o.case_id,
                 "field": f"f{i}", "correct": int(o.correct), "tp": o.tp,
                 "fp": o.fp, "fn": o.fn, "judge_model": self.cfg.judge.model,
                 "prompt_hash": "", "latency_ms": 0}
                for i, o in enumerate(outcomes)
            ]
        )
        (self.run_dir / "scores.jsonl").write_text(
            "".join(json.dumps(asdict(o), sort_keys=True) + "\n" for o in outcomes)
        )
        if label_fields:
            # Persist raw (gold, pred) label pairs so resume can rebuild the
            # classification report without re-running inference or the judge.
            (self.run_dir / "labels.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in label_rows)
            )

    def _load_outcomes(self) -> list[FieldOutcome]:
        """Read back the score stage's output so resume doesn't recompute it."""
        path = self.run_dir / "scores.jsonl"
        if not path.exists():
            return []
        out: list[FieldOutcome] = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(
                FieldOutcome(case_id=d["case_id"], tp=d["tp"], fp=d["fp"], fn=d["fn"])
            )
        return out

    def _classification_reports(self) -> dict[str, Any]:
        """Per-label-field classification reports, or {} when none declared.

        Built from labels.jsonl (raw gold/pred pairs), so a resumed run reports
        identically to a fresh one.
        """
        path = self.run_dir / "labels.jsonl"
        if not path.exists() or not self.cfg.fields.label_fields:
            return {}
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        reports: dict[str, Any] = {}
        for fname in self.cfg.fields.label_fields:
            fr = [r for r in rows if r["field"] == fname]
            if not fr:
                continue
            reports[fname] = classification_report(
                [str(r["gold"]) for r in fr], [str(r["pred"]) for r in fr]
            ).to_dict()
        return reports

    def _stratum_metrics(
        self, holdout: list[dict], outcomes: list[FieldOutcome]
    ) -> list[StratumResult]:
        if not self.cfg.strata.keys:
            return []
        primary = self.cfg.strata.keys[0]
        by_case = {o.case_id: [] for o in outcomes}
        for o in outcomes:
            by_case.setdefault(o.case_id, []).append(o)
        groups: dict[str, list[FieldOutcome]] = {}
        for c in holdout:
            val = c["strata"].get(primary, "unknown")
            groups.setdefault(val, []).extend(by_case.get(c["case_id"], []))
        results = []
        for value, outs in sorted(groups.items()):
            m = bootstrap_micro_f1(outs, self.cfg.stats.resamples, self.cfg.stats.seed)
            results.append(
                StratumResult(
                    key=f"{primary}={value}", n=m.n, micro_f1=m.point,
                    ci_lower=m.ci_lower, ci_upper=m.ci_upper,
                )
            )
        return results

    def _write_gate(
        self, model_path, model_sha256, split, overall, stratum_results,
        calibration, decision, *, defect_rate: float = 0.0, defect_n: int = 0,
        classification: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "schema_version": 1,
            "model": {"path": model_path, "sha256": model_sha256},
            "holdout": {
                "n": len(split.holdout), "hash": split.holdout_hash,
                "split_seed": self.cfg.split.seed,
            },
            "decision": {"promote": decision.promote, "reason": decision.reason,
                         "status": decision.status},
            "metrics": {
                "micro_f1": round(overall.point, 4),
                "ci_lower": round(overall.ci_lower, 4),
                "ci_upper": round(overall.ci_upper, 4),
                "resamples": overall.resamples,
                "seed": self.cfg.stats.seed,
            },
            "strata": [
                {"key": s.key, "n": s.n, "micro_f1": round(s.micro_f1, 4),
                 "ci_lower": round(s.ci_lower, 4), "ci_upper": round(s.ci_upper, 4),
                 "status": s.status}
                for s in stratum_results
            ],
            "dataset_defect": {
                "rate": round(defect_rate, 4),
                "defect_n": defect_n,
                "total_n": len(split.holdout),
                "scored_n": len(split.holdout) - defect_n,
                "quarantined": defect_n > 0,
            },
            "judge": (
                {"kappa": calibration.kappa, "pairs": calibration.pairs,
                 "model": calibration.model}
                if calibration else {"kappa": None, "pairs": 0, "model": self.cfg.judge.model}
            ),
            "blockers": decision.blockers,
            "warnings": decision.warnings,
            # Additive: only present when config declares label_fields, so
            # existing readers of metrics.micro_f1 are unaffected.
            **({"classification": classification} if classification else {}),
            "pending_reviews": self.store.pending_review_count(self.run_id),
            "cost_usd": round(self.store.cost_total(self.run_id), 6),
            "provenance": {"config_hash": self.cfg.config_hash, "code_version": self.code_version},
        }
        (self.run_dir / "gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    def _post_process(
        self,
        dataset_path: str,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
    ) -> dict[str, Any] | None:
        """Run AI Data Science Team post-processing analysis.

        This is an optional post-processing stage that runs after the gate decision
        to provide enhanced analysis, visualization, and semantic insights.

        Args:
            dataset_path: Path to the original dataset (cases.jsonl or its dir)
            llm_base_url: Optional LLM endpoint for semantic analysis
            llm_model: Optional LLM model name

        Returns:
            Dict with AI analysis results, or None if disabled
        """
        if self.config_path is None:
            self._emit("post_process", "skipped", level="warning",
                       reason="no config path supplied")
            return None
        try:
            cases_path = fixture.resolve_dataset_path(dataset_path)
            score_path = self.run_dir / "scores.jsonl"
            result = ai_data_integration_agent(
                run_dir=self.run_dir,
                config_path=self.config_path,
                dataset_path=cases_path,
                score_path=score_path if score_path.exists() else None,
                llm_base_url=llm_base_url,
                llm_model=llm_model,
            )

            # Write AI analysis report alongside the other run artifacts.
            report_dir = self.run_dir / "viz"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / "ai_analysis_report.json"
            report_path.write_text(json.dumps(result, indent=2))
            self._emit("post_process", "completed", report_path=str(report_path))

            return result
        except Exception as e:  # noqa: BLE001 — advisory stage must never fail the run
            self._emit("post_process", "failed", level="error", error=str(e))
            return None
