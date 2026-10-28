"""SQLite state store (SPEC_V3 §4.2 schema, §4.3 invariants).

Single-writer, multi-reader, WAL mode. ``run.version`` implements optimistic
locking: every state write bumps it with ``UPDATE … WHERE version = ?`` and
raises ``StateConflict`` on mismatch.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from slm_eval.errors import SealedArtifactError, StateConflict

SCHEMA = """
CREATE TABLE IF NOT EXISTS run (
  run_id TEXT PRIMARY KEY, model_path TEXT, model_sha256 TEXT,
  dataset_path TEXT, dataset_hash TEXT, holdout_hash TEXT,
  status TEXT CHECK (status IN ('created','running','completed','failed')),
  config_hash TEXT, code_version TEXT,
  created_at TEXT, updated_at TEXT, version INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS stage_checkpoint (
  run_id TEXT, stage TEXT, input_hash TEXT, status TEXT,
  output_ref TEXT, error TEXT, started_at TEXT, completed_at TEXT,
  PRIMARY KEY (run_id, stage)
);
CREATE TABLE IF NOT EXISTS score (
  run_id TEXT, round INTEGER, case_id TEXT, field TEXT,
  correct INTEGER, tp INTEGER, fp INTEGER, fn INTEGER,
  judge_model TEXT, prompt_hash TEXT, latency_ms INTEGER,
  UNIQUE (run_id, round, case_id, field, prompt_hash)
);
CREATE TABLE IF NOT EXISTS stratum_case (
  run_id TEXT, case_id TEXT, stratum_key TEXT, stratum_value TEXT
);
CREATE TABLE IF NOT EXISTS calibration (
  id INTEGER PRIMARY KEY, pair_hash TEXT UNIQUE, gold TEXT,
  judge TEXT, label TEXT, kappa REAL, pairs INTEGER, created_at TEXT
);
CREATE TABLE IF NOT EXISTS review_task (
  run_id TEXT, case_id TEXT, reason TEXT, state TEXT,
  assigned_at TEXT, due_at TEXT, decided_at TEXT, decision TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS cost_ledger (
  run_id TEXT, stage TEXT, usd REAL, tokens_in INTEGER, tokens_out INTEGER, ts TEXT
);
CREATE TABLE IF NOT EXISTS proposal (
  run_id TEXT, round INTEGER, agent TEXT, kind TEXT, payload TEXT,
  state TEXT, created_at TEXT,
  UNIQUE (run_id, round, agent, kind, payload)
);
CREATE TABLE IF NOT EXISTS error_category (
  id INTEGER PRIMARY KEY, run_id TEXT, case_id TEXT, field TEXT,
  category TEXT, subcategory TEXT, confidence REAL, reviewer TEXT, created_at TEXT,
  UNIQUE (run_id, case_id, field, category)
);
CREATE TABLE IF NOT EXISTS review_comment (
  id INTEGER PRIMARY KEY, run_id TEXT, case_id TEXT, reason TEXT,
  reviewer TEXT, comment TEXT, created_at TEXT
);
"""

VALID_RUN_STATUSES = ("created", "running", "completed", "failed")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    """Wrapper over the harness database. All writes go through it."""

    def __init__(self, db_path: str | Path = "slm_eval.db") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- runs ---------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        model_path: str,
        model_sha256: str,
        dataset_path: str,
        dataset_hash: str,
        config_hash: str,
        code_version: str,
    ) -> None:
        self._conn.execute(
            """INSERT INTO run (run_id, model_path, model_sha256, dataset_path,
               dataset_hash, config_hash, code_version, status, created_at, updated_at, version)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'created', ?, ?, 0)""",
            (run_id, model_path, model_sha256, dataset_path, dataset_hash, config_hash,
             code_version, _now(), _now()),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self._conn.execute("SELECT * FROM run WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"run not found: {run_id}")
        return dict(row)

    def get_run_safe(self, run_id: str) -> dict[str, Any] | None:
        """Like get_run but returns None instead of raising (idempotent create)."""
        row = self._conn.execute("SELECT * FROM run WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def update_run(self, run_id: str, expected_version: int, **fields: Any) -> int:
        """Optimistic-locked state write: bumps version, raises StateConflict (§4.3)."""
        allowed = {"status", "holdout_hash", "dataset_hash", "model_sha256"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"immutable or unknown run fields: {sorted(bad)}")
        if "status" in fields and fields["status"] not in VALID_RUN_STATUSES:
            raise ValueError(f"invalid status: {fields['status']}")
        sets = ", ".join(f"{k} = ?" for k in fields)
        params = [*fields.values(), _now(), run_id, expected_version]
        cur = self._conn.execute(
            f"UPDATE run SET {sets}, updated_at = ?, version = version + 1 "
            f"WHERE run_id = ? AND version = ?",
            params,
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise StateConflict(
                f"run {run_id}: version {expected_version} is stale "
                f"(current {self.get_run(run_id)['version']})"
            )
        return expected_version + 1

    # -- sealed holdout (§4.3: write-once) ----------------------------------

    def seal_holdout(self, run_id: str, holdout_hash: str, expected_version: int) -> int:
        current = self.get_run(run_id).get("holdout_hash")
        if current is not None and current != holdout_hash:
            raise SealedArtifactError(
                f"run {run_id}: holdout already sealed with different hash "
                f"({current[:12]}… != {holdout_hash[:12]}…)"
            )
        return self.update_run(run_id, expected_version, holdout_hash=holdout_hash)

    # -- stage checkpoints ---------------------------------------------------

    def upsert_checkpoint(
        self,
        run_id: str,
        stage: str,
        status: str,
        input_hash: str | None = None,
        output_ref: str | None = None,
        error: str | None = None,
        started: bool = False,
        completed: bool = False,
    ) -> None:
        now = _now()
        self._conn.execute(
            """INSERT INTO stage_checkpoint
               (run_id, stage, input_hash, status, output_ref, error, started_at, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (run_id, stage) DO UPDATE SET
                 input_hash = excluded.input_hash, status = excluded.status,
                 output_ref = excluded.output_ref, error = excluded.error,
                 started_at = COALESCE(stage_checkpoint.started_at, excluded.started_at),
                 completed_at = CASE WHEN ? THEN excluded.completed_at
                                     ELSE stage_checkpoint.completed_at END""",
            (run_id, stage, input_hash, status, output_ref, error,
             now if started else None, now if completed else None, int(completed)),
        )
        self._conn.commit()

    def get_checkpoint(self, run_id: str, stage: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM stage_checkpoint WHERE run_id = ? AND stage = ?", (run_id, stage)
        ).fetchone()
        return dict(row) if row else None

    def stage_is_fresh(self, run_id: str, stage: str, input_hash: str) -> bool:
        """Resume rule (§5.2): COMPLETED and input_hash matches."""
        cp = self.get_checkpoint(run_id, stage)
        return bool(cp and cp["status"] == "completed" and cp["input_hash"] == input_hash)

    # -- scores (§4.3 uniqueness = judge dedup) ------------------------------

    def insert_scores(self, rows: Iterable[dict[str, Any]]) -> int:
        """Idempotent insert; UNIQUE dedup makes judge resume safe (§5.4)."""
        cur = self._conn.executemany(
            """INSERT OR IGNORE INTO score
               (run_id, round, case_id, field, correct, tp, fp, fn,
                judge_model, prompt_hash, latency_ms)
               VALUES (:run_id, :round, :case_id, :field, :correct, :tp, :fp, :fn,
                       :judge_model, :prompt_hash, :latency_ms)""",
            rows,
        )
        self._conn.commit()
        return cur.rowcount

    def scores(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM score WHERE run_id = ? ORDER BY case_id, field", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # -- strata ---------------------------------------------------------------

    def insert_stratum_cases(self, run_id: str, rows: Iterable[dict[str, str]]) -> int:
        cur = self._conn.executemany(
            """INSERT OR IGNORE INTO stratum_case (run_id, case_id, stratum_key, stratum_value)
               VALUES (?, ?, ?, ?)""",
            [(run_id, r["case_id"], r["stratum_key"], r["stratum_value"]) for r in rows],
        )
        self._conn.commit()
        return cur.rowcount

    # -- calibration ----------------------------------------------------------

    def latest_calibration(self, judge: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM calibration WHERE judge = ? ORDER BY id DESC LIMIT 1", (judge,)
        ).fetchone()
        return dict(row) if row else None

    def record_calibration(
        self, pair_hash: str, gold: str, judge: str, label: str, kappa: float, pairs: int
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO calibration
               (pair_hash, gold, judge, label, kappa, pairs, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (pair_hash, gold, judge, label, kappa, pairs, _now()),
        )
        self._conn.commit()

    # -- review tasks ----------------------------------------------------------

    def pending_review_count(self, run_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM review_task WHERE run_id = ? AND state = 'pending'",
            (run_id,),
        ).fetchone()
        return int(row["n"])

    def insert_review_tasks(self, run_id: str, rows: list[dict[str, Any]]) -> int:
        """Insert review tasks (idempotent by case_id + reason)."""
        cur = self._conn.executemany(
            """INSERT OR IGNORE INTO review_task
               (run_id, case_id, reason, state, assigned_at, due_at, decided_at, decision, notes)
               VALUES (?, ?, ?, 'pending', ?, ?, NULL, NULL, '')""",
            [(r["run_id"], r["case_id"], r["reason"], r.get("assigned_at", _now()), r.get("due_at")) for r in rows],
        )
        self._conn.commit()
        return cur.rowcount

    def get_review_tasks(self, run_id: str, state: str | None = None) -> list[dict[str, Any]]:
        if state:
            rows = self._conn.execute(
                "SELECT * FROM review_task WHERE run_id = ? AND state = ? ORDER BY assigned_at",
                (run_id, state),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM review_task WHERE run_id = ? ORDER BY assigned_at",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_review_task(self, run_id: str, case_id: str, reason: str,
                           decision: str, notes: str = "") -> None:
        self._conn.execute(
            """UPDATE review_task SET state = 'decided', decided_at = ?, decision = ?, notes = ?
               WHERE run_id = ? AND case_id = ? AND reason = ?""",
            (_now(), decision, notes, run_id, case_id, reason),
        )
        self._conn.commit()

    def reopen_review_task(self, run_id: str, case_id: str, reason: str) -> None:
        self._conn.execute(
            """UPDATE review_task SET state = 'pending', decided_at = NULL, decision = NULL
               WHERE run_id = ? AND case_id = ? AND reason = ?""",
            (run_id, case_id, reason),
        )
        self._conn.commit()

    # -- error categorization (analyst lane) -----------------------------------

    def insert_error_category(
        self, run_id: str, case_id: str, field: str, category: str,
        subcategory: str = "", confidence: float = 1.0, reviewer: str = ""
    ) -> bool:
        """Record an error category. Returns True if new, False if duplicate."""
        cur = self._conn.execute(
            """INSERT OR IGNORE INTO error_category
               (run_id, case_id, field, category, subcategory, confidence, reviewer, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, case_id, field, category, subcategory, confidence, reviewer, _now()),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get_error_categories(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM error_category WHERE run_id = ? ORDER BY created_at", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_error_summary(self, run_id: str) -> dict[str, int]:
        """Aggregate counts by category."""
        rows = self._conn.execute(
            "SELECT category, COUNT(*) as n FROM error_category WHERE run_id = ? GROUP BY category",
            (run_id,),
        ).fetchall()
        return {r["category"]: r["n"] for r in rows}

    # -- review comments ---------------------------------------------------------

    def add_review_comment(
        self, run_id: str, case_id: str, reason: str, reviewer: str, comment: str
    ) -> None:
        self._conn.execute(
            """INSERT INTO review_comment (run_id, case_id, reason, reviewer, comment, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, case_id, reason, reviewer, comment, _now()),
        )
        self._conn.commit()

    def get_review_comments(self, run_id: str, case_id: str | None = None) -> list[dict[str, Any]]:
        if case_id:
            rows = self._conn.execute(
                "SELECT * FROM review_comment WHERE run_id = ? AND case_id = ? ORDER BY created_at",
                (run_id, case_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM review_comment WHERE run_id = ? ORDER BY created_at", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- proposals (advisory lane, §7.2) --------------------------------------

    def insert_proposal(
        self, run_id: str, round: int, agent: str, kind: str, payload: str
    ) -> bool:
        """Returns True if newly recorded, False if duplicate (content-addressed)."""
        cur = self._conn.execute(
            """INSERT OR IGNORE INTO proposal
               (run_id, round, agent, kind, payload, state, created_at)
               VALUES (?, ?, ?, ?, ?, 'proposed', ?)""",
            (run_id, round, agent, kind, payload, _now()),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # -- cost ledger ------------------------------------------------------------

    def add_cost(
        self, run_id: str, stage: str, usd: float, tokens_in: int = 0, tokens_out: int = 0
    ) -> None:
        self._conn.execute(
            "INSERT INTO cost_ledger (run_id, stage, usd, tokens_in, tokens_out, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, stage, usd, tokens_in, tokens_out, _now()),
        )
        self._conn.commit()

    def cost_total(self, run_id: str) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(usd), 0) AS total FROM cost_ledger WHERE run_id = ?", (run_id,)
        ).fetchone()
        return float(row["total"])
