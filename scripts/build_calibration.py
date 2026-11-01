"""Build judge-calibration pairs from dataset gold, then label them with a judge.

Pair truth is KNOWN BY CONSTRUCTION (identical list → match, dropped item →
mismatch, altered price → mismatch), so ``human`` is an objective answer key,
not the judge grading itself. For a real promotion these should be replaced by
hand-reviewed labels — this script is the dev-loop stand-in that gets κ onto
the board.

Usage:
  build_calibration.py build              # write calibration/pairs.jsonl
  build_calibration.py judge <model-id>   # label pairs with one NIM judge
  build_calibration.py score <model-id>   # print κ for a labeled file

Labeled output: calibration/judge_<slug>.jsonl with {human, judge, ...} rows,
the format ``slm-eval calibrate apply`` and ``_load_calibration`` expect.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CAL = ROOT / "calibration"
PAIRS = CAL / "pairs.jsonl"
DATA = ROOT / "data" / "cord" / "cases.jsonl"

from slm_eval.config import EvalConfig
from slm_eval.judge import CalibrationGate, CalibrationStatus, Judge
from slm_eval.llm import OpenAICompatClient


def _perturbations(gold_items: list[dict], i: int) -> list[tuple[str, list[dict], str]]:
    """Deterministic (gold, human_label) variants for one case."""
    if not gold_items:
        return []
    kind = i % 4
    if kind == 0:                       # identical → match
        return [("identical", copy.deepcopy(gold_items), "match")]
    if kind == 1:                       # case/space variation → match
        varied = copy.deepcopy(gold_items)
        for it in varied:
            it["name"] = "  " + str(it["name"]).title() + "  "
        return [("case_variation", varied, "match")]
    if kind == 2:                       # drop one item → mismatch
        return [("dropped_item", copy.deepcopy(gold_items[:-1]), "mismatch")]
    # alter a price → mismatch
    altered = copy.deepcopy(gold_items)
    price = altered[0].get("price")
    altered[0]["price"] = (price or 0) + 100
    return [("altered_price", altered, "mismatch")]


def cmd_build() -> int:
    cases = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    rows = []
    for i, c in enumerate(cases):
        gold_items = c["gold"].get("line_items") or []
        for kind, pred, human in _perturbations(gold_items, i):
            rows.append({
                "pair_id": f"p{len(rows):04d}",
                "case_id": c["case_id"],
                "kind": kind,
                "field": "line_items",
                "gold": gold_items,
                "pred": pred,
                "human": human,
            })
    CAL.mkdir(exist_ok=True)
    PAIRS.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    n_match = sum(1 for r in rows if r["human"] == "match")
    print(f"wrote {len(rows)} pairs to {PAIRS} ({n_match} match / {len(rows) - n_match} mismatch)")
    return 0


def _slug(model: str) -> str:
    return model.replace("/", "_")


def cmd_judge(model: str) -> int:
    cfg = EvalConfig.from_yaml(ROOT / "configs" / "cord.yaml")
    pairs = [json.loads(l) for l in PAIRS.read_text().splitlines() if l.strip()]
    out_path = CAL / f"judge_{_slug(model)}.jsonl"

    # Resume-friendly: keep labels already produced for this model.
    done: dict[str, str] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["pair_id"]] = r["judge"]

    client = OpenAICompatClient(
        base_url=cfg.judge.endpoint, model=model,
        api_key=os.environ["NVIDIA_API_KEY"], timeout=90,
        max_attempts=3, backoff_base=2.0,
    )
    # prompt_hash keys off the judge model — stamp this invocation's model so
    # verdicts cached under one model are never attributed to another (AC-03).
    judge = Judge(cfg.judge.model_copy(update={"model": model}), client,
                  cfg.task_description)

    pending = [p for p in pairs if p["pair_id"] not in done]
    print(f"{model}: {len(done)} cached, {len(pending)} to label → {out_path.name}")
    t0 = time.time()
    with out_path.open("a") as f:
        for n, p in enumerate(pending, 1):
            v = judge.judge_field(p["field"], p["gold"], p["pred"])
            rec = {**{k: p[k] for k in
                      ("pair_id", "case_id", "kind", "field", "human")},
                   "judge": v.verdict, "judge_model": model,
                   "prompt_hash": v.phash, "latency_ms": v.latency_ms}
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(pending):
                el = time.time() - t0
                eta = el / n * (len(pending) - n)
                print(f"  [{n}/{len(pending)}] elapsed {el:.0f}s eta {eta:.0f}s", flush=True)
    print("done")
    return 0


def cmd_score(model: str) -> int:
    cfg = EvalConfig.from_yaml(ROOT / "configs" / "cord.yaml")
    path = CAL / f"judge_{_slug(model)}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    human = [r["human"] for r in rows]
    judge = [r["judge"] for r in rows]
    gate = CalibrationGate(cfg.judge)
    # Evaluate against this model's identity, not the config's default.
    status = gate.evaluate(human, judge, resamples=2000)
    status = CalibrationStatus(
        kappa=status.kappa, pairs=status.pairs, ci_lower=status.ci_lower,
        ci_upper=status.ci_upper, model=model,
    )
    blocked, reason = gate.blocks_promotion(status)
    print(json.dumps({
        "model": model, "pairs": status.pairs,
        "kappa": round(status.kappa, 4),
        "kappa_ci": [round(status.ci_lower, 4), round(status.ci_upper, 4)],
        "blocks_promotion": blocked, "reason": reason or None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        raise SystemExit(cmd_build())
    if cmd == "judge" and len(sys.argv) > 2:
        raise SystemExit(cmd_judge(sys.argv[2]))
    if cmd == "score" and len(sys.argv) > 2:
        raise SystemExit(cmd_score(sys.argv[2]))
    print(__doc__)
    raise SystemExit(2)
