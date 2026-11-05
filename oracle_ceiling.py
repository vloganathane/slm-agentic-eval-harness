"""Oracle ceiling: extract EXACTLY what the OCR text says, score with real pipeline.

If the oracle cannot reach ci_lower >= target_f1, no model can.
"""

from __future__ import annotations

import json
import re

from slm_eval.config import EvalConfig
from slm_eval.scoring import score_case

CFG = EvalConfig.from_yaml("configs/cord.yaml")

# "1 POTATO SAUSAGE BREAD 19,000" / "1 OREO GREEN TEA SPREAD  2,000"
LINE_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s+'?([\d][\d,\.]*)\s*$")
TOTAL_RE = re.compile(r"TOTAL\s+'?([\d][\d,\.]*)", re.I)


def _num(s: str) -> float | None:
    s = re.sub(r"[^\d.]", "", s)
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def oracle_predict(text: str) -> dict:
    items = []
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        qty, name, price = m.group(1), m.group(2).strip(), _num(m.group(3))
        if price is None or len(name) < 3:
            continue
        items.append({"name": name, "price": price, "qty": qty})

    total = None
    tm = TOTAL_RE.search(text)
    if tm:
        total = _num(tm.group(1))
    if total is None and items:
        total = sum(i["price"] for i in items)

    return {"line_items": items, "total": total}


def main() -> None:
    cases = [json.loads(l) for l in open("data/cord/cases.jsonl")]
    holdout_ids = {
        json.loads(l)["case_id"] for l in open("data/cord/splits/holdout.jsonl")
    }
    cases = [c for c in cases if c["case_id"] in holdout_ids]

    tp = fp = fn = 0
    per_case = []
    for c in cases:
        pred = oracle_predict(c["text"])
        outcomes = score_case(c["case_id"], pred, c["gold"], CFG)
        ctp = sum(o.tp for o in outcomes)
        cfp = sum(o.fp for o in outcomes)
        cfn = sum(o.fn for o in outcomes)
        tp += ctp
        fp += cfp
        fn += cfn
        per_case.append((c["case_id"], ctp, cfp, cfn))

    micro_f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + 0.5 * (fp + fn)) else 0.0

    # bootstrap CI, same resamples/seed as config
    import random

    rng = random.Random(CFG.stats.seed)
    scores = []
    n = len(per_case)
    for _ in range(CFG.stats.resamples):
        sample = [per_case[rng.randrange(n)] for _ in range(n)]
        stp = sum(x[1] for x in sample)
        sfp = sum(x[2] for x in sample)
        sfn = sum(x[3] for x in sample)
        den = stp + 0.5 * (sfp + sfn)
        scores.append(stp / den if den else 0.0)
    scores.sort()
    lo_idx = int(0.025 * len(scores))
    hi_idx = int(0.975 * len(scores))
    ci_lo, ci_hi = scores[lo_idx], scores[hi_idx]

    target = CFG.thresholds.target_f1
    print(f"holdout n          : {n}")
    print(f"tp/fp/fn           : {tp}/{fp}/{fn}")
    print(f"micro_f1           : {micro_f1:.3f}")
    print(f"CI                 : [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"target             : {target}")
    print(f"ORACLE WOULD PROMOTE: {ci_lo >= target}")

    hit = sum(1 for _, a, b, cc in per_case if b == 0 and cc == 0)
    print(f"perfect cases      : {hit}/{n}")
    worst = sorted(per_case, key=lambda x: x[1] - x[2] - x[3])[:5]
    print("worst cases        :", worst)


if __name__ == "__main__":
    main()
