"""Build data/cord from CORD v2 parquet: OCR words (input) + parsed annotations (gold).

Both come from the same receipt, so input and gold correspond by construction.
Numbers are parsed with scoring._to_float on BOTH sides so a model that reads a
token straight from the text lands on the same value as the gold.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from slm_eval.scoring import _to_float

SPLITS = ["train", "valid", "test"]
OUT = Path("data/cord")


def _words(raw: dict) -> list[str]:
    return [w for w in ast.literal_eval(raw["ocr_words"]) if w and str(w).strip()]


def _quality(words: list[str], boxes: str) -> str:
    try:
        boxes_list = ast.literal_eval(boxes)
        confs = [b[1][1] for b in boxes_list if len(b) > 1 and len(b[1]) > 1]
        mean = sum(confs) / len(confs) if confs else 0.0
    except Exception:
        mean = 0.0
    if mean >= 0.92:
        return "high"
    if mean >= 0.80:
        return "medium"
    return "low"


def _line_items(parsed: dict) -> list[dict]:
    raw = parsed.get("line_items")
    if isinstance(raw, dict):
        raw = [raw]
    items = []
    for it in raw or []:
        if not isinstance(it, dict):
            continue
        name = it.get("item_name") or it.get("name")
        if not name:
            continue
        price = _to_float(it.get("item_value") or it.get("price") or 0)
        qty = str(it.get("item_quantity") or it.get("qty") or "1")
        if price is None:
            continue
        items.append({"name": str(name), "price": price, "qty": qty})
    return items


def _gold(parsed_data: str) -> dict | None:
    try:
        node = parsed_data if isinstance(parsed_data, dict) else json.loads(parsed_data)
        if isinstance(node, dict) and "json" in node:
            node = node["json"]
        # HF mirrors nest this further: the value can be a JSON string of a
        # JSON string of the annotation object. Unwrap until it is a dict.
        for _ in range(5):
            if not isinstance(node, str):
                break
            node = json.loads(node)
        if not isinstance(node, dict):
            return None
        parsed = node
    except Exception:
        return None

    total_block = parsed.get("total") or {}
    total_raw = total_block.get("total") if isinstance(total_block, dict) else total_block
    total = _to_float(total_raw) if total_raw is not None else None

    items = _line_items(parsed)
    if total is None and items:
        total = sum(i["price"] for i in items)
    if total is None and not items:
        return None
    return {"line_items": items, "total": total}


def build() -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        path = Path(f".scratch/cord_{split}.parquet")
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for _, r in df.iterrows():
            try:
                raw = ast.literal_eval(r["raw_data"])
            except Exception:
                continue
            words = _words(raw)
            if len(words) < 5:
                continue
            gold = _gold(r["parsed_data"])
            if gold is None:
                continue
            if not gold["line_items"] and gold["total"] is None:
                continue
            rows.append(
                {
                    "case_id": f"cord-{int(r['id']) if str(r['id']).isdigit() else len(rows):04d}",
                    "gold": gold,
                    "strata": {
                        "quality": _quality(words, raw.get("ocr_boxes", "[]")),
                        "ocr_signal": "usable",
                        "split_source": split,
                    },
                    "text": "\n".join(words),
                }
            )
    return pd.DataFrame(rows)


def _num_tokens(text: str) -> set[float]:
    """Every value in the text that the numeric validator could accept.

    Tokenised per line so a value can never swallow a newline and its
    neighbour. CORD writes thousands with a dot (``129.000``) while the
    validator reads a dot as a decimal, so both spellings are offered: the
    naive parse and the IDR reading.
    """
    import re

    out = set()
    for line in text.splitlines():
        for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", line):
            v = _to_float(tok)
            if v is not None:
                out.add(v)
            # "129.000" in an IDR receipt means 129000, not 129.0.
            if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", tok):
                idr = _to_float(tok.replace(".", ","))
                if idr is not None:
                    out.add(idr)
    return out


def ceiling(df: pd.DataFrame) -> None:
    """What would a perfect extractor of the input text score?"""
    tp = fn = 0
    tot_ok = tot_n = li_ok = li_n = 0
    for _, c in df.iterrows():
        toks = _num_tokens(c["text"])
        g = c["gold"]
        if g["total"] is not None:
            tot_n += 1
            if g["total"] in toks:
                tot_ok += 1
                tp += 1
            else:
                fn += 1
        if g["line_items"]:
            li_n += 1
            if all(i["price"] in toks for i in g["line_items"]):
                li_ok += 1
                tp += 1
            else:
                fn += 1
    f1 = tp / (tp + 0.5 * fn) if (tp + 0.5 * fn) else 0.0
    print(f"  total  : {tot_ok}/{tot_n} = {tot_ok / tot_n:.3f}" if tot_n else "  total  : n/a")
    print(f"  items  : {li_ok}/{li_n} = {li_ok / li_n:.3f}" if li_n else "  items  : n/a")
    print(f"  CEILING micro-F1 = {f1:.3f}  (target 0.85) -> {'REACHABLE' if f1 >= 0.85 else 'UNREACHABLE'}")


if __name__ == "__main__":
    df = build()
    print(f"built {len(df)} cases")
    print("quality:", dict(Counter(c["strata"]["quality"] for _, c in df.iterrows())))
    print("has line_items:", sum(1 for _, c in df.iterrows() if c["gold"]["line_items"]))
    print("has total:", sum(1 for _, c in df.iterrows() if c["gold"]["total"] is not None))
    print("ceiling:")
    ceiling(df)
