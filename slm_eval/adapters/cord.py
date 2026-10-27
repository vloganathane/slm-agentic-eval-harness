"""CORD-v2 receipt dataset → ``cases.jsonl`` (SPEC_V3 §2).

CORD-v2 is a real-world receipt dataset (Singapore/Indonesia receipts). Each row
carries a receipt ``image`` (PNG) and ``ground_truth`` JSON. The image is OCR'd
with Tesseract to produce the noisy ``text`` the model actually sees — this is the
"bad dataset" whose extraction quality we want to measure and improve.

Gold fields we emit (only those reliably labelled in CORD):
  - ``total``       → numeric, from ``gt_parse.total.total_price``
  - ``line_items``  → list of {name, qty, price}, from ``gt_parse.menu``

Strata are derived from the OCR text and receipt structure (R-3), since CORD
ships no language/quality labels:
  - ``quality``     → low / medium / high, from OCR character cleanliness
  - ``item_count``  → small / medium / large, bucketed by line-item count
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

from slm_eval.hashing import sha256_json


def _ocr_image(png_bytes: bytes) -> str:
    """OCR a PNG receipt with Tesseract, returning the raw text."""
    import pytesseract
    from PIL import Image

    img = Image.open(io.BytesIO(png_bytes))
    # Receipts are small; upscale 2x for better OCR on thermal-printed text.
    img = img.resize((img.width * 2, img.height * 2))
    return pytesseract.image_to_string(img)


def _ocr_signal_stratum(text: str) -> str:
    """``usable`` if OCR produced enough text to extract from, else ``unextractable``.

    Blank/thin OCR is a *dataset* defect, not a model failure: those cases carry
    gold values but the input contains no signal. The gate quarantines them out of
    the micro-F1 denominator and reports them as ``dataset_defect`` instead, so a
    broken input never masquerades as a weak model.
    """
    return "usable" if len(text.strip()) >= 60 else "unextractable"


def _quality_stratum(text: str) -> str:
    """Bucket OCR text quality from character cleanliness.

    High = mostly alnum/whitespace (clean OCR); low = heavy symbol/garble noise.
    """
    if not text:
        return "low"
    total = len(text)
    clean = sum(1 for ch in text if ch.isalnum() or ch.isspace() or ch in ".,₹$:-/")
    ratio = clean / total
    if ratio >= 0.92:
        return "high"
    if ratio >= 0.82:
        return "medium"
    return "low"


def _item_count_stratum(n: int) -> str:
    if n <= 3:
        return "small"
    if n <= 8:
        return "medium"
    return "large"


def _clean_money(value: Any) -> float | None:
    """CORD money is a string like ``"45,500"``; return a float."""
    if value is None:
        return None
    s = re.sub(r"[^\d.\-]", "", str(value))
    if s in ("", "-", "."):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _extract_gold(gt_parse: dict[str, Any]) -> dict[str, Any]:
    total = _clean_money((gt_parse.get("total") or {}).get("total_price"))
    menu = gt_parse.get("menu") or []
    if isinstance(menu, dict):  # CORD sometimes collapses a single item
        menu = [menu]
    line_items = [
        {
            "name": str(item.get("nm", "")).strip(),
            "qty": str(item.get("cnt", "")).strip(),
            "price": _clean_money(item.get("price")),
        }
        for item in menu
        if str(item.get("nm", "")).strip()
    ]
    return {"total": total, "line_items": line_items}


def build_cases(
    parquet_path: str | Path,
    *,
    ocr_cache: dict[int, str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Convert CORD-v2 validation rows into eval cases.

    OCR is the slow part (~1s/image); pass an ``ocr_cache`` dict keyed by row
    index to reuse results across calls.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if limit is not None:
        rows = rows[:limit]
    cache = ocr_cache if ocr_cache is not None else {}

    cases: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if i in cache:
            text = cache[i]
        else:
            image_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
            text = _ocr_image(image_bytes)
            cache[i] = text

        gt = json.loads(row["ground_truth"]) if isinstance(row["ground_truth"], str) else row["ground_truth"]
        gold = _extract_gold(gt.get("gt_parse", {}))

        cases.append(
            {
                "case_id": f"cord-{i:04d}",
                "text": text,
                "gold": gold,
                "strata": {
                    "quality": _quality_stratum(text),
                    "item_count": _item_count_stratum(len(gold["line_items"])),
                    "ocr_signal": _ocr_signal_stratum(text),
                },
            }
        )
    return cases


def write_cases(
    parquet_path: str | Path,
    out_dir: str | Path,
    *,
    limit: int | None = None,
) -> tuple[Path, str]:
    """Build CORD cases and write cases.jsonl; returns (path, content_hash)."""
    cases = build_cases(parquet_path, limit=limit)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "cases.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in cases))
    tmp.replace(path)
    return path, sha256_json(cases)
