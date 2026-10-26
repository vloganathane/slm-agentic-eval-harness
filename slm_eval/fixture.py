"""Synthetic golden fixture: GST invoices with strata (SPEC_V3 §10 M0).

Deterministic: same seed + same count ⇒ byte-identical ``cases.jsonl``.
Used by the golden regression test (§6.6) and as the dev dataset when the real
invoice data isn't wired in yet.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from slm_eval.hashing import sha256_json

VENDORS = [
    "Apex Polymers Pvt Ltd", "Bharat Logistics Co", "Chandra Textiles",
    "Deccan Steel Works", "Everest Paper Mills", "Fineprint Services",
    "Ganga Electricals", "Hyderabad Fasteners",
]
CITIES = ["Mumbai", "Pune", "Nagpur", "Surat", "Indore", "Kochi"]
LANGUAGES = ["en", "hi", "mr"]
LAYOUTS = ["clean", "rotated_90", "handwritten", "multi_column"]
QUALITIES = ["high", "medium", "low"]

# Field pool per case — exact-match fields
EXACT_FIELDS = ["invoice_number", "date", "total", "tax_id", "line_items"]
SUBJECTIVE_FIELDS = ["vendor_name", "address", "description"]


def _make_case(rng: random.Random, i: int) -> dict[str, Any]:
    layout = rng.choice(LAYOUTS)
    language = rng.choice(LANGUAGES)
    quality = rng.choice(QUALITIES)

    line_items = [
        {"item": f"SKU-{rng.randint(1000, 9999)}", "qty": rng.randint(1, 50),
         "rate": round(rng.uniform(10, 500), 2)}
        for _ in range(rng.randint(1, 5))
    ]
    total = round(sum(li["qty"] * li["rate"] for li in line_items), 2)
    city = rng.choice(CITIES)
    vendor = rng.choice(VENDORS)
    inv_no = f"INV-{2024 + i % 3}-{i:06d}"
    date = f"{2024 + i % 3}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    # 15-char GSTIN: 2-digit state + 5 letters + 4 digits + 1 letter + 1 alnum + Z + 1 alnum
    gstin = (
        rng.choice(["27", "29", "33"])
        + "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))
        + "".join(rng.choice("0123456789") for _ in range(4))
        + rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        + rng.choice("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ")
        + "Z"
        + rng.choice("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ")
    )

    case: dict[str, Any] = {
        "case_id": f"inv-{i:05d}",
        "text": (
            f"TAX INVOICE\nVendor: {vendor}\n"
            f"Ship to: {city}\nInvoice No: {inv_no}\n"
            f"Date: {date}\n"
            f"GSTIN: {gstin}\n"
            + "\n".join(f"  {li['item']} x{li['qty']} @ {li['rate']}" for li in line_items)
            + f"\nTOTAL: {total}"
        ),
        "gold": {
            "invoice_number": inv_no,
            "date": date,
            "total": total,
            "tax_id": gstin,
            "line_items": line_items,
            "vendor_name": vendor,
            "address": f"{city}, India",
            "description": "Supply of goods as per purchase order",
        },
        "strata": {"layout_type": layout, "language": language, "quality": quality},
    }
    return case


def generate(count: int = 500, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return [_make_case(rng, i) for i in range(count)]


def write_dataset(
    out_dir: str | Path, count: int = 500, seed: int = 42
) -> tuple[Path, str]:
    """Write cases.jsonl atomically; returns (path, content_hash)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "cases.jsonl"
    cases = generate(count, seed)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in cases))
    tmp.replace(path)
    return path, sha256_json(cases)


def read_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read cases.jsonl; raises ValueError on malformed content."""
    cases: list[dict[str, Any]] = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: malformed JSON: {e}") from e
    return cases


def iter_strata(cases: list[dict[str, Any]], key: str) -> Iterator[tuple[str, list[dict]]]:
    """Group cases by one stratum key (single_key mode, §6.2)."""
    groups: dict[str, list[dict]] = {}
    for c in cases:
        groups.setdefault(c["strata"][key], []).append(c)
    yield from sorted(groups.items())
