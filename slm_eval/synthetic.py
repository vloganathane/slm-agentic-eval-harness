"""Synthetic case generation with MinHash deduplication (M3).

Generates novel cases via LLM, filters by MinHash similarity against holdout,
routes through human review queue, then injects into training set.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_eval.hashing import minhash_signature, minhash_similarity
from slm_eval.llm import LLMClient, OpenAICompatClient, WebEnabledClient, WebSearchProvider
from slm_eval.store import Store


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    text: str
    gold: dict[str, Any]
    strata: dict[str, str]
    source: str = "synthetic"
    minhash: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    proposed: int
    novel: int
    duplicate: int
    cases: list[SyntheticCase]


# Default merchant/item data for fallback when web search unavailable
DEFAULT_MERCHANTS = [
    {"name": "WALMART", "categories": ["grocery", "retail"], "items": [("Milk", 2.99), ("Bread", 1.99), ("Eggs", 3.49), ("Chicken", 5.99), ("Rice", 2.49)]},
    {"name": "TARGET", "categories": ["retail", "grocery"], "items": [("Chicken", 5.99), ("Rice", 2.49), ("Detergent", 8.99), ("Toothpaste", 3.49)]},
    {"name": "STARBUCKS", "categories": ["cafe"], "items": [("Latte", 4.95), ("Croissant", 3.50), ("Muffin", 2.95), ("Americano", 3.25)]},
    {"name": "HOME DEPOT", "categories": ["hardware"], "items": [("Hammer", 12.99), ("Nails", 3.49), ("Screwdriver", 7.99), ("Tape Measure", 5.99)]},
    {"name": "CVS PHARMACY", "categories": ["pharmacy"], "items": [("Aspirin", 5.99), ("Bandages", 2.99), ("Vitamins", 12.99), ("Cough Syrup", 7.99)]},
    {"name": "MCDONALDS", "categories": ["fast_food"], "items": [("Big Mac", 4.99), ("Large Fries", 2.49), ("Coke", 1.99), ("McChicken", 3.99)]},
    {"name": "BEST BUY", "categories": ["electronics"], "items": [("USB Cable", 8.99), ("Phone Case", 14.99), ("Headphones", 29.99), ("Charger", 19.99)]},
    {"name": "SUBWAY", "categories": ["fast_food"], "items": [("Turkey Sub", 6.99), ("Chips", 1.50), ("Water", 1.25), ("Cookie", 0.99)]},
    {"name": "WHOLE FOODS", "categories": ["grocery"], "items": [("Organic Milk", 4.99), ("Avocado", 1.99), ("Salmon", 12.99), ("Quinoa", 6.99)]},
    {"name": "COSTCO", "categories": ["warehouse"], "items": [("Rotisserie Chicken", 4.99), ("Toilet Paper", 24.99), ("Gasoline", 3.49), ("Hot Dog", 1.50)]},
]


def _load_holdout_minhashes(holdout_dir: Path) -> list[tuple[str, str]]:
    """Load (case_id, minhash) for all holdout cases."""
    out = []
    # Check cases.jsonl first (pipeline format)
    cases_file = holdout_dir / "cases.jsonl"
    if cases_file.exists():
        for line in cases_file.open():
            case = json.loads(line)
            text = case.get("text", "")
            if text:
                sig = minhash_signature(text)
                mh = "".join(str(x) for x in sig[:16])
                out.append((case["case_id"], mh))
    else:
        # Fallback to individual .json files
        for p in sorted(holdout_dir.glob("*.json")):
            case = json.loads(p.read_text())
            text = case.get("text", "")
            if text:
                sig = minhash_signature(text)
                mh = "".join(str(x) for x in sig[:16])
                out.append((case["case_id"], mh))
    return out


def _case_minhash(case: SyntheticCase) -> str:
    sig = minhash_signature(case.text)
    return "".join(str(x) for x in sig[:16])


def _fetch_real_merchants(client: LLMClient | WebSearchProvider | None, cfg) -> list[dict]:
    """Fetch real merchant/item data from web search or use defaults."""
    merchants = DEFAULT_MERCHANTS.copy()
    
    if client is None:
        return merchants
    
    # Try to get real merchant data via web search
    try:
        # Search for current popular merchants and items
        queries = [
            "popular retail stores receipt items 2024",
            "common grocery store receipt format items prices",
            "restaurant receipt format menu items prices",
        ]
        
        for query in queries:
            if hasattr(client, 'web_search'):
                results = client.web_search(query, max_results=3)
            elif isinstance(client, WebSearchProvider):
                results = client.search(query, max_results=3)
            else:
                continue
                
            # Parse results to extract merchant/item info (simplified)
            for r in results:
                content = r.get("content", "").lower()
                # Look for merchant names in content
                for merchant in DEFAULT_MERCHANTS:
                    if merchant["name"].lower() in content and merchant not in merchants:
                        merchants.append(merchant)
    except Exception:  # noqa: BLE001, S110 - web search failure, fall back to defaults
        pass  # Fall back to defaults
    
    return merchants


def _build_prompt_from_real_data(merchants: list[dict], n_propose: int) -> str:
    """Build a prompt using real merchant data for the LLM."""
    merchant_lines = []
    for m in merchants[:10]:  # Limit to top 10
        items_str = ", ".join([f"{item[0]} ${item[1]:.2f}" for item in m["items"][:5]])
        merchant_lines.append(f"- {m['name']}: {items_str}")
    
    merchant_list = "\n".join(merchant_lines)
    
    return f"""Generate {n_propose} diverse receipt/invoice OCR text samples for training an extraction model.

Use these real-world merchants and items as inspiration (but vary them):
{merchant_list}

Each sample should be a realistic receipt or invoice with varied:
- Merchants (from above or similar real businesses)
- Formats (itemized, simple, with discounts/taxes)
- Quality (clean, noisy OCR artifacts, truncated)

Return as JSON array of objects with fields:
- text: the OCR text (realistic, 200-800 chars)
- gold: extracted fields (total, line_items array of {{name, qty, price}}, date, merchant)
- strata: {{quality: "high|medium|low", layout: "itemized|simple|mixed"}}

Example:
{{
  "text": "WALMART\\n123 Main St\\nITEM: Milk 2.99\\nITEM: Bread 1.99\\nTOTAL: 4.98",
  "gold": {{"total": 4.98, "line_items": [{{"name": "Milk", "qty": 1, "price": 2.99}}, {{"name": "Bread", "qty": 1, "price": 1.99}}], "date": "2024-01-15", "merchant": "WALMART"}},
  "strata": {{"quality": "high", "layout": "itemized"}}
}}

Return ONLY the JSON array.""".strip()


def _generate_with_llm(client: LLMClient, prompt: str, temperature: float = 0.8, timeout_seconds: int = 30) -> list[dict] | None:
    """Generate synthetic cases using LLM with web-enhanced prompt.
    
    Uses a short timeout to fall back quickly to template generation if LLM is slow.
    """
    import queue
    import threading
    
    result_queue = queue.Queue()
    
    def worker():
        try:
            full_prompt = f"System: You generate realistic receipt/invoice OCR samples for ML training. Output ONLY a JSON array.\n\nUser: {prompt}"
            resp = client.complete(full_prompt, temperature=temperature, json_mode=True)
            proposed = json.loads(resp.text)
            if isinstance(proposed, list):
                result_queue.put(proposed)
            else:
                result_queue.put(None)
        except Exception:  # noqa: BLE001 - LLM failure, fall back to templates
            result_queue.put(None)
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        # Timeout - fall back to templates
        return None
    
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return None


def _generate_from_templates(merchants: list[dict], n_propose: int) -> list[dict]:
    """Generate cases from templates using real merchant data (fallback)."""
    templates = []
    for merchant in merchants[:n_propose]:
        name = merchant["name"]
        items = merchant["items"]
        # Create varied formats
        layout = random.choice(["itemized", "simple", "mixed"])
        quality = random.choice(["high", "medium", "low"])
        
        if layout == "itemized":
            lines = [f"{name}\n123 Main St"]
            for item_name, price in items[:random.randint(2, 4)]:
                qty = random.randint(1, 3)
                lines.append(f"ITEM: {item_name} {price:.2f} x{qty}")
            total = sum(price * random.randint(1, 3) for _, price in items[:3])
            lines.append(f"TOTAL: {total:.2f}")
            text = "\n".join(lines)
        elif layout == "simple":
            lines = [name]
            for item_name, price in items[:random.randint(1, 3)]:
                lines.append(f"{item_name} ${price:.2f}")
            total = sum(price for _, price in items[:2])
            lines.append(f"TOTAL: {total:.2f}")
            text = "\n".join(lines)
        else:  # mixed
            lines = [f"{name}\n456 Oak Ave"]
            for item_name, price in items[:random.randint(2, 3)]:
                lines.append(f"PROD: {item_name} {price:.2f}")
            tax = sum(price for _, price in items[:2]) * 0.08
            total = sum(price for _, price in items[:2]) + tax
            lines.append(f"TAX: {tax:.2f}")
            lines.append(f"TOTAL: {total:.2f}")
            text = "\n".join(lines)
        
        # Build gold
        line_items = [{"name": item[0], "qty": random.randint(1, 3), "price": item[1]} for item in items[:random.randint(1, 4)]]
        total = sum(li["qty"] * li["price"] for li in line_items)
        
        from datetime import date
        today = date.today().isoformat()
        
        templates.append({
            "text": text,
            "gold": {"total": round(total, 2), "line_items": line_items, "date": today, "merchant": name},
            "strata": {"quality": quality, "layout": layout}
        })
    
    return templates


def generate_synthetic_cases(
    cfg,
    holdout_dir: Path,
    n_propose: int,
    minhash_threshold: float = 0.6,
    client: LLMClient | None = None,
    model: str | None = None,
    temperature: float = 0.8,
    use_web: bool = False,
    web_provider: str = "tavily",
    web_api_key: str | None = None,
) -> GenerationResult:
    """Generate synthetic cases, deduplicate via MinHash against holdout.
    
    If use_web=True and client supports web_search, fetches real merchant data first.
    """
    # Create web-enabled client if requested
    if use_web and client is None:
        client = WebEnabledClient(
            base_url=cfg.inference.base_url,
            model=model or "synthetic-generator",
            api_key="local",
            timeout=30,  # Short timeout for synthetic generation
            web_search_provider=web_provider,
            web_search_api_key=web_api_key,
        )
    elif client is None:
        client = OpenAICompatClient(
            base_url=cfg.inference.base_url,
            model=model or "synthetic-generator",
            api_key="local",
            timeout=30,  # Short timeout for synthetic generation
        )

    holdout_minhashes = _load_holdout_minhashes(holdout_dir)
    
    # Fetch real merchant data
    merchants = _fetch_real_merchants(client, cfg)
    
    # Try LLM generation first
    proposed_data = None
    if hasattr(client, 'complete'):
        prompt = _build_prompt_from_real_data(merchants, n_propose)
        proposed_data = _generate_with_llm(client, prompt, temperature)
    
    # Fallback to template-based generation
    if proposed_data is None:
        proposed_data = _generate_from_templates(merchants, n_propose)
    
    # Limit to n_propose
    proposed_data = proposed_data[:n_propose]

    novel_cases = []
    duplicate = 0

    for i, item in enumerate(proposed_data):
        text = item.get("text", "").strip()
        if not text or len(text) < 50:
            continue

        case = SyntheticCase(
            case_id=f"synth_{hashlib.md5(text.encode()).hexdigest()[:8]}",
            text=text,
            gold=item.get("gold", {}),
            strata=item.get("strata", {"quality": "medium", "layout": "mixed"}),
        )

        mh = _case_minhash(case)
        case = SyntheticCase(
            case_id=case.case_id,
            text=case.text,
            gold=case.gold,
            strata=case.strata,
            minhash=mh,
        )

        is_dup = False
        for hid, hmh in holdout_minhashes:
            sim = minhash_similarity(mh, hmh)
            if sim >= minhash_threshold:
                is_dup = True
                break

        if is_dup:
            duplicate += 1
        else:
            novel_cases.append(case)

    return GenerationResult(
        proposed=len(proposed_data),
        novel=len(novel_cases),
        duplicate=duplicate,
        cases=novel_cases,
    )


def inject_synthetic_cases(
    store: Store,
    run_id: str,
    cases: list[SyntheticCase],
    reviewer: str = "synthetic_pipeline",
) -> int:
    """Inject approved synthetic cases as review tasks for human approval."""
    if not cases:
        return 0

    rows = []
    for c in cases:
        rows.append({
            "run_id": run_id,
            "case_id": c.case_id,
            "reason": "synthetic_injection",
            "assigned_at": store._now() if hasattr(store, "_now") else "",
            "due_at": None,
        })

    inserted = store.insert_review_tasks(run_id, rows)

    for c in cases:
        store.add_review_comment(
            run_id, c.case_id, "synthetic_injection",
            reviewer,
            f"Synthetic case proposed for injection. Gold: {json.dumps(c.gold)}. Strata: {json.dumps(c.strata)}"
        )

    return inserted


def resolve_synthetic_review(
    store: Store,
    run_id: str,
    case_id: str,
    decision: str,
    notes: str = "",
) -> bool:
    """Resolve a synthetic injection review. Returns True if approved for injection."""
    store.update_review_task(run_id, case_id, "synthetic_injection", decision, notes)
    return decision == "approve"


# Convenience CLI
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate synthetic cases")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--holdout-dir", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--model")
    args = p.parse_args()

    from slm_eval.config import EvalConfig
    cfg = EvalConfig.from_yaml(args.config)

    result = generate_synthetic_cases(cfg, Path(args.holdout_dir), args.n, args.threshold, model=args.model)

    print(f"Proposed: {result.proposed}, Novel: {result.novel}, Duplicate: {result.duplicate}")

    out_path = Path(args.run_dir) / "synthetic_proposals.jsonl"
    with out_path.open("w") as f:
        for c in result.cases:
            f.write(json.dumps({
                "case_id": c.case_id,
                "text": c.text,
                "gold": c.gold,
                "strata": c.strata,
                "minhash": c.minhash,
            }) + "\n")

    print(f"Written to {out_path}")