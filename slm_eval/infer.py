"""Real inference client: the model under test (SPEC_V3 §5 stage 3).

Separate from the *judge* (slm_eval.llm) because this is the model being measured,
not the instrument measuring it. One prompt per case: extract the configured
fields as JSON. Deterministic (temperature 0) so a rerun scores the same.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from slm_eval.config import EvalConfig
from slm_eval.errors import SlmEvalError
from slm_eval.llm import OpenAICompatClient, parse_json_response

EXTRACT_TEMPLATE = """Extract the following fields from this receipt/invoice.
Return ONLY a JSON object with exactly these keys and no others.
Use an empty string for a field that is not present in the text. Do not invent values.

Fields and their expected types:
{spec}

Text:
{text}

JSON:"""


def _field_spec(cfg: EvalConfig, fname: str) -> str:
    """Describe one field's expected JSON shape.

    A list field with a declared item schema spells out the object shape, which
    is what stops the model from emitting a bare string for it. Scalar types
    come from the field's validator where one is declared.
    """
    schema = cfg.fields.item_schemas.get(fname)
    if schema:
        attrs = ", ".join(f'"{a}": <{t}>' for a, t in schema.items())
        return f'  "{fname}": [{{{attrs}}}]'
    validator = cfg.fields.validators.get(fname)
    if validator is not None and getattr(validator, "type", None) == "numeric":
        return f'  "{fname}": <number>'
    return f'  "{fname}": <string>'


def build_extract_prompt(cfg: EvalConfig, text: str) -> str:
    fields = sorted(set(cfg.fields.exact_match_fields + cfg.fields.subjective_fields))
    spec = ",\n".join(_field_spec(cfg, f) for f in fields)
    return EXTRACT_TEMPLATE.format(spec=spec, text=text)


def infer_case(client: OpenAICompatClient, cfg: EvalConfig, text: str, *, images: list[str] | None = None) -> dict[str, Any]:
    """Run one case. Returns predicted fields; missing/malformed keys → empty string."""
    prompt = build_extract_prompt(cfg, text)
    resp = client.complete(prompt, json_mode=True, images=images)
    try:
        obj = parse_json_response(resp.text)
    except (ValueError, json.JSONDecodeError):
        # Repair once with an explicit reminder, then give up on structure.
        resp = client.complete(
            prompt + "\n\nReturn ONLY the JSON object with the requested keys.",
            json_mode=True,
        )
        try:
            obj = parse_json_response(resp.text)
        except (ValueError, json.JSONDecodeError) as e:
            raise SlmEvalError(f"model returned malformed JSON: {e}") from e

    wanted = set(cfg.fields.exact_match_fields + cfg.fields.subjective_fields)
    return {k: obj.get(k, "") for k in wanted}


def run_inference(
    cfg: EvalConfig,
    holdout: list[dict[str, Any]],
    *,
    base_url: str | None = None,
    model: str = "model",
    cache_path: str | Path | None = None,
    progress: bool = True,
) -> dict[str, dict[str, Any]]:
    """Infer every holdout case; optional on-disk cache keyed by case_id.

    The cache makes reruns/resume free and is written atomically as results land.
    """
    cache: dict[str, dict[str, Any]] = {}
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.exists():
        cache = json.loads(cache_file.read_text())

    client = OpenAICompatClient(
        base_url=base_url or cfg.inference.base_url,
        model=model,
        api_key="local",
        timeout=cfg.inference.timeout_seconds,
        max_attempts=cfg.inference.retry.max_attempts,
        backoff_base=cfg.inference.retry.backoff_base,
    )

    preds: dict[str, dict[str, Any]] = {}
    for i, c in enumerate(holdout):
        cid = c["case_id"]
        if cid in cache:
            preds[cid] = cache[cid]
            continue
        start = time.monotonic()
        preds[cid] = infer_case(client, cfg, c["text"], images=c.get("images"))
        cache[cid] = preds[cid]
        if cache_file:
            cache_file.write_text(json.dumps(cache, sort_keys=True))
        if progress:
            n_fields = sum(
                1 for k, v in preds[cid].items()
                if v not in ("", None, [], {})
            )
            print(
                f"  [{i + 1}/{len(holdout)}] {cid} "
                f"{time.monotonic() - start:.1f}s filled={n_fields}",
                flush=True,
            )
    return preds
