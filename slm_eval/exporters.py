"""Format exporters for fine‑tuning datasets."""

from __future__ import annotations

import json
from typing import Any


class BaseExporter:
    """Base class for format exporters."""

    def format(self, case: dict[str, Any]) -> str:
        raise NotImplementedError


class SFTExporter(BaseExporter):
    """SFT (Simple Fine‑Tuning) format: plain text output."""

    def format(self, case: dict[str, Any]) -> str:
        # Minimal SFT: just the gold fields as a JSON string
        gold = case.get("gold", {})
        return json.dumps(gold, ensure_ascii=False)


class AlpacaExporter(BaseExporter):
    """Alpaca format: instruction / input / output."""

    def format(self, case: dict[str, Any]) -> str:
        gold = case.get("gold", {})
        text = case.get("text", "")

        # Build instruction based on available gold fields
        instruction = "Extract the following fields from the receipt text."
        # Format input as the raw text, output as gold JSON
        output = json.dumps(gold, ensure_ascii=False)

        # Simple alpaca template
        return json.dumps(
            {
                "instruction": instruction,
                "input": text[:512] if text else "",
                "output": output,
            },
            ensure_ascii=False,
        )


class ChatMLExporter(BaseExporter):
    """ChatML format: system + user (text) + assistant (gold JSON)."""

    def format(self, case: dict[str, Any]) -> str:
        gold = case.get("gold", {})
        text = case.get("text", "")

        messages = [
            {"role": "system", "content": "You are a receipt information extraction assistant."},
            {"role": "user", "content": text[:1024] if text else ""},
            {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)},
        ]

        return json.dumps({"messages": messages}, ensure_ascii=False)
