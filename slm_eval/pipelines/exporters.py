"""Domain-agnostic exporters for fine-tuning formats.

Supports SFT, Alpaca, ChatML, ShareGPT, and OpenAI chat formats.
Templates are configurable for any domain.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class BaseExporter(ABC):
    """Base class for format exporters."""

    def __init__(self, template: dict[str, Any] | None = None):
        self.template = template or {}

    @abstractmethod
    def format(self, case: dict[str, Any]) -> str:
        """Format a single case for fine-tuning."""
        pass


class SFTExporter(BaseExporter):
    """SFT (Simple Fine-Tuning) format: plain text output or JSON.

    Default: outputs gold as JSON string.
    Can be configured with a custom template.
    """

    def format(self, case: dict[str, Any]) -> str:
        gold = case.get("gold", {})
        text = case.get("text", "")

        # Default: just the gold as JSON
        if "output_template" not in self.template:
            return json.dumps(gold, ensure_ascii=False)

        # Custom template (e.g., "{text}\n\n### Response:\n{gold}")
        output = self.template["output_template"]
        # Prepare format variables with defaults
        fmt_vars = {"text": text, "gold": json.dumps(gold, ensure_ascii=False), **gold}
        for key, value in self.template.get("defaults", {}).items():
            if key not in fmt_vars:
                fmt_vars[key] = value.format(**fmt_vars) if isinstance(value, str) else value
        return output.format(**fmt_vars)


class AlpacaExporter(BaseExporter):
    """Alpaca format: instruction / input / output.

    Configurable instruction template and field mapping.
    """

    def format(self, case: dict[str, Any]) -> str:
        gold = case.get("gold", {})
        text = case.get("text", "")

        # Default instruction (configurable)
        instruction = self.template.get(
            "instruction",
            "Extract the relevant information from the following text."
        )

        # Default input is the text (configurable truncation)
        input_text = text
        if "max_input_length" in self.template:
            input_text = text[:self.template["max_input_length"]]

        # Default output is gold as JSON
        output = json.dumps(gold, ensure_ascii=False)
        if "output_template" in self.template:
            # Prepare format variables with defaults
            fmt_vars = {"text": text, "gold": json.dumps(gold, ensure_ascii=False), **gold}
            for key, value in self.template.get("defaults", {}).items():
                if key not in fmt_vars:
                    fmt_vars[key] = value.format(**fmt_vars) if isinstance(value, str) else value
            output = self.template["output_template"].format(**fmt_vars)

        record = {
            "instruction": instruction,
            "input": input_text,
            "output": output,
        }

        # Add any extra fields from template
        for key, value in self.template.get("extra_fields", {}).items():
            record[key] = value.format(text=text, gold=gold, **gold) if isinstance(value, str) else value

        return json.dumps(record, ensure_ascii=False)


class ChatMLExporter(BaseExporter):
    """ChatML format: system + user + assistant messages.

    Configurable system prompt and message templates.
    """

    def format(self, case: dict[str, Any]) -> str:
        gold = case.get("gold", {})
        text = case.get("text", "")

        # Configurable system prompt
        system_prompt = self.template.get(
            "system_prompt",
            "You are a helpful assistant that extracts structured information from text."
        )

        # Configurable user prompt template
        user_template = self.template.get(
            "user_template",
            "{text}"
        )

        # Configurable assistant response template
        assistant_template = self.template.get(
            "assistant_template",
            "{gold}"
        )

        # Prepare format variables with defaults
        fmt_vars = {"text": text, "gold": json.dumps(gold, ensure_ascii=False), **gold}
        for key, value in self.template.get("defaults", {}).items():
            if key not in fmt_vars:
                fmt_vars[key] = value.format(**fmt_vars) if isinstance(value, str) else value

        user_content = user_template.format(**fmt_vars)
        assistant_content = assistant_template.format(**fmt_vars)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        # Add extra messages if configured
        for msg in self.template.get("extra_messages", []):
            messages.append(msg)

        return json.dumps({"messages": messages}, ensure_ascii=False)


class ShareGPTExporter(BaseExporter):
    """ShareGPT format: conversations with human/gpt roles.

    Popular for chat fine-tuning (e.g., LLaMA-Factory, FastChat).
    """

    def format(self, case: dict[str, Any]) -> str:
        gold = case.get("gold", {})
        text = case.get("text", "")

        # Configurable templates
        human_template = self.template.get(
            "human_template",
            "{text}"
        )
        gpt_template = self.template.get(
            "gpt_template",
            "{gold}"
        )

        # Prepare format variables with defaults
        fmt_vars = {"text": text, "gold": json.dumps(gold, ensure_ascii=False), **gold}
        # Add defaults from template
        for key, value in self.template.get("defaults", {}).items():
            if key not in fmt_vars:
                fmt_vars[key] = value.format(**fmt_vars) if isinstance(value, str) else value

        conversations = [
            {"from": "human", "value": human_template.format(**fmt_vars)},
            {"from": "gpt", "value": gpt_template.format(**fmt_vars)},
        ]

        # Add system prompt if configured
        if "system_prompt" in self.template:
            conversations.insert(0, {"from": "system", "value": self.template["system_prompt"]})

        return json.dumps({"conversations": conversations}, ensure_ascii=False)


class OpenAIChatExporter(BaseExporter):
    """OpenAI Chat Completions format.

    Matches the OpenAI API format for chat completions training.
    """

    def format(self, case: dict[str, Any]) -> str:
        gold = case.get("gold", {})
        text = case.get("text", "")

        system_prompt = self.template.get(
            "system_prompt",
            "You are a helpful assistant that extracts structured information from text."
        )

        user_template = self.template.get("user_template", "{text}")
        assistant_template = self.template.get("assistant_template", "{gold}")

        # Prepare format variables with defaults
        fmt_vars = {"text": text, "gold": json.dumps(gold, ensure_ascii=False), **gold}
        for key, value in self.template.get("defaults", {}).items():
            if key not in fmt_vars:
                fmt_vars[key] = value.format(**fmt_vars) if isinstance(value, str) else value

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template.format(**fmt_vars)},
            {"role": "assistant", "content": assistant_template.format(**fmt_vars)},
        ]

        return json.dumps({"messages": messages}, ensure_ascii=False)


class CompletionExporter(BaseExporter):
    """Plain completion format: prompt → completion.

    For causal LM fine-tuning without chat template.
    """

    def format(self, case: dict[str, Any]) -> str:
        gold = case.get("gold", {})
        text = case.get("text", "")

        prompt_template = self.template.get("prompt_template", "{text}\n\n### Response:\n")
        completion_template = self.template.get("completion_template", "{gold}")

        # Prepare format variables with defaults
        fmt_vars = {"text": text, "gold": json.dumps(gold, ensure_ascii=False), **gold}
        for key, value in self.template.get("defaults", {}).items():
            if key not in fmt_vars:
                fmt_vars[key] = value.format(**fmt_vars) if isinstance(value, str) else value

        prompt = prompt_template.format(**fmt_vars)
        completion = completion_template.format(**fmt_vars)

        return json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False)


# Registry of available exporters
EXPORTERS = {
    "sft": SFTExporter,
    "alpaca": AlpacaExporter,
    "chatml": ChatMLExporter,
    "sharegpt": ShareGPTExporter,
    "openai": OpenAIChatExporter,
    "completion": CompletionExporter,
}


def get_exporter(fmt: str, template: dict[str, Any] | None = None) -> BaseExporter:
    """Get exporter instance by format name."""
    if fmt not in EXPORTERS:
        raise ValueError(f"Unsupported export format: {fmt}. Available: {list(EXPORTERS.keys())}")
    return EXPORTERS[fmt](template)


def list_formats() -> list[str]:
    """List available export formats."""
    return list(EXPORTERS.keys())


# Default templates for common domains
DEFAULT_TEMPLATES = {
    "receipt_extraction": {
        "system_prompt": "You are a receipt information extraction assistant.",
        "user_template": "Extract the following fields from the receipt text:\n\n{text}",
        "gpt_template": "{gold}",
        "instruction": "Extract the following fields from the receipt text.",
        "output_template": "{gold}",
    },
    "qa": {
        "system_prompt": "You are a helpful assistant that answers questions based on the provided context.",
        "user_template": "Context: {text}\n\nQuestion: {question}",
        "gpt_template": "{answer}",
        "instruction": "Answer the question based on the context.",
        # Provide defaults for missing fields
        "defaults": {"question": "What is the answer?", "answer": "{gold}"},
    },
    "summarization": {
        "system_prompt": "You are a helpful assistant that summarizes text.",
        "user_template": "Summarize the following text:\n\n{text}",
        "gpt_template": "{summary}",
        "instruction": "Summarize the following text.",
        "output_template": "{summary}",
        "defaults": {"summary": "{gold}"},
    },
    "classification": {
        "system_prompt": "You are a text classification assistant.",
        "user_template": "Classify the following text:\n\n{text}",
        "gpt_template": "{label}",
        "instruction": "Classify the following text.",
        "output_template": "{label}",
        "defaults": {"label": "{gold}"},
    },
    "generic": {
        "system_prompt": "You are a helpful assistant.",
        "user_template": "{text}",
        "gpt_template": "{gold}",
        "instruction": "Process the following text.",
        "output_template": "{gold}",
    },
}
