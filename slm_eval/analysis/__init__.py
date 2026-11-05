"""Automatic Data Analysis Agent for SLM Evaluation Harness.

Three analyzers answering different questions at different pipeline stages:

1. DatasetAnalyzer (before eval):  Is this data ready? Duplicates? Defects? Difficulty?
2. ErrorAnalyzer (after scoring):  Why did the model fail? Error patterns? Root causes?
3. GateAnalyzer (after gate):      Why blocked? What changes would flip the decision?

All analyzers return structured JSON for agent consumption.
"""

from __future__ import annotations

from .dataset import DatasetAnalysis, DatasetAnalyzer
from .error import ErrorAnalysis, ErrorAnalyzer
from .gate import GateAnalysis, GateAnalyzer

__all__ = [
    "DatasetAnalysis",
    "DatasetAnalyzer",
    "ErrorAnalysis",
    "ErrorAnalyzer",
    "GateAnalysis",
    "GateAnalyzer",
]
