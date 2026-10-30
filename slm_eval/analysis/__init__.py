"""Automatic Data Analysis Agent for SLM Evaluation Harness.

Three analyzers answering different questions at different pipeline stages:

1. DatasetAnalyzer (before eval):  Is this data ready? Duplicates? Defects? Difficulty?
2. ErrorAnalyzer (after scoring):  Why did the model fail? Error patterns? Root causes?
3. GateAnalyzer (after gate):      Why blocked? What changes would flip the decision?

All analyzers return structured JSON for agent consumption.
"""

from __future__ import annotations

from .dataset import DatasetAnalyzer, DatasetAnalysis
from .error import ErrorAnalyzer, ErrorAnalysis
from .gate import GateAnalyzer, GateAnalysis

__all__ = [
    "DatasetAnalyzer",
    "DatasetAnalysis",
    "ErrorAnalyzer",
    "ErrorAnalysis",
    "GateAnalyzer",
    "GateAnalysis",
]