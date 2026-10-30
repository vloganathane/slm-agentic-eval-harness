"""Tests for analysis module."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from slm_eval.analysis.dataset import DatasetAnalyzer, DatasetAnalysis
from slm_eval.analysis.error import ErrorAnalyzer, ErrorAnalysis
from slm_eval.analysis.gate import GateAnalyzer, GateAnalysis
from slm_eval.analysis.llm_analyzer import LLMAnalyzer, create_llm_analyzer


def test_dataset_analysis_basic():
    """Test basic dataset analysis functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cases_path = Path(tmpdir) / "cases.jsonl"
        with cases_path.open("w") as f:
            f.write('{"case_id": "c1", "text": "test text", "strata": {"ocr_signal": "usable", "quality": "high"}}\n')
            f.write('{"case_id": "c2", "text": "test text", "strata": {"ocr_signal": "unextractable", "quality": "low"}}\n')
            f.write('{"case_id": "c3", "text": "test text", "strata": {"ocr_signal": "usable", "quality": "medium"}}\n')
        
        analyzer = DatasetAnalyzer(cases_path)
        analysis = analyzer.analyze()
        
        assert analysis.n_cases == 3
        assert analysis.n_fields == 0  # No fields defined in this test
        assert analysis.ocr_signal_dist["usable"] == 2
        assert analysis.ocr_signal_dist["unextractable"] == 1


def test_dataset_analysis_with_fields():
    """Test dataset analysis with fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cases_path = Path(tmpdir) / "cases.jsonl"
        with cases_path.open("w") as f:
            f.write('{"case_id": "c1", "text": "test text", "gold": {"field1": "value"}, "strata": {"ocr_signal": "usable", "quality": "high"}}\n')
            f.write('{"case_id": "c2", "text": "test text", "gold": {"field2": "value"}, "strata": {"ocr_signal": "unextractable", "quality": "low"}}\n')
        
        analyzer = DatasetAnalyzer(cases_path)
        analysis = analyzer.analyze()
        
        assert analysis.n_cases == 2
        assert "field1" in analysis.fields_seen
        assert "field2" in analysis.fields_seen
        assert analysis.ocr_signal_dist["usable"] == 1
        assert analysis.ocr_signal_dist["unextractable"] == 1


def test_dataset_analysis_duplicate_detection():
    """Test duplicate detection with MinHash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cases_path = Path(tmpdir) / "cases.jsonl"
        with cases_path.open("w") as f:
            f.write('{"case_id": "c1", "text": "identical text", "strata": {"ocr_signal": "usable"}}\n')
            f.write('{"case_id": "c2", "text": "identical text", "strata": {"ocr_signal": "usable"}}\n')
            f.write('{"case_id": "c3", "text": "different text", "strata": {"ocr_signal": "usable"}}\n')
        
        analyzer = DatasetAnalyzer(cases_path)
        analysis = analyzer.analyze()
        
        # Should find 1 duplicate pair (c1 and c2)
        assert len(analysis.duplicate_pairs) == 1
        assert analysis.duplicate_pairs[0][2] >= 0.6  # similarity threshold


def test_error_analysis_basic():
    """Test error analysis functionality."""
    # Create mock scores file
    with tempfile.TemporaryDirectory() as tmpdir:
        score_path = Path(tmpdir) / "scores.jsonl"
        with score_path.open("w") as f:
            f.write('{"case_id": "c1", "field": "field1", "tp": 1, "fp": 0, "fn": 0}\n')
            f.write('{"case_id": "c1", "field": "field1", "tp": 0, "fp": 1, "fn": 0}\n')
            f.write('{"case_id": "c2", "field": "field2", "tp": 0, "fp": 0, "fn": 1}\n')
        
        # Create mock cases.jsonl in config parent
        cases_path = Path(tmpdir) / "cases.jsonl"
        with cases_path.open("w") as f:
            f.write('{"case_id": "c1", "strata": {"mode": "test"}}\n')
            f.write('{"case_id": "c2", "strata": {"mode": "test"}}\n')
        
        # Create mock config
        config_path = Path(tmpdir) / "config.yaml"
        with config_path.open("w") as f:
            f.write("task_description: test\n")
            f.write("fields:\n")
            f.write("  field1: {}\n")
            f.write("  field2: {}\n")
            f.write("strata:\n")
            f.write("  keys:\n")
            f.write("    - mode\n")
        
        analyzer = ErrorAnalyzer(score_path, config_path, "test-run")
        analysis = analyzer.analyze()
        
        assert analysis.n_cases == 2
        assert analysis.total_tp == 1
        assert analysis.total_fp == 1
        assert analysis.total_fn == 1
        assert analysis.f1_point > 0.0
        assert analysis.f1_lower >= 0.0
        assert analysis.f1_upper <= 1.0


def test_gate_analysis_basic():
    """Test gate analysis functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gate_path = Path(tmpdir) / "gate.json"
        config_path = Path(tmpdir) / "config.yaml"
        score_path = Path(tmpdir) / "scores.jsonl"
        
        # Create gate.json
        with gate_path.open("w") as f:
            json.dump({
                "status": "blocked",
                "promote": False,
                "metrics": {"micro_f1": 0.5, "ci_lower": 0.3, "ci_upper": 0.7},
                "n": 10,
                "blockers": ["ci_lower_below_target"],
                "warnings": [],
                "strata": [],
                "run_id": "test-run"
            }, f)
            
        # Create scores.jsonl
        with score_path.open("w") as f:
            f.write('{"case_id": "c1", "field": "field1", "tp": 0, "fp": 1, "fn": 0}\n')
            f.write('{"case_id": "c2", "field": "field1", "tp": 1, "fp": 0, "fn": 0}\n')
            
        # Create config.yaml
        with config_path.open("w") as f:
            f.write("task_description: test\n")
            f.write("fields:\n")
            f.write("  field1: {}\n")
            f.write("strata:\n")
            f.write("  keys:\n")
            f.write("    - mode\n")
            
        analyzer = GateAnalyzer(gate_path, config_path, score_path)
        analysis = analyzer.analyze()
        
        assert analysis.passed == False
        assert analysis.micro_f1 == 0.5
        assert analysis.ci_lower == 0.3
        assert analysis.ci_upper == 0.7
        assert analysis.root_cause is not None
        assert len(analysis.remediation) > 0


def test_llm_analyzer_basic():
    """Test LLM analyzer functionality."""
    # This test verifies that our LLM analyzer can be instantiated
    # The actual LLM client requires network access, so we'll test instantiation
    
    analyzer = create_llm_analyzer()
    assert analyzer is not None