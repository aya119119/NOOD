"""
Pipeline-level tests: timing metadata, report structure.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Speech Analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestReportStructure:
    """Verify the output report contains expected metadata."""

    @pytest.fixture
    def report_path(self):
        """Check if an existing test report exists (from prior runs)."""
        p = Path(__file__).resolve().parent.parent / "test_optimized_report.json"
        if not p.exists():
            pytest.skip("No test report found — run the pipeline first")
        return p

    def test_report_has_timings(self, report_path):
        with open(report_path) as f:
            report = json.load(f)

        assert "meta" in report
        assert "timings" in report["meta"]
        timings = report["meta"]["timings"]
        assert "total" in timings
        assert "audio_extraction" in timings

    def test_report_has_process_id(self, report_path):
        with open(report_path) as f:
            report = json.load(f)

        assert "process_id" in report["meta"]
        assert isinstance(report["meta"]["process_id"], int)

    def test_report_has_audio_duration(self, report_path):
        with open(report_path) as f:
            report = json.load(f)

        assert "audio_duration_s" in report["meta"]
        assert report["meta"]["audio_duration_s"] > 0

    def test_report_has_speech_detail_timings(self, report_path):
        with open(report_path) as f:
            report = json.load(f)

        speech_detail = report["meta"]["timings"].get("speech_detail", {})
        expected_keys = ["vad_load", "vad_inference", "asr_load",
                         "asr_inference", "prosody_inference"]
        for key in expected_keys:
            assert key in speech_detail, f"Missing timing key: {key}"

    def test_report_has_cold_start_flag(self, report_path):
        with open(report_path) as f:
            report = json.load(f)

        assert "cold_start" in report["meta"]
        assert isinstance(report["meta"]["cold_start"], bool)


class TestVadUtils:
    """Unit tests for VAD utility functions."""

    def test_merge_segments_bridges_gaps(self):
        import torch
        from vad_utils import merge_speech_segments

        boundaries = torch.tensor([
            [1.0, 2.0],
            [2.2, 3.0],   # gap = 0.2s (should merge)
            [5.0, 6.0],   # gap = 2.0s (should NOT merge)
        ])

        merged = merge_speech_segments(boundaries, min_gap=0.3)
        assert len(merged) == 2
        assert merged[0] == (1.0, 3.0)
        assert merged[1] == (5.0, 6.0)

    def test_merge_discards_short_segments(self):
        import torch
        from vad_utils import merge_speech_segments

        boundaries = torch.tensor([
            [1.0, 1.2],   # 0.2s — too short
            [3.0, 5.5],   # 2.5s — keep
        ])

        merged = merge_speech_segments(boundaries, min_duration=0.5)
        assert len(merged) == 1
        assert merged[0] == (3.0, 5.5)

    def test_chunk_for_asr(self):
        from vad_utils import chunk_for_asr

        segments = [(0.0, 45.0), (60.0, 75.0)]
        chunks = chunk_for_asr(segments, max_chunk_s=30)

        # First segment should split into 2
        assert len(chunks) == 3
        assert chunks[0] == (0.0, 30.0)
        assert chunks[1] == (30.0, 45.0)
        assert chunks[2] == (60.0, 75.0)
