"""
Tests for model_registry: singleton behavior, thread safety, device consistency.
"""
import sys
import threading
from pathlib import Path

import pytest

# Ensure the Speech Analysis directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Speech Analysis"))


class TestModelSingleton:
    """Verify that repeated calls to get_*() return the exact same object."""

    def test_singleton_vad(self):
        from model_registry import get_vad
        vad1 = get_vad()
        vad2 = get_vad()
        assert vad1 is vad2, "get_vad() should return same object on repeated calls"

    def test_singleton_asr(self):
        from model_registry import get_asr
        asr1 = get_asr()
        asr2 = get_asr()
        assert asr1 is asr2, "get_asr() should return same object on repeated calls"

    def test_singleton_emotion(self):
        from model_registry import get_emotion
        emo1 = get_emotion()
        emo2 = get_emotion()
        assert emo1 is emo2, "get_emotion() should return same object on repeated calls"


class TestDeviceConsistency:
    """Verify that model params and model.device agree."""

    def test_vad_device(self):
        from model_registry import get_vad, DEVICE
        vad = get_vad()
        assert str(vad.device) == DEVICE
        # Check a parametric module
        for mod in vad.mods.values():
            params = list(mod.parameters())
            if params:
                assert str(params[0].device) == DEVICE

    def test_asr_device(self):
        from model_registry import get_asr, DEVICE
        asr = get_asr()
        for mod in asr.mods.values():
            params = list(mod.parameters())
            if params:
                assert str(params[0].device) == DEVICE


class TestConcurrentInit:
    """Verify no duplicate initialization under concurrent calls."""

    def test_concurrent_vad(self):
        """5 threads call get_vad() simultaneously — all get same object."""
        from model_registry import get_vad

        results = [None] * 5

        def worker(idx):
            results[idx] = get_vad()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for r in results:
            assert r is results[0], "Concurrent calls returned different objects"


class TestModelStatus:
    """Verify model_status() returns expected structure."""

    def test_status_after_warmup(self):
        from model_registry import warmup_models, model_status

        warmup_models()
        status = model_status()

        for name in ("vad", "asr", "emotion"):
            assert name in status
            assert status[name]["loaded"] is True
            assert "cold_load_s" in status[name]
            assert "device" in status[name]
