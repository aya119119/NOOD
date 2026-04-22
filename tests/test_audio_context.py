"""
Tests for AudioContext: single-load behavior, format correctness.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Speech Analysis"))


@pytest.fixture
def sample_wav(tmp_path):
    """Create a short 16kHz mono WAV for testing."""
    sr = 16000
    duration = 2.0
    t = torch.linspace(0, duration, int(sr * duration))
    waveform = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)  # 440 Hz sine
    path = str(tmp_path / "test_audio.wav")
    torchaudio.save(path, waveform, sr)
    return path, sr, duration


@pytest.fixture
def sample_wav_44k(tmp_path):
    """Create a short 44.1kHz stereo WAV for testing resampling."""
    sr = 44100
    duration = 1.5
    t = torch.linspace(0, duration, int(sr * duration))
    mono = torch.sin(2 * 3.14159 * 440 * t)
    waveform = torch.stack([mono, mono])  # stereo
    path = str(tmp_path / "test_audio_44k.wav")
    torchaudio.save(path, waveform, sr)
    return path, sr, duration


class TestAudioContextLoad:
    """Verify AudioContext loads files correctly."""

    def test_single_load_16k(self, sample_wav):
        from audio_context import AudioContext
        path, sr, duration = sample_wav

        ctx = AudioContext.from_wav(path)

        assert ctx.sr_native == 16000
        assert abs(ctx.duration_s - duration) < 0.1
        assert ctx.waveform_16k.shape[0] == 1
        assert ctx.waveform_16k.shape[1] > 0
        assert isinstance(ctx.waveform_native, np.ndarray)
        assert len(ctx.waveform_native) > 0
        # For 16kHz input, no temp file needed
        assert ctx.wav_16k_path == path

    def test_resampling_44k(self, sample_wav_44k):
        from audio_context import AudioContext
        path, sr, duration = sample_wav_44k

        ctx = AudioContext.from_wav(path)

        assert ctx.sr_native == 44100
        # 16k waveform should be shorter than native
        assert ctx.waveform_16k.shape[1] < len(ctx.waveform_native)
        # Should have created a temp file
        assert ctx.wav_16k_path != path
        assert Path(ctx.wav_16k_path).exists()

        ctx.cleanup()
        assert not Path(ctx.wav_16k_path).exists()

    def test_duration_accuracy(self, sample_wav):
        from audio_context import AudioContext
        path, sr, duration = sample_wav

        ctx = AudioContext.from_wav(path)
        assert abs(ctx.duration_s - duration) < 0.05


class TestNoStaleFiles:
    """Verify cleanup works and no temp files are left behind."""

    def test_cleanup_removes_temp(self, sample_wav_44k):
        from audio_context import AudioContext
        path, _, _ = sample_wav_44k

        ctx = AudioContext.from_wav(path)
        temp_path = ctx.wav_16k_path

        assert Path(temp_path).exists()
        ctx.cleanup()
        assert not Path(temp_path).exists()

    def test_16k_no_temp(self, sample_wav):
        from audio_context import AudioContext
        path, _, _ = sample_wav

        ctx = AudioContext.from_wav(path)
        # cleanup should be a no-op for 16kHz input
        ctx.cleanup()
        assert Path(path).exists()
