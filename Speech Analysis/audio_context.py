"""
audio_context.py
---------------------------------------------------------------------------
Single-load audio context shared across all analysis stages.

Eliminates the previous pattern of loading the same WAV 4× via different
loaders (ffmpeg → librosa → torchaudio → _16k_tmp.wav).

Usage
-----
    ctx = AudioContext.from_wav("/tmp/audio.wav")
    # ctx.waveform_16k  -> torch.Tensor (1, samples) @ 16 kHz
    # ctx.waveform_native -> np.ndarray (samples,) @ native sr
    # ctx.sr_native      -> int
    # ctx.duration_s     -> float
---------------------------------------------------------------------------
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
import torchaudio


@dataclass
class AudioContext:
    """Immutable audio data loaded once per job."""

    source_path: str                    # path to the original extracted WAV
    waveform_16k: torch.Tensor          # (1, samples) @ 16 kHz mono — for SpeechBrain
    waveform_native: np.ndarray         # (samples,) @ native sr — for librosa/prosody
    sr_native: int
    duration_s: float

    # SpeechBrain VAD.get_speech_segments() requires a file path.
    # This holds the path to pass (the original WAV if already 16 kHz,
    # or a temp copy otherwise).
    wav_16k_path: str = ""

    @classmethod
    def from_wav(cls, wav_path: str) -> "AudioContext":
        """
        Load a WAV file once and build both representations in-memory.

        If the file is already 16 kHz mono (as produced by our ffmpeg
        extraction), no resampling is needed and wav_16k_path == wav_path.
        """
        path = Path(wav_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        # ── Primary load via torchaudio (efficient, returns tensor) ──────
        waveform, sr = torchaudio.load(wav_path)

        # Stereo → mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Derive 16 kHz tensor
        if sr == 16000:
            waveform_16k = waveform
            wav_16k_path = wav_path
        else:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=16000
            )
            waveform_16k = resampler(waveform)
            # SpeechBrain VAD needs a file path — write one temp WAV
            import tempfile
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", prefix="nood_16k_", dir=str(path.parent),
                delete=False,
            )
            torchaudio.save(tmp.name, waveform_16k, 16000)
            wav_16k_path = tmp.name

        # ── Native-sr numpy array for librosa/prosody ────────────────────
        waveform_native = waveform.squeeze(0).numpy().astype(np.float32)

        duration_s = len(waveform_native) / sr

        return cls(
            source_path=wav_path,
            waveform_16k=waveform_16k,
            waveform_native=waveform_native,
            sr_native=sr,
            duration_s=duration_s,
            wav_16k_path=wav_16k_path,
        )

    def cleanup(self):
        """Remove any temp files created during loading."""
        if self.wav_16k_path and self.wav_16k_path != self.source_path:
            Path(self.wav_16k_path).unlink(missing_ok=True)
