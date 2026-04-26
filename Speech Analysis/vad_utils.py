"""
vad_utils.py
---------------------------------------------------------------------------
Utilities for working with VAD (Voice Activity Detection) boundaries.

Converts raw VAD output into usable speech regions and provides helpers
to extract speech-only audio for downstream stages (ASR, emotion, prosody).
---------------------------------------------------------------------------
"""

import torch
import numpy as np
from typing import List, Tuple


def merge_speech_segments(
    boundaries: torch.Tensor,
    min_gap: float = 0.3,
    min_duration: float = 0.5,
) -> List[Tuple[float, float]]:
    """
    Merge nearby VAD segments and discard very short ones.

    Parameters
    ----------
    boundaries : torch.Tensor
        Shape (N, 2) — start/end times in seconds from SpeechBrain VAD.
    min_gap : float
        Gaps shorter than this (seconds) are bridged.
    min_duration : float
        Segments shorter than this after merging are discarded.

    Returns
    -------
    List of (start_s, end_s) tuples — merged speech regions.
    """
    if len(boundaries) == 0:
        return []

    segments = [(float(b[0]), float(b[1])) for b in boundaries]
    segments.sort(key=lambda x: x[0])

    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= min_gap:
            # Bridge the gap
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    # Filter out very short segments
    merged = [(s, e) for s, e in merged if (e - s) >= min_duration]
    return merged


def compute_pause_stats(
    merged_segments: List[Tuple[float, float]],
    total_duration: float,
    min_pause: float = 0.15,
) -> Tuple[float, List[float]]:
    """
    Compute pause ratio and individual pause durations from merged segments.

    Returns
    -------
    pause_ratio : float
        Fraction of total duration that is silence (0.0–1.0).
    pauses : list of float
        Individual pause durations in seconds.
    """
    pauses = []
    for i in range(1, len(merged_segments)):
        gap = merged_segments[i][0] - merged_segments[i - 1][1]
        if gap > min_pause:
            pauses.append(gap)

    speech_duration = sum(e - s for s, e in merged_segments)
    silence_duration = total_duration - speech_duration
    pause_ratio = silence_duration / max(total_duration, 1e-9)

    return pause_ratio, pauses


def extract_speech_audio_torch(
    waveform_16k: torch.Tensor,
    sr: int,
    merged_segments: List[Tuple[float, float]],
) -> torch.Tensor:
    """
    Concatenate speech-only regions from a 16 kHz waveform tensor.

    Parameters
    ----------
    waveform_16k : torch.Tensor
        Shape (1, total_samples) at 16 kHz.
    sr : int
        Sample rate (should be 16000).
    merged_segments : list of (start_s, end_s)

    Returns
    -------
    torch.Tensor — shape (1, speech_samples), speech-only audio.
    """
    if not merged_segments:
        return waveform_16k  # fallback: return everything

    chunks = []
    for start_s, end_s in merged_segments:
        s = int(start_s * sr)
        e = int(end_s * sr)
        e = min(e, waveform_16k.shape[1])
        if s < e:
            chunks.append(waveform_16k[:, s:e])

    if not chunks:
        return waveform_16k

    return torch.cat(chunks, dim=1)


def extract_speech_audio_numpy(
    waveform: np.ndarray,
    sr: int,
    merged_segments: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Concatenate speech-only regions from a numpy waveform (any sr).

    Parameters
    ----------
    waveform : np.ndarray
        Shape (total_samples,) — mono audio.
    sr : int
        Sample rate of this waveform.
    merged_segments : list of (start_s, end_s)

    Returns
    -------
    np.ndarray — shape (speech_samples,), speech-only audio.
    """
    if not merged_segments:
        return waveform

    chunks = []
    for start_s, end_s in merged_segments:
        s = int(start_s * sr)
        e = int(end_s * sr)
        e = min(e, len(waveform))
        if s < e:
            chunks.append(waveform[s:e])

    if not chunks:
        return waveform

    return np.concatenate(chunks)


def chunk_for_asr(
    merged_segments: List[Tuple[float, float]],
    max_chunk_s: float = 30.0,
) -> List[Tuple[float, float]]:
    """
    Split long speech regions into bounded windows for ASR.

    Preserves segment ordering.  Regions already shorter than max_chunk_s
    are passed through unchanged.

    Returns
    -------
    List of (start_s, end_s) tuples — each ≤ max_chunk_s long.
    """
    chunks = []
    for start, end in merged_segments:
        duration = end - start
        if duration <= max_chunk_s:
            chunks.append((start, end))
        else:
            pos = start
            while pos < end:
                chunk_end = min(pos + max_chunk_s, end)
                chunks.append((pos, chunk_end))
                pos = chunk_end
    return chunks
