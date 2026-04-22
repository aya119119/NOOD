"""
---------------------------------------------------------------------------
Speech performance analyzer.

Analyse ---> WPM, filler words, tone and energy variance

Usage:
    python speech_analyzer.py path/to/audio.wav
    python speech_analyzer.py path/to/audio.wav --segment-duration 30
    python speech_analyzer.py path/to/audio.wav --json
 
Dependencies:
    pip install speechbrain librosa numpy scipy torch torchaudio
 
Models downloaded automatically on first run (~1-2 GB total):
    • speechbrain/vad-crdnn-libriparty          (VAD / pause detection)
    • speechbrain/asr-crdnn-rnnlm-librispeech   (ASR / transcript)
    • speechbrain/emotion-recognition-wav2vec2-IEMOCAP  (vocal emotion)
---------------------------------------------------------------------------
"""
 
import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
import torchaudio
from scipy.interpolate import interp1d
 
warnings.filterwarnings("ignore")
 
# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class Marker:
    """A single performance marker with score, raw value, and human feedback."""
    score: float          # normalised −1.0 → 1.0
    raw: float            # raw measurement in natural units
    unit: str             # e.g. "wpm", "%", "Hz std-dev"
    label: str            # short human label
    feedback: str         # one-line actionable note
 

@dataclass
class SpeechReport:
    overall: float        # weighted composite −1.0 → 1.0
    grade: str            # letter grade A–F
    wpm: Marker
    filler_rate: Marker
    pitch_variation: Marker
    energy_variation: Marker
    pause_ratio: Marker
    vocal_emotion: Marker
    transcript_preview: str
    segments: list        # per-segment breakdowns (optional)
    timings: dict = None  # per-stage timing breakdown
 

# ─── Scorer helpers ──────────────────────────────────────────────────────────

def tanh_score(value: float, ideal: float, std: float, higher_is_better: bool = True) -> float:
    """
    Maps `value` to [−1, 1].
    • When value == ideal  → 0.0  (neutral / average)
    • Deviating from ideal decreases score symmetrically.
    • For one-sided metrics (higher_is_better=False), ideal should be 0.
    """
    deviation = (value - ideal) / max(std, 1e-9)
    if higher_is_better:
        return float(np.tanh(deviation))
    else:
        # Penalise anything above ideal (e.g. filler rate — lower is always better)
        return float(np.tanh(-deviation))
 

def bell_score(value: float, ideal: float, std: float) -> float:
    """
    Centred Gaussian bell: +1.0 at the ideal, smoothly decays toward −1.0.
    Symmetric and much more forgiving than the old tanh variant.
    """
    z = (value - ideal) / max(std, 1e-9)
    return float(np.exp(-0.5 * z * z) * 2 - 1)
 

def grade(score: float) -> str:
    thresholds = [(0.75, "A"), (0.45, "B"), (0.10, "C"), (-0.25, "D")]
    for threshold, letter in thresholds:
        if score >= threshold:
            return letter
    return "F"
 

def feedback_wpm(wpm: float) -> str:
    if wpm < 100:
        return f"Very slow ({wpm:.0f} wpm) — pick up the pace to maintain audience engagement."
    if wpm < 120:
        return f"Slightly slow ({wpm:.0f} wpm) — aim for 130–160 wpm."
    if wpm <= 165:
        return f"Good pace ({wpm:.0f} wpm) — comfortably in the ideal range."
    if wpm <= 190:
        return f"Slightly fast ({wpm:.0f} wpm) — slow down for clarity."
    return f"Too fast ({wpm:.0f} wpm) — audience will struggle to follow."
 

def feedback_fillers(rate: float) -> str:
    pct = rate * 100
    if pct < 1.0:
        return f"Excellent — almost no filler words ({pct:.1f}%)."
    if pct < 3.0:
        return f"Good — low filler usage ({pct:.1f}%)."
    if pct < 6.0:
        return f"Moderate filler usage ({pct:.1f}%) — practise pausing instead of filling."
    return f"High filler usage ({pct:.1f}%) — significantly undermines credibility."
 

def feedback_pitch(std_hz: float) -> str:
    if std_hz < 8:
        return f"Monotone delivery (pitch σ={std_hz:.1f} Hz) — vary your intonation more."
    if std_hz < 18:
        return f"Limited pitch variation ({std_hz:.1f} Hz) — add more rises/falls for emphasis."
    if std_hz <= 45:
        return f"Good pitch variation ({std_hz:.1f} Hz) — dynamic and engaging."
    return f"Very high pitch variation ({std_hz:.1f} Hz) — may sound erratic; aim for intentional changes."
 

def feedback_energy(std_rms: float) -> str:
    if std_rms < 0.008:
        return f"Flat energy (RMS σ={std_rms:.4f}) — punch key words with more volume."
    if std_rms < 0.018:
        return f"Some energy variation ({std_rms:.4f}) — add more deliberate emphasis."
    if std_rms <= 0.045:
        return f"Good energy variation ({std_rms:.4f}) — natural emphasis on key points."
    return f"Very uneven energy ({std_rms:.4f}) — smooth out erratic volume changes."
 

def feedback_pause(ratio: float) -> str:
    pct = ratio * 100
    if pct < 5:
        return f"Almost no pauses ({pct:.1f}%) — use silence strategically for impact."
    if pct < 10:
        return f"Few pauses ({pct:.1f}%) — allow more breathing room."
    if pct <= 22:
        return f"Good pause usage ({pct:.1f}%) — confident and measured delivery."
    return f"Excessive pausing ({pct:.1f}%) — may signal hesitation; tighten up."
 

EMOTION_FEEDBACK = {
    "hap": ("Positive / engaged",  0.8,  "Vocal tone sounds engaged and positive — great for audience connection."),
    "neu": ("Neutral",              0.0,  "Neutral vocal tone — consider adding more warmth and enthusiasm."),
    "ang": ("Tense / aggressive",  -0.4,  "Vocal tone sounds tense — try to relax and lower your larynx."),
    "sad": ("Flat / disengaged",   -0.7,  "Vocal tone sounds flat or disengaged — project more energy."),
}


# ─── Filler words ────────────────────────────────────────────────────────────

FILLER_WORDS = {
    "um", "uh", "like", "basically", "literally", "right",
    "okay", "so", "you know", "actually", "honestly",
    "i mean", "kind of", "sort of", "just",
}


# ─── Stage functions (operate on shared context) ─────────────────────────────

def _run_vad(ctx: dict):
    """Stage 1: VAD → speech boundaries + pause stats."""
    from model_registry import get_vad
    from vad_utils import merge_speech_segments, compute_pause_stats

    t0 = time.time()
    vad = get_vad()
    ctx["timings"]["vad_load"] = round(time.time() - t0, 3)

    t0 = time.time()
    boundaries = vad.get_speech_segments(ctx["wav_16k_path"])
    ctx["timings"]["vad_inference"] = round(time.time() - t0, 3)

    # Merge and compute pauses
    merged = merge_speech_segments(boundaries)
    pause_ratio, pauses = compute_pause_stats(
        merged, ctx["duration_s"]
    )

    ctx["raw_boundaries"] = boundaries
    ctx["speech_segments"] = merged
    ctx["pause_ratio"] = pause_ratio
    ctx["pauses"] = pauses


def _build_speech_audio(ctx: dict):
    """Build speech-only waveforms from VAD segments (for ASR, emotion)."""
    from vad_utils import extract_speech_audio_torch, extract_speech_audio_numpy

    merged = ctx["speech_segments"]

    ctx["speech_waveform_16k"] = extract_speech_audio_torch(
        ctx["waveform_16k"], 16000, merged,
    )
    ctx["speech_waveform_native"] = extract_speech_audio_numpy(
        ctx["waveform_native"], ctx["sr_native"], merged,
    )


def _run_asr(ctx: dict):
    """Stage 2: ASR on speech-only regions → WPM + filler detection.

    Chunks speech segments into ≤15s windows before decoding.
    The CRDNN-RNNLM decoder uses beam search which is O(n²) on sequence
    length — feeding the full concatenated speech as one tensor is
    catastrophically slow.  Chunking keeps each decode fast and also
    preserves natural word boundaries (no cross-segment fusion).
    """
    from model_registry import get_asr
    from vad_utils import chunk_for_asr

    t0 = time.time()
    asr = get_asr()
    ctx["timings"]["asr_load"] = round(time.time() - t0, 3)

    t0 = time.time()
    transcript_parts = []

    try:
        waveform = ctx["waveform_16k"]   # (1, total_samples) — full audio
        sr = 16000

        # Chunk merged speech segments into ≤15s windows
        asr_chunks = chunk_for_asr(ctx["speech_segments"], max_chunk_s=15.0)

        for i, (start_s, end_s) in enumerate(asr_chunks):
            s = int(start_s * sr)
            e = int(end_s * sr)
            e = min(e, waveform.shape[1])
            if s >= e:
                continue

            chunk_wav = waveform[:, s:e]           # (1, chunk_samples)
            wav_lens = torch.tensor([1.0])

            predicted_words, _ = asr.transcribe_batch(chunk_wav, wav_lens)
            chunk_text = predicted_words[0] if predicted_words else ""
            if chunk_text.strip():
                transcript_parts.append(chunk_text.strip())

    except Exception as e:
        print(f"  [ASR warning] {e} — continuing with partial transcript.",
              flush=True)

    transcript = " ".join(transcript_parts)
    ctx["timings"]["asr_inference"] = round(time.time() - t0, 3)
    ctx["timings"]["asr_chunks"] = len(transcript_parts)

    # WPM and filler analysis
    words = [w for w in transcript.lower().split() if w.strip()]
    total_words = len(words)

    # Use speech duration (not total) for more accurate WPM
    speech_duration_min = sum(
        e - s for s, e in ctx["speech_segments"]
    ) / 60.0 if ctx["speech_segments"] else ctx["duration_s"] / 60.0

    wpm = total_words / max(speech_duration_min, 1e-9)

    filler_count = 0
    for i, word in enumerate(words):
        if word in FILLER_WORDS:
            filler_count += 1
        if i < len(words) - 1:
            bigram = word + " " + words[i + 1]
            if bigram in FILLER_WORDS:
                filler_count += 1

    filler_rate = filler_count / max(total_words, 1)

    ctx["wpm"] = wpm
    ctx["filler_rate"] = filler_rate
    ctx["transcript"] = transcript


def _run_prosody(ctx: dict):
    """Stage 3: Prosody (pitch + energy) on speech-active regions."""
    t0 = time.time()

    y = ctx["speech_waveform_native"]
    sr = ctx["sr_native"]

    # Pitch (F0) via pYIN
    f0_voiced = np.array([], dtype=np.float64)
    pitch_std = 0.0
    pitch_mean = 0.0
    pitch_cv = 0.0
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
            frame_length=2048,
        )
        f0_voiced = f0[voiced_flag & ~np.isnan(f0)]
        if len(f0_voiced) > 10:
            pitch_std = float(np.std(f0_voiced))
            pitch_mean = float(np.mean(f0_voiced))
            pitch_cv = pitch_std / max(pitch_mean, 1e-9)
    except Exception:
        pass

    # RMS energy
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    active_frames = rms[rms > rms.max() * 0.05] if rms.max() > 0 else rms
    energy_std = float(np.std(active_frames)) if len(active_frames) > 10 else 0.0

    ctx["timings"]["prosody_inference"] = round(time.time() - t0, 3)

    ctx["pitch_std"] = pitch_std
    ctx["pitch_mean"] = pitch_mean
    ctx["pitch_cv"] = pitch_cv
    ctx["energy_std"] = energy_std
    ctx["n_voiced"] = len(f0_voiced)
    ctx["n_active"] = len(active_frames)


def _run_emotion(ctx: dict):
    """Stage 4: Vocal emotion on representative speech samples.

    The emotion model (wav2vec2-IEMOCAP) has modules:
        wav2vec2 → avg_pool → output_mlp
    NOT the generic compute_features → mean_var_norm → embedding_model
    that EncoderClassifier.encode_batch() expects.  We call the modules
    directly to avoid the 'compute_features' AttributeError.

    To avoid O(n) wav2vec2 processing for long audio, we sample up to
    MAX_CHUNKS representative 10s windows from the speech segments and
    aggregate predictions via confidence-weighted majority vote.
    """
    from model_registry import get_emotion, DEVICE

    MAX_CHUNKS = 5
    CHUNK_S = 10.0
    SR = 16000

    t0 = time.time()
    emotion_model = get_emotion()
    ctx["timings"]["emotion_load"] = round(time.time() - t0, 3)

    t0 = time.time()
    try:
        waveform = ctx["waveform_16k"]  # (1, total_samples) — full audio
        segments = ctx["speech_segments"]

        # Build representative sample chunks from speech segments
        chunk_tensors = []
        for start_s, end_s in segments:
            if len(chunk_tensors) >= MAX_CHUNKS:
                break
            s = int(start_s * SR)
            e = int(end_s * SR)
            e = min(e, waveform.shape[1])
            seg_len = e - s

            # Skip very short segments
            if seg_len < SR * 2:  # < 2s
                continue

            # Cap each chunk to CHUNK_S seconds
            chunk_samples = int(CHUNK_S * SR)
            if seg_len > chunk_samples:
                # Take from the middle of the segment
                mid = s + seg_len // 2
                half = chunk_samples // 2
                s = max(0, mid - half)
                e = min(waveform.shape[1], s + chunk_samples)

            chunk_tensors.append(waveform[:, s:e])

        if not chunk_tensors:
            # Fallback: take first 10s of audio
            max_e = min(int(CHUNK_S * SR), waveform.shape[1])
            chunk_tensors.append(waveform[:, :max_e])

        # Classify each chunk independently
        votes = {}  # label -> total confidence
        with torch.no_grad():
            for chunk in chunk_tensors:
                chunk = chunk.to(DEVICE).float()
                feats = emotion_model.mods.wav2vec2(chunk)
                pooled = emotion_model.mods.avg_pool(feats)
                logits = emotion_model.mods.output_mlp(pooled).squeeze(1)
                out_probs = emotion_model.hparams.softmax(logits)
                score_val, index_val = torch.max(out_probs, dim=-1)
                lab = emotion_model.hparams.label_encoder.decode_torch(index_val)
                lab_str = lab[0] if isinstance(lab, (list, tuple)) else str(lab)
                conf = float(score_val[0]) if hasattr(score_val, "__len__") else float(score_val)
                votes[lab_str] = votes.get(lab_str, 0.0) + conf

        # Majority vote weighted by confidence
        label_str = max(votes, key=votes.get) if votes else "neu"
        confidence = votes.get(label_str, 0.5) / max(len(chunk_tensors), 1)

    except Exception as e:
        print(f"  [Emotion warning] {e}", flush=True)
        label_str = "neu"
        confidence = 0.5
    ctx["timings"]["emotion_inference"] = round(time.time() - t0, 3)
    ctx["timings"]["emotion_chunks"] = len(chunk_tensors) if 'chunk_tensors' in dir() else 0

    ctx["emotion_label"] = label_str
    ctx["emotion_confidence"] = confidence


def _build_report(ctx: dict) -> SpeechReport:
    """Assemble all metrics into a scored SpeechReport."""
    wpm = ctx["wpm"]
    filler_rate = ctx["filler_rate"]
    pause_ratio = ctx["pause_ratio"]
    pitch_std = ctx["pitch_std"]
    pitch_cv = ctx["pitch_cv"]
    energy_std = ctx["energy_std"]
    n_voiced = ctx["n_voiced"]
    n_active = ctx["n_active"]
    emotion_label = ctx["emotion_label"]
    emotion_confidence = ctx["emotion_confidence"]
    transcript = ctx["transcript"]

    # Scoring
    wpm_score    = bell_score(wpm, ideal=145, std=50)
    filler_score = tanh_score(filler_rate, ideal=0.0, std=0.04, higher_is_better=False)
    pitch_score  = bell_score(pitch_cv,    ideal=0.18, std=0.12)
    energy_score = bell_score(energy_std,  ideal=0.028, std=0.025)
    pause_score  = bell_score(pause_ratio, ideal=0.15, std=0.10)

    # Confidence dampener
    pitch_score  *= min(1.0, n_voiced / 50)
    energy_score *= min(1.0, n_active / 50)

    emo_label, emo_coeff, emo_feedback = EMOTION_FEEDBACK.get(
        emotion_label, ("Unknown", 0.0, "Could not determine vocal emotion.")
    )
    emotion_score = emo_coeff

    # Weighted composite
    weights = {
        "wpm":    0.20,
        "filler": 0.20,
        "pause":  0.20,
        "pitch":  0.15,
        "energy": 0.15,
        "emotion":0.10,
    }
    overall = (
        weights["wpm"]    * wpm_score     +
        weights["filler"] * filler_score  +
        weights["pitch"]  * pitch_score   +
        weights["energy"] * energy_score  +
        weights["pause"]  * pause_score   +
        weights["emotion"]* emotion_score
    )
    overall = float(np.clip(overall, -1.0, 1.0))

    # Baseline shift: "average = good"
    overall_01 = (overall + 1) / 2.0
    overall = round(overall_01 * 2 - 1, 3)

    return SpeechReport(
        overall=round(overall, 3),
        grade=grade(overall),
        wpm=Marker(
            score=round(wpm_score, 3),
            raw=round(wpm, 1),
            unit="wpm",
            label="Speaking rate",
            feedback=feedback_wpm(wpm),
        ),
        filler_rate=Marker(
            score=round(filler_score, 3),
            raw=round(filler_rate * 100, 2),
            unit="% of words",
            label="Filler words",
            feedback=feedback_fillers(filler_rate),
        ),
        pitch_variation=Marker(
            score=round(pitch_score, 3),
            raw=round(pitch_std, 2),
            unit="Hz σ",
            label="Pitch variation",
            feedback=feedback_pitch(pitch_std),
        ),
        energy_variation=Marker(
            score=round(energy_score, 3),
            raw=round(energy_std, 5),
            unit="RMS σ",
            label="Energy variation",
            feedback=feedback_energy(energy_std),
        ),
        pause_ratio=Marker(
            score=round(pause_score, 3),
            raw=round(pause_ratio * 100, 1),
            unit="% of duration",
            label="Pause ratio",
            feedback=feedback_pause(pause_ratio),
        ),
        vocal_emotion=Marker(
            score=round(emotion_score, 3),
            raw=round(emotion_confidence, 3),
            unit="confidence",
            label=f"Vocal emotion ({emo_label})",
            feedback=emo_feedback,
        ),
        transcript_preview=transcript[:300] + ("…" if len(transcript) > 300 else ""),
        segments=[],
        timings=ctx["timings"],
    )


# ─── Core analysis function ─────────────────────────────────────────────────

def analyze(audio_input) -> SpeechReport:
    """
    Analyse a speech audio file or AudioContext.

    Parameters
    ----------
    audio_input : str | AudioContext
        Path to a WAV file, or a pre-loaded AudioContext object.

    Returns
    -------
    SpeechReport
    """
    # Accept both a path string and an AudioContext object
    from audio_context import AudioContext

    if isinstance(audio_input, AudioContext):
        actx = audio_input
    elif isinstance(audio_input, (str, Path)):
        actx = AudioContext.from_wav(str(audio_input))
    else:
        raise TypeError(f"Expected str or AudioContext, got {type(audio_input)}")

    print(f"\n── Analyzing: {Path(actx.source_path).name} "
          f"({actx.duration_s:.1f}s) ──")

    # Build shared context dict for all stages
    ctx = {
        "wav_16k_path": actx.wav_16k_path,
        "waveform_16k": actx.waveform_16k,
        "waveform_native": actx.waveform_native,
        "sr_native": actx.sr_native,
        "duration_s": actx.duration_s,
        "timings": {},
    }

    t_total = time.time()

    print("  [1/4] VAD + pause analysis…", flush=True)
    _run_vad(ctx)

    _build_speech_audio(ctx)

    print("  [2/4] ASR + filler detection…", flush=True)
    _run_asr(ctx)

    print("  [3/4] Prosody (pitch + energy)…", flush=True)
    _run_prosody(ctx)

    print("  [4/4] Vocal emotion…", flush=True)
    _run_emotion(ctx)

    ctx["timings"]["total_speech_analysis"] = round(time.time() - t_total, 3)

    report = _build_report(ctx)

    # Clean up only if we created the AudioContext internally
    if not isinstance(audio_input, AudioContext):
        actx.cleanup()

    return report


# ─── Segmented analysis ─────────────────────────────────────────────────────

def _process_segment(i, chunk, sr, t_start, t_end):
    """Process a single segment for prosody analysis (no disk I/O)."""
    f0_voiced = np.array([], dtype=np.float64)
    pitch_std = 0.0
    pitch_cv = 0.0
    energy_std = 0.0
    n_voiced = 0
    n_active = 0

    try:
        f0, voiced_flag, _ = librosa.pyin(
            chunk,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
            frame_length=2048,
        )
        f0_voiced = f0[voiced_flag & ~np.isnan(f0)]
        if len(f0_voiced) > 10:
            pitch_std = float(np.std(f0_voiced))
            pitch_mean = float(np.mean(f0_voiced))
            pitch_cv = pitch_std / max(pitch_mean, 1e-9)
        n_voiced = len(f0_voiced)
    except Exception:
        pass

    rms = librosa.feature.rms(y=chunk, frame_length=2048, hop_length=512)[0]
    active_frames = rms[rms > rms.max() * 0.05] if rms.max() > 0 else rms
    energy_std = float(np.std(active_frames)) if len(active_frames) > 10 else 0.0
    n_active = len(active_frames)

    p_score = bell_score(pitch_cv,   ideal=0.18, std=0.12)
    e_score = bell_score(energy_std, ideal=0.028, std=0.025)
    p_score *= min(1.0, n_voiced / 50)
    e_score *= min(1.0, n_active / 50)

    return {
        "segment": i + 1,
        "time_start": round(t_start, 1),
        "time_end":   round(t_end,   1),
        "pitch_std":  round(pitch_std, 2),
        "pitch_cv":   round(pitch_cv, 4),
        "energy_std": round(energy_std, 5),
        "pitch_score":  round(p_score, 3),
        "energy_score": round(e_score, 3),
    }


def analyze_segments(
    audio_input,
    segment_duration: int = 30,
    max_segments: int = 20,
) -> list:
    """
    Splits audio into N-second chunks and runs prosody analysis sequentially.

    Accepts either a file path or AudioContext.  When an AudioContext is
    provided, the waveform is sliced in-memory (no file reload).

    Parameters
    ----------
    audio_input : str | AudioContext
        Path to audio file or pre-loaded AudioContext.
    segment_duration : int
        Window size in seconds.
    max_segments : int
        Cap on number of segments to process (for long files).

    Returns
    -------
    list of dicts with timestamp + pitch/energy scores.
    """
    from audio_context import AudioContext

    if isinstance(audio_input, AudioContext):
        y = audio_input.waveform_native
        sr = audio_input.sr_native
    else:
        y = librosa.load(str(audio_input), sr=None, mono=True)[0]
        sr = librosa.get_samplerate(str(audio_input))

    total = len(y) / sr
    segment_samples = int(segment_duration * sr)

    t0 = time.time()
    segments = []
    seg_idx = 0

    for start_sample in range(0, len(y), segment_samples):
        if seg_idx >= max_segments:
            break

        chunk = y[start_sample : start_sample + segment_samples]
        if len(chunk) < sr * 2:   # skip clips shorter than 2s
            continue

        t_start = start_sample / sr
        t_end   = min(t_start + segment_duration, total)

        result = _process_segment(seg_idx, chunk, sr, t_start, t_end)
        segments.append(result)
        seg_idx += 1

    elapsed = round(time.time() - t0, 3)
    print(f"  Segmented prosody: {len(segments)} segments in {elapsed}s", flush=True)

    return segments
 

# ─── Pretty printer ─────────────────────────────────────────────────────────
 
BAR_WIDTH = 30

def score_bar(score: float) -> str:
    """Renders a text bar: e.g.  ███████░░  0.42"""
    filled = int((score + 1) / 2 * BAR_WIDTH)
    filled = max(0, min(BAR_WIDTH, filled))
    return "█" * filled + "░" * (BAR_WIDTH - filled)
 

def print_report(report: SpeechReport):
    sep = "─" * 62
 
    print(f"\n{'═' * 62}")
    print(f"  SPEECH PERFORMANCE REPORT")
    print(f"{'═' * 62}")
    print(f"  Overall score : {report.overall:+.3f}  │  Grade: {report.grade}")
    print(f"  {score_bar(report.overall)}")
    print(sep)
 
    markers = [
        report.wpm,
        report.filler_rate,
        report.pitch_variation,
        report.energy_variation,
        report.pause_ratio,
        report.vocal_emotion,
    ]
 
    for m in markers:
        print(f"\n  {m.label}")
        print(f"  {score_bar(m.score)}  {m.score:+.3f}")
        print(f"  Raw value : {m.raw} {m.unit}")
        print(f"  ↳ {m.feedback}")
 
    print(f"\n{sep}")
    if report.transcript_preview:
        print(f"  Transcript preview:\n  \"{report.transcript_preview}\"")
 
    if report.segments:
        print(f"\n{sep}")
        print(f"  Segment breakdown ({len(report.segments)} segments):")
        print(f"  {'Seg':>4}  {'Start':>6}  {'End':>6}  {'Pitch σ':>8}  {'Energy σ':>10}  {'P.Score':>8}  {'E.Score':>8}")
        for s in report.segments:
            print(
                f"  {s['segment']:>4}  {s['time_start']:>5.0f}s  {s['time_end']:>5.0f}s"
                f"  {s['pitch_std']:>8.2f}  {s['energy_std']:>10.5f}"
                f"  {s['pitch_score']:>+8.3f}  {s['energy_score']:>+8.3f}"
            )

    if report.timings:
        print(f"\n{sep}")
        print(f"  Timings:")
        for k, v in report.timings.items():
            print(f"    {k:.<30} {v:.3f}s")
 
    print(f"\n{'═' * 62}\n")
 

# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze public speaking performance from an audio file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("audio", help="Path to audio file (.wav, .mp3, .flac, …)")
    parser.add_argument(
        "--segment-duration",
        type=int,
        default=0,
        metavar="SECS",
        help="If > 0, also run per-segment analysis in N-second windows (default: off)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of formatted text",
    )
    args = parser.parse_args()
 
    report = analyze(args.audio)
 
    if args.segment_duration > 0:
        print(f"\n  Running segmented analysis ({args.segment_duration}s windows)…", flush=True)
        report.segments = analyze_segments(args.audio, args.segment_duration)
 
    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print_report(report)
 

if __name__ == "__main__":
    main()
