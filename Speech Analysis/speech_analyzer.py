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
    • speechbrain/asr-wav2vec2-commonvoice-en   (ASR / transcript)
    • speechbrain/emotion-recognition-wav2vec2-IEMOCAP  (vocal emotion)
---------------------------------------------------------------------------
"""
 
import argparse
import json
import sys
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
 
#data classes
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
 
 
#scorer helpers
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
    Gives +1 at the ideal, falls to −1 far from it.
    Used for metrics that have a sweet-spot (e.g. WPM, pause ratio).
    """
    z = (value - ideal) / max(std, 1e-9)
    return float(np.tanh(1.5) - 2 * np.tanh(abs(z) * 1.2))
 
 
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
 
 
#model loaders
_vad_model = None
_asr_model = None
"""
from speechbrain.pretrained import EncoderDecoderASR
_asr_model = EncoderDecoderASR.from_hparams(
        source="../pretrained_models/asr/",
        savedir="../tmp",
        run_opts={"device":"cuda"}
        )
"""
_emotion_model = None
 
 
def load_vad():
    global _vad_model
    if _vad_model is None:
        from speechbrain.pretrained import VAD
        print("  Loading VAD model…", flush=True)
        _vad_model = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir="pretrained_models/vad",
        )
    return _vad_model
 
 
def load_asr():
    global _asr_model
    if _asr_model is None:
        from speechbrain.pretrained import EncoderDecoderASR
        print("  Loading ASR model…", flush=True)
        _asr_model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-crdnn-rnnlm-librispeech",
            savedir="pretrained_models/asr2",
            run_opts={"device": "cuda"},
        )
    return _asr_model
 
 
def load_emotion():
    global _emotion_model
    if _emotion_model is None:
        from speechbrain.pretrained import EncoderClassifier
        print("  Loading emotion model…", flush=True)
        _emotion_model = EncoderClassifier.from_hparams(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            savedir="pretrained_models/emotion",
        )
    return _emotion_model
 
 
#audio utilities
def load_audio_16k(path: str):
    """Load audio and resample to 16 kHz mono (required by SpeechBrain models)."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # stereo → mono
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    return waveform, 16000
 
 
def save_tmp_wav(waveform: torch.Tensor, sr: int, path: str):
    torchaudio.save(path, waveform, sr)
 
 
# Stage 1: VAD ---> speech boundaries + pause stats 
def analyze_pauses(audio_path: str, total_duration: float):
    """Returns pause_ratio and list of pause durations."""
    vad = load_vad()
    boundaries = vad.get_speech_segments(audio_path)
 
    pauses = []
    for i in range(1, len(boundaries)):
        gap = float(boundaries[i][0]) - float(boundaries[i - 1][1])
        if gap > 0.15:          #(< 150ms are breath noise)
            pauses.append(gap)
 
    speech_duration = sum(
        float(b[1]) - float(b[0]) for b in boundaries
    )
    silence_duration = total_duration - speech_duration
    pause_ratio = silence_duration / max(total_duration, 1e-9)
 
    return pause_ratio, pauses, boundaries
 
 
# Stage 2: ASR ----> WPM + filler detection
FILLER_WORDS = {
    "um", "uh", "like", "basically", "literally", "right",
    "okay", "so", "you know", "actually", "honestly",
    "i mean", "kind of", "sort of", "just",
}
 
 
def analyze_speech_content(audio_path: str, total_duration_minutes: float):
    """Returns wpm, filler_rate, transcript string."""
    asr = load_asr()
 
    try:
        transcripts = asr.transcribe_file(audio_path)
        if isinstance(transcripts, (list, tuple)):
            transcript = transcripts[0] if transcripts else ""
        else:
            transcript = str(transcripts)
    except Exception as e:
        print(f"  [ASR warning] {e} — continuing with empty transcript.", flush=True)
        transcript = ""
 
    words = [w for w in transcript.lower().split() if w.strip()]
    total_words = len(words)
 
    wpm = total_words / max(total_duration_minutes, 1e-9)
 
    #count filler tokens (single-word and two-word phrases)
    filler_count = 0
    for i, word in enumerate(words):
        if word in FILLER_WORDS:
            filler_count += 1
        if i < len(words) - 1:
            bigram = word + " " + words[i + 1]
            if bigram in FILLER_WORDS:
                filler_count += 1
 
    filler_rate = filler_count / max(total_words, 1)
 
    return wpm, filler_rate, transcript
 
 
# Stage 3: Prosody - pitch + energy via librosa
def analyze_prosody(audio_path: str):
    """
    Returns:
        pitch_std   ----> standard deviation of fundamental frequency (Hz)
        energy_std  ----> standard deviation of RMS amplitude
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
 
    # ── Pitch (F0) via pYIN ──────────────────────────────────────────────────
    # pYIN separates voiced/unvoiced frames, so we only measure pitch where
    # the speaker is actually phonating — not during silence or fricatives.
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz  (low male voice)
            fmax=librosa.note_to_hz("C6"),   # ~1047 Hz (high female voice)
            sr=sr,
            frame_length=2048,
        )
        f0_voiced = f0[voiced_flag & ~np.isnan(f0)]
        pitch_std = float(np.std(f0_voiced)) if len(f0_voiced) > 10 else 0.0
    except Exception:
        pitch_std = 0.0
 
    # RMS
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    #we exclude near-silence frames so pauses don't artificially inflate variance
    active_frames = rms[rms > rms.max() * 0.05]
    energy_std = float(np.std(active_frames)) if len(active_frames) > 10 else 0.0
 
    return pitch_std, energy_std
 
 
# Stage 4: Vocal emotion (SpeechBrain wav2vec2 IEMOCAP)
def analyze_emotion(audio_path: str) -> tuple[str, float]:
    """Returns (emotion_label, confidence 0–1)."""
    emotion_model = load_emotion()
    try:
        out_probs, score, _, label = emotion_model.classify_file(audio_path)
        # label is a list; take first element
        label_str = label[0] if isinstance(label, (list, tuple)) else str(label)
        confidence = float(score[0]) if hasattr(score, "__len__") else float(score)
        return label_str, confidence
    except Exception as e:
        print(f"  [Emotion warning] {e}", flush=True)
        return "neu", 0.5
 
 
# Core analysis function
def analyze(audio_path: str) -> SpeechReport:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
 
    print(f"\n── Analyzing: {path.name} ──")
 
    # Load once for duration; 16k version saved for SpeechBrain
    y_native, sr_native = librosa.load(str(path), sr=None, mono=True)
    total_duration = len(y_native) / sr_native
    total_duration_minutes = total_duration / 60.0
 
    # SpeechBrain needs 16 kHz wav — write a temp file if needed
    tmp_path = str(path.with_suffix("")) + "_16k_tmp.wav"
    waveform_16k, _ = load_audio_16k(str(path))
    save_tmp_wav(waveform_16k, 16000, tmp_path)
 
    print("  [1/4] VAD + pause analysis…", flush=True)
    pause_ratio, pauses, _ = analyze_pauses(tmp_path, total_duration)
 
    print("  [2/4] ASR + filler detection…", flush=True)
    wpm, filler_rate, transcript = analyze_speech_content(tmp_path, total_duration_minutes)
 
    print("  [3/4] Prosody (pitch + energy)…", flush=True)
    pitch_std, energy_std = analyze_prosody(str(path))   # use native sr for librosa
 
    print("  [4/4] Vocal emotion…", flush=True)
    emotion_label, emotion_confidence = analyze_emotion(tmp_path)
 

    Path(tmp_path).unlink(missing_ok=True)
 
 
    wpm_score     = bell_score(wpm, ideal=145, std=28)
    filler_score  = tanh_score(filler_rate, ideal=0.0, std=0.04, higher_is_better=False)
    pitch_score   = bell_score(pitch_std,   ideal=28,  std=14)
    energy_score  = bell_score(energy_std,  ideal=0.028, std=0.014)
    pause_score   = bell_score(pause_ratio, ideal=0.15, std=0.07)
 
    emo_label, emo_coeff, emo_feedback = EMOTION_FEEDBACK.get(
        emotion_label, ("Unknown", 0.0, "Could not determine vocal emotion.")
    )
    emotion_score = emo_coeff
 
    #Weighted composite
    weights = {
        "wpm":    0.18,
        "filler": 0.15,
        "pitch":  0.25,  
        "energy": 0.20,
        "pause":  0.12,
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
 
    #Build report
    report = SpeechReport(
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
    )
 
    return report
 
 
# Segmented analysis (optional: run over 30-second windows)
def analyze_segments(audio_path: str, segment_duration: int = 30) -> list:
    """
    Splits audio into N-second chunks and runs prosody analysis on each.
    Useful for tracking how energy/pitch change over the talk.
    Returns a list of dicts with timestamp + pitch/energy scores.
    """
    import tempfile, os
 
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    total = len(y) / sr
    segment_samples = segment_duration * sr
    segments = []
 
    for i, start_sample in enumerate(range(0, len(y), int(segment_samples))):
        chunk = y[start_sample : start_sample + int(segment_samples)]
        if len(chunk) < sr * 2:   # skip clips shorter than 2 s
            continue
 
        t_start = start_sample / sr
        t_end   = min(t_start + segment_duration, total)
 
        # Write temp chunk
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        torchaudio.save(tmp, torch.tensor(chunk).unsqueeze(0), sr)
 
        try:
            pitch_std, energy_std = analyze_prosody(tmp)
            p_score = bell_score(pitch_std,  ideal=28,   std=14)
            e_score = bell_score(energy_std, ideal=0.028, std=0.014)
        finally:
            os.unlink(tmp)
 
        segments.append({
            "segment": i + 1,
            "time_start": round(t_start, 1),
            "time_end":   round(t_end,   1),
            "pitch_std":  round(pitch_std, 2),
            "energy_std": round(energy_std, 5),
            "pitch_score":  round(p_score, 3),
            "energy_score": round(e_score, 3),
        })
 
    return segments
 
 
# Pretty printer
 
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
 
    print(f"\n{'═' * 62}\n")
 
 
# CLI 
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
 
