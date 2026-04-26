"""
---------------------------------------------------------------------------
Pipeline:
    ┌─────────────────────────────────────────────────────────┐
    │  INPUT: video.mp4                                       │
    │         ┌───────────────┐   ┌────────────────────────┐  │
    │  video ─┤ Body Language ├   │ Speech  --> Tone (LLM) │  │
    │         │   Detector    │   │ Analyzer    Analyzer   │  │
    │         └───────┬───────┘   └───────────┬────────────┘  │
    │                 │                       │               │
    │                 └──────────┬────────────┘               │
    │                    MERGE + SCORE                        │
    │                            │                            │
    │                    report.json                          │
    └─────────────────────────────────────────────────────────┘

Usage:
    python presentation_analyzer.py --video path/to/presentation.mp4
    python presentation_analyzer.py --video talk.mp4 --output my_report.json
    python presentation_analyzer.py --video talk.mp4 --segment-duration 30

Dependencies:
    - ffmpeg (system binary, used to extract audio)
    - All dependencies from body_language_detector.py, speech_analyzer.py,
      and tone_analyzer.py

---------------------------------------------------------------------------
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


# ─── Path setup ──────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent
_BODY_DIR = _PROJECT_ROOT / "Body Analysis"
_SPEECH_DIR = _PROJECT_ROOT / "Speech Analysis"
_TMP_DIR = _PROJECT_ROOT / "tmp" / "audio"

sys.path.insert(0, str(_BODY_DIR))
sys.path.insert(0, str(_SPEECH_DIR))

# Ensure tmp directory exists
_TMP_DIR.mkdir(parents=True, exist_ok=True)


# ─── Timing utility ─────────────────────────────────────────────────────────

class PipelineTimer:
    """Collects named stage timings into a dict."""

    def __init__(self):
        self._timings = {}
        self._starts = {}

    def start(self, name: str):
        self._starts[name] = time.time()

    def stop(self, name: str):
        if name in self._starts:
            self._timings[name] = round(time.time() - self._starts.pop(name), 3)

    def record(self, name: str, value: float):
        self._timings[name] = round(value, 3)

    @property
    def timings(self) -> dict:
        return dict(self._timings)


# ─── Audio extraction ───────────────────────────────────────────────────────

def extract_audio(video_path: str, output_path: str) -> str:
    """
    Extract audio from a video file to WAV using ffmpeg.
    Output is 16 kHz, mono, 16-bit PCM — directly usable by SpeechBrain.
    """
    cmd = [
        "ffmpeg",
        "-y",                   # overwrite without asking
        "-i", video_path,
        "-vn",                  # discard video stream
        "-acodec", "pcm_s16le", # 16-bit PCM
        "-ar", "16000",         # 16 kHz (what SpeechBrain needs)
        "-ac", "1",             # mono
        output_path,
    ]

    print(f"  Extracting audio → {output_path}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n{result.stderr[:500]}"
        )

    if not os.path.isfile(output_path):
        raise FileNotFoundError(f"ffmpeg did not produce output file: {output_path}")

    return os.path.abspath(output_path)


# ─── Worker functions ────────────────────────────────────────────────────────

def _run_body_analysis(video_path: str) -> dict:
    """Thread A: body language detection on the raw video."""
    print("\n══ [Thread A] Body language analysis starting…", flush=True)
    t0 = time.time()

    from body_language_detector import run_analysis
    result = run_analysis(video_path)

    elapsed = time.time() - t0
    n_frames = result["summary"]["total_frames_analyzed"]
    dominant = result["summary"]["dominant_emotion"]
    print(
        f"══ [Thread A] Body language done  "
        f"({n_frames} frames, dominant: {dominant}, {elapsed:.1f}s)",
        flush=True,
    )
    return result


def _run_speech_and_tone(audio_context, segment_duration: int = 0) -> dict:
    """
    Thread B: speech analysis --> tone analysis (sequential, since tone
    depends on the speech report).
    """
    print("\n══ [Thread B] Speech analysis starting…", flush=True)
    t0 = time.time()

    from speech_analyzer import analyze as speech_analyze, analyze_segments
    speech_report = speech_analyze(audio_context)

    if segment_duration > 0:
        print(f"  Running segmented analysis ({segment_duration}s windows)…", flush=True)
        speech_report.segments = analyze_segments(
            audio_context, segment_duration
        )

    speech_elapsed = time.time() - t0
    print(f"══ [Thread B] Speech done ({speech_elapsed:.1f}s), starting tone analysis…", flush=True)

    from tone_analyzer import analyze_tone
    tone_report = analyze_tone(speech_report, verbose=True)

    total_elapsed = time.time() - t0
    print(f"══ [Thread B] Tone analysis done ({total_elapsed:.1f}s total)", flush=True)

    return {
        "speech": asdict(speech_report),
        "tone": asdict(tone_report),
    }


# ─── Scoring ─────────────────────────────────────────────────────────────────

_POSITIVE_EMOTIONS = {"Happy", "Excited", "Surprised"}
_NEGATIVE_EMOTIONS = {"Angry", "Sad", "Pain", "Tension"}
_NEUTRAL_EMOTIONS  = {"Confused"}


def compute_body_language_score(summary: dict) -> float:
    """
    Convert body language summary into a normalised score (0.0 ----> 1.0).
    """
    distribution = summary.get("emotion_distribution", {})
    confidence = summary.get("average_confidence", 0.5)

    positive_pct = sum(distribution.get(e, 0) for e in _POSITIVE_EMOTIONS) / 100
    negative_pct = sum(distribution.get(e, 0) for e in _NEGATIVE_EMOTIONS) / 100
    neutral_pct  = sum(distribution.get(e, 0) for e in _NEUTRAL_EMOTIONS)  / 100

    # Base: positive weight + half of neutral; penalty for negatives
    raw = (positive_pct * 1.0) + (neutral_pct * 0.5) - (negative_pct * 0.6)
    # Factor in model confidence
    raw = raw * (0.5 + 0.5 * confidence)
    # Clamp to [0, 1]
    return max(0.0, min(1.0, round(raw, 4)))


def compute_overall_score(
    speech_overall: float,
    body_score: float,
    tone_fit_score: float,
    speech_emotion_score: float = 0.0,
) -> tuple[float, str]:
    """
    Weighted combination → 0–100 score + letter grade.
    """
    # speech_overall is in [-1, 1]; map to [0, 1]
    speech_norm = (speech_overall + 1) / 2.0

    # Context-aware emotion adjustment
    emotion_context_bonus = speech_emotion_score * tone_fit_score * 0.10
    speech_norm = max(0.0, min(1.0, speech_norm + emotion_context_bonus))

    raw = (
        0.40 * speech_norm +
        0.30 * body_score +
        0.30 * tone_fit_score
    )
    score_100 = max(0, min(100, round(raw * 100, 1)))

    if score_100 >= 85:
        letter = "A"
    elif score_100 >= 70:
        letter = "B"
    elif score_100 >= 55:
        letter = "C"
    elif score_100 >= 40:
        letter = "D"
    else:
        letter = "F"

    return score_100, letter


# ─── Timeline builder ───────────────────────────────────────────────────────

def build_timeline(body_frames: list, speech_segments: list) -> list:
    """
    Merges body language per-frame events and speech per-segment events
    into a single chronological timeline.
    """
    timeline = []

    # Body language: sample every ~1 second to avoid thousands of entries
    if body_frames:
        last_ts = -1.0
        for f in body_frames:
            ts = f["timestamp_s"]
            if ts - last_ts >= 1.0:  # 1 event per second
                timeline.append({
                    "timestamp_s": ts,
                    "source": "body_language",
                    "event": f"Emotion: {f['emotion']}",
                    "confidence": f["confidence"],
                })
                last_ts = ts

    # Speech segments
    for seg in speech_segments:
        timeline.append({
            "timestamp_s": seg.get("time_start", 0),
            "source": "speech_prosody",
            "event": (
                f"Segment {seg['segment']}: "
                f"pitch_σ={seg['pitch_std']} Hz, "
                f"energy_σ={seg['energy_std']}"
            ),
            "pitch_score": seg.get("pitch_score", 0),
            "energy_score": seg.get("energy_score", 0),
        })

    # Sort chronologically
    timeline.sort(key=lambda x: x["timestamp_s"])
    return timeline


# ─── Pretty console summary ─────────────────────────────────────────────────

def print_summary(report: dict):
    """Print a human-readable overview to the console."""
    sep = "─" * 62

    print(f"\n{'═' * 62}")
    print("  PRESENTATION PERFORMANCE REPORT")
    print(f"{'═' * 62}")
    print(f"  Overall score : {report['overall_score']} / 100  │  Grade: {report['overall_grade']}")
    print(sep)

    # Body language
    bl = report["body_language"]["summary"]
    print(f"\n  ▸ Body Language")
    print(f"    Dominant emotion : {bl['dominant_emotion']} ({bl['dominant_emotion_pct']}%)")
    print(f"    Avg confidence   : {bl['average_confidence']}")
    print(f"    Score            : {report['body_language_score']}")
    dist_str = ", ".join(f"{e}: {p}%" for e, p in bl.get("emotion_distribution", {}).items())
    print(f"    Distribution     : {dist_str}")

    # Speech
    sp = report["speech"]
    print(f"\n  ▸ Speech Performance (grade: {sp['grade']})")
    print(f"    WPM              : {sp['wpm']['raw']} ({sp['wpm']['feedback']})")
    print(f"    Filler rate      : {sp['filler_rate']['raw']}% ({sp['filler_rate']['feedback']})")
    print(f"    Pitch variation  : {sp['pitch_variation']['raw']} Hz σ")
    print(f"    Pause ratio      : {sp['pause_ratio']['raw']}%")

    # Tone
    tn = report["tone"]
    print(f"\n  ▸ Tone-Content Fit")
    print(f"    Topic            : {tn['detected_topic']}")
    print(f"    Context          : {tn['detected_context']}")
    print(f"    Fit              : {tn['overall_tone_fit']} ({tn['tone_fit_score']:.2f})")
    if tn.get("coaching_tips"):
        print(f"    Top tip          : {tn['coaching_tips'][0]}")

    # Timings
    if "timings" in report.get("meta", {}):
        print(f"\n  ▸ Timings")
        for k, v in report["meta"]["timings"].items():
            if isinstance(v, dict):
                print(f"    {k}:")
                for sk, sv in v.items():
                    print(f"      {sk:.<30} {sv:.3f}s")
            else:
                print(f"    {k:.<32} {v:.3f}s")

    # Timeline count
    print(f"\n  ▸ Timeline: {len(report['timeline'])} events")
    print(f"{'═' * 62}\n")


# ─── Main pipeline ──────────────────────────────────────────────────────────

def run_pipeline(
    video_path: str,
    output_path: str = "presentation_report.json",
    segment_duration: int = 0,
) -> dict:
    """
    Full pipeline: extract audio → parallel analysis → score → JSON.

    Parameters
    ----------
    video_path       : path to input .mp4
    output_path      : where to write the JSON report
    segment_duration : seconds per speech segment window (0 to disable)

    Returns
    -------
    dict : the complete report
    """
    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    timer = PipelineTimer()

    print(f"\n{'═' * 62}")
    print(f"  PRESENTATION ANALYZER — pipeline starting")
    print(f"  Video : {video_path}")
    print(f"{'═' * 62}\n")

    timer.start("total")

    # ── Extract audio ────────────────────────────────────────────────────
    timer.start("audio_extraction")
    tmp_audio = tempfile.NamedTemporaryFile(
        suffix=".wav", prefix="pres_audio_", dir=str(_TMP_DIR), delete=False
    ).name
    try:
        audio_path = extract_audio(video_path, tmp_audio)
    except RuntimeError as e:
        print(f"\n  [ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    timer.stop("audio_extraction")

    # ── Load audio once ──────────────────────────────────────────────────
    timer.start("audio_load")
    from audio_context import AudioContext
    audio_context = AudioContext.from_wav(audio_path)
    timer.stop("audio_load")

    # ── Parallel analysis ────────────────────────────────────────────────
    body_result = None
    speech_tone_result = None

    with ThreadPoolExecutor(max_workers=1) as pool:
        # Body language is CPU-only (MediaPipe + TFLite) — run on thread
        timer.start("body_analysis")
        future_body = pool.submit(_run_body_analysis, video_path)

        # Speech + tone uses GPU (SpeechBrain) — run on main thread
        timer.start("speech_tone_analysis")
        try:
            speech_tone_result = _run_speech_and_tone(audio_context, segment_duration)
        except Exception as e:
            print(f"\n  [ERROR] Speech/tone analyzer failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        timer.stop("speech_tone_analysis")

        # Collect body result
        try:
            body_result = future_body.result()
        except Exception as e:
            print(f"\n  [ERROR] Body analyzer failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        timer.stop("body_analysis")

    # ── Clean up temp files ──────────────────────────────────────────────
    Path(tmp_audio).unlink(missing_ok=True)
    audio_context.cleanup()

    if body_result is None or speech_tone_result is None:
        print("\n  [FATAL] One or more analyzers failed. Cannot produce report.",
              file=sys.stderr)
        sys.exit(1)

    # ── Scoring ──────────────────────────────────────────────────────────
    body_score = compute_body_language_score(body_result["summary"])
    speech_overall = speech_tone_result["speech"]["overall"]
    tone_fit_score = speech_tone_result["tone"]["tone_fit_score"]
    speech_emotion_score = speech_tone_result["speech"].get(
        "vocal_emotion", {}
    ).get("score", 0.0)

    overall_score, overall_grade = compute_overall_score(
        speech_overall, body_score, tone_fit_score,
        speech_emotion_score=speech_emotion_score,
    )

    # ── Timeline ─────────────────────────────────────────────────────────
    timeline = build_timeline(
        body_result.get("frames", []),
        speech_tone_result["speech"].get("segments", []),
    )

    # ── Assemble report ──────────────────────────────────────────────────
    tone_data = speech_tone_result["tone"]
    tone_data.pop("raw_response", None)

    timer.stop("total")

    # Merge speech-level timings into pipeline timings
    pipeline_timings = timer.timings
    speech_timings = speech_tone_result["speech"].get("timings", {})
    if speech_timings:
        pipeline_timings["speech_detail"] = speech_timings

    report = {
        "meta": {
            "video": video_path,
            "generated_at": datetime.now().isoformat(),
            "pipeline_duration_s": pipeline_timings.get("total", 0),
            "segment_duration": segment_duration,
            "process_id": os.getpid(),
            "cold_start": True,       # will be False in service mode
            "audio_duration_s": round(audio_context.duration_s, 2),
            "timings": pipeline_timings,
        },
        "overall_score": overall_score,
        "overall_grade": overall_grade,
        "component_scores": {
            "speech_score": round((speech_overall + 1) / 2 * 100, 1),
            "body_language_score": round(body_score * 100, 1),
            "tone_fit_score": round(tone_fit_score * 100, 1),
        },
        "body_language_score": round(body_score, 4),
        "body_language": body_result,
        "speech": speech_tone_result["speech"],
        "tone": tone_data,
        "timeline": timeline,
    }

    # ── Write output ─────────────────────────────────────────────────────
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ Report written to: {os.path.abspath(output_path)}")
    print_summary(report)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Analyze presentation performance from a video file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file (.mp4, .mkv, etc.)",
    )
    parser.add_argument(
        "--output",
        default="presentation_report.json",
        help="Path for the output JSON report (default: presentation_report.json)",
    )
    parser.add_argument(
        "--segment-duration",
        type=int,
        default=0,
        metavar="SECS",
        help="Per-segment analysis window in seconds (default: 0 = disabled)",
    )
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        output_path=args.output,
        segment_duration=args.segment_duration,
    )


if __name__ == "__main__":
    main()
