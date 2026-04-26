#!/usr/bin/env python3
"""
SIMPLE PUBLIC SPEAKING ANALYZER (CLI Only)
Analyzes public speaking skills from audio files using Hugging Face models.

INSTALLATION REQUIRED:

1. Install Python dependencies:
    pip install --upgrade pip
    pip install torch torchaudio librosa soundfile transformers speechbrain numpy scipy

2. Install ffmpeg (for MP4/video support):
    
    Windows (with Chocolatey):
        choco install ffmpeg
    
    Windows (manual):
        Download from: https://ffmpeg.org/download.html
        Add to PATH
    
    Mac:
        brew install ffmpeg
    
    Linux:
        sudo apt-get install ffmpeg

Models:
  - Transcription: openai/whisper-large-v3-turbo
  - Emotion: speechbrain/emotion-recognition-wav2vec2-IEMOCAP

SUPPORTED FORMATS:
    WAV, MP3, MP4, MOV, FLAC

USAGE:
    python analyzer.py path/to/audio.wav
    python analyzer.py path/to/audio.mp3
    python analyzer.py path/to/video.mp4
"""

import sys
import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from speechbrain.pretrained import EncoderClassifier

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION & DEVICE SETUP
# =============================================================================

# Detect device (GPU or CPU)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Device: {DEVICE} | PyTorch dtype: {TORCH_DTYPE}\n")

# MODEL IDs - Change these to swap models
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
EMOTION_MODEL = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

# Filler words to detect
FILLERS = {
    'um': ['um', 'umm', 'ummm'],
    'uh': ['uh', 'uhh'],
    'like': ['like'],
    'you know': ['you', 'know'],
    'so': ['so'],
    'basically': ['basically'],
    'actually': ['actually'],
    'i mean': ['i', 'mean'],
    'well': ['well'],
    'right': ['right'],
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_audio(audio_path, sr=16000):
    """Load audio file (WAV, MP3, MP4, etc.) and resample to 16kHz."""
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        return audio, sr
    except FileNotFoundError:
        print(f"❌ Error: Audio file not found: {audio_path}")
        sys.exit(1)
    except Exception as e:
        if "ffmpeg" in str(e).lower():
            print(f"❌ Error: ffmpeg is required to process this audio format.")
            print("\nPlease install ffmpeg:")
            print("  Windows (Chocolatey): choco install ffmpeg")
            print("  Windows (manual): https://ffmpeg.org/download.html")
            print("  Mac: brew install ffmpeg")
            print("  Linux: sudo apt-get install ffmpeg")
        else:
            print(f"❌ Error loading audio: {e}")
        sys.exit(1)


def seconds_to_timestamp(seconds):
    """Convert seconds to MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


# =============================================================================
# TRANSCRIPTION
# =============================================================================

def transcribe(audio_path):
    """Transcribe audio using Whisper with word-level timestamps."""
    print("Loading Whisper model...")
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL, torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True
    ).to(DEVICE)
    
    processor = AutoProcessor.from_pretrained(WHISPER_MODEL)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=TORCH_DTYPE,
        device=DEVICE,
        return_timestamps=True,
    )
    
    print("Transcribing audio...")
    result = pipe(audio_path, generate_kwargs={"language": "en"})
    
    return result.get("text", ""), result.get("chunks", [])


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_speed(text, duration_sec):
    """Calculate speaking speed in WPM."""
    word_count = len(text.split())
    wpm = (word_count / duration_sec) * 60 if duration_sec > 0 else 0
    return wpm, word_count


def analyze_pauses(audio, sr, threshold_db=-40):
    """Detect long pauses (> 0.8s) in audio."""
    # Use mel-spectrogram to detect silence
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_db = librosa.power_to_db(S)
    power_db = np.mean(S_db, axis=0)
    
    # Find silent frames
    is_silent = power_db < threshold_db
    frames = np.arange(len(is_silent))
    times = librosa.frames_to_time(frames, sr=sr)
    
    # Detect pause regions
    pauses = []
    pause_start = None
    
    for i, silent in enumerate(is_silent):
        if silent and pause_start is None:
            pause_start = times[i]
        elif not silent and pause_start is not None:
            pause_end = times[i]
            duration = pause_end - pause_start
            if duration > 0.2:  # Only count pauses > 0.2s
                pauses.append((pause_start, pause_end, duration))
            pause_start = None
    
    # Handle end of audio
    if pause_start is not None:
        pause_end = len(audio) / sr
        duration = pause_end - pause_start
        if duration > 0.2:
            pauses.append((pause_start, pause_end, duration))
    
    # Count long pauses (> 0.8s)
    long_pauses = [p for p in pauses if p[2] > 0.8]
    
    return long_pauses


def analyze_fillers(text):
    """Count filler words in text."""
    text_lower = text.lower()
    words = text_lower.split()
    
    filler_counts = {}
    for filler_name, variants in FILLERS.items():
        count = sum(words.count(v) for v in variants)
        if count > 0:
            filler_counts[filler_name] = count
    
    return filler_counts


def analyze_voice_features(audio, sr):
    """Analyze volume (dB) and pitch variation (Hz)."""
    # Volume
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_db = librosa.power_to_db(S)
    avg_volume_db = np.mean(S_db)
    
    is_low_volume = avg_volume_db < -25
    
    # Pitch variation (using yin pitch tracking)
    f0 = librosa.yin(audio, fmin=80, fmax=400, sr=sr)
    f0_voiced = f0[f0 > 0]
    
    pitch_variation = np.std(f0_voiced) if len(f0_voiced) > 0 else 0
    is_trembling = pitch_variation > 20
    
    return {
        'avg_volume_db': avg_volume_db,
        'is_low_volume': is_low_volume,
        'pitch_variation_hz': pitch_variation,
        'is_trembling': is_trembling,
    }


def classify_emotion(audio_path):
    """Classify emotion using SpeechBrain."""
    print("Loading emotion classifier...")
    
    try:
        classifier = EncoderClassifier.from_hparams(
            source=EMOTION_MODEL,
            savedir="pretrained_models/",
            run_opts={"device": DEVICE}
        )
        
        print("Analyzing emotion...")
        out_prob, score, index, text_lab = classifier.classify_file(audio_path)
        
        primary_emotion = text_lab[int(index)]
        confidence = float(out_prob[int(index)])
        
        return primary_emotion, confidence
    except Exception as e:
        print(f"Warning: Emotion classification failed: {e}")
        return "neutral", 0.0


# =============================================================================
# CONFIDENCE SCORE CALCULATION
# =============================================================================

def calculate_score(wpm, long_pause_count, fillers_count, voice_features, emotion):
    """Calculate confidence score (1-10)."""
    score = 100.0
    
    # Speaking speed (ideal 120-150 WPM)
    if wpm < 80:
        score -= 20
    elif wpm < 120:
        score -= 5
    elif wpm > 200:
        score -= 25
    elif wpm > 150:
        score -= 5
    
    # Long pauses
    if long_pause_count > 3:
        score -= 15
    elif long_pause_count > 0:
        score -= 5
    
    # Filler words
    total_fillers = sum(fillers_count.values())
    if total_fillers > 15:
        score -= 20
    elif total_fillers > 8:
        score -= 10
    elif total_fillers > 3:
        score -= 5
    
    # Volume
    if voice_features['is_low_volume']:
        score -= 10
    
    # Pitch variation
    if voice_features['is_trembling']:
        score -= 10
    
    # Emotion
    stress_emotions = {'angry', 'sad', 'fear', 'disgust'}
    if emotion.lower() in stress_emotions:
        score -= 15
    
    # Normalize to 1-10
    final_score = max(1, min(10, score / 10))
    return final_score


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(
    audio_name, wpm, word_count, duration_sec, long_pauses,
    fillers_count, voice_features, emotion, emotion_conf, confidence_score
):
    """Generate formatted terminal report."""
    
    report = f"\n{'='*50}\n"
    report += f"📝 Public Speaking Analysis: {audio_name}\n"
    report += f"{'='*50}\n\n"
    
    report += f"🎯 Confidence Score: {confidence_score:.0f}/10\n"
    
    if confidence_score >= 8:
        report += "   ✨ Excellent delivery!\n"
    elif confidence_score >= 6:
        report += "   👍 Good job! Room for improvement.\n"
    elif confidence_score >= 4:
        report += "   💪 Keep practicing!\n"
    else:
        report += "   🚀 Great potential—let's work on these areas!\n"
    
    report += f"\n{'─'*50}\n"
    report += "KEY OBSERVATIONS:\n"
    report += f"{'─'*50}\n"
    
    # Speaking speed
    if wpm > 150:
        report += f"- You spoke quite rapidly at {wpm:.0f} WPM (ideal: 120-150).\n"
    elif wpm < 120:
        report += f"- You spoke slowly at {wpm:.0f} WPM (ideal: 120-150).\n"
    else:
        report += f"- Great speaking pace: {wpm:.0f} WPM (ideal: 120-150).\n"
    
    # Volume
    if voice_features['is_low_volume']:
        report += f"- Voice was low volume ({voice_features['avg_volume_db']:.1f} dB).\n"
    else:
        report += f"- Good vocal projection ({voice_features['avg_volume_db']:.1f} dB).\n"
    
    # Pitch
    if voice_features['is_trembling']:
        report += f"- Voice trembled in parts (pitch variation: {voice_features['pitch_variation_hz']:.1f} Hz).\n"
    else:
        report += f"- Steady voice (pitch variation: {voice_features['pitch_variation_hz']:.1f} Hz).\n"
    
    # Emotion
    stress_emotions = {'angry', 'sad', 'fear', 'disgust'}
    if emotion.lower() in stress_emotions:
        report += f"- You sounded stressed (detected emotion: {emotion}).\n"
    else:
        report += f"- You sounded {emotion.lower()} ({emotion_conf*100:.0f}%).\n"
    
    # Fillers
    if fillers_count:
        filler_str = ", ".join([f'"{k}" x{v}' for k, v in sorted(fillers_count.items(), key=lambda x: x[1], reverse=True)[:3]])
        report += f"- Filler words: {filler_str}\n"
    else:
        report += f"- Great! Minimal filler words.\n"
    
    # Pauses
    if long_pauses:
        pause_times = ", ".join([seconds_to_timestamp(p[0]) for p in long_pauses[:5]])
        report += f"- Long pauses: {len(long_pauses)}x (at approx {pause_times})\n"
    else:
        report += f"- No long pauses detected. ✓\n"
    
    report += f"\n{'─'*50}\n"
    report += "DETAILED FEEDBACK:\n"
    report += f"{'─'*50}\n"
    
    feedback = []
    
    if wpm > 150:
        feedback.append("- Speaking speed was too fast. Focus on breathing between sentences.")
    elif wpm < 120:
        feedback.append("- Speaking speed was too slow. Pick up the pace slightly.")
    
    if voice_features['is_low_volume']:
        feedback.append("- Vocal projection needs work. Project from your diaphragm.")
    
    if voice_features['is_trembling']:
        feedback.append("- Voice showed signs of nervousness. Practice breathing exercises.")
    
    if len(long_pauses) > 2:
        feedback.append("- You had several long pauses. This can lose audience engagement.")
    
    total_fillers = sum(fillers_count.values())
    if total_fillers > 8:
        feedback.append(f"- High filler count ({total_fillers}). Replace fillers with intentional pauses.")
    
    if emotion.lower() in stress_emotions:
        feedback.append("- You conveyed stress. Relax and focus on your message.")
    
    if feedback:
        for f in feedback:
            report += f"{f}\n"
    else:
        report += "- Great delivery overall! Keep up the good work.\n"
    
    report += f"\n{'─'*50}\n"
    report += "ACTIONABLE TIPS TO IMPROVE:\n"
    report += f"{'─'*50}\n"
    
    tips = []
    
    if wpm > 150:
        tips.append("1. Slow down — aim for 120-150 WPM. Pause between sentences.")
    elif wpm < 120:
        tips.append("1. Speed up slightly — aim for 120-150 WPM.")
    else:
        tips.append("1. Maintain your current speaking pace.")
    
    if voice_features['is_low_volume']:
        tips.append("2. Project your voice louder and more confidently.")
    else:
        tips.append("2. Maintain your current vocal projection.")
    
    if total_fillers > 5:
        tips.append("3. Replace fillers (um, like) with intentional pauses.")
    else:
        tips.append("3. Keep reducing fillers—you're doing well.")
    
    if voice_features['is_trembling']:
        tips.append("4. Practice deep breathing to steady your voice.")
    else:
        tips.append("4. Keep up your steady vocal delivery.")
    
    if len(long_pauses) > 2:
        tips.append("5. Shorten pauses; use them strategically for impact.")
    else:
        tips.append("5. Your pause usage is good.")
    
    for tip in tips[:5]:
        report += f"{tip}\n"
    
    report += f"\nGreat effort! Keep practicing daily. 💪\n"
    report += f"{'='*50}\n\n"
    
    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Analyze public speaking skills from an audio file.",
        usage="python analyzer.py path/to/audio.wav"
    )
    parser.add_argument("audio_file", nargs="?", help="Path to audio file (WAV or MP3)")
    
    args = parser.parse_args()
    
    # Check if audio file provided
    if not args.audio_file:
        print("\n" + "="*50)
        print("USAGE:")
        print("  python analyzer.py path/to/audio.wav")
        print("  python analyzer.py path/to/audio.mp3")
        print("  python analyzer.py path/to/video.mp4")
        print("\nSupported formats: WAV, MP3, MP4, MOV, FLAC, OGG, and more")
        print("\nExample:")
        print("  python analyzer.py my_speech.mp4")
        print("="*50 + "\n")
        sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(args.audio_file):
        print(f"❌ Error: File not found: {args.audio_file}\n")
        sys.exit(1)
    
    # Check file format
    supported_formats = ('.wav', '.mp3', '.mp4', '.mov', '.flac', '.ogg', '.m4a', '.aac')
    if not args.audio_file.lower().endswith(supported_formats):
        print(f"❌ Error: Unsupported audio format.")
        print(f"   Supported: {', '.join(supported_formats)}\n")
        sys.exit(1)
    
    audio_name = Path(args.audio_file).name
    
    print(f"\n{'='*50}")
    print(f"Analyzing: {audio_name}")
    print(f"{'='*50}\n")
    
    # Load audio
    print("Loading audio...")
    audio, sr = load_audio(args.audio_file)
    duration_sec = len(audio) / sr
    print(f"Duration: {seconds_to_timestamp(duration_sec)}\n")
    
    # Transcription
    text, chunks = transcribe(args.audio_file)
    print(f"Transcription complete: {len(text)} characters\n")
    
    # Analyses
    print("Analyzing speaking speed...")
    wpm, word_count = analyze_speed(text, duration_sec)
    
    print("Detecting pauses...")
    long_pauses = analyze_pauses(audio, sr)
    
    print("Detecting filler words...")
    fillers_count = analyze_fillers(text)
    
    print("Analyzing voice features...")
    voice_features = analyze_voice_features(audio, sr)
    
    # Emotion
    emotion, emotion_conf = classify_emotion(args.audio_file)
    
    # Score
    print("Calculating confidence score...\n")
    confidence_score = calculate_score(
        wpm, len(long_pauses), fillers_count, voice_features, emotion
    )
    
    # Generate report
    report = generate_report(
        audio_name, wpm, word_count, duration_sec, long_pauses,
        fillers_count, voice_features, emotion, emotion_conf, confidence_score
    )
    
    print(report)


if __name__ == "__main__":
    main()
