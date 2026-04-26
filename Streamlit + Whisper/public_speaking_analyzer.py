"""
PUBLIC SPEAKING SKILLS ANALYZER
A friendly AI coach for improving your public speaking abilities.

INSTALLATION REQUIRED:
Before running, install dependencies:
    pip install torch torchaudio librosa soundfile transformers speechbrain gradio tqdm numpy scipy

For GPU support (optional, enables faster processing):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Models Used:
  - Transcription: openai/whisper-large-v3-turbo
  - Emotion: speechbrain/emotion-recognition-wav2vec2-IEMOCAP
"""

import os
import sys
import argparse
import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from pathlib import Path
from typing import Tuple, Dict, List
import gradio as gr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from speechbrain.pretrained import EncoderClassifier
import warnings
from datetime import timedelta

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & DEVICE SETUP
# ============================================================================

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Model IDs - Easy to swap
WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
EMOTION_MODEL_ID = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

# Filler words to detect
FILLER_WORDS = {
    'um': ['um', 'umm', 'ummm'],
    'uh': ['uh', 'uhh'],
    'like': ['like'],
    'you know': ['you', 'know'],  # Special case: two words
    'actually': ['actually'],
    'basically': ['basically'],
    'i mean': ['i', 'mean'],
    'sort of': ['sort', 'of'],
    'kind of': ['kind', 'of'],
    'well': ['well'],
    'right': ['right'],
    'er': ['er', 'err'],
    'ah': ['ah', 'ahh'],
}

print(f"✓ Device: {DEVICE}")
print(f"✓ PyTorch dtype: {TORCH_DTYPE}")


# ============================================================================
# AUDIO LOADING & PREPROCESSING
# ============================================================================

def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file (WAV or MP3) and resample to target sample rate.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default 16kHz for Whisper/SpeechBrain)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr)
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")


def convert_to_wav(audio_path: str) -> str:
    """Convert MP3 to WAV if necessary. Returns path to WAV file."""
    if audio_path.lower().endswith('.wav'):
        return audio_path
    
    print(f"  Converting {Path(audio_path).name} to WAV...")
    audio, sr = librosa.load(audio_path, sr=16000)
    wav_path = audio_path.rsplit('.', 1)[0] + '_converted.wav'
    sf.write(wav_path, audio, sr)
    return wav_path


# ============================================================================
# TRANSCRIPTION WITH TIMESTAMPS
# ============================================================================

def transcribe_audio(audio_path: str, device: str = DEVICE) -> Dict:
    """
    Transcribe audio using Whisper with word-level timestamps.
    
    Args:
        audio_path: Path to audio file
        device: Device to use ('cuda:0' or 'cpu')
    
    Returns:
        Dict with 'text', 'chunks' (with timestamps), and 'language'
    """
    print(f"  Loading Whisper model...")
    model_id = WHISPER_MODEL_ID
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=TORCH_DTYPE,
        device=device,
        return_timestamps=True,  # Enable word-level timestamps
    )
    
    print(f"  Transcribing audio...")
    # Whisper pipeline with timestamps
    result = pipe(audio_path, generate_kwargs={"language": "en"})
    
    # Extract word-level chunks
    chunks = []
    if "chunks" in result:
        chunks = result["chunks"]
    
    return {
        "text": result.get("text", ""),
        "chunks": chunks,
        "language": "en"
    }


# ============================================================================
# SPEAKING SPEED ANALYSIS
# ============================================================================

def analyze_speaking_speed(transcription: Dict, audio_duration_sec: float) -> Dict:
    """
    Analyze speaking speed (WPM) from transcription.
    
    Args:
        transcription: Result from transcribe_audio()
        audio_duration_sec: Duration of audio in seconds
    
    Returns:
        Dict with overall WPM and segment info
    """
    text = transcription["text"]
    
    # Count words (simple split)
    words = text.split()
    word_count = len(words)
    
    # Overall WPM
    duration_minutes = audio_duration_sec / 60.0
    overall_wpm = word_count / duration_minutes if duration_minutes > 0 else 0
    
    # Per-chunk analysis if available
    chunk_info = []
    if transcription.get("chunks"):
        for chunk in transcription["chunks"]:
            if "timestamp" in chunk and isinstance(chunk["timestamp"], (list, tuple)):
                start, end = chunk["timestamp"]
                duration = (end - start) / 60.0  # minutes
                words_in_chunk = len(chunk["text"].split())
                wpm = words_in_chunk / duration if duration > 0 else 0
                chunk_info.append({
                    "time": f"{start:.1f}s-{end:.1f}s",
                    "words": words_in_chunk,
                    "wpm": wpm,
                    "text": chunk["text"]
                })
    
    return {
        "overall_wpm": overall_wpm,
        "word_count": word_count,
        "duration_sec": audio_duration_sec,
        "chunks": chunk_info
    }


# ============================================================================
# PAUSE DETECTION
# ============================================================================

def detect_pauses(audio: np.ndarray, sr: int, silence_threshold_db: float = -40) -> Dict:
    """
    Detect pauses in audio by analyzing silence periods.
    
    Args:
        audio: Audio array
        sr: Sample rate
        silence_threshold_db: Threshold below which is considered silence
    
    Returns:
        Dict with pause count, average duration, and long pauses
    """
    # Convert to dB
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_db = librosa.power_to_db(S)
    
    # Mean power across frequencies
    power_db = np.mean(S_db, axis=0)
    
    # Detect silence frames
    is_silent = power_db < silence_threshold_db
    
    # Convert frame indices to time
    frames = np.arange(len(is_silent))
    times = librosa.frames_to_time(frames, sr=sr)
    
    # Find pause regions
    pauses = []
    pause_start = None
    
    for i, silent in enumerate(is_silent):
        if silent and pause_start is None:
            pause_start = times[i]
        elif not silent and pause_start is not None:
            pause_end = times[i]
            duration = pause_end - pause_start
            if duration > 0.2:  # Only count pauses > 0.2 seconds
                pauses.append({
                    "start": pause_start,
                    "end": pause_end,
                    "duration": duration
                })
            pause_start = None
    
    if pause_start is not None:
        pause_end = len(audio) / sr
        duration = pause_end - pause_start
        if duration > 0.2:
            pauses.append({
                "start": pause_start,
                "end": pause_end,
                "duration": duration
            })
    
    # Statistics
    long_pauses = [p for p in pauses if p["duration"] > 0.8]
    avg_pause = np.mean([p["duration"] for p in pauses]) if pauses else 0
    
    return {
        "total_pauses": len(pauses),
        "long_pauses": len(long_pauses),  # > 0.8 seconds
        "avg_pause_duration": avg_pause,
        "pauses": pauses,
        "long_pause_details": long_pauses
    }


# ============================================================================
# FILLER WORDS DETECTION
# ============================================================================

def detect_filler_words(transcription: Dict) -> Dict:
    """
    Detect and count filler words in transcription.
    
    Args:
        transcription: Result from transcribe_audio()
    
    Returns:
        Dict with filler word counts and timestamps
    """
    text = transcription["text"].lower()
    words = text.split()
    
    filler_counts = {}
    
    for filler_name, filler_variants in FILLER_WORDS.items():
        count = 0
        for variant in filler_variants:
            count += words.count(variant)
        if count > 0:
            filler_counts[filler_name] = count
    
    # Sort by count descending
    filler_counts = dict(sorted(filler_counts.items(), key=lambda x: x[1], reverse=True))
    
    total_fillers = sum(filler_counts.values())
    
    return {
        "filler_words": filler_counts,
        "total_fillers": total_fillers,
        "filler_percentage": (total_fillers / len(words) * 100) if len(words) > 0 else 0
    }


# ============================================================================
# AUDIO FEATURE ANALYSIS (Volume, Pitch)
# ============================================================================

def analyze_audio_features(audio: np.ndarray, sr: int) -> Dict:
    """
    Analyze volume (dB) and pitch variation (for shakiness/trembling).
    
    Args:
        audio: Audio array
        sr: Sample rate
    
    Returns:
        Dict with volume and pitch metrics
    """
    # VOLUME (dB)
    # RMS energy
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_db = librosa.power_to_db(S)
    avg_volume_db = np.mean(S_db)
    
    # Detect low volume
    is_low_volume = avg_volume_db < -25  # Arbitrary threshold
    
    # PITCH VARIATION (using librosa's piptrack)
    f0 = librosa.yin(audio, fmin=80, fmax=400, sr=sr)
    f0_voiced = f0[f0 > 0]  # Remove unvoiced frames
    
    pitch_variation = np.std(f0_voiced) if len(f0_voiced) > 0 else 0
    pitch_mean = np.mean(f0_voiced) if len(f0_voiced) > 0 else 0
    
    # Detect trembling (high variation = trembling voice)
    is_trembling = pitch_variation > 20  # Hz
    
    return {
        "avg_volume_db": float(avg_volume_db),
        "is_low_volume": bool(is_low_volume),
        "pitch_mean_hz": float(pitch_mean),
        "pitch_variation_hz": float(pitch_variation),
        "is_trembling": bool(is_trembling),
        "volume_range_db": float(np.max(S_db) - np.min(S_db))
    }


# ============================================================================
# EMOTION RECOGNITION
# ============================================================================

def classify_emotion(audio_path: str, device: str = DEVICE) -> Dict:
    """
    Classify emotion using SpeechBrain.
    
    Args:
        audio_path: Path to audio file
        device: Device to use
    
    Returns:
        Dict with emotion scores and primary emotion
    """
    print(f"  Loading emotion classifier...")
    try:
        classifier = EncoderClassifier.from_hparams(
            source=EMOTION_MODEL_ID,
            savedir="pretrained_models/",
            run_opts={"device": device}
        )
        
        print(f"  Analyzing emotion...")
        out_prob, score, index, text_lab = classifier.classify_file(audio_path)
        
        # Get emotion label
        emotion_predictions = {
            text_lab[i]: float(out_prob[i]) for i in range(len(text_lab))
        }
        
        primary_emotion = text_lab[int(index)]
        
        return {
            "primary_emotion": primary_emotion,
            "scores": emotion_predictions,
            "confidence": float(out_prob[int(index)])
        }
    except Exception as e:
        print(f"  ⚠️  Emotion classification failed: {e}")
        return {
            "primary_emotion": "neutral",
            "scores": {"neutral": 1.0},
            "confidence": 0.0
        }


# ============================================================================
# CONFIDENCE SCORE CALCULATION
# ============================================================================

def calculate_confidence_score(
    speed_analysis: Dict,
    pause_analysis: Dict,
    filler_analysis: Dict,
    audio_features: Dict,
    emotion_analysis: Dict
) -> Tuple[float, List[str]]:
    """
    Calculate confidence score (1-10) based on multiple factors.
    
    Returns:
        Tuple of (score, explanation_points)
    """
    score = 100.0  # Start at perfect, deduct for issues
    reasons = []
    
    # 1. SPEAKING SPEED (ideal: 120-150 WPM)
    wpm = speed_analysis["overall_wpm"]
    if wpm < 80:
        score -= 15
        reasons.append("Speaking very slowly (below 80 WPM)")
    elif wpm < 120:
        score -= 5
        reasons.append(f"Speaking slowly ({wpm:.0f} WPM, ideal 120-150)")
    elif wpm > 200:
        score -= 20
        reasons.append(f"Speaking too fast ({wpm:.0f} WPM, ideal 120-150)")
    elif wpm > 150:
        score -= 5
        reasons.append(f"Speaking a bit fast ({wpm:.0f} WPM, ideal 120-150)")
    else:
        reasons.append(f"Great speaking pace ({wpm:.0f} WPM)")
    
    # 2. PAUSES
    if pause_analysis["long_pauses"] > 3:
        score -= 10
        reasons.append(f"Too many long pauses ({pause_analysis['long_pauses']})")
    elif pause_analysis["long_pauses"] > 0:
        score -= 3
        reasons.append(f"Some long pauses ({pause_analysis['long_pauses']})")
    
    # 3. FILLER WORDS
    filler_pct = filler_analysis["filler_percentage"]
    if filler_pct > 10:
        score -= 20
        reasons.append(f"Many filler words ({filler_pct:.1f}% of words)")
    elif filler_pct > 5:
        score -= 10
        reasons.append(f"Moderate fillers ({filler_pct:.1f}% of words)")
    elif filler_pct > 2:
        score -= 3
        reasons.append(f"Few fillers ({filler_pct:.1f}% of words)")
    else:
        reasons.append(f"Minimal fillers ({filler_pct:.1f}%)")
    
    # 4. VOLUME
    if audio_features["is_low_volume"]:
        score -= 10
        reasons.append(f"Low volume ({audio_features['avg_volume_db']:.1f} dB)")
    else:
        reasons.append(f"Good volume ({audio_features['avg_volume_db']:.1f} dB)")
    
    # 5. PITCH VARIATION (trembling = less confident)
    if audio_features["is_trembling"]:
        score -= 10
        reasons.append(f"Voice trembling (pitch variation: {audio_features['pitch_variation_hz']:.1f} Hz)")
    else:
        reasons.append(f"Steady voice (variation: {audio_features['pitch_variation_hz']:.1f} Hz)")
    
    # 6. EMOTION
    emotion = emotion_analysis["primary_emotion"].lower()
    stress_emotions = {"angry", "sad", "fear", "disgust"}
    confident_emotions = {"neutral", "happy", "calm"}
    
    if emotion in stress_emotions:
        score -= 15
        reasons.append(f"Detected stress (emotion: {emotion})")
    elif emotion in confident_emotions:
        reasons.append(f"Sounded {emotion}")
    else:
        reasons.append(f"Emotion: {emotion}")
    
    # Normalize to 1-10
    final_score = max(1, min(10, score / 10))
    
    return final_score, reasons


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(
    audio_path: str,
    speed_analysis: Dict,
    pause_analysis: Dict,
    filler_analysis: Dict,
    audio_features: Dict,
    emotion_analysis: Dict,
    confidence_score: float,
    reasoning: List[str]
) -> str:
    """Generate a friendly, actionable report."""
    
    audio_name = Path(audio_path).name
    
    report = f"""
╔════════════════════════════════════════════════════════════════╗
║         YOUR PUBLIC SPEAKING ANALYSIS - {audio_name}            ║
║                 (AI Soft Skills Coach)                         ║
╚════════════════════════════════════════════════════════════════╝

🎯 CONFIDENCE SCORE: {confidence_score:.1f}/10

{'✨ Excellent delivery!' if confidence_score >=8 else
  '👍 Good job! Room for improvement.' if confidence_score >= 6 else
  '💪 Keep practicing, you\'re on the right track.' if confidence_score >= 4 else
  '🚀 Great potential—let\'s work on these areas!'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 YOUR METRICS:

• Speaking Speed:
  └─ {speed_analysis['overall_wpm']:.0f} WPM (ideal: 120-150 WPM)
  └─ {speed_analysis['word_count']} words in {speed_analysis['duration_sec']:.1f} seconds

• Pauses:
  └─ {pause_analysis['total_pauses']} total pauses
  └─ {pause_analysis['long_pauses']} long pauses (>0.8 sec)
  └─ Average pause: {pause_analysis['avg_pause_duration']:.2f} sec

• Filler Words:
  └─ Total fillers: {filler_analysis['total_fillers']} ({filler_analysis['filler_percentage']:.1f}% of words)"""
    
    if filler_analysis['filler_words']:
        report += "\n  └─ Breakdown:"
        for word, count in list(filler_analysis['filler_words'].items())[:5]:
            report += f"\n     • '{word}': {count}x"
    
    report += f"""

• Voice Quality:
  └─ Average Volume: {audio_features['avg_volume_db']:.1f} dB
  └─ Pitch Variation: {audio_features['pitch_variation_hz']:.1f} Hz (steadiness)
  └─ {'⚠️ Voice sounds low' if audio_features['is_low_volume'] else '✓ Good projection'}
  └─ {'⚠️ Voice sounds trembling' if audio_features['is_trembling'] else '✓ Steady delivery'}

• Emotion Detected:
  └─ {emotion_analysis['primary_emotion'].capitalize()} ({emotion_analysis['confidence']*100:.0f}% confidence)"""
    
    if emotion_analysis['scores']:
        report += "\n  └─ Full breakdown:"
        for em, score in emotion_analysis['scores'].items():
            report += f"\n     • {em.capitalize()}: {score*100:.0f}%"
    
    report += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 KEY OBSERVATIONS:

"""
    
    for i, reason in enumerate(reasoning, 1):
        report += f"  {i}. {reason}\n"
    
    report += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 ACTIONABLE TIPS TO IMPROVE:

"""
    
    tips = []
    
    # Speed tips
    if speed_analysis['overall_wpm'] > 150:
        tips.append("⏱️  Pace: Slow down! Take deliberate pauses between sentences.")
    elif speed_analysis['overall_wpm'] < 120:
        tips.append("⏱️  Pace: Pick up the pace slightly to maintain engagement.")
    else:
        tips.append("⏱️  Pace: Excellent! Maintain this speaking speed.")
    
    # Pause tips
    if pause_analysis['long_pauses'] > 2:
        tips.append("🫁 Pauses: Instead of long silences, keep going or use 'umm' less.")
    
    # Filler tips
    if filler_analysis['filler_percentage'] > 5:
        tips.append("🤐 Fillers: Practice pausing instead of saying 'um' or 'like'.")
    
    # Volume tips
    if audio_features['is_low_volume']:
        tips.append("📢 Volume: Speak louder and with more confidence—project from your diaphragm.")
    else:
        tips.append("📢 Volume: Great vocal projection. Keep it up!")
    
    # Pitch tips
    if audio_features['is_trembling']:
        tips.append("🎙️  Trembling: Take deep breaths and slow down. Trembling usually means nerves.")
    else:
        tips.append("🎙️  Pitch: Steady voice—great sign of confidence!")
    
    # Emotion tips
    if emotion_analysis['primary_emotion'].lower() in {'angry', 'sad', 'fear'}:
        tips.append("😌 Emotion: You sounded stressed. Relax, breathe, and remember your message.")
    else:
        tips.append(f"😌 Emotion: You conveyed {emotion_analysis['primary_emotion'].lower()}—great!")
    
    for tip in tips:
        report += f"  {tip}\n"
    
    report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 NEXT STEPS:

  1. Record yourself again and compare your progress—see if you hit 120-150 WPM.
  2. Practice slow, intentional pauses (catch yourself before saying fillers).
  3. Record in a quiet space and check your volume levels.
  4. Work on voice steadiness via breathing exercises.
  5. Watch videos of confident speakers to calibrate your emotion.

Remember: Public speaking is a skill that improves with practice. 
You're on the right track! 💪

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    return report


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def analyze_public_speaking(audio_path: str) -> Tuple[str, Dict]:
    """
    Complete pipeline: load audio → analyze → report.
    
    Args:
        audio_path: Path to audio file (WAV or MP3)
    
    Returns:
        Tuple of (report_text, detailed_dict)
    """
    print(f"\n{'='*65}")
    print(f"📝 ANALYZING: {Path(audio_path).name}")
    print(f"{'='*65}\n")
    
    # Convert MP3 to WAV if needed
    audio_path = convert_to_wav(audio_path)
    
    # Load audio
    print("Step 1/7: Loading audio...")
    audio, sr = load_audio(audio_path, target_sr=16000)
    audio_duration = len(audio) / sr
    print(f"  ✓ Audio loaded: {audio_duration:.1f} seconds @ {sr} Hz")
    
    # Transcription
    print("\nStep 2/7: Transcribing audio...")
    transcription = transcribe_audio(audio_path, device=DEVICE)
    print(f"  ✓ Transcribed: {len(transcription['text'])} characters")
    print(f"  Text preview: {transcription['text'][:100]}...")
    
    # Speaking speed
    print("\nStep 3/7: Analyzing speaking speed...")
    speed_analysis = analyze_speaking_speed(transcription, audio_duration)
    print(f"  ✓ WPM: {speed_analysis['overall_wpm']:.1f}")
    
    # Pauses
    print("\nStep 4/7: Detecting pauses...")
    pause_analysis = detect_pauses(audio, sr)
    print(f"  ✓ Found {pause_analysis['total_pauses']} pauses")
    
    # Fillers
    print("\nStep 5/7: Detecting filler words...")
    filler_analysis = detect_filler_words(transcription)
    print(f"  ✓ Found {filler_analysis['total_fillers']} fillers")
    
    # Audio features
    print("\nStep 6/7: Analyzing voice features...")
    audio_features = analyze_audio_features(audio, sr)
    print(f"  ✓ Volume: {audio_features['avg_volume_db']:.1f} dB")
    print(f"  ✓ Pitch variation: {audio_features['pitch_variation_hz']:.1f} Hz")
    
    # Emotion
    print("\nStep 7/7: Classifying emotion...")
    emotion_analysis = classify_emotion(audio_path, device=DEVICE)
    print(f"  ✓ Primary emotion: {emotion_analysis['primary_emotion']}")
    
    # Score & Report
    print("\nCalculating confidence score...")
    confidence_score, reasoning = calculate_confidence_score(
        speed_analysis, pause_analysis, filler_analysis,
        audio_features, emotion_analysis
    )
    
    print(f"  ✓ Score: {confidence_score:.1f}/10")
    
    report = generate_report(
        audio_path, speed_analysis, pause_analysis, filler_analysis,
        audio_features, emotion_analysis, confidence_score, reasoning
    )
    
    # Detailed output
    detailed = {
        "speed": speed_analysis,
        "pauses": pause_analysis,
        "fillers": filler_analysis,
        "audio_features": audio_features,
        "emotion": emotion_analysis,
        "score": confidence_score,
        "transcription": transcription["text"]
    }
    
    print("\n✅ Analysis complete!\n")
    
    return report, detailed


# ============================================================================
# GRADIO WEB INTERFACE
# ============================================================================

def gradio_interface(audio_file):
    """Wrapper for Gradio."""
    if audio_file is None:
        return "Please upload an audio file." , ""
    
    try:
        report, detailed = analyze_public_speaking(audio_file)
        
        # Format detailed metrics
        metrics = f"""
**CONFIDENCE SCORE:** {detailed['score']:.1f}/10

**Speaking Speed:** {detailed['speed']['overall_wpm']:.0f} WPM ({detailed['speed']['word_count']} words)

**Pauses:** {detailed['pauses']['total_pauses']} total, {detailed['pauses']['long_pauses']} long pauses

**Fillers:** {detailed['fillers']['total_fillers']} total ({detailed['fillers']['filler_percentage']:.1f}%)

**Volume:** {detailed['audio_features']['avg_volume_db']:.1f} dB

**Pitch Variation:** {detailed['audio_features']['pitch_variation_hz']:.1f} Hz

**Emotion:** {detailed['emotion']['primary_emotion']}

---
**Transcription:** {detailed['transcription'][:500]}...
        """
        
        return report, metrics
    except Exception as e:
        return f"Error: {str(e)}", ""


def launch_gradio():
    """Launch the Gradio interface."""
    print("\n" + "="*65)
    print("🚀 LAUNCHING GRADIO INTERFACE")
    print("="*65)
    print("\nOpen your browser and go to: http://localhost:7860")
    print("\nFeatures:")
    print("  • Upload WAV or MP3 audio files")
    print("  • Get instant analysis and feedback")
    print("  • See detailed metrics and transcription")
    print("\nPress Ctrl+C to stop.\n")
    
    with gr.Blocks(title="Public Speaking Analyzer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🎤 Public Speaking Skills Analyzer
## Your AI Soft Skills Coach

Upload an audio file and get instant feedback on your speaking skills!

**Supported formats:** WAV, MP3 | **Max duration:** 5 minutes
        """)
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Upload your audio", 
                    type="filepath",
                    sources=["upload"]
                )
                analyze_btn = gr.Button("📊 Analyze My Speaking", size="lg", variant="primary")
            
            with gr.Column():
                confidence_output = gr.Textbox(
                    label="Analysis Report",
                    lines=30,
                    max_lines=50
                )
        
        metrics_output = gr.Markdown(label="Metrics Summary")
        
        analyze_btn.click(
            fn=gradio_interface,
            inputs=[audio_input],
            outputs=[confidence_output, metrics_output]
        )
    
    demo.launch(share=False, show_error=True)


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Public Speaking Skills Analyzer - Your AI soft skills coach"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Path to audio file (WAV or MP3)"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch Gradio web interface instead"
    )
    
    args = parser.parse_args()
    
    if args.web or args.audio_file is None:
        launch_gradio()
    else:
        # CLI mode
        audio_file = args.audio_file
        
        if not os.path.exists(audio_file):
            print(f"❌ File not found: {audio_file}")
            sys.exit(1)
        
        # Check file format
        if not audio_file.lower().endswith(('.wav', '.mp3')):
            print("❌ Unsupported format. Please use WAV or MP3.")
            sys.exit(1)
        
        report, detailed = analyze_public_speaking(audio_file)
        print(report)
        
        # Save report
        report_path = audio_file.rsplit('.', 1)[0] + "_analysis.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
