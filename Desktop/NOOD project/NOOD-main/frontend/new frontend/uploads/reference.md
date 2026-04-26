# NOOD — Project Reference

**NOOD** (Precision in Public Speaking) is an AI-powered presentation analysis system that evaluates speaker performance across three dimensions: **body language**, **speech metrics**, and **tone-content fit**.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Directory Structure](#directory-structure)
3. [Backend](#backend)
4. [Presentation Analyzer Pipeline](#presentation-analyzer-pipeline)
5. [Body Language Analysis](#body-language-analysis)
6. [Speech Analysis](#speech-analysis)
7. [Tone Analysis](#tone-analysis)
8. [Frontend](#frontend)
9. [Network Relay & Batch Processing](#network-relay--batch-processing)
10. [AI/ML Models](#aiml-models)
11. [Dependencies](#dependencies)
12. [Configuration & Environment Variables](#configuration--environment-variables)
13. [Data Flow](#data-flow)
14. [Scoring System](#scoring-system)
15. [Sample Report Structure](#sample-report-structure)
16. [Running the Project](#running-the-project)

---

## Architecture Overview

```
React/Electron Frontend
        ↕ (HTTP)
FastAPI Backend (port 8000)
        ↕ (async job)
presentation_analyzer.py (orchestrator)
    ├── Body Language Thread (MediaPipe + TFLite)
    └── Speech + Tone Thread (SpeechBrain + Pollinations AI)
```

The system is **modular**: each analysis component (body, speech, tone) can run standalone or as part of the full pipeline.

---

## Directory Structure

```
NOOD-main/
├── backend/                         # FastAPI REST API
│   ├── main.py                      # App entry point, CORS, router registration
│   ├── routers/
│   │   ├── analysis.py              # POST /api/analyze, GET /api/analyze/{job_id}
│   │   └── health.py                # GET /health
│   ├── schemas/
│   │   └── analysis.py              # Pydantic request/response models
│   └── services/
│       ├── job_manager.py           # Thread-safe in-memory job store
│       └── pipeline.py              # Calls presentation_analyzer.py in background thread
│
├── Body Analysis/
│   ├── body_language_detector.py    # MediaPipe + TFLite emotion classifier
│   ├── body_language.tflite         # Custom TFLite model (8 emotion classes, 534 KB)
│   ├── body_language.pkl            # Older pickle model (unused in pipeline)
│   ├── emotion_data.xlsx            # Training data reference
│   └── mediapipe_models/            # Auto-downloaded MediaPipe .task files
│       ├── pose_landmarker_lite.task
│       └── face_landmarker.task
│
├── Speech Analysis/
│   ├── speech_analyzer.py           # VAD, ASR, prosody, vocal emotion (SpeechBrain)
│   └── tone_analyzer.py             # LLM-based tone-content fit (Pollinations AI)
│
├── frontend/
│   ├── index.html                   # Vite entry point
│   ├── electron.d.ts                # TypeScript definitions for Electron API
│   ├── HEADER_SETUP.md              # React Router page structure docs
│   ├── electron/
│   │   ├── main.js                  # Electron main process (window 1200×780)
│   │   └── preload.js               # contextBridge → window.noodElectron.platform
│   └── dist/                        # Production build (Vite output)
│       ├── index.html
│       └── assets/
│           ├── index-BaYns6Mw.js    # Minified React bundle
│           └── index-BGkf90W0.css   # Minified Tailwind styles
│
├── pretrained_models/               # SpeechBrain model cache (HuggingFace)
│   ├── asr/hyperparams.yaml
│   └── vad/hyperparams.yaml
├── pretrained_model_checkpoints/    # Empty (unused)
│
├── presentation_analyzer.py         # Main pipeline orchestrator (CLI entry point)
├── requirements.txt                 # Python dependencies
├── nood_relay.sh                    # Relay HTTP server for remote submissions (port 5050)
├── nood_watcher.sh                  # File-based batch watcher (polls every 3s)
├── test_imports.py                  # Import sanity check utility
├── combined_analysis_report.json    # Sample full output
├── report.json                      # Sample output
└── test_report.json                 # Sample output
```

---

## Backend

**File**: `backend/main.py`
- FastAPI app titled "NOOD Presentation Analyzer API" v1.0.0
- CORS: `allow_origins=["*"]` (open for dev)
- Default host: `0.0.0.0:8000`

### API Endpoints

| Method | Endpoint | Purpose | Response |
|--------|----------|---------|----------|
| `POST` | `/api/analyze` | Upload video, start analysis job | `202 Accepted` + `job_id` |
| `GET` | `/api/analyze/{job_id}` | Poll job status / fetch report | Job status + report |
| `GET` | `/health` | Health check | Status + active job count |

**Upload constraints** (`backend/routers/analysis.py`):
- Max file size: **500 MB**
- Formats: `.mp4`, `.mkv`, `.mov`, `.webm`, `.avi`
- Optional param: `segment_duration` (seconds per analysis window; 0 = disabled)

### Job Manager (`backend/services/job_manager.py`)
- Thread-safe in-memory store using locks
- UUID-based job IDs
- States: `queued` → `processing` → `done` | `failed`
- Stores: timestamps (`created_at`, `started_at`, `completed_at`), report dict, error message

### Pipeline Wrapper (`backend/services/pipeline.py`)
1. Saves uploaded video bytes to temp file
2. Calls `run_pipeline()` from `presentation_analyzer.py` in a background thread
3. Cleans up temp files after completion

### Pydantic Schemas (`backend/schemas/analysis.py`)
- `AnalyzeResponse` — immediate POST response (`job_id`)
- `JobStatusResponse` — poll response (status + optional report/error)
- `AnalysisReport` — full structured report (meta, scores, body, speech, tone, timeline)

---

## Presentation Analyzer Pipeline

**File**: `presentation_analyzer.py` — CLI entry point and main orchestrator

### Pipeline Flow
```
video input
    ↓
extract_audio()  →  16 kHz mono WAV (via ffmpeg)
    ↓
┌─────────────────────────┐   ┌──────────────────────────────┐
│ Thread A (CPU)          │   │ Thread B (main thread)        │
│ _run_body_analysis()    │   │ _run_speech_and_tone()        │
│ - MediaPipe landmarks   │   │ - VAD pause detection         │
│ - TFLite inference      │   │ - ASR transcription           │
│ → 8-class emotion/frame │   │ - Prosody (pitch, energy)     │
│                         │   │ - Vocal emotion               │
│                         │   │ - LLM tone fit                │
└─────────────────────────┘   └──────────────────────────────┘
        ↓ (both complete)
compute_overall_score()
build_timeline()
JSON report output
```

### Key Functions
| Function | Purpose |
|----------|---------|
| `extract_audio()` | ffmpeg wrapper → WAV 16 kHz mono |
| `_run_body_analysis()` | Thread A worker |
| `_run_speech_and_tone()` | Thread B worker (speech then tone, sequential) |
| `compute_body_language_score()` | Emotion distribution → [0,1] score |
| `compute_overall_score()` | Weighted combination of all scores |
| `build_timeline()` | Merge body + speech events chronologically |
| `print_summary()` | Pretty-print console summary |
| `run_pipeline()` | Entry point called by backend pipeline.py |

---

## Body Language Analysis

**File**: `Body Analysis/body_language_detector.py`

### Model
- **Type**: TensorFlow Lite (custom trained)
- **File**: `body_language.tflite` (534 KB)
- **Input**: [1, 2004] feature vector (132 pose + 1872 face landmarks)
- **Output**: 8 emotion classes with softmax probabilities

### Emotion Classes
`Angry`, `Confused`, `Excited`, `Happy`, `Pain`, `Sad`, `Surprised`, `Tension`

### Landmark Extraction
- **Pose**: MediaPipe PoseLandmarker → 33 landmarks (x, y, z, visibility) = 132 features
- **Face**: MediaPipe FaceLandmarker → 478 landmarks, truncated to 468 (legacy Holistic format) + visibility=0 = 1872 features
- **Combined**: 2004-feature vector per frame

### Processing
- Video sampled at ~5 FPS (frame skipping)
- Batch inference: `BATCH_SIZE=32`
- MediaPipe `.task` models downloaded automatically on first run

### Key Class: `EmotionClassifier`
- Supports dynamic batch resizing
- Falls back to sequential inference if model doesn't support batching
- Auto-downloads MediaPipe models if missing from `mediapipe_models/`

### Output
```json
{
  "frames": [
    {"timestamp_s": 1.2, "emotion": "Happy", "confidence": 0.92}
  ],
  "summary": {
    "total_frames_analyzed": 150,
    "dominant_emotion": "Happy",
    "dominant_emotion_pct": 45.3,
    "average_confidence": 0.87,
    "emotion_distribution": {"Happy": 45.3, "Excited": 30.1},
    "duration_s": 45.0
  }
}
```

---

## Speech Analysis

**File**: `Speech Analysis/speech_analyzer.py`

### Models Used
| Stage | Model | Source |
|-------|-------|--------|
| VAD | `speechbrain/vad-crdnn-libriparty` | HuggingFace |
| ASR | `speechbrain/asr-crdnn-rnnlm-librispeech` | HuggingFace |
| Emotion | `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` | HuggingFace |

### Analysis Stages

**Stage 1/4 — VAD + Pause Analysis**
- Detect speech/silence boundaries
- Pause threshold: >150 ms
- `pause_ratio = silence_duration / total_duration`

**Stage 2/4 — ASR + Filler Detection**
- Transcribe audio to text
- Filler words: `"um"`, `"uh"`, `"like"`, `"basically"`, `"you know"`, etc.
- `filler_rate = filler_count / total_words`
- Transcript preview: first 300 characters

**Stage 3/4 — Prosody Analysis (librosa)**
- **Pitch (F0)**: pYIN algorithm, range C2–C6 (65–1047 Hz)
  - Only voiced frames counted
  - Metrics: std-dev, mean, coefficient of variation
- **Energy (RMS)**:
  - Excludes near-silence frames (<5% of max RMS)
  - Metric: std-dev only

**Stage 4/4 — Vocal Emotion Recognition**
- 4 classes: `happy`, `neutral`, `angry`, `sad`
- Mapped to emotional coefficient for context blending

### Scoring Functions
```python
# Bell curve (Gaussian) — for balanced metrics (WPM, pitch, pause)
bell_score(value, ideal, std) = exp(-0.5 * z²) * 2 - 1

# Tanh — for one-sided metrics (fillers: lower is always better)
tanh_score(value, ideal, std, higher_is_better)

# Confidence dampening (unreliable if <50 voiced/active frames)
pitch_score  *= min(1.0, n_voiced / 50)
energy_score *= min(1.0, n_active / 50)
```

### Score Weights
| Metric | Weight | Ideal | Std |
|--------|--------|-------|-----|
| WPM | 20% | 145 wpm | 50 |
| Filler rate | 20% | 0% | 4% |
| Pause ratio | 20% | 15% | 10% |
| Pitch variation (CV) | 15% | 0.18 | 0.12 |
| Energy variation (σ) | 15% | 0.028 | 0.025 |
| Vocal emotion | 10% | — | — |

### Output: `SpeechReport` dataclass
```python
@dataclass
class SpeechReport:
    overall: float           # [-1, 1]
    grade: str               # A–F
    wpm: Marker
    filler_rate: Marker
    pitch_variation: Marker
    energy_variation: Marker
    pause_ratio: Marker
    vocal_emotion: Marker
    transcript_preview: str
    segments: list           # Per-window breakdown (if segment_duration > 0)
```

Each `Marker` has: `score`, `raw`, `unit`, `label`, `feedback`.

### Segmented Analysis
- Splits audio into N-second chunks (controlled by `segment_duration`)
- Prosody computed in parallel CPU threads per chunk

---

## Tone Analysis

**File**: `Speech Analysis/tone_analyzer.py`

### LLM Integration
- **Provider**: Pollinations AI (free, no mandatory auth)
- **Models**: OpenAI (default), Mistral (fallback)
- **Endpoint**: `https://text.pollinations.ai/`
- **Auth**: Optional `POLLINATIONS_API_KEY` env var
- **Retry logic**: 3 attempts, POST → GET fallback on auth error
- **Seed**: 42 (reproducible output)

### Analysis Process
1. Build prompt from transcript preview + vocal markers + grade
2. POST to Pollinations API with system + user messages
3. Parse JSON response → `ToneReport`
4. Extract structured fields

### System Prompt Goal
> Assess whether the speaker's vocal tone matches the content and context of their speech. Identify mismatches by severity (high/medium/low). Provide actionable, specific coaching tips.

### Output: `ToneReport` dataclass
```python
@dataclass
class ToneReport:
    detected_topic: str           # e.g., "product launch"
    detected_context: str         # e.g., "corporate event"
    overall_tone_fit: str         # "appropriate" | "partially appropriate" | "inappropriate"
    tone_fit_score: float         # 0.0 to 1.0
    mismatches: list[ToneMismatch]
    coaching_tips: list[str]      # Ranked by impact
    model_used: str
    raw_response: str             # Full LLM output (for debugging)
```

### Mismatch Example
```json
{
  "severity": "high",
  "observed_tone": "cheerful, upbeat, high energy",
  "expected_tone": "solemn, respectful, subdued",
  "reason": "Eulogy requires respectful tone; cheerful delivery undermines sincerity",
  "moment": "opening and throughout"
}
```

---

## Frontend

### Technology Stack
| Layer | Technology |
|-------|-----------|
| Framework | React + TypeScript |
| Bundler | Vite |
| Desktop | Electron |
| Styling | Tailwind CSS |
| Routing | React Router DOM |
| Fonts | Plus Jakarta Sans (headers), Inter (body), Material Symbols |

### Electron Configuration (`frontend/electron/main.js`)
- Window: **1200×780 px** (min 900×640)
- Background: `#07041b` (dark purple)
- Context isolation: enabled (security)
- Dev mode: loads from `http://localhost:5173` (Vite dev server)
- Prod mode: loads `dist/index.html`
- Preload exposes: `window.noodElectron.platform`

### Pages (per `HEADER_SETUP.md`)
| Route | Page | Header |
|-------|------|--------|
| `/` | Landing Page | Logo + nav (Platform, Services, Pricing, About) + "Get Started" |
| `/onboarding` | Analysis Page | Logo + icons (upload, bookmark, history, edit) + Sign In/Register |

### Build Artifacts
- `dist/index.html` — production HTML
- `dist/assets/index-BaYns6Mw.js` — minified React bundle
- `dist/assets/index-BGkf90W0.css` — minified Tailwind styles

---

## Network Relay & Batch Processing

### Relay Server (`nood_relay.sh`)
Standalone HTTP server (port 5050) for remote video submission over shared network (Samba/NFS).

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/upload` | POST | Receive video file |
| `/status` | GET | Check processing status |
| `/result` | GET | Fetch JSON result |
| `/health` | GET | Server health |
| `/` | OPTIONS | CORS preflight |

**States**: `idle` → `pending` → `processing` → `done` | `error`

### Watcher Script (`nood_watcher.sh`)
File-based batch processor that polls the shared directory every 3 seconds.

**Flow**:
1. Watch for `input.mp4` in shared dir
2. Write `"processing"` to status file
3. Run `presentation_analyzer.py`
4. Write `"done"` or `"error"` to status file
5. Clean up input file

**Environment Variables**:
- `NOOD_SHARED_DIR` — Samba mount path (default: `/mnt/shared`)
- `NOOD_PROJECT_DIR` — NOOD source root
- `NOOD_VENV` — Python virtual environment path

---

## AI/ML Models

| Component | Model | Source | Size | Location |
|-----------|-------|--------|------|----------|
| Body Language | `body_language.tflite` (custom) | Local | 534 KB | `Body Analysis/` |
| Pose Landmarks | `pose_landmarker_lite.task` | MediaPipe | ~4 MB | `Body Analysis/mediapipe_models/` |
| Face Landmarks | `face_landmarker.task` | MediaPipe | ~8 MB | `Body Analysis/mediapipe_models/` |
| VAD | `vad-crdnn-libriparty` | SpeechBrain/HuggingFace | ~50 MB | `pretrained_models/vad/` |
| ASR | `asr-crdnn-rnnlm-librispeech` | SpeechBrain/HuggingFace | ~150 MB | `pretrained_models/asr/` |
| Vocal Emotion | `emotion-recognition-wav2vec2-IEMOCAP` | SpeechBrain/HuggingFace | ~400 MB | `pretrained_models/emotion/` |
| Tone Analysis | `openai` / `mistral` | Pollinations AI (cloud) | — | API |

**Download behavior**:
- MediaPipe models: auto-downloaded on first run if missing
- SpeechBrain models: downloaded via HuggingFace Hub, cached in `pretrained_models/`
- Custom TFLite: pre-included in repo

---

## Dependencies

**File**: `requirements.txt`

```
# Core ML
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.30.0
speechbrain>=0.5.13

# Audio
librosa>=0.10.0
soundfile>=0.12.0
scipy>=1.10.0
numpy>=1.24.0

# Backend
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9

# Optional UI alternatives
streamlit>=1.28.0
gradio>=3.40.0
```

**System dependency**: `ffmpeg` must be installed and on PATH.

---

## Configuration & Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `POLLINATIONS_API_KEY` | (none) | Optional auth for tone LLM |
| `NOOD_RELAY_PORT` | `5050` | Relay server port |
| `NOOD_SHARED_DIR` | `/mnt/shared` | Samba/NFS shared directory |
| `NOOD_PROJECT_DIR` | — | NOOD project root |
| `NOOD_VENV` | — | Python venv path (watcher script) |
| `NODE_ENV` | — | `production` disables Electron dev tools |

---

## Data Flow

```
1. User uploads video via Frontend (React/Electron)
         ↓ POST /api/analyze (multipart/form-data)
2. FastAPI saves video to temp file, creates job_id, starts background thread
         ↓ 202 Accepted + job_id
3. Frontend polls GET /api/analyze/{job_id} every N seconds

4. Background thread runs presentation_analyzer.py:
   a. ffmpeg extracts 16 kHz mono WAV
   b. Thread A: body_language_detector.py
      - MediaPipe extracts pose + face landmarks
      - TFLite classifies emotion per frame (8 classes)
   c. Thread B: speech_analyzer.py (sequential stages)
      - VAD detects speech boundaries + pause ratio
      - ASR transcribes speech to text
      - librosa computes pitch + energy variation
      - SpeechBrain classifies vocal emotion
   d. (after speech) tone_analyzer.py
      - Builds prompt from transcript + metrics
      - Calls Pollinations AI LLM
      - Returns tone fit score + coaching tips
   e. compute_overall_score() combines all scores
   f. build_timeline() merges events chronologically
   g. Returns JSON report → stored in JobManager

5. Frontend receives completed report
6. Display results: overall score, grade, component breakdowns, timeline, coaching tips
```

---

## Scoring System

### Component Scores
```python
speech_score     = SpeechReport.overall  # [-1, 1] from weighted bell/tanh
body_score       = compute_body_language_score()  # [0, 1] from emotion distribution
tone_fit_score   = ToneReport.tone_fit_score      # [0, 1] from LLM

# Normalize speech to [0, 1]
speech_norm = (speech_score + 1) / 2
```

### Overall Score Calculation
```python
overall = (speech_norm * 0.40) + (body_score * 0.30) + (tone_fit_score * 0.30)

# Context-aware emotion adjustment
emotion_adjustment = speech_emotion_coeff × tone_fit_score × 0.10
final_score = overall + emotion_adjustment  # clamped to [0, 1]

# Scale to 0–100
overall_score = final_score * 100
```

### Grade Thresholds
| Grade | Score |
|-------|-------|
| A | 85+ |
| B | 70–84 |
| C | 55–69 |
| D | 40–54 |
| F | <40 |

### Scoring Weights Summary
| Component | Weight |
|-----------|--------|
| Speech (overall) | 40% |
| Body language | 30% |
| Tone fit | 30% |

Within speech:
| Metric | Weight |
|--------|--------|
| WPM | 20% |
| Filler rate | 20% |
| Pause ratio | 20% |
| Pitch variation | 15% |
| Energy variation | 15% |
| Vocal emotion | 10% |

---

## Sample Report Structure

```json
{
  "meta": {
    "video": "/path/to/video.mp4",
    "generated_at": "2026-03-28T12:00:00",
    "pipeline_duration_s": 120.45,
    "segment_duration": 30
  },
  "overall_score": 72.5,
  "overall_grade": "B",
  "component_scores": {
    "speech_score": 68.0,
    "body_language_score": 75.0,
    "tone_fit_score": 78.0
  },
  "body_language": {
    "frames": [{"timestamp_s": 1.2, "emotion": "Happy", "confidence": 0.92}],
    "summary": {
      "total_frames_analyzed": 500,
      "dominant_emotion": "Happy",
      "dominant_emotion_pct": 42.0,
      "average_confidence": 0.87,
      "emotion_distribution": {"Happy": 42.0, "Excited": 28.0},
      "duration_s": 120.0
    }
  },
  "speech": {
    "overall": 0.36,
    "grade": "B",
    "wpm": {"score": 0.2, "raw": 142, "unit": "wpm", "label": "Speaking rate", "feedback": "..."},
    "filler_rate": {"score": 0.5, "raw": 0.02, "unit": "%", "label": "Filler rate", "feedback": "..."},
    "pitch_variation": {"score": 0.3, "raw": 0.19, "unit": "CV", "label": "Pitch variation", "feedback": "..."},
    "energy_variation": {"score": 0.4, "raw": 0.03, "unit": "σ", "label": "Energy variation", "feedback": "..."},
    "pause_ratio": {"score": 0.6, "raw": 0.14, "unit": "%", "label": "Pause ratio", "feedback": "..."},
    "vocal_emotion": {"score": 0.8, "raw": "happy", "unit": null, "label": "Vocal emotion", "feedback": "..."},
    "transcript_preview": "Good morning everyone, today I would like to...",
    "segments": [
      {"index": 0, "start_s": 0, "end_s": 30, "pitch_score": 0.2, "energy_score": 0.4}
    ]
  },
  "tone": {
    "detected_topic": "Product launch presentation",
    "detected_context": "corporate event",
    "overall_tone_fit": "appropriate",
    "tone_fit_score": 0.78,
    "mismatches": [],
    "coaching_tips": ["Slow down slightly in technical sections", "Add more deliberate pauses for emphasis"],
    "model_used": "openai"
  },
  "timeline": [
    {"timestamp_s": 0.5, "source": "body_language", "event": "Emotion: Happy", "confidence": 0.92},
    {"timestamp_s": 10.0, "source": "speech_prosody", "event": "Segment 1: pitch_σ=0.03", "pitch_score": 0.15}
  ]
}
```

---

## Running the Project

### Option 1: Full Stack (Backend + Frontend)
```bash
# Terminal 1: Start FastAPI backend
cd "C:\Users\pc\Desktop\NOOD project\NOOD-main"
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Vite dev server
cd frontend && npm run dev

# Terminal 3: Start Electron (if desktop app)
npm run electron-dev
```

### Option 2: CLI (Standalone Analysis)
```bash
python presentation_analyzer.py --video my_speech.mp4 --output report.json --segment-duration 30
```

### Option 3: Distributed (Relay + Watcher)
```bash
# On network machine: start relay server
bash nood_relay.sh

# On GPU machine: start watcher
bash nood_watcher.sh
```

### Environment Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# ffmpeg must be installed and on PATH
# Windows: winget install ffmpeg
# Linux:   apt install ffmpeg

# (Optional) Set Pollinations API key
export POLLINATIONS_API_KEY=your_key_here
```

### Checking Imports
```bash
python test_imports.py
```
