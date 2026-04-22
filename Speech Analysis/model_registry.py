"""
model_registry.py
---------------------------------------------------------------------------
Centralized, thread-safe singleton manager for all SpeechBrain models.

Every model is loaded exactly once per process lifetime.  Absolute cache
paths prevent "works-from-one-cwd" surprises, and the VAD device-mismatch
patch is applied automatically.

Public API
----------
    get_vad()        -> VAD
    get_asr()        -> EncoderDecoderASR
    get_emotion()    -> EncoderClassifier
    warmup_models()  -> dict          # preload all, return timing info
    model_status()   -> dict          # per-model metadata
    DEVICE           -> str           # "cuda:0" or "cpu"
---------------------------------------------------------------------------
"""

import threading
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_ROOT = _PROJECT_ROOT / "pretrained_models"

DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"

# Canonical absolute cache paths (no more relative dirs)
_CACHE_PATHS = {
    "vad":     _CACHE_ROOT / "vad",
    "asr":     _CACHE_ROOT / "asr2",
    "emotion": _CACHE_ROOT / "emotion",
}

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_models: dict = {}          # name -> model instance
_metadata: dict = {}        # name -> {loaded_at, device, cold_load_s, cache_source}
_locks: dict = {            # per-model init locks
    "vad":     threading.Lock(),
    "asr":     threading.Lock(),
    "emotion": threading.Lock(),
}


def _record(name: str, model, load_time: float, cache_source: str):
    """Store model instance and metadata after a successful load."""
    _models[name] = model
    _metadata[name] = {
        "loaded_at": time.time(),
        "device": DEVICE,
        "savedir": str(_CACHE_PATHS[name]),
        "cold_load_s": round(load_time, 2),
        "cache_source": cache_source,
    }


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------
def get_vad():
    """Return the VAD model (thread-safe singleton)."""
    if "vad" in _models:
        return _models["vad"]

    with _locks["vad"]:
        if "vad" in _models:          # double-check after acquiring lock
            return _models["vad"]

        from speechbrain.pretrained import VAD

        savedir = str(_CACHE_PATHS["vad"])
        cache_source = "disk" if (_CACHE_PATHS["vad"] / "model.ckpt").exists() else "download"

        print(f"  [model_registry] Loading VAD (device={DEVICE}, "
              f"cache={cache_source})…", flush=True)
        t0 = time.time()

        model = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir=savedir,
            run_opts={"device": DEVICE},
        )
        # SpeechBrain VAD.__init__ overwrites self.device with
        # hparams.device (defaults to "cpu" in the YAML).  Patch it
        # so internal .to(self.device) calls target the correct device.
        model.device = DEVICE
        model.hparams.device = DEVICE

        _record("vad", model, time.time() - t0, cache_source)
        print(f"  [model_registry] VAD ready ({_metadata['vad']['cold_load_s']}s)",
              flush=True)
        return model


# ---------------------------------------------------------------------------
# ASR  (speechbrain/asr-crdnn-rnnlm-librispeech)
# ---------------------------------------------------------------------------
def get_asr():
    """Return the ASR model (thread-safe singleton)."""
    if "asr" in _models:
        return _models["asr"]

    with _locks["asr"]:
        if "asr" in _models:
            return _models["asr"]

        from speechbrain.pretrained import EncoderDecoderASR

        savedir = str(_CACHE_PATHS["asr"])
        cache_source = "disk" if (_CACHE_PATHS["asr"] / "asr.ckpt").exists() else "download"

        print(f"  [model_registry] Loading ASR (device={DEVICE}, "
              f"cache={cache_source})…", flush=True)
        t0 = time.time()

        model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-crdnn-rnnlm-librispeech",
            savedir=savedir,
            run_opts={"device": DEVICE},
        )

        _record("asr", model, time.time() - t0, cache_source)
        print(f"  [model_registry] ASR ready ({_metadata['asr']['cold_load_s']}s)",
              flush=True)
        return model


# ---------------------------------------------------------------------------
# Emotion  (speechbrain/emotion-recognition-wav2vec2-IEMOCAP)
# ---------------------------------------------------------------------------
def get_emotion():
    """Return the emotion classifier (thread-safe singleton)."""
    if "emotion" in _models:
        return _models["emotion"]

    with _locks["emotion"]:
        if "emotion" in _models:
            return _models["emotion"]

        from speechbrain.pretrained import EncoderClassifier

        savedir = str(_CACHE_PATHS["emotion"])
        cache_source = ("disk" if (_CACHE_PATHS["emotion"] / "model.ckpt").exists()
                        else "download")

        print(f"  [model_registry] Loading emotion model (device={DEVICE}, "
              f"cache={cache_source})…", flush=True)
        t0 = time.time()

        model = EncoderClassifier.from_hparams(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            savedir=savedir,
            run_opts={"device": DEVICE},
        )

        _record("emotion", model, time.time() - t0, cache_source)
        print(f"  [model_registry] Emotion ready "
              f"({_metadata['emotion']['cold_load_s']}s)", flush=True)
        return model


# ---------------------------------------------------------------------------
# Bulk operations
# ---------------------------------------------------------------------------
def warmup_models() -> dict:
    """Pre-load all models.  Returns combined timing metadata."""
    t0 = time.time()
    get_vad()
    get_asr()
    get_emotion()
    total = round(time.time() - t0, 2)
    return {
        "total_warmup_s": total,
        "models": dict(_metadata),
    }


def model_status() -> dict:
    """Return current model metadata (for /health endpoint)."""
    return {
        name: {
            "loaded": name in _models,
            **_metadata.get(name, {}),
        }
        for name in ("vad", "asr", "emotion")
    }
