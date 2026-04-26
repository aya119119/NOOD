"""
startup_check.py
---------------------------------------------------------------------------
Run at service boot to validate the environment.
Reports GPU, ffmpeg, model cache, and directory status.
---------------------------------------------------------------------------
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_CHECKS_PASSED = True


def _check(label: str, ok: bool, detail: str = ""):
    global _CHECKS_PASSED
    icon = "✓" if ok else "✗"
    msg = f"  {icon} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if not ok:
        _CHECKS_PASSED = False


def main():
    print("  ── Startup Checks ──\n")

    # ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    _check("ffmpeg", ffmpeg_path is not None,
           ffmpeg_path or "NOT FOUND — install ffmpeg")

    # CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            _check("CUDA", True, f"{name}, {mem:.1f} GB")
        else:
            _check("CUDA", False, "not available — will use CPU")
    except Exception as e:
        _check("CUDA", False, str(e))

    # Model cache directories
    cache_root = _PROJECT_ROOT / "pretrained_models"
    for name, marker in [("vad", "model.ckpt"), ("asr", "asr.ckpt"),
                          ("emotion", "model.ckpt")]:
        d = cache_root / name
        has_marker = (d / marker).exists() or (d / marker).is_symlink()
        _check(f"Model cache: {name}", d.exists() and has_marker,
               "cached on disk" if has_marker else "will download on first use")

    # Temp directories
    for d in ["tmp/audio", "tmp/uploads"]:
        p = _PROJECT_ROOT / d
        p.mkdir(parents=True, exist_ok=True)
        writable = os.access(p, os.W_OK)
        _check(f"Directory: {d}", writable,
               "writable" if writable else "NOT WRITABLE")

    # Stale temp file count
    stale_wavs = list(_PROJECT_ROOT.glob("pres_audio_*_16k_tmp.wav"))
    if stale_wavs:
        _check(f"Stale temp files", False,
               f"{len(stale_wavs)} orphan WAVs in project root")
    else:
        _check("Stale temp files", True, "none found")

    print()
    if _CHECKS_PASSED:
        print("  All checks passed.\n")
    else:
        print("  Some checks failed — review above.\n")

    return 0 if _CHECKS_PASSED else 1


if __name__ == "__main__":
    sys.exit(main())
