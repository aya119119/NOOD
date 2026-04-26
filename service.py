"""
service.py
---------------------------------------------------------------------------
Persistent FastAPI backend for NOOD presentation analysis.

Models are loaded once at startup and reused across all requests.
Jobs are processed serially in a background thread (GPU is shared).

Endpoints
---------
    POST /analyze         Submit a video for analysis
    GET  /jobs/{id}       Check job status
    GET  /jobs/{id}/result  Get completed report
    GET  /health          Service health + model status
    POST /upload          Upload video file (multipart, compat with relay)
    GET  /status          Compat with relay (returns latest job status)
    GET  /result          Compat with relay (returns latest result)

Run
---
    uvicorn service:app --host 0.0.0.0 --port 5050 --workers 1
---------------------------------------------------------------------------
"""

import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# ── Path setup ──────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent
_BODY_DIR = _PROJECT_ROOT / "Body Analysis"
_SPEECH_DIR = _PROJECT_ROOT / "Speech Analysis"
_TMP_DIR = _PROJECT_ROOT / "tmp" / "audio"
_UPLOAD_DIR = _PROJECT_ROOT / "tmp" / "uploads"

sys.path.insert(0, str(_BODY_DIR))
sys.path.insert(0, str(_SPEECH_DIR))

_TMP_DIR.mkdir(parents=True, exist_ok=True)
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Job state ───────────────────────────────────────────────────────────────

_job_queue: queue.Queue = queue.Queue()
_jobs: dict = {}            # job_id -> {status, result, error, timings, ...}
_latest_job_id: str = ""    # for relay-compat /status and /result endpoints
_warmup_info: dict = {}
_start_time: float = 0


class AnalyzeRequest(BaseModel):
    video_path: str
    segment_duration: int = 0
    output_path: Optional[str] = None


# ── Background worker ───────────────────────────────────────────────────────

def _process_job(job_id: str, video_path: str, segment_duration: int,
                 output_path: Optional[str]):
    """Run the full pipeline for one job (called from worker thread)."""
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["started_at"] = datetime.now().isoformat()

    try:
        from presentation_analyzer import run_pipeline

        out = output_path or str(
            _PROJECT_ROOT / "tmp" / f"report_{job_id}.json"
        )

        report = run_pipeline(
            video_path=video_path,
            output_path=out,
            segment_duration=segment_duration,
        )

        # Mark as warm (not cold start) in the report
        report["meta"]["cold_start"] = False

        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["result"] = report
        _jobs[job_id]["output_path"] = out
        _jobs[job_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        import traceback
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
        _jobs[job_id]["traceback"] = traceback.format_exc()
        _jobs[job_id]["completed_at"] = datetime.now().isoformat()
        print(f"  [ERROR] Job {job_id} failed: {e}", file=sys.stderr)
        traceback.print_exc()


def _worker_loop():
    """Serial job processor — one at a time."""
    while True:
        try:
            job_id, video_path, segment_duration, output_path = _job_queue.get()
            print(f"\n  ── Processing job {job_id} ──", flush=True)
            _process_job(job_id, video_path, segment_duration, output_path)
            _job_queue.task_done()
        except Exception as e:
            print(f"  [WORKER ERROR] {e}", file=sys.stderr)


# ── App lifecycle ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: preload models.  Shutdown: clean up."""
    global _warmup_info, _start_time
    _start_time = time.time()

    print("\n" + "═" * 62)
    print("  NOOD Analysis Service — starting up")
    print("═" * 62 + "\n")

    # Preload all models
    from model_registry import warmup_models
    _warmup_info = warmup_models()

    print(f"\n  ✓ All models loaded in {_warmup_info['total_warmup_s']}s")
    print(f"  ✓ Service ready for requests\n")

    # Start background worker
    worker = threading.Thread(target=_worker_loop, daemon=True)
    worker.start()

    yield

    print("\n  Service shutting down…")


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="NOOD Analysis Service",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Service health check with model status."""
    from model_registry import model_status
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _start_time, 1),
        "models": model_status(),
        "warmup": _warmup_info,
        "pending_jobs": _job_queue.qsize(),
    }


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """Submit a video for analysis. Returns job ID immediately."""
    global _latest_job_id

    if not os.path.isfile(request.video_path):
        raise HTTPException(404, f"Video file not found: {request.video_path}")

    job_id = str(uuid.uuid4())[:8]
    _latest_job_id = job_id
    _jobs[job_id] = {
        "status": "queued",
        "video_path": request.video_path,
        "submitted_at": datetime.now().isoformat(),
    }

    _job_queue.put((
        job_id,
        request.video_path,
        request.segment_duration,
        request.output_path,
    ))

    return {"job_id": job_id, "status": "queued"}


@app.post("/upload")
async def upload(video: UploadFile = File(...)):
    """
    Upload a video file and start analysis.
    Compatible with the Electron frontend and nood_relay.sh.
    """
    global _latest_job_id

    # Save uploaded file
    dest = _UPLOAD_DIR / f"upload_{uuid.uuid4().hex[:8]}.mp4"
    with open(dest, "wb") as f:
        shutil.copyfileobj(video.file, f)

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  ✓ Video uploaded: {dest} ({size_mb:.1f} MB)", flush=True)

    # Submit job
    job_id = str(uuid.uuid4())[:8]
    _latest_job_id = job_id
    _jobs[job_id] = {
        "status": "queued",
        "video_path": str(dest),
        "submitted_at": datetime.now().isoformat(),
    }

    _job_queue.put((job_id, str(dest), 0, None))

    return {"ok": True, "job_id": job_id, "size_mb": round(size_mb, 1)}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    """Check status of a specific job."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job not found: {job_id}")
    job = _jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "submitted_at": job.get("submitted_at"),
        "completed_at": job.get("completed_at"),
        "error": job.get("error"),
    }


@app.get("/jobs/{job_id}/result")
async def job_result(job_id: str):
    """Get the result of a completed job."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job not found: {job_id}")
    job = _jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(
            409, f"Job is not done yet (status: {job['status']})"
        )
    return job["result"]


# ── Relay-compatible endpoints ──────────────────────────────────────────────
# These match the nood_relay.sh interface so the Electron frontend works
# without changes.

@app.get("/status")
async def relay_status():
    """Relay-compatible status endpoint."""
    if not _latest_job_id or _latest_job_id not in _jobs:
        return {"status": "idle"}
    return {"status": _jobs[_latest_job_id]["status"]}


@app.get("/result")
async def relay_result():
    """Relay-compatible result endpoint."""
    if not _latest_job_id or _latest_job_id not in _jobs:
        return JSONResponse(status_code=404, content={"error": "Result not ready"})
    job = _jobs[_latest_job_id]
    if job["status"] != "done":
        return JSONResponse(status_code=404, content={"error": "Result not ready"})
    return job["result"]
