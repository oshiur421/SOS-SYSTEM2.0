"""
====================================================================
SOS SYSTEM 2.0 — Fall Detection AI Microservice
====================================================================
Microservice 2: Real-time Computer Vision Pose Analysis, Body Angle
Calculation, Sudden Collapse & Sustained Horizontal Posture Detection.
Auto-pushes CRITICAL alerts to Central Alert Hub (port 8000).
Runs locally on port 8001 and deployable to Render.
====================================================================
"""

import os
import sys
import time
import math
import base64
import urllib.request
import threading
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
import requests
from fastapi import FastAPI, Request, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ====================================================================
# APP SETUP & CORS CONFIGURATION
# ====================================================================
app = FastAPI(
    title="SOS SYSTEM 2.0 — Fall Detection AI Microservice",
    description="Vision-based optical telemetry body angle tracker & human collapse detector.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Central Alert Hub Dispatch URL (Configurable via env var)
CENTRAL_ALERT_URL = os.environ.get("CENTRAL_ALERT_URL", "http://localhost:8000/api/alerts")
ALERT_COOLDOWN_SECONDS = 4.0

# ====================================================================
# MODEL RESOLUTION & AUTOMATIC FALLBACK DOWNLOAD
# ====================================================================
MODEL_FILENAME = "pose_landmarker_lite.task"
OFFICIAL_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

def resolve_model_path() -> Path:
    """Find existing model file or automatically download it from Google."""
    base_dir = Path(__file__).resolve().parent
    candidate_paths = [
        base_dir / "models" / MODEL_FILENAME,
        base_dir.parent / "models" / MODEL_FILENAME,
        Path("C:/SOS-AI-Sensor/models") / MODEL_FILENAME,
        Path("models") / MODEL_FILENAME,
    ]

    for p in candidate_paths:
        if p.exists() and p.stat().st_size > 1000000:
            print(f"[FallAI] Found existing model at: {p}")
            return p

    # If missing everywhere, download to local models directory
    target_dir = base_dir / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / MODEL_FILENAME

    print(f"[FallAI] Model not found locally. Downloading from official Google CDN: {OFFICIAL_MODEL_URL}...")
    try:
        urllib.request.urlretrieve(OFFICIAL_MODEL_URL, str(target_file))
        print(f"[FallAI] Successfully downloaded model to {target_file} ({target_file.stat().st_size} bytes)")
        return target_file
    except Exception as e:
        print(f"[FallAI ERROR] Failed to download model: {e}")
        raise RuntimeError(f"Could not load or download {MODEL_FILENAME}: {e}")

MODEL_PATH = resolve_model_path()

# Initialize MediaPipe PoseLandmarker in IMAGE running mode
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.PoseLandmarker.create_from_options(options)
LANDMARKER_LOCK = threading.Lock()
print("[FallAI] PoseLandmarker initialized successfully in IMAGE mode.")

# ====================================================================
# SECTOR STATE TRACKER (THREAD-SAFE)
# ====================================================================
# Track previous angles, posture timestamps, and cooldown per camera/sector
SECTOR_STATES: Dict[str, Dict[str, Any]] = {}
STATE_LOCK = threading.Lock()

def get_or_create_sector_state(sector: str) -> Dict[str, Any]:
    with STATE_LOCK:
        if sector not in SECTOR_STATES:
            SECTOR_STATES[sector] = {
                "previous_angle": None,
                "previous_shoulder_y": None,
                "horizontal_start_time": None,
                "last_alert_time": 0.0,
            }
        return SECTOR_STATES[sector]

# ====================================================================
# EXACT BODY ANGLE & COLLAPSE LOGIC
# ====================================================================
def calculate_body_angle(landmarks):
    """
    Exact body angle computation from landmarks:
    left_shoulder = 11, right_shoulder = 12
    left_hip = 23, right_hip = 24
    """
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip = landmarks[23]
    right_hip = landmarks[24]

    shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
    hip_x = (left_hip.x + right_hip.x) / 2.0
    hip_y = (left_hip.y + right_hip.y) / 2.0

    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y

    angle = math.degrees(math.atan2(abs(dy), abs(dx)))
    return angle, shoulder_x, shoulder_y, hip_x, hip_y

def dispatch_central_alert(sector: str, angle: float, angle_change: float):
    """Asynchronously post critical fall alert to Central Alert Hub."""
    def _worker():
        payload = {
            "type": "HUMAN FALL DETECTED",
            "severity": "CRITICAL",
            "message": f"Subject posture collapse detected! Body Angle: {angle:.1f}° (Delta: {angle_change:.1f}°). Status: POSSIBLE FALL!",
            "sector": sector,
            "source": "FALL_AI",
        }
        try:
            resp = requests.post(CENTRAL_ALERT_URL, json=payload, timeout=2.0)
            print(f"[FallAI -> Central] Alert dispatched: Status {resp.status_code}")
        except Exception as err:
            print(f"[FallAI -> Central WARN] Failed to reach Central Hub: {err}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

# ====================================================================
# PYDANTIC SCHEMAS
# ====================================================================
class DetectFallRequest(BaseModel):
    image: Optional[str] = Field(default=None, description="Base64 encoded frame (data:image/jpeg;base64,...)")
    image_base64: Optional[str] = Field(default=None, description="Alternative base64 field")
    sector: Optional[str] = Field(default="DRONE-01 [AERIAL]", description="Active camera/sector identifier")

# ====================================================================
# API ENDPOINTS
# ====================================================================

# 1. Health Check
@app.get("/", tags=["Health"])
def health_check():
    """Health check confirming model availability and server status."""
    return {
        "status": "ONLINE",
        "service": "SOS Fall Detection AI",
        "model_loaded": True,
        "model_file": str(MODEL_PATH.name),
        "central_alert_url": CENTRAL_ALERT_URL,
        "active_sectors": list(SECTOR_STATES.keys()),
    }

# 2. Fall Detection Endpoint (Supports JSON with Base64 OR Multipart Upload)
@app.post("/api/detect-fall", tags=["Fall Detection"])
@app.post("/detect-fall", tags=["Fall Detection"])
async def detect_fall(
    request: Request,
    file: Optional[UploadFile] = File(None),
    sector_form: Optional[str] = Form(None),
):
    """
    Process incoming frame, detect pose landmarks, calculate body angle,
    and trigger emergency alert if human collapse conditions are met.
    """
    frame_bgr: Optional[np.ndarray] = None
    sector_name: str = "DRONE-01 [AERIAL]"

    # Case A: Multipart form upload (e.g. from canvas.toBlob)
    if file is not None:
        file_bytes = await file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if sector_form:
            sector_name = sector_form

    # Case B: JSON payload with Base64 image string
    else:
        try:
            body = await request.json()
            raw_b64 = body.get("image") or body.get("image_base64") or body.get("frame")
            if not raw_b64:
                raise HTTPException(status_code=400, detail="No image provided in request")

            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]

            img_bytes = base64.b64decode(raw_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            sector_name = body.get("sector") or sector_name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {str(e)}")

    if frame_bgr is None or frame_bgr.size == 0:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Perform thread-safe landmark inference
    with LANDMARKER_LOCK:
        detection_result = landmarker.detect(mp_image)

    # If no human pose detected in frame
    if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
        return {
            "fall_detected": False,
            "angle": 90.0,
            "angle_change": 0.0,
            "position": "NO_SUBJECT_DETECTED",
            "status": "CLEAR",
            "subject_detected": False,
        }

    # Extract primary subject landmarks (33 landmarks)
    landmarks = detection_result.pose_landmarks[0]
    angle, shoulder_x, shoulder_y, hip_x, hip_y = calculate_body_angle(landmarks)

    # Retrieve sector telemetry state
    state = get_or_create_sector_state(sector_name)
    now = time.time()

    prev_angle = state["previous_angle"]
    prev_shoulder_y = state["previous_shoulder_y"]
    angle_change = abs(angle - prev_angle) if prev_angle is not None else 0.0

    # Horizontal duration tracking
    if angle < 35.0:
        if state["horizontal_start_time"] is None:
            state["horizontal_start_time"] = now
        horizontal_duration = now - state["horizontal_start_time"]
    else:
        state["horizontal_start_time"] = None
        horizontal_duration = 0.0

    # Fall Trigger Conditions:
    # 1. Sudden Collapse: Rapid angle drop
    is_sudden_collapse = (angle_change > 22.0 and angle < 35.0)
    # 2. Sustained Lying/Horizontal Posture
    is_sustained_horizontal = (angle < 30.0 and horizontal_duration >= 1.2)
    # 3. Vertical Collapse in Frame: Shoulder center sudden drop down
    is_vertical_collapse = (
        shoulder_y > 0.78 and (
            (prev_shoulder_y is not None and (shoulder_y - prev_shoulder_y) > 0.12)
            or angle < 40.0
        )
    )

    fall_detected = is_sudden_collapse or is_sustained_horizontal or is_vertical_collapse

    # Update state history
    state["previous_angle"] = angle
    state["previous_shoulder_y"] = shoulder_y

    # Push to Central Hub if cooldown passed
    alert_dispatched = False
    if fall_detected:
        if (now - state["last_alert_time"]) >= ALERT_COOLDOWN_SECONDS:
            state["last_alert_time"] = now
            dispatch_central_alert(sector_name, angle, angle_change)
            alert_dispatched = True

    # Posture classification
    position = "VERTICAL" if angle > 60.0 else ("HORIZONTAL / LYING" if angle < 35.0 else "TILTED")
    status = "POSSIBLE FALL!" if fall_detected else ("HORIZONTAL" if angle < 35.0 else "NORMAL")

    return {
        "fall_detected": fall_detected,
        "angle": round(angle, 1),
        "angle_change": round(angle_change, 1),
        "position": position,
        "status": status,
        "shoulder_y": round(shoulder_y, 3),
        "horizontal_duration": round(horizontal_duration, 2),
        "alert_dispatched": alert_dispatched,
        "subject_detected": True,
    }

# 3. Reset State Endpoint
@app.post("/api/reset-state", tags=["Telemetry"])
@app.post("/reset-state", tags=["Telemetry"])
def reset_state():
    """Reset sector tracking states."""
    with STATE_LOCK:
        SECTOR_STATES.clear()
    return {"status": "SUCCESS", "message": "All sector telemetry states reset."}

# ====================================================================
# ENTRYPOINT FOR DIRECT EXECUTION
# ====================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
