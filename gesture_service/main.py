"""
====================================================================
SOS SYSTEM 2.0 — Hand Gesture Distress AI Microservice
====================================================================
Microservice 3: Real-time Computer Vision Hand Landmark Tracking,
35-Geometric Feature Extraction & Random Forest Gesture Classifier.
Detects Universal SOS Signal for Help, FIST, and OPEN_PALM.
Auto-pushes HIGH severity alerts to Central Alert Hub (port 8000).
Runs locally on port 8002 and deployable to Render.
====================================================================
"""

import os
import sys
import time
import math
import base64
import urllib.request
import threading
from itertools import combinations
from pathlib import Path
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
import pandas as pd
import requests
import joblib
from fastapi import FastAPI, Request, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ====================================================================
# APP SETUP & CORS CONFIGURATION
# ====================================================================
app = FastAPI(
    title="SOS SYSTEM 2.0 — Hand Gesture AI Microservice",
    description="Vision-based optical telemetry hand landmarker & distress gesture classification engine.",
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
last_alert_time: float = 0.0
ALERT_LOCK = threading.Lock()

# ====================================================================
# MODEL RESOLUTION & AUTOMATIC FALLBACK DOWNLOAD
# ====================================================================
MODEL_FILENAME = "hand_landmarker.task"
OFFICIAL_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

def resolve_hand_model_path() -> Path:
    """Find existing hand landmarker model or download from Google CDN."""
    base_dir = Path(__file__).resolve().parent
    candidate_paths = [
        base_dir / "models" / MODEL_FILENAME,
        base_dir.parent / "models" / MODEL_FILENAME,
        Path("C:/SOS-AI-Sensor/models") / MODEL_FILENAME,
        Path("models") / MODEL_FILENAME,
    ]

    for p in candidate_paths:
        if p.exists() and p.stat().st_size > 1000000:
            print(f"[GestureAI] Found existing hand model at: {p}")
            return p

    target_dir = base_dir / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / MODEL_FILENAME

    print(f"[GestureAI] Hand model not found locally. Downloading from official Google CDN: {OFFICIAL_MODEL_URL}...")
    try:
        urllib.request.urlretrieve(OFFICIAL_MODEL_URL, str(target_file))
        print(f"[GestureAI] Successfully downloaded hand model to {target_file}")
        return target_file
    except Exception as e:
        print(f"[GestureAI ERROR] Failed to download hand model: {e}")
        raise RuntimeError(f"Could not load or download {MODEL_FILENAME}: {e}")

HAND_MODEL_PATH = resolve_hand_model_path()

# Initialize MediaPipe HandLandmarker in IMAGE running mode
base_options = python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
hand_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(hand_options)
LANDMARKER_LOCK = threading.Lock()
print("[GestureAI] HandLandmarker initialized successfully in IMAGE mode.")

# ====================================================================
# 35-FEATURE EXTRACTION MATHEMATICS
# ====================================================================
def calculate_joint_angle(p1, p2, p3) -> float:
    """Calculate joint angle in radians between p1-p2 and p3-p2."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def create_features(hand_landmarks) -> np.ndarray:
    """
    Extract exact 35 geometric coordinates and features from 21 hand landmarks:
    a) 5 fingertip distances from wrist (tips: 4, 8, 12, 16, 20)
    b) 10 pairwise distances between fingertips
    c) 10 finger joint angles
    d) 5 fingertip heights relative to wrist (y - wrist_y)
    e) 5 fingertip horizontal positions relative to wrist (x - wrist_x)
    """
    coords = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
    wrist = coords[0]
    tips = [4, 8, 12, 16, 20]
    features = []

    # a) 5 fingertip distances from wrist
    for tip in tips:
        dist = np.linalg.norm(np.array(coords[tip]) - np.array(wrist))
        features.append(float(dist))

    # b) 10 pairwise distances between fingertips
    for t1, t2 in combinations(tips, 2):
        dist = np.linalg.norm(np.array(coords[t1]) - np.array(coords[t2]))
        features.append(float(dist))

    # c) 10 finger joint angles
    joint_triplets = [
        (1, 2, 3), (2, 3, 4),
        (5, 6, 7), (6, 7, 8),
        (9, 10, 11), (10, 11, 12),
        (13, 14, 15), (14, 15, 16),
        (17, 18, 19), (18, 19, 20),
    ]
    for j1, j2, j3 in joint_triplets:
        ang = calculate_joint_angle(coords[j1], coords[j2], coords[j3])
        features.append(float(ang))

    # d) 5 fingertip heights relative to wrist
    for tip in tips:
        features.append(float(coords[tip][1] - wrist[1]))

    # e) 5 fingertip horizontal positions relative to wrist
    for tip in tips:
        features.append(float(coords[tip][0] - wrist[0]))

    return np.array(features, dtype=np.float32)

# ====================================================================
# MACHINE LEARNING CLASSIFIER AUTO-LOAD / AUTO-TRAIN
# ====================================================================
def initialize_classifier() -> RandomForestClassifier:
    """
    Loads pre-trained model or automatically trains RandomForestClassifier(300)
    from hand_features.csv to guarantee 100% version compatibility.
    """
    base_dir = Path(__file__).resolve().parent
    model_pkl_paths = [
        base_dir / "models" / "gesture_features_model.pkl",
        base_dir / "gesture_features_model.pkl",
        Path("C:/SOS-AI-Sensor/gesture_features_model.pkl"),
    ]
    csv_paths = [
        base_dir / "hand_features.csv",
        base_dir / "models" / "hand_features.csv",
        Path("c:/SOS-SYSTEM2.0/gesture_service/hand_features.csv"),
    ]

    # Try loading pre-trained pkl first
    for mpkl in model_pkl_paths:
        if mpkl.exists() and mpkl.stat().st_size > 100000:
            try:
                clf = joblib.load(str(mpkl))
                if hasattr(clf, "classes_") and len(clf.classes_) >= 3:
                    print(f"[GestureAI] Pre-trained classifier loaded successfully from {mpkl}")
                    print(f"[GestureAI] Classes: {clf.classes_}")
                    return clf
            except Exception as e:
                print(f"[GestureAI] Warning loading {mpkl} ({e}). Falling back to auto-training from dataset...")

    # Fallback to training fresh RandomForest from hand_features.csv
    for cp in csv_paths:
        if cp.exists() and cp.stat().st_size > 1000:
            print(f"[GestureAI] Training fresh RandomForestClassifier from {cp}...")
            df = pd.read_csv(str(cp))
            feature_cols = [str(i) for i in range(35)]
            X = df[feature_cols].values
            y = df["label"].values

            clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
            clf.fit(X, y)
            print(f"[GestureAI] Trained successfully on {len(df)} samples! Classes: {clf.classes_}")

            # Save to models folder for fast reuse
            out_model = base_dir / "models" / "gesture_features_model.pkl"
            out_model.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, str(out_model))
            return clf

    raise RuntimeError("Could not find gesture_features_model.pkl or hand_features.csv to initialize classifier!")

classifier = initialize_classifier()

# ====================================================================
# AUTO-PUSH ALERT TO CENTRAL HUB
# ====================================================================
def dispatch_central_gesture_alert(sector: str, gesture_class: str, confidence: float):
    """Asynchronously post confirmed SOS gesture alert to Central Alert Hub."""
    def _worker():
        payload = {
            "type": "SOS HAND GESTURE",
            "severity": "HIGH",
            "message": (
                f"Visual SOS Hand Signal (Distress Gesture) confirmed on optical telemetry feed. "
                f"Class: {gesture_class} (Confidence: {confidence * 100:.1f}%)"
            ),
            "sector": sector,
            "source": "GESTURE_AI",
        }
        try:
            resp = requests.post(CENTRAL_ALERT_URL, json=payload, timeout=2.0)
            print(f"[GestureAI -> Central] Alert dispatched: Status {resp.status_code}")
        except Exception as err:
            print(f"[GestureAI -> Central WARN] Failed to reach Central Hub: {err}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

# ====================================================================
# API ENDPOINTS
# ====================================================================

# 1. Health Check
@app.get("/", tags=["Health"])
def health_check():
    """Health check returning service and classifier status."""
    return {
        "status": "ONLINE",
        "service": "SOS Hand Gesture AI",
        "model_trained": True,
        "classes": list(classifier.classes_),
        "n_features": 35,
        "central_alert_url": CENTRAL_ALERT_URL,
    }

# 2. Gesture Detection Endpoint (Supports JSON with Base64 OR Multipart Upload)
@app.post("/api/detect-gesture", tags=["Gesture Detection"])
@app.post("/detect-gesture", tags=["Gesture Detection"])
async def detect_gesture(
    request: Request,
    file: Optional[UploadFile] = File(None),
    sector_form: Optional[str] = Form(None),
):
    """
    Analyze frame for hand gestures using MediaPipe + 35-feature Random Forest.
    Dispatches HIGH severity alert to Central Hub if universal SOS signal is confirmed.
    """
    global last_alert_time
    frame_bgr: Optional[np.ndarray] = None
    sector_name: str = "DRONE-01 [AERIAL]"

    # Case A: Multipart form upload
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

    # If no hand detected
    if not detection_result.hand_landmarks or len(detection_result.hand_landmarks) == 0:
        return {
            "gesture_detected": False,
            "gesture": "NO_HAND",
            "confidence": 0.0,
            "probabilities": {},
            "status": "CLEAR",
            "hand_detected": False,
        }

    # Extract 35 features from first hand
    hand_landmarks = detection_result.hand_landmarks[0]
    features_35 = create_features(hand_landmarks)
    X_input = features_35.reshape(1, -1)

    # Predict class & probabilities
    prediction = classifier.predict(X_input)[0]
    probs = classifier.predict_proba(X_input)[0]
    confidence = float(np.max(probs))

    prob_dict = {cls_name: round(float(p), 3) for cls_name, p in zip(classifier.classes_, probs)}

    # Check for SOS distress gesture condition
    is_sos = (prediction == "SOS" and confidence > 0.65)
    alert_dispatched = False

    if is_sos:
        now = time.time()
        with ALERT_LOCK:
            if (now - last_alert_time) >= ALERT_COOLDOWN_SECONDS:
                last_alert_time = now
                dispatch_central_gesture_alert(sector_name, prediction, confidence)
                alert_dispatched = True

    status_str = (
        "SOS SIGNAL CONFIRMED" if is_sos
        else f"ACTIVE GESTURE: {prediction}"
    )

    return {
        "gesture_detected": is_sos,
        "gesture": prediction,
        "confidence": round(confidence, 3),
        "probabilities": prob_dict,
        "status": status_str,
        "alert_dispatched": alert_dispatched,
        "hand_detected": True,
    }

# ====================================================================
# ENTRYPOINT FOR DIRECT EXECUTION
# ====================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
