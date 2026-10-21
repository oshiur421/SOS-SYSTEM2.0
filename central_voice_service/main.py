"""
====================================================================
SOS SYSTEM 2.0 — Central Alert & Voice Distress Detection Hub
====================================================================
Microservice 1: Central State Management, In-Memory Telemetry Database,
Duplicate Cooldown Filtering, & Emergency Voice NLP Detection Engine.
Deployable to Render (e.g., https://sos-ai-sensor-1.onrender.com)
and testable locally on port 8000.
====================================================================
"""

import time
import uuid
import random
import re
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ====================================================================
# FASTAPI APP INITIALIZATION & CORS SETUP
# ====================================================================
app = FastAPI(
    title="SOS SYSTEM 2.0 — Central Alert & Voice Hub",
    description="Mission-critical alert telemetry queue and multilingual emergency voice detection service.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable ultra-permissive CORS for zero-friction local and production cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================================================
# IN-MEMORY TELEMETRY DATABASE & THREAD SAFETY
# ====================================================================
ALERTS_DB: List[Dict[str, Any]] = []
DB_LOCK = threading.Lock()

# Smart Duplicate Cooldown Tracking (4 seconds per alert type)
COOLDOWN_SECONDS: float = 4.0
LAST_ALERT_TIMESTAMPS: Dict[str, float] = {}
LAST_ALERT_OBJECTS: Dict[str, Dict[str, Any]] = {}

# Emergency Voice Distress Keywords (Multilingual: English + Bengali + Phonetic variants)
EMERGENCY_KEYWORDS: List[str] = [
    "help",
    "bachao",
    "bachau",
    "banchao",
    "bacha",
    "বাঁচাও",
    "বাঁচান",
    "হেল্প",
    "সাহায্য",
    "emergency",
    "save me",
    "fire",
    "danger",
    "police",
    "attack",
    "sos",
]

# Default Sector & Geolocation Constants
DEFAULT_SECTOR = "DRONE-01 // SECTOR 7"
DEFAULT_LOCATION = "SECTOR 7 [GAZIPUR/DHAKA]"
BASE_LAT = 23.8759
BASE_LNG = 90.3795

# ====================================================================
# PYDANTIC DATA SCHEMAS
# ====================================================================
class AlertCreateRequest(BaseModel):
    id: Optional[str] = None
    type: str = Field(default="MANUAL SOS DISPATCH", description="Tactical alert type / headline")
    severity: str = Field(default="CRITICAL", description="CRITICAL, HIGH, MODERATE, or LOW")
    message: Optional[str] = Field(default=None, description="Detailed tactical alert description")
    sector: Optional[str] = Field(default=DEFAULT_SECTOR, description="Active surveillance sector or camera")
    location: Optional[str] = Field(default=DEFAULT_LOCATION, description="Geo-location tag")
    lat: Optional[float] = None
    lng: Optional[float] = None
    status: Optional[str] = Field(default="ACTIVE", description="ACTIVE, ACK'D, or RESOLVED")
    source: Optional[str] = Field(default="MANUAL", description="VOICE_AI, FALL_AI, GESTURE_AI, or MANUAL")

class VoiceDetectRequest(BaseModel):
    transcript: Optional[str] = Field(default="", description="Recognized speech text")
    text: Optional[str] = Field(default=None, description="Alternative field for speech text")
    sector: Optional[str] = Field(default=DEFAULT_SECTOR, description="Originating microphone/sector")
    location: Optional[str] = Field(default=DEFAULT_LOCATION, description="Geo-location tag")

class AcknowledgeRequest(BaseModel):
    alert_id: Optional[str] = Field(default=None, description="Target alert ID, or null/empty for all")

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================
def generate_tactical_id() -> str:
    """Generate unique tactical ID (e.g., 'SOS-8F4A21')."""
    return f"SOS-{uuid.uuid4().hex[:6].upper()}"

def build_alert_object(
    alert_type: str,
    severity: str,
    message: str,
    sector: Optional[str] = None,
    location: Optional[str] = None,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    source: str = "MANUAL",
    status: str = "ACTIVE",
    custom_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a tactical alert compliant with frontend schema."""
    now = datetime.now()
    
    # Slight randomized offset around Dhaka/Gazipur if not provided
    calculated_lat = lat if lat is not None else round(BASE_LAT + random.uniform(-0.012, 0.012), 6)
    calculated_lng = lng if lng is not None else round(BASE_LNG + random.uniform(-0.012, 0.012), 6)

    return {
        "id": custom_id or generate_tactical_id(),
        "type": alert_type.strip().upper(),
        "severity": severity.strip().upper(),
        "message": message.strip(),
        "sector": sector or DEFAULT_SECTOR,
        "location": location or DEFAULT_LOCATION,
        "lat": calculated_lat,
        "lng": calculated_lng,
        "timestamp": now.strftime("%H:%M:%S"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status.strip().upper(),
        "source": source.strip().upper(),
    }

def check_cooldown(alert_type: str) -> Optional[Dict[str, Any]]:
    """
    Check if an alert of the same type arrived within COOLDOWN_SECONDS (4s).
    Returns the existing alert object if spam is detected, or None if clear.
    """
    normalized_type = alert_type.strip().upper()
    now_ts = time.time()
    last_ts = LAST_ALERT_TIMESTAMPS.get(normalized_type, 0.0)

    if (now_ts - last_ts) < COOLDOWN_SECONDS:
        return LAST_ALERT_OBJECTS.get(normalized_type)
    return None

def record_alert_timestamp(alert_type: str, alert_obj: Dict[str, Any]) -> None:
    """Record timestamp and latest object for cooldown tracking."""
    normalized_type = alert_type.strip().upper()
    LAST_ALERT_TIMESTAMPS[normalized_type] = time.time()
    LAST_ALERT_OBJECTS[normalized_type] = alert_obj

def compute_counts() -> Dict[str, int]:
    """Compute real-time alert severity breakdown."""
    counts = {
        "CRITICAL": 0,
        "HIGH": 0,
        "MODERATE": 0,
        "LOW": 0,
        "TOTAL": len(ALERTS_DB),
    }
    for item in ALERTS_DB:
        sev = item.get("severity", "").upper()
        if sev in counts:
            counts[sev] += 1
        elif sev == "MOD":
            counts["MODERATE"] += 1
    return counts

# ====================================================================
# API ENDPOINTS
# ====================================================================

# --- 1. Health Check ---
@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint indicating service state and active telemetry."""
    active_count = sum(1 for a in ALERTS_DB if a.get("status") == "ACTIVE")
    return {
        "status": "ONLINE",
        "service": "SOS Central Alert & Voice Hub",
        "active_alerts": active_count,
        "total_alerts": len(ALERTS_DB),
        "cooldown_window_sec": COOLDOWN_SECONDS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# --- 2. Alerts Queue Retrieval ---
@app.get("/api/alerts", tags=["Alerts"])
@app.get("/alerts", tags=["Alerts"])
def get_alerts():
    """
    Fetch all recorded alerts and severity statistics.
    100% compatible with frontend app.js poll loop.
    """
    with DB_LOCK:
        alerts_copy = list(ALERTS_DB)
        counts = compute_counts()

    return {
        "alerts": alerts_copy,
        "counts": counts,
        "server_time": datetime.now().strftime("%H:%M:%S"),
    }

# --- 3. Alert Ingestion (From Fall Backend, Gesture Backend, or Manual Frontend) ---
@app.post("/api/alerts", tags=["Alerts"])
@app.post("/alerts", tags=["Alerts"])
async def create_alert(payload: AlertCreateRequest):
    """
    Ingest a new emergency alert. Applies thread-safe 4-second cooldown filter.
    Inserts at the top of ALERTS_DB (insert(0, ...)).
    """
    # 1. Cooldown spam check
    existing_alert = check_cooldown(payload.type)
    if existing_alert:
        return {
            "status": "COOLDOWN_IGNORED",
            "message": f"Alert cooldown active ({COOLDOWN_SECONDS}s). Duplicate ignored.",
            "alert": existing_alert,
        }

    # 2. Build tactical alert
    default_msg = f"{payload.type} triggered at {payload.sector or DEFAULT_SECTOR}"
    new_alert = build_alert_object(
        alert_type=payload.type,
        severity=payload.severity or "CRITICAL",
        message=payload.message or default_msg,
        sector=payload.sector,
        location=payload.location,
        lat=payload.lat,
        lng=payload.lng,
        source=payload.source or "MANUAL",
        status=payload.status or "ACTIVE",
        custom_id=payload.id,
    )

    # 3. Thread-safe prepend
    with DB_LOCK:
        ALERTS_DB.insert(0, new_alert)
        record_alert_timestamp(payload.type, new_alert)

    return {
        "status": "SUCCESS",
        "message": "Alert logged and dispatched",
        "alert": new_alert,
        "total_active": sum(1 for a in ALERTS_DB if a.get("status") == "ACTIVE"),
    }

# --- 4. Voice Distress Detection Engine ---
@app.post("/api/detect-voice", tags=["Voice AI"])
@app.post("/detect-voice", tags=["Voice AI"])
async def detect_voice_distress(payload: VoiceDetectRequest):
    """
    Analyze spoken transcript for emergency distress keywords in Bengali and English.
    If matched, generates a CRITICAL severity 'VOICE DISTRESS DETECTED' alert.
    """
    raw_text = (payload.transcript or payload.text or "").strip()
    if not raw_text:
        return {"detected": False, "status": "CLEAR", "message": "No voice transcript received"}

    text_lower = raw_text.lower()

    # Search for emergency keywords via substring & regex word boundary
    matched_keyword: Optional[str] = None
    for kw in EMERGENCY_KEYWORDS:
        kw_clean = kw.lower().strip()
        # Word boundary or substring search
        pattern = r"(?:\b|_|^)" + re.escape(kw_clean) + r"(?:\b|_|$)"
        if re.search(pattern, text_lower, re.IGNORECASE) or kw_clean in text_lower:
            matched_keyword = kw
            break

    if not matched_keyword:
        return {
            "detected": False,
            "status": "CLEAR",
            "transcript_analyzed": raw_text,
        }

    alert_type = "VOICE DISTRESS DETECTED"

    # Cooldown check for voice distress
    existing_alert = check_cooldown(alert_type)
    if existing_alert:
        return {
            "detected": True,
            "keyword": matched_keyword,
            "status": "COOLDOWN_IGNORED",
            "message": f"Voice alert cooldown active ({COOLDOWN_SECONDS}s)",
            "alert": existing_alert,
        }

    # Construct tactical voice alert
    alert_msg = (
        f"Acoustic distress trigger: '{matched_keyword.upper()}' detected in transcript: \"{raw_text}\""
    )
    new_alert = build_alert_object(
        alert_type=alert_type,
        severity="CRITICAL",
        message=alert_msg,
        sector=payload.sector or DEFAULT_SECTOR,
        location=payload.location or DEFAULT_LOCATION,
        source="VOICE_AI",
        status="ACTIVE",
    )

    with DB_LOCK:
        ALERTS_DB.insert(0, new_alert)
        record_alert_timestamp(alert_type, new_alert)

    return {
        "detected": True,
        "keyword": matched_keyword,
        "transcript": raw_text,
        "alert": new_alert,
    }

# --- 5. Acknowledge Alerts (All or Specific ID) ---
@app.post("/api/alerts/acknowledge", tags=["Alerts"])
@app.post("/alerts/acknowledge", tags=["Alerts"])
async def acknowledge_alerts(payload: Optional[AcknowledgeRequest] = None):
    """Acknowledge either a single alert by ID or all active alerts."""
    target_id = payload.alert_id if payload else None
    updated_count = 0

    with DB_LOCK:
        for alert in ALERTS_DB:
            if target_id:
                if alert.get("id") == target_id:
                    alert["status"] = "ACK'D"
                    return {"status": "SUCCESS", "message": f"Alert {target_id} acknowledged", "alert": alert}
            else:
                if alert.get("status") == "ACTIVE":
                    alert["status"] = "ACK'D"
                    updated_count += 1

    return {
        "status": "SUCCESS",
        "message": f"Acknowledged {updated_count} alerts" if not target_id else "Alert not found",
        "acknowledged_count": updated_count,
    }

@app.put("/api/alerts/{alert_id}/ack", tags=["Alerts"])
@app.post("/api/alerts/{alert_id}/ack", tags=["Alerts"])
@app.post("/api/alerts/{alert_id}/acknowledge", tags=["Alerts"])
def acknowledge_single_alert(alert_id: str):
    """Acknowledge an individual alert via path parameter."""
    with DB_LOCK:
        for alert in ALERTS_DB:
            if alert.get("id") == alert_id:
                alert["status"] = "ACK'D"
                return {"status": "SUCCESS", "alert": alert}

    raise HTTPException(status_code=404, detail=f"Alert with ID {alert_id} not found")

# --- 6. Resolve Alerts ---
@app.post("/api/alerts/{alert_id}/resolve", tags=["Alerts"])
@app.put("/api/alerts/{alert_id}/resolve", tags=["Alerts"])
@app.post("/alerts/{alert_id}/resolve", tags=["Alerts"])
def resolve_alert(alert_id: str):
    """Mark an individual alert as RESOLVED."""
    with DB_LOCK:
        for alert in ALERTS_DB:
            if alert.get("id") == alert_id:
                alert["status"] = "RESOLVED"
                return {"status": "SUCCESS", "alert": alert}

    raise HTTPException(status_code=404, detail=f"Alert with ID {alert_id} not found")

# --- 7. Purge Alerts (Clear Logs) ---
@app.delete("/api/alerts", tags=["Alerts"])
@app.delete("/alerts", tags=["Alerts"])
@app.post("/api/alerts/purge", tags=["Alerts"])
@app.post("/alerts/purge", tags=["Alerts"])
def purge_alerts():
    """Purge all recorded alerts from in-memory database."""
    with DB_LOCK:
        cleared_count = len(ALERTS_DB)
        ALERTS_DB.clear()
        LAST_ALERT_TIMESTAMPS.clear()
        LAST_ALERT_OBJECTS.clear()

    return {
        "status": "PURGED",
        "cleared_count": cleared_count,
        "active_alerts": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ====================================================================
# SEED INITIAL DEMO DATA (FOR IMMEDIATE TACTICAL PREVIEW)
# ====================================================================
@app.on_event("startup")
def seed_sample_alerts():
    """Seed sample alerts on startup so frontend immediately displays tactical data."""
    sample_alerts = [
        build_alert_object(
            alert_type="HUMAN FALL DETECTED",
            severity="CRITICAL",
            message="Rapid vertical trajectory collapse detected in perimeter gate corridor",
            sector="CAM-01 // SECTOR 4B",
            location="PERIMETER GATE 4B [GAZIPUR]",
            lat=23.8772,
            lng=90.3810,
            source="FALL_AI",
            status="ACTIVE",
        ),
        build_alert_object(
            alert_type="VOICE DISTRESS DETECTED",
            severity="CRITICAL",
            message="Acoustic distress trigger: 'HELP' detected in transcript: \"Help me please\"",
            sector="DRONE-01 // SECTOR 7",
            location="SECTOR 7 [GAZIPUR/DHAKA]",
            lat=23.8759,
            lng=90.3795,
            source="VOICE_AI",
            status="ACTIVE",
        ),
        build_alert_object(
            alert_type="SOS HAND GESTURE",
            severity="HIGH",
            message="Universal signal for help hand gesture recognized by vision sensor",
            sector="CAM-02 // SECTOR 2A",
            location="STORAGE ZONE BRAVO [DHAKA]",
            lat=23.8741,
            lng=90.3778,
            source="GESTURE_AI",
            status="ACK'D",
        ),
    ]
    with DB_LOCK:
        for sa in sample_alerts:
            ALERTS_DB.append(sa)

# ====================================================================
# ENTRYPOINT FOR DIRECT EXECUTION
# ====================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
