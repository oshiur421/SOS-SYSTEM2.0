/* ================================================================
   SOS SYSTEM 2.0 — Real-Time AI Detection & Multi-Backend Sync
   Production-Ready Command Center Frontend Engine
   ================================================================ */

// ====================================================================
// BACKEND CONFIGURATION
// ====================================================================
// Automatically route to local port 8000 when running locally, or Render cloud in production
const isLocalEnv = typeof window !== "undefined" && (
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1" ||
  window.location.protocol === "file:"
);

const BACKEND_CONFIG = {
  CENTRAL_AND_VOICE_API: isLocalEnv ? "http://localhost:8000" : "https://sos-ai-sensor-1.onrender.com",
  FALL_DETECTION_API: isLocalEnv ? "http://localhost:8001" : "https://sos-fall-detector.onrender.com",
  GESTURE_DETECTION_API: isLocalEnv ? "http://localhost:8002" : "https://sos-gesture-detector.onrender.com"
};

// ====================================================================
// CAMERA FEED METADATA
// ====================================================================
const CAM_DATA = [
  { label: "CAM-01 [GATE-4B]", subtitle: "CAM-01 // SECTOR 4B — PERIMETER GATE COVERAGE (FIXED)" },
  { label: "CAM-02 [STORAGE-B]", subtitle: "CAM-02 // SECTOR 2A — STORAGE ZONE BRAVO (WIDE ANGLE)" },
  { label: "CAM-03 [HELIPAD]", subtitle: "CAM-03 // SECTOR 5C — HELIPAD TARMAC (PAN-TILT)" },
  { label: "DRONE-01 [AERIAL]", subtitle: "DRONE-01 // SECTOR 7 — DYNAMIC ORBIT (ALT 120M)" },
];

// ====================================================================
// SOS SYSTEM MAIN MODULE
// ====================================================================
const SOS = (() => {
  // --- State ---
  let audioEnabled = true;
  let aiTrackingEnabled = true;
  let activeCam = 3;
  let currentVision = "optical";
  let currentFilter = "all";
  let currentStatus = "all";
  let searchQuery = "";

  // PTZ state
  let panAngle = 0;
  let tiltAngle = 0;
  let zoomLevel = 1;
  const PAN_STEP = 15;
  const TILT_STEP = 10;
  const ZOOM_STEP = 0.25;
  const MAX_ZOOM = 4;
  const MIN_ZOOM = 1;

  // Video / Canvas
  let videoStream = null;
  let videoEl = null;
  let canvasEl = null;
  let canvasCtx = null;

  // FPS / Bitrate tracking
  let lastFrameTime = 0;
  let frameCount = 0;
  let fpsDisplay = 0;
  let lastFrameBytes = 0;

  // Alerts
  let allAlerts = [];
  let alertIdCounter = 0;
  let knownAlertIds = new Set();

  // Audio context
  let audioCtx = null;

  // Leaflet map
  let map = null;
  let incidentMarkers = [];
  let radarMarker = null;

  // Backend poll intervals
  let centralPollInterval = null;
  let fallPollInterval = null;
  let gesturePollInterval = null;
  let frameSendInterval = null;
  let luxInterval = null;
  let fpsInterval = null;

  // ================================================================
  // INITIALIZATION
  // ================================================================
  function init() {
    videoEl = document.getElementById("liveVideo");
    canvasEl = document.getElementById("analysisCanvas");
    canvasCtx = canvasEl.getContext("2d", { willReadFrequently: true });

    initWebcam();
    initMap();
    renderEmptyState();
    startPolling();
    startFPSCounter();
    startLuxSensor();

    // Frame sender for AI backends (500ms intervals)
    frameSendInterval = setInterval(sendFrameToBackends, 500);
  }

  // ================================================================
  // WEBCAM
  // ================================================================
  async function initWebcam() {
    try {
      videoStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false
      });
      videoEl.srcObject = videoStream;
      videoEl.play();
    } catch (err) {
      console.warn("SOS: Webcam access denied or unavailable:", err.message);
    }
  }

  // ================================================================
  // THEME TOGGLE
  // ================================================================
  function toggleTheme() {
    const html = document.documentElement;
    const isLight = html.classList.contains("light");
    html.classList.toggle("dark", isLight);
    html.classList.toggle("light", !isLight);
    document.getElementById("themeIcon").textContent = isLight ? "☀" : "🌙";
    document.getElementById("themeLabel").textContent = isLight ? "LIGHT MODE" : "DARK MODE";

    // Re-init map tiles
    if (map) {
      map.eachLayer(layer => {
        if (layer._url) map.removeLayer(layer);
      });
      addMapTiles();
    }
  }

  // ================================================================
  // AUDIO TOGGLE
  // ================================================================
  function toggleAudio() {
    audioEnabled = !audioEnabled;
    document.getElementById("audioIcon").textContent = audioEnabled ? "🔊" : "🔇";
    document.getElementById("audioLabel").textContent = audioEnabled ? "AUDIO ON" : "AUDIO MUTED";
  }

  function playAlertSound() {
    if (!audioEnabled) return;
    try {
      if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      // Tactical beep sequence
      const now = audioCtx.currentTime;
      for (let i = 0; i < 3; i++) {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.type = "square";
        osc.frequency.value = i % 2 === 0 ? 880 : 660;
        gain.gain.setValueAtTime(0.12, now + i * 0.18);
        gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.18 + 0.15);
        osc.start(now + i * 0.18);
        osc.stop(now + i * 0.18 + 0.15);
      }
    } catch (e) { /* Audio not supported */ }
  }

  // ================================================================
  // CAMERA SWITCHER
  // ================================================================
  function switchCam(idx) {
    activeCam = idx;
    document.getElementById("camSubtitle").textContent = CAM_DATA[idx].subtitle;
    document.querySelectorAll(".cam-tab").forEach((tab, i) => {
      tab.classList.toggle("active", i === idx);
    });
  }

  // ================================================================
  // VISION MODES
  // ================================================================
  function setVision(mode) {
    currentVision = mode;
    document.querySelectorAll(".vision-btn").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.vision === mode);
    });
    videoEl.classList.remove("night-vision-filter", "thermal-filter");
    if (mode === "night") videoEl.classList.add("night-vision-filter");
    if (mode === "thermal") videoEl.classList.add("thermal-filter");
  }

  // ================================================================
  // PTZ CONTROLS
  // ================================================================
  function ptz(action) {
    switch (action) {
      case "left":  panAngle = Math.max(panAngle - PAN_STEP, -180); break;
      case "right": panAngle = Math.min(panAngle + PAN_STEP, 180); break;
      case "up":    tiltAngle = Math.max(tiltAngle - TILT_STEP, -90); break;
      case "down":  tiltAngle = Math.min(tiltAngle + TILT_STEP, 90); break;
      case "zoomin":  zoomLevel = Math.min(zoomLevel + ZOOM_STEP, MAX_ZOOM); break;
      case "zoomout": zoomLevel = Math.max(zoomLevel - ZOOM_STEP, MIN_ZOOM); break;
      case "reset":   panAngle = 0; tiltAngle = 0; zoomLevel = 1; break;
    }
    // Apply CSS transform
    const translateX = -(panAngle / 180) * 30;
    const translateY = (tiltAngle / 90) * 20;
    videoEl.style.transform = `scale(${zoomLevel}) translate(${translateX}%, ${translateY}%)`;

    // Update telemetry
    document.getElementById("panAngleText").textContent = `PAN ANGLE: ${panAngle.toFixed(1)}°`;
    document.getElementById("tiltAngleText").textContent = `TILT ANGLE: ${tiltAngle.toFixed(1)}°`;
  }

  // ================================================================
  // AI TRACKING TOGGLE
  // ================================================================
  function toggleAiTracking() {
    aiTrackingEnabled = !aiTrackingEnabled;
    const btn = document.getElementById("btnAiTracking");
    if (aiTrackingEnabled) {
      btn.classList.remove("bg-gray-800/50", "border-gray-600", "text-gray-500");
      btn.classList.add("bg-emerald-900/40", "border-emerald-600", "text-emerald-400");
    } else {
      btn.classList.remove("bg-emerald-900/40", "border-emerald-600", "text-emerald-400");
      btn.classList.add("bg-gray-800/50", "border-gray-600", "text-gray-500");
    }
  }

  // ================================================================
  // SNAPSHOT
  // ================================================================
  function snapshot() {
    if (!videoEl || videoEl.readyState < 2) return;
    canvasEl.width = videoEl.videoWidth || 640;
    canvasEl.height = videoEl.videoHeight || 480;
    canvasCtx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
    const link = document.createElement("a");
    link.download = `SOS_SNAPSHOT_${Date.now()}.png`;
    link.href = canvasEl.toDataURL("image/png");
    link.click();
  }

  // ================================================================
  // FPS COUNTER & BITRATE ESTIMATOR
  // ================================================================
  function startFPSCounter() {
    let prevTime = performance.now();
    let frames = 0;

    function tick() {
      frames++;
      const now = performance.now();
      const elapsed = now - prevTime;
      if (elapsed >= 1000) {
        fpsDisplay = frames;
        frames = 0;
        prevTime = now;

        document.getElementById("fpsBadge").textContent = `${fpsDisplay.toFixed(1)} FPS`;

        // Estimate bitrate from canvas data (rough approximation)
        if (videoEl && videoEl.readyState >= 2) {
          canvasEl.width = 320;
          canvasEl.height = 240;
          canvasCtx.drawImage(videoEl, 0, 0, 320, 240);
          const dataUrl = canvasEl.toDataURL("image/jpeg", 0.5);
          const byteLen = Math.round((dataUrl.length - 22) * 0.75);
          const mbps = ((byteLen * 8 * fpsDisplay) / 1e6).toFixed(2);
          document.getElementById("bitrateBadge").textContent = `${mbps} Mbps`;
        }
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  // ================================================================
  // AMBIENT LIGHT (LUX) SENSOR FROM FRAME BRIGHTNESS
  // ================================================================
  function startLuxSensor() {
    luxInterval = setInterval(() => {
      if (!videoEl || videoEl.readyState < 2) return;
      canvasEl.width = 160;
      canvasEl.height = 120;
      canvasCtx.drawImage(videoEl, 0, 0, 160, 120);
      const imgData = canvasCtx.getImageData(0, 0, 160, 120);
      const pixels = imgData.data;
      let totalBrightness = 0;
      const pixelCount = pixels.length / 4;
      for (let i = 0; i < pixels.length; i += 4) {
        totalBrightness += (pixels[i] * 0.299 + pixels[i + 1] * 0.587 + pixels[i + 2] * 0.114);
      }
      const avgBrightness = totalBrightness / pixelCount;
      const lux = (avgBrightness / 255 * 100).toFixed(1);
      document.getElementById("luxText").textContent = `AMBIENT LIGHT: ${lux} LUX`;

      // Simulated thermal from brightness
      const temp = (18 + avgBrightness / 255 * 12).toFixed(1);
      document.getElementById("thermalText").textContent = `THERMAL SENSOR: CALIBRATED (${temp}°C)`;
    }, 2000);
  }

  // ================================================================
  // SEND FRAMES TO AI BACKENDS (500MS BASE64 STREAM)
  // ================================================================
  let isSendingFrame = false;

  async function sendFrameToBackends() {
    if (!aiTrackingEnabled || !videoEl || videoEl.readyState < 2 || isSendingFrame) return;
    isSendingFrame = true;

    try {
      canvasEl.width = videoEl.videoWidth || 640;
      canvasEl.height = videoEl.videoHeight || 480;
      canvasCtx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);

      const base64Data = canvasEl.toDataURL("image/jpeg", 0.6);
      const currentSector = CAM_DATA[activeCam]?.subtitle?.split("//")[1]?.trim()?.split("—")[0]?.trim() || "DRONE-01 // SECTOR 7";

      // 1. Send Base64 to Fall Detection API
      try {
        const fallRes = await fetch(`${BACKEND_CONFIG.FALL_DETECTION_API}/api/detect-fall`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: base64Data,
            sector: currentSector
          })
        });

        if (fallRes.ok) {
          const data = await fallRes.json();
          if (data.fall_detected) {
            createAlertFromDetection(
              "HUMAN FALL DETECTED",
              "CRITICAL",
              `Subject posture collapse detected! Body Angle: ${data.angle}° (Delta: ${data.angle_change}°). Status: ${data.status}`,
              data
            );
          }
        }
      } catch (err) {
        // Fall service may be initializing
      }

      // 2. Send Base64 to Gesture Detection API
      try {
        const gestRes = await fetch(`${BACKEND_CONFIG.GESTURE_DETECTION_API}/api/detect-gesture`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: base64Data,
            sector: currentSector
          })
        });

        if (gestRes.ok) {
          const data = await gestRes.json();
          if (data.gesture_detected || (data.gesture === "SOS" && data.confidence > 0.65)) {
            createAlertFromDetection(
              "SOS HAND GESTURE",
              "HIGH",
              `Visual SOS Hand Signal (Distress Gesture) confirmed on optical telemetry feed. Class: ${data.gesture} (Confidence: ${(data.confidence * 100).toFixed(1)}%)`,
              data
            );
          }
        }
      } catch (err) {
        // Gesture service may be initializing
      }

    } finally {
      isSendingFrame = false;
    }
  }

  // ================================================================
  // POLL CENTRAL/VOICE API FOR ALERTS
  // ================================================================
  function startPolling() {
    // Poll central API every 1s
    centralPollInterval = setInterval(async () => {
      try {
        const res = await fetch(`${BACKEND_CONFIG.CENTRAL_AND_VOICE_API}/alerts`);
        if (res.ok) {
          const data = await res.json();
          const alerts = Array.isArray(data) ? data : (data.alerts || []);
          processIncomingAlerts(alerts);
        }
      } catch (e) { /* Central API unreachable */ }
    }, 1000);
  }

  function processIncomingAlerts(incoming) {
    let newCount = 0;
    let statusChanged = false;
    incoming.forEach(alert => {
      const id = alert.id || alert._id || `backend-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      if (!knownAlertIds.has(id)) {
        knownAlertIds.add(id);
        const normalized = normalizeAlert(alert, id);
        allAlerts.unshift(normalized);
        newCount++;
      } else {
        // Sync status if backend marked alert as ACK'D or RESOLVED
        const existing = allAlerts.find(a => a.id === id);
        if (existing && alert.status) {
          const rawStatus = (alert.status || "").toUpperCase();
          const newStatus = (rawStatus === "ACK'D" || rawStatus === "ACKNOWLEDGED") ? "acknowledged" :
                            (rawStatus === "RESOLVED") ? "resolved" : "active";
          if (existing.status !== newStatus) {
            existing.status = newStatus;
            statusChanged = true;
          }
        }
      }
    });
    if (newCount > 0) {
      playAlertSound();
      renderAlerts();
      updateMapIncidents();
    } else if (statusChanged) {
      renderAlerts();
      updateMapIncidents();
    }
  }

  function normalizeAlert(raw, id) {
    const type = (raw.type || raw.alert_type || raw.event_type || "UNKNOWN").toUpperCase();
    let severity = (raw.severity || "").toUpperCase();
    if (!severity) {
      if (type.includes("FALL") || type.includes("VOICE")) severity = "CRITICAL";
      else if (type.includes("SOS") || type.includes("GESTURE")) severity = "HIGH";
      else severity = "MODERATE";
    }

    const rawStatus = (raw.status || "ACTIVE").toUpperCase();
    const normalizedStatus = (rawStatus === "ACK'D" || rawStatus === "ACKNOWLEDGED") ? "acknowledged" :
                             (rawStatus === "RESOLVED") ? "resolved" : "active";

    return {
      id: id,
      type: type,
      severity: severity,
      message: raw.message || raw.description || raw.detected_message || type,
      sector: raw.sector || raw.location || `SECTOR ${Math.floor(Math.random() * 9) + 1}`,
      timestamp: raw.timestamp || new Date().toISOString(),
      status: normalizedStatus,
      lat: raw.lat || (23.8759 + (Math.random() - 0.5) * 0.02),
      lng: raw.lng || (90.3795 + (Math.random() - 0.5) * 0.02),
    };
  }

  function createAlertFromDetection(type, severity, message, rawData) {
    const id = `local-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    if (knownAlertIds.has(id)) return;
    knownAlertIds.add(id);

    const alert = {
      id,
      type,
      severity,
      message,
      sector: CAM_DATA[activeCam]?.subtitle?.split("//")[1]?.trim()?.split("—")[0]?.trim() || "SECTOR 7",
      timestamp: new Date().toISOString(),
      status: "active",
      lat: 23.8759 + (Math.random() - 0.5) * 0.02,
      lng: 90.3795 + (Math.random() - 0.5) * 0.02,
    };
    allAlerts.unshift(alert);
    playAlertSound();
    renderAlerts();
    updateMapIncidents();
  }

  // ================================================================
  // ALERT RENDERING
  // ================================================================
  function getFilteredAlerts() {
    return allAlerts.filter(a => {
      if (currentFilter !== "all" && a.severity.toLowerCase() !== currentFilter) return false;
      if (currentStatus !== "all" && a.status !== currentStatus) return false;
      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        const searchable = `${a.id} ${a.type} ${a.message} ${a.sector} ${a.severity}`.toLowerCase();
        if (!searchable.includes(q)) return false;
      }
      return true;
    });
  }

  function renderAlerts() {
    const container = document.getElementById("alertQueue");
    const filtered = getFilteredAlerts();

    updateCounters();

    if (allAlerts.length === 0) {
      renderEmptyState();
      return;
    }

    if (filtered.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="shield-icon">🔍</div>
          <p class="text-sm font-bold text-gray-400 mb-1">NO MATCHING ALERTS</p>
          <p class="text-[10px] text-gray-600 max-w-md">No alerts match the current filter criteria. Adjust filters or search query.</p>
        </div>`;
      updateFooter(0);
      return;
    }

    container.innerHTML = filtered.map(a => buildAlertCard(a)).join("");
    updateFooter(filtered.length);
  }

  function renderEmptyState() {
    const container = document.getElementById("alertQueue");
    container.innerHTML = `
      <div class="empty-state">
        <div class="shield-icon">🛡️</div>
        <p class="text-sm font-bold text-gray-300 mb-1">NO ALERTS IN SYSTEM</p>
        <p class="text-[10px] text-gray-500 max-w-lg leading-relaxed">
          All monitored sectors operating within expected parameters. Alerts strictly sync from FastAPI backend at
          <span class="text-sos-cyan">${BACKEND_CONFIG.CENTRAL_AND_VOICE_API}</span>.
        </p>
      </div>`;
    updateCounters();
    updateFooter(0);
  }

  function buildAlertCard(a) {
    const sevClass = a.severity.toLowerCase();
    const timeStr = formatTimestamp(a.timestamp);
    const statusBadge = a.status === "acknowledged"
      ? '<span class="alert-badge" style="background:#422006;color:#fbbf24;border:1px solid #ca8a04;">ACK\'D</span>'
      : a.status === "resolved"
        ? '<span class="alert-badge" style="background:#052e16;color:#4ade80;border:1px solid #16a34a;">RESOLVED</span>'
        : '';

    return `
      <div class="alert-card alert-card-enter severity-${sevClass}" data-id="${a.id}">
        <div class="flex items-center justify-between mb-1.5">
          <div class="flex items-center gap-2">
            <span class="alert-badge ${sevClass}">${a.severity}</span>
            <span class="text-[10px] font-bold text-white">${a.type}</span>
            ${statusBadge}
          </div>
          <span class="text-[9px] text-gray-500 font-semibold">${timeStr}</span>
        </div>
        <div class="flex items-center justify-between">
          <div class="min-w-0">
            <p class="text-[10px] text-gray-400 truncate"><span class="text-gray-500">SECTOR:</span> ${a.sector}</p>
            <p class="text-[10px] text-gray-400 truncate mt-0.5">${a.message}</p>
          </div>
          <div class="flex items-center gap-1.5 flex-shrink-0 ml-2">
            ${a.status === "active" ? `
              <button onclick="SOS.ackAlert('${a.id}')" class="alert-action-btn ack-btn">ACK</button>
              <button onclick="SOS.resolveAlert('${a.id}')" class="alert-action-btn resolve-btn">RESOLVE</button>
            ` : a.status === "acknowledged" ? `
              <button onclick="SOS.resolveAlert('${a.id}')" class="alert-action-btn resolve-btn">RESOLVE</button>
            ` : ''}
          </div>
        </div>
      </div>`;
  }

  function formatTimestamp(ts) {
    try {
      const d = new Date(ts);
      return d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" }) +
        " " + d.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" }).toUpperCase();
    } catch { return ts; }
  }

  function updateCounters() {
    const counts = { critical: 0, high: 0, moderate: 0, low: 0 };
    allAlerts.forEach(a => {
      const sev = a.severity.toLowerCase();
      if (counts[sev] !== undefined) counts[sev]++;
    });
    document.getElementById("cntCritical").textContent = counts.critical;
    document.getElementById("cntHigh").textContent = counts.high;
    document.getElementById("cntMod").textContent = counts.moderate;
    document.getElementById("cntLow").textContent = counts.low;
    document.getElementById("filterAllCnt").textContent = allAlerts.length;
  }

  function updateFooter(shown) {
    document.getElementById("alertFooterInfo").textContent =
      `SHOWING ${shown} OF ${allAlerts.length} ALERTS • AUTO-REFRESH: 1 SEC`;
  }

  // ================================================================
  // ALERT ACTIONS & BACKEND SYNC
  // ================================================================
  async function ackAlert(id) {
    const alert = allAlerts.find(a => a.id === id);
    if (alert) {
      alert.status = "acknowledged";
      renderAlerts();
    }
    try {
      await fetch(`${BACKEND_CONFIG.CENTRAL_AND_VOICE_API}/api/alerts/${id}/ack`, { method: "POST" });
    } catch (e) { /* Backend offline */ }
  }

  async function resolveAlert(id) {
    const alert = allAlerts.find(a => a.id === id);
    if (alert) {
      alert.status = "resolved";
      renderAlerts();
      updateMapIncidents();
    }
    try {
      await fetch(`${BACKEND_CONFIG.CENTRAL_AND_VOICE_API}/api/alerts/${id}/resolve`, { method: "POST" });
    } catch (e) { /* Backend offline */ }
  }

  async function ackAll() {
    allAlerts.forEach(a => { if (a.status === "active") a.status = "acknowledged"; });
    renderAlerts();
    try {
      await fetch(`${BACKEND_CONFIG.CENTRAL_AND_VOICE_API}/api/alerts/acknowledge`, { method: "POST" });
    } catch (e) { /* Backend offline */ }
  }

  async function purgeLogs() {
    // Attempt backend purge
    try {
      await fetch(`${BACKEND_CONFIG.CENTRAL_AND_VOICE_API}/api/alerts`, { method: "DELETE" });
    } catch (e) {
      try {
        await fetch(`${BACKEND_CONFIG.CENTRAL_AND_VOICE_API}/alerts`, { method: "DELETE" });
      } catch (err) { /* Offline */ }
    }
    allAlerts = [];
    knownAlertIds.clear();
    renderEmptyState();
    updateMapIncidents();
  }

  async function newDispatch() {
    const currentSector = CAM_DATA[activeCam]?.subtitle?.split("//")[1]?.trim()?.split("—")[0]?.trim() || "DRONE-01 // SECTOR 7";
    const alertData = {
      type: "MANUAL SOS DISPATCH",
      severity: "CRITICAL",
      message: `Manual emergency dispatch triggered by command operator at ${new Date().toLocaleTimeString()}`,
      sector: currentSector,
      source: "MANUAL"
    };

    try {
      const res = await fetch(`${BACKEND_CONFIG.CENTRAL_AND_VOICE_API}/api/alerts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(alertData)
      });
      if (res.ok) {
        const data = await res.json();
        if (data.alert) {
          processIncomingAlerts([data.alert]);
          return;
        }
      }
    } catch (e) {
      console.warn("SOS: Central backend offline, creating local dispatch:", e.message);
    }

    // Fallback if backend offline
    createAlertFromDetection(alertData.type, alertData.severity, alertData.message, {});
  }

  // ================================================================
  // VOICE DISTRESS DETECTION CLIENT
  // ================================================================
  async function detectVoice(transcript) {
    if (!transcript) return null;
    const currentSector = CAM_DATA[activeCam]?.subtitle?.split("//")[1]?.trim()?.split("—")[0]?.trim() || "DRONE-01 // SECTOR 7";
    try {
      const res = await fetch(`${BACKEND_CONFIG.CENTRAL_AND_VOICE_API}/api/detect-voice`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript: transcript,
          sector: currentSector
        })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.detected && data.alert) {
          processIncomingAlerts([data.alert]);
        }
        return data;
      }
    } catch (e) {
      console.warn("SOS: Voice detection endpoint unreachable:", e.message);
    }
    return null;
  }

  // ================================================================
  // FILTERS & SEARCH
  // ================================================================
  function filterAlerts(filter) {
    currentFilter = filter;
    document.querySelectorAll(".filter-pill").forEach(p => {
      p.classList.toggle("active", p.dataset.filter === filter);
    });
    renderAlerts();
  }

  function filterStatus(status) {
    currentStatus = status;
    document.querySelectorAll(".status-tab").forEach(t => {
      t.classList.toggle("active", t.dataset.status === status);
    });
    renderAlerts();
  }

  function searchAlerts() {
    searchQuery = document.getElementById("alertSearch").value;
    renderAlerts();
  }

  // ================================================================
  // EXPORT LOG
  // ================================================================
  function exportLog() {
    if (allAlerts.length === 0) return;
    const jsonStr = JSON.stringify(allAlerts, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const link = document.createElement("a");
    link.download = `SOS_ALERT_LOG_${Date.now()}.json`;
    link.href = URL.createObjectURL(blob);
    link.click();
    URL.revokeObjectURL(link.href);
  }

  // ================================================================
  // LEAFLET MAP
  // ================================================================
  function initMap() {
    map = L.map("radarMap", {
      center: [23.8759, 90.3795],
      zoom: 13,
      zoomControl: true,
      attributionControl: false,
    });
    addMapTiles();

    // Radar pulse marker
    const radarIcon = L.divIcon({
      className: "",
      html: '<div class="radar-pulse"></div>',
      iconSize: [18, 18],
      iconAnchor: [9, 9],
    });
    radarMarker = L.marker([23.8759, 90.3795], { icon: radarIcon }).addTo(map);
  }

  function addMapTiles() {
    const isDark = document.documentElement.classList.contains("dark");
    const tileUrl = isDark
      ? "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
      : "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png";
    L.tileLayer(tileUrl, {
      maxZoom: 19,
      subdomains: "abcd",
    }).addTo(map);
  }

  function updateMapIncidents() {
    // Clear old markers
    incidentMarkers.forEach(m => map.removeLayer(m));
    incidentMarkers = [];

    const activeAlerts = allAlerts.filter(a => a.status !== "resolved");
    const radarStatusEl = document.getElementById("radarStatus");

    if (activeAlerts.length === 0) {
      radarStatusEl.textContent = "NO ACTIVE INCIDENTS";
      radarStatusEl.className = "text-[10px] font-semibold text-gray-500 tracking-wider";
    } else {
      radarStatusEl.textContent = "ACTIVE INCIDENT DETECTED";
      radarStatusEl.className = "text-[10px] font-semibold text-red-400 tracking-wider";
    }

    activeAlerts.forEach(a => {
      const sevClass = a.severity.toLowerCase();
      const icon = L.divIcon({
        className: "",
        html: `<div class="incident-marker ${sevClass}"></div>`,
        iconSize: [14, 14],
        iconAnchor: [7, 7],
      });
      const marker = L.marker([a.lat, a.lng], { icon }).addTo(map);
      marker.bindPopup(`
        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;">
          <strong>${a.type}</strong><br/>
          <span style="color:#888;">${a.sector}</span><br/>
          <span>${a.message}</span>
        </div>
      `);
      incidentMarkers.push(marker);
    });
  }

  // ================================================================
  // PUBLIC API
  // ================================================================
  return {
    init,
    toggleTheme,
    toggleAudio,
    switchCam,
    setVision,
    ptz,
    toggleAiTracking,
    snapshot,
    filterAlerts,
    filterStatus,
    searchAlerts,
    ackAlert,
    resolveAlert,
    ackAll,
    purgeLogs,
    newDispatch,
    detectVoice,
    exportLog,
  };
})();

// ================================================================
// BOOT
// ================================================================
document.addEventListener("DOMContentLoaded", SOS.init);
