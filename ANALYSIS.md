# Kenshō — Codebase Analysis

## Project Overview

**Kenshō** (見性 — "seeing one's true nature") is a browser-based physical therapy and fitness AI platform that performs real-time pose analysis and exercise form correction. The project targets clinical use (therapists + patients) and home fitness, with a Turkish-language backend and interface.

---

## Architecture

```
Kenshō
├── main.py                  FastAPI REST + WebSocket backend
├── pose_analyzer.py         Computer vision core (CV2 + DTW)
├── dashboard.html           Clinical therapist hub (1290 lines)
├── exercises.html           Live exercise tracking (1080 lines)
├── exercises_fixed.html     Enhanced exercise tracker (1186 lines)
├── posture_analyze.html     3D biomechanical analysis (1667 lines)
├── posture_report.html      Before/after comparison (928 lines)
└── data/
    └── templates/
        └── quadruped_hip_extension_template.json   Reference motion data
```

**Stack summary:**
- **Backend:** Python — FastAPI, OpenCV (cv2), NumPy, SciPy, dtaidistance
- **Frontend:** Vanilla JavaScript — TensorFlow.js (MoveNet), Three.js, Canvas API, WebSockets
- **Pose detection (server):** Color-based segmentation (HSV masking)
- **Pose detection (client):** TensorFlow.js MoveNet (17 keypoints)

---

## Backend: `main.py`

A FastAPI server providing REST endpoints and a WebSocket channel for real-time analysis.

### REST API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Status check |
| GET | `/patients` | List all patients |
| GET/POST | `/patients/{id}` | Get/create patient |
| GET/POST | `/programs/{patient_id}` | Get/set exercise program |
| GET | `/sessions/{session_id}` | Session detail |
| GET | `/sessions/patient/{patient_id}` | Patient's sessions |
| GET | `/exercises` | List exercise templates |
| GET | `/exercises/{type}/template` | Template summary |
| POST | `/exercises/{type}/upload-reference` | Upload reference video → build template |
| GET | `/therapist/dashboard/{therapist_id}` | Therapist stats |

### WebSocket: `/ws/session/{patient_id}/{exercise_type}`

Real-time frame-by-frame analysis loop:
1. Client sends base64-encoded video frames as JSON `{type: "frame", data: "..."}`
2. Server decodes frame → extracts keypoints via `BodyTracker`
3. Server sends back `frame_result` with detected overlay coordinates and feedback messages
4. On `end_session` message: runs DTW comparison against reference template, generates full report, persists to `data/sessions/`

### In-memory Database

The current DB is a Python dict (production note in the code says PostgreSQL). Seeded with 3 Turkish patients (`p001–p003`) and one therapist (`t001`).

**Security note:** CORS is configured with `allow_origins=["*"]` — intentionally permissive for development.

---

## Pose Analysis Core: `pose_analyzer.py`

### `BodyTracker`

Color-based keypoint extraction — not model-based. It detects body parts via HSV color segmentation:
- **Yellow** → torso (shirt)
- **Green** → lower body / pants
- **Blue** → exercise mat (defined but unused in keypoint extraction)

Extracted keypoints per frame:
- `torso_x/y` — centroid of yellow (shirt) region
- `hip_x/y` — centroid of green (pants) region
- `ext_leg_x/y` — rightmost point of green region (extended leg tip)
- `ext_leg_height` — normalized height of extended leg (0–1)
- `spine_flatness` — 1.0 = flat, 0 = curved (based on y-variance of torso region)

**Limitation:** This approach is entirely color-dependent. It will fail with different clothing colors and is not generalized pose detection.

### `MovementAnalyzer`

- `extract_time_series()` — aggregates per-frame keypoints into smoothed time series (Savitzky-Golay filter, window=11)
- `detect_phases()` — uses `scipy.signal.find_peaks` to identify rep peaks in `leg_heights` signal
- `compare_with_reference()` — DTW (Dynamic Time Warping) comparison of user vs reference `leg_heights` series; returns distance and similarity %

### `FeedbackEngine`

- `realtime_feedback()` — per-frame rule-based feedback based on `ext_leg_height` and `spine_flatness` thresholds
- `session_report()` — end-of-session report with quality score formula:
  ```
  quality_score = (good_reps/total_reps * 50) + (avg_spine * 30) + (dtw_similarity/100 * 20)
  ```

### `ReferenceTemplateBuilder`

Processes a reference video file → extracts keypoints frame-by-frame → builds and saves a JSON template. The template stores the full time-series data and detected rep phases.

---

## Frontend Pages

### `dashboard.html` — Clinical Therapist Hub

Multi-page SPA for therapists:
- Patient list and profile management
- Live session initiation (connects to WebSocket backend)
- Session reports with DTW similarity, quality scores, grade labels
- Embeds reference template data for comparison visualization

### `exercises.html` / `exercises_fixed.html` — Live Exercise Tracker

Real-time workout form checker:
- Side-by-side reference video vs. live camera feed
- TensorFlow.js MoveNet extracts 17 keypoints from camera
- Angle similarity score: `sim = 100 - (avgAngleDiff * 1.5)`
- Rep detection, real-time feedback messages
- `exercises_fixed.html` is an enhanced version with better video loading, error handling, and debug logging

### `posture_analyze.html` — 3D Biomechanical Analysis

Single-image posture assessment:
- User uploads a lateral (side-view) photo
- TensorFlow.js extracts keypoints from photo
- Three.js renders a 3D skeleton with OrbitControls
- 15+ metrics: forward head posture, spinal curves, shoulder/hip symmetry, knee alignment
- Color-coded joint indicators: green (optimal), amber (caution), red (concern)

### `posture_report.html` — Before/After Comparison

Progress tracking report:
- Upload "before" and "after" posture photos
- Interactive slider to compare images side-by-side
- MediaPipe-based landmark detection with SVG skeleton overlay
- Improvement metrics with sparkline charts
- Preloaded with demo data (patient: Ayşe Hanım)

---

## Reference Template Data

`data/templates/quadruped_hip_extension_template.json`:
- Derived from a 322-frame (~10.7 second) reference video at 30 FPS
- 6 detected reps
- Average peak extension: **0.775** (normalized height)
- Average hold duration: **0.48 seconds** (notably short — ideal is 2+ seconds)
- Contains full smoothed time-series arrays: `leg_heights`, `spine_flat`, `torso_y`

---

## Patient Data

`patient/ayse/` contains before/after posture photos (`ayse_before.png`, `ayse_after.png`) used as demo/test data for `posture_report.html`.

---

## Issues and Observations

### Technical Issues

1. **Color-dependent tracking:** `BodyTracker` relies on specific clothing colors (yellow shirt, green pants). This is brittle and non-generalizable. The frontend uses TensorFlow.js MoveNet — the two approaches are entirely different.

2. **Reference hold duration too short:** The reference template shows avg hold duration of 0.48s. The feedback rules flag holds under 1.5s as errors, so the reference itself would fail its own quality check.

3. **Phase detection hardcoded:** In `websocket_session`, the feedback phase is hardcoded as `"uzatma"` (extension), ignoring actual phase detection.

4. **In-memory DB:** All patient/session data is lost on server restart. No persistence layer.

5. **`exercises.html` vs `exercises_fixed.html`:** Two versions of the same page exist — likely a development artifact. The "fixed" version has improved loading but they are otherwise parallel.

6. **No authentication:** The API has no auth layer. Any client can read/write patient data.

7. **`ext_leg_height` direction:** The code identifies the rightmost green pixel as the "extended leg tip." For quadruped exercises this assumes the patient is facing a specific direction (right side toward camera).

### Design Observations

- The system has two distinct pose detection stacks: color-segmentation (server, `pose_analyzer.py`) and TensorFlow.js MoveNet (client, HTML pages). They do not share data or algorithms.
- The HTML pages are largely self-contained and can run without the backend (browser-side computation with demo fallbacks).
- The platform is clearly designed for a specific clinical use case (physical therapy in Turkey) with a specific reference exercise.

---

## Summary

Kenshō is a functional proof-of-concept for AI-assisted physical therapy. The frontend (TensorFlow.js + Three.js) is the more sophisticated component — it does proper model-based pose detection. The backend uses a simpler, color-dependent approach that is tailored to specific reference video conditions. The project is well-structured as a demo/prototype but needs authentication, a real database, generalized pose tracking, and consolidated frontend versions before production use.
