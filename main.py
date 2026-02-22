"""
Kensho Body & Breath — FastAPI Backend
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import cv2, numpy as np, json, base64, uuid, asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys, os

sys.path.insert(0, str(Path(__file__).parent))
from pose_analyzer import BodyTracker, MovementAnalyzer, FeedbackEngine, ReferenceTemplateBuilder

app = FastAPI(title="Kensho API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
DATA = BASE / "data"
TEMPLATES_DIR = DATA / "templates"
SESSIONS_DIR  = DATA / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
(DATA / "videos").mkdir(parents=True, exist_ok=True)

# ─── In-memory DB (production'da PostgreSQL kullanılır) ───────────────────────
db = {
    "patients": {
        "p001": {"id": "p001", "name": "Ahmet Yılmaz", "age": 45, "therapist_id": "t001",
                 "conditions": ["postür bozukluğu", "bel ağrısı"], "joined": "2025-01-15"},
        "p002": {"id": "p002", "name": "Fatma Demir",  "age": 67, "therapist_id": "t001",
                 "conditions": ["kifoz", "denge problemi"], "joined": "2025-02-01"},
        "p003": {"id": "p003", "name": "Murat Çelik",  "age": 32, "therapist_id": "t001",
                 "conditions": ["sırt kasları zayıflığı"], "joined": "2025-02-10"},
    },
    "therapists": {
        "t001": {"id": "t001", "name": "Dr. Kensho Terapist", "email": "therapist@kensho.com"},
    },
    "programs": {},
    "sessions": {},
}

# ─── Loaded Templates ────────────────────────────────────────────────────────
templates: dict = {}

def load_templates():
    for f in TEMPLATES_DIR.glob("*_template.json"):
        key = f.stem.replace("_template", "")
        with open(f) as fh:
            templates[key] = json.load(fh)
    print(f"[Kensho] {len(templates)} şablon yüklendi: {list(templates.keys())}")

load_templates()

# ─── Active WebSocket Sessions ────────────────────────────────────────────────
active_sessions: dict = {}


# ════════════════════════════════════════════════════════════════════════════════
#  REST API
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "Kensho API çalışıyor", "templates": list(templates.keys())}

# ── Patients ──────────────────────────────────────────────────────────────────
@app.get("/patients")
def get_patients():
    return list(db["patients"].values())

@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    p = db["patients"].get(patient_id)
    if not p:
        raise HTTPException(404, "Hasta bulunamadı")
    sessions = [s for s in db["sessions"].values() if s.get("patient_id") == patient_id]
    return {**p, "sessions": sessions}

@app.post("/patients")
def create_patient(data: dict):
    pid = f"p{len(db['patients'])+1:03d}"
    db["patients"][pid] = {**data, "id": pid, "joined": datetime.now().isoformat()[:10]}
    return db["patients"][pid]

# ── Programs ──────────────────────────────────────────────────────────────────
@app.get("/programs/{patient_id}")
def get_program(patient_id: str):
    return db["programs"].get(patient_id, {"exercises": [], "notes": ""})

@app.post("/programs/{patient_id}")
def set_program(patient_id: str, data: dict):
    db["programs"][patient_id] = {
        **data,
        "patient_id": patient_id,
        "created_by": "t001",
        "created_at": datetime.now().isoformat(),
    }
    return db["programs"][patient_id]

# ── Sessions ──────────────────────────────────────────────────────────────────
@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    s = db["sessions"].get(session_id)
    if not s:
        raise HTTPException(404, "Seans bulunamadı")
    return s

@app.get("/sessions/patient/{patient_id}")
def get_patient_sessions(patient_id: str):
    return [s for s in db["sessions"].values() if s.get("patient_id") == patient_id]

# ── Templates ─────────────────────────────────────────────────────────────────
@app.get("/exercises")
def list_exercises():
    return list(templates.keys())

@app.get("/exercises/{exercise_type}/template")
def get_template(exercise_type: str):
    if exercise_type not in templates:
        raise HTTPException(404, "Egzersiz şablonu bulunamadı")
    t = templates[exercise_type]
    # Sadece özet dön
    return {
        "exercise_type": exercise_type,
        "stats": t.get("stats", {}),
        "phase_count": len(t.get("phases", [])),
        "series_length": len(t.get("series", {}).get("times", [])),
    }

# ── Video upload & template build ─────────────────────────────────────────────
@app.post("/exercises/{exercise_type}/upload-reference")
async def upload_reference(exercise_type: str, file: UploadFile = File(...)):
    path = DATA / "videos" / f"{exercise_type}_ref.mp4"
    with open(path, "wb") as f:
        f.write(await file.read())
    
    builder = ReferenceTemplateBuilder(exercise_type)
    template = builder.build_from_video(str(path))
    
    out = TEMPLATES_DIR / f"{exercise_type}_template.json"
    with open(out, "w") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    
    templates[exercise_type] = template
    return {"message": "Şablon oluşturuldu", "stats": template["stats"]}

# ── Terapist rapor endpointi ──────────────────────────────────────────────────
@app.get("/therapist/dashboard/{therapist_id}")
def therapist_dashboard(therapist_id: str):
    therapist = db["therapists"].get(therapist_id)
    if not therapist:
        raise HTTPException(404, "Terapist bulunamadı")
    
    patients = [p for p in db["patients"].values() if p.get("therapist_id") == therapist_id]
    
    all_sessions = []
    for patient in patients:
        psessions = [s for s in db["sessions"].values() if s.get("patient_id") == patient["id"]]
        all_sessions.extend(psessions)
    
    # İstatistikler
    total_sessions = len(all_sessions)
    avg_score = 0
    if all_sessions:
        scores = [s.get("report", {}).get("quality_score", 0) for s in all_sessions]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    
    return {
        "therapist": therapist,
        "patient_count": len(patients),
        "patients": patients,
        "total_sessions": total_sessions,
        "avg_quality_score": avg_score,
        "recent_sessions": sorted(all_sessions, key=lambda x: x.get("started_at",""), reverse=True)[:10],
    }


# ════════════════════════════════════════════════════════════════════════════════
#  WebSocket — Gerçek Zamanlı Analiz
# ════════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/session/{patient_id}/{exercise_type}")
async def websocket_session(ws: WebSocket, patient_id: str, exercise_type: str):
    await ws.accept()
    
    session_id = str(uuid.uuid4())[:8]
    tracker  = BodyTracker()
    analyzer = MovementAnalyzer(exercise_type)
    feedback = FeedbackEngine(exercise_type)
    
    # Referans şablon
    ref_template = templates.get(exercise_type, {})
    ref_series   = ref_template.get("series", {}).get("leg_heights", [])
    
    frames_data = []
    frame_count = 0
    started_at  = datetime.now().isoformat()
    
    print(f"[WS] Seans başladı: {session_id} | Hasta: {patient_id} | Egzersiz: {exercise_type}")
    
    await ws.send_json({
        "type": "session_start",
        "session_id": session_id,
        "exercise": exercise_type,
        "reference_loaded": bool(ref_series),
    })
    
    try:
        while True:
            data = await ws.receive_text()
            msg  = json.loads(data)
            
            if msg["type"] == "frame":
                # Base64 frame decode
                img_data = base64.b64decode(msg["data"].split(",")[-1])
                nparr    = np.frombuffer(img_data, np.uint8)
                frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Keypoint extraction
                kp = tracker.extract_keypoints(frame)
                
                entry = {
                    "frame": frame_count,
                    "time": round(frame_count / 30.0, 3),
                    "detected": kp is not None,
                }
                if kp:
                    entry.update(kp)
                frames_data.append(entry)
                
                # Gerçek zamanlı feedback
                fb_messages = []
                if kp:
                    # Mevcut faz tahmini
                    phase = "uzatma"  # basit: her zaman uzatma fazını kontrol et
                    fb_messages = feedback.realtime_feedback(kp, phase)
                
                # Overlay data: skeleton çizmek için noktalar
                overlay = {}
                if kp:
                    overlay = {
                        "torso":    [kp.get("torso_x",  0.5), kp.get("torso_y",  0.5)],
                        "hip":      [kp.get("hip_x",    0.5), kp.get("hip_y",    0.5)],
                        "ext_leg":  [kp.get("ext_leg_x",0.8), kp.get("ext_leg_y",0.5)],
                        "leg_h":    kp.get("ext_leg_height", 0),
                        "spine":    kp.get("spine_flatness",  1),
                    }
                
                await ws.send_json({
                    "type":     "frame_result",
                    "frame":    frame_count,
                    "detected": kp is not None,
                    "feedback": fb_messages,
                    "overlay":  overlay,
                })
                
                frame_count += 1
            
            elif msg["type"] == "end_session":
                # Seans sonu analiz
                series  = analyzer.extract_time_series(frames_data)
                phases  = analyzer.detect_phases(series)
                
                # DTW karşılaştırma
                user_series = series.get("leg_heights", [])
                dtw_result  = analyzer.compare_with_reference(user_series, ref_series) if ref_series else {"similarity_pct": 0, "dtw_distance": 999}
                
                # Tam rapor
                report = feedback.session_report(phases, series, dtw_result)
                
                # Kaydet
                session_record = {
                    "session_id":   session_id,
                    "patient_id":   patient_id,
                    "exercise":     exercise_type,
                    "started_at":   started_at,
                    "ended_at":     datetime.now().isoformat(),
                    "total_frames": frame_count,
                    "report":       report,
                    "phases":       phases,
                }
                db["sessions"][session_id] = session_record
                
                # Dosyaya da kaydet
                session_file = SESSIONS_DIR / f"{session_id}.json"
                with open(session_file, "w", encoding="utf-8") as f:
                    json.dump(session_record, f, ensure_ascii=False, indent=2)
                
                await ws.send_json({
                    "type":    "session_report",
                    "session_id": session_id,
                    "report":  report,
                })
                break
    
    except WebSocketDisconnect:
        print(f"[WS] Bağlantı koptu: {session_id}")
    finally:
        active_sessions.pop(session_id, None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
