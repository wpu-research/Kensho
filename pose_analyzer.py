"""
Kensho Body & Breath - Pose Analyzer
Renk tabanlı vücut takibi + DTW hareket analizi
"""

import cv2
import numpy as np
import json
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
from dtaidistance import dtw
from typing import Optional


# ─── Renk Aralıkları (HSV) ────────────────────────────────────────────────────
COLOR_PROFILES = {
    "yellow_shirt": {"lower": np.array([18, 80, 80]),  "upper": np.array([40, 255, 255])},
    "green_pants":  {"lower": np.array([35, 30, 30]),  "upper": np.array([90, 255, 180])},
    "blue_mat":     {"lower": np.array([100, 80, 50]), "upper": np.array([130, 255, 255])},
}

# ─── Egzersiz Tanımları ───────────────────────────────────────────────────────
EXERCISE_DEFINITIONS = {
    "quadruped_hip_extension": {
        "name": "Quadruped Hip Extension",
        "name_tr": "4 Nokta Destek Kalça Uzatma",
        "description": "4 nokta destekte bir bacağı geriye uzatma",
        "target_joints": ["hip", "extended_leg"],
        "feedback_rules": {
            "leg_too_low":     {"threshold": 0.35, "message": "Bacağını daha yukarı kaldır (~15° daha)"},
            "leg_too_high":    {"threshold": 0.85, "message": "Bacağı çok yüksek kaldırıyorsun, beli zorluyor"},
            "spine_not_flat":  {"threshold": 0.15, "message": "Sırtını düzleştir, beli çöktürme"},
            "hold_too_short":  {"threshold": 1.5,  "message": "Pozisyonu en az 2 saniye tut"},
        },
        "ideal_extension": 0.65,  # normalize edilmiş ideal yükseklik
        "phases": ["başlangıç", "uzatma", "tutma", "geri_dönüş"]
    },
    "plank_hold": {
        "name": "Plank Hold",
        "name_tr": "Plank",
        "description": "Ön plank pozisyonu tutma",
        "target_joints": ["spine", "shoulders", "hips"],
        "ideal_extension": 0.5,
        "phases": ["başlangıç", "tutma", "bitiş"]
    }
}


class BodyTracker:
    """Renk tabanlı vücut parçası takibi"""

    def __init__(self, color_profile: dict = None):
        self.colors = color_profile or COLOR_PROFILES
        self.frame_history = []

    def extract_keypoints(self, frame: np.ndarray) -> Optional[dict]:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Renk maskelerini oluştur
        yellow_mask = cv2.inRange(hsv, self.colors["yellow_shirt"]["lower"],
                                      self.colors["yellow_shirt"]["upper"])
        green_mask  = cv2.inRange(hsv, self.colors["green_pants"]["lower"],
                                       self.colors["green_pants"]["upper"])

        # Gürültü temizleme
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN,  kernel)
        green_mask  = cv2.morphologyEx(green_mask,  cv2.MORPH_CLOSE, kernel)
        green_mask  = cv2.morphologyEx(green_mask,  cv2.MORPH_OPEN,  kernel)

        body_mask = cv2.bitwise_or(yellow_mask, green_mask)
        contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        large = [c for c in contours if cv2.contourArea(c) > 500]
        if not large:
            return None

        all_pts = np.vstack(large)
        x, y, bw, bh = cv2.boundingRect(all_pts)
        pts = all_pts.reshape(-1, 2)

        kp = {
            "bbox": [int(x), int(y), int(bw), int(bh)],
            "body_width_norm":  float(bw / w),
            "body_height_norm": float(bh / h),
            "center": [float((x + bw/2) / w), float((y + bh/2) / h)],
        }

        # Üst vücut (sarı gömlek = gövde)
        yp = np.where(yellow_mask > 0)
        if len(yp[0]) > 100:
            kp["torso_x"] = float(np.mean(yp[1]) / w)
            kp["torso_y"] = float(np.mean(yp[0]) / h)

        # Alt vücut (yeşil pantolon = kalça + bacaklar)
        gp = np.where(green_mask > 0)
        if len(gp[0]) > 100:
            gc = np.column_stack([gp[1], gp[0]])
            kp["hip_x"]    = float(np.mean(gc[:, 0]) / w)
            kp["hip_y"]    = float(np.mean(gc[:, 1]) / h)
            # En sağdaki nokta = uzanan bacak ucu
            rightmost_g    = gc[np.argmax(gc[:, 0])]
            kp["ext_leg_x"] = float(rightmost_g[0] / w)
            kp["ext_leg_y"] = float(rightmost_g[1] / h)
            # Uzanan bacak yüksekliği (düşük y = yüksek = iyi uzanım)
            if bh > 0:
                kp["ext_leg_height"] = float(1 - (rightmost_g[1] - y) / bh)

        # Omurga düzlüğü: sarı bölgenin y varyansı
        if len(yp[0]) > 100 and bh > 0:
            spine_var = float(np.std(yp[0]) / bh)
            kp["spine_flatness"] = float(max(0, 1 - spine_var * 3))  # 1=düz, 0=eğri

        return kp


class MovementAnalyzer:
    """Hareket fazı tespiti ve DTW analizi"""

    def __init__(self, exercise_type: str):
        self.exercise = EXERCISE_DEFINITIONS.get(exercise_type, {})
        self.exercise_type = exercise_type

    def extract_time_series(self, frames: list) -> dict:
        """Frame listesinden açısal zaman serileri çıkar"""
        detected = [f for f in frames if f.get("detected") and "ext_leg_height" in f]

        if len(detected) < 10:
            return {}

        times       = np.array([f["time"] for f in detected])
        leg_heights = np.array([f["ext_leg_height"] for f in detected])
        spine_flat  = np.array([f.get("spine_flatness", 1.0) for f in detected])
        torso_y     = np.array([f.get("torso_y", 0.5) for f in detected])

        # Smooth
        if len(leg_heights) > 11:
            leg_heights = savgol_filter(leg_heights, 11, 2)
            spine_flat  = savgol_filter(spine_flat,  11, 2)

        return {
            "times":       times.tolist(),
            "leg_heights": leg_heights.tolist(),
            "spine_flat":  spine_flat.tolist(),
            "torso_y":     torso_y.tolist(),
        }

    def detect_phases(self, series: dict) -> list:
        """Hareket fazlarını tespit et"""
        if not series:
            return []

        leg_h = np.array(series["leg_heights"])
        times  = np.array(series["times"])
        threshold = np.mean(leg_h) + np.std(leg_h) * 0.3

        peaks, props = find_peaks(leg_h, height=threshold, distance=15, prominence=0.05)

        phases = []
        for i, p in enumerate(peaks):
            t_peak = float(times[p])
            peak_val = float(leg_h[p])

            # Bu tekin başlangıcı: peak öncesi minimum
            left = max(0, p - 20)
            start_idx = left + int(np.argmin(leg_h[left:p]))
            t_start = float(times[start_idx])

            # Bu tekin bitişi: peak sonrası minimum
            right = min(len(leg_h)-1, p + 20)
            end_idx = p + int(np.argmin(leg_h[p:right]))
            t_end = float(times[end_idx])

            phases.append({
                "rep_number": i + 1,
                "phase_start":  round(t_start, 2),
                "peak_time":    round(t_peak, 2),
                "phase_end":    round(t_end, 2),
                "duration":     round(t_end - t_start, 2),
                "peak_extension": round(peak_val, 3),
                "hold_duration":  round(t_end - t_peak, 2),
            })

        return phases

    def compare_with_reference(self, user_series: list, reference_series: list) -> dict:
        """DTW ile kullanıcı vs referans karşılaştırması"""
        if len(user_series) < 5 or len(reference_series) < 5:
            return {"error": "Yetersiz veri"}

        u = np.array(user_series, dtype=np.double)
        r = np.array(reference_series, dtype=np.double)

        # Normalize
        u = (u - u.min()) / (u.max() - u.min() + 1e-8)
        r = (r - r.min()) / (r.max() - r.min() + 1e-8)

        distance = dtw.distance_fast(u, r)
        similarity = max(0, 100 - distance * 100)

        return {
            "dtw_distance":  round(float(distance), 4),
            "similarity_pct": round(float(similarity), 1),
        }


class FeedbackEngine:
    """Gerçek zamanlı ve seans sonu geri bildirim üreteci"""

    def __init__(self, exercise_type: str):
        self.exercise = EXERCISE_DEFINITIONS.get(exercise_type, {})
        self.rules = self.exercise.get("feedback_rules", {})
        self.ideal = self.exercise.get("ideal_extension", 0.65)

    def realtime_feedback(self, kp: dict, phase: str = "uzatma") -> list:
        """Anlık geri bildirim — bir frame için"""
        messages = []

        leg_h = kp.get("ext_leg_height", None)
        spine = kp.get("spine_flatness", 1.0)

        if leg_h is not None and phase == "uzatma":
            diff_deg = (self.ideal - leg_h) * 90  # yaklaşık derece

            if leg_h < self.rules.get("leg_too_low", {}).get("threshold", 0.35):
                messages.append({
                    "type": "correction",
                    "severity": "high",
                    "text": f"Bacağını {abs(diff_deg):.0f}° daha yukarı kaldır",
                    "joint": "kalça",
                })
            elif leg_h > self.rules.get("leg_too_high", {}).get("threshold", 0.85):
                messages.append({
                    "type": "correction",
                    "severity": "medium",
                    "text": "Bacağı biraz aşağı al, beli korumak için",
                    "joint": "kalça",
                })
            elif leg_h >= self.ideal * 0.9:
                messages.append({
                    "type": "positive",
                    "severity": "info",
                    "text": "Harika! Pozisyon doğru",
                    "joint": "kalça",
                })

        if spine < 0.6:
            messages.append({
                "type": "correction",
                "severity": "high",
                "text": "Sırtını düzleştir — omurga nötr pozisyonda olmalı",
                "joint": "omurga",
            })

        return messages

    def session_report(self, phases: list, series: dict, dtw_result: dict) -> dict:
        """Seans sonu tam rapor"""
        if not phases:
            return {"error": "Hareket fazı tespit edilemedi"}

        total_reps = len(phases)
        extensions = [p["peak_extension"] for p in phases]
        holds      = [p["hold_duration"] for p in phases]

        # Kalite değerlendirmesi
        good_reps    = sum(1 for e in extensions if self.ideal*0.85 <= e <= self.ideal*1.15)
        low_reps     = sum(1 for e in extensions if e < self.ideal * 0.85)
        high_reps    = sum(1 for e in extensions if e > self.ideal * 1.15)
        short_holds  = sum(1 for h in holds if h < 1.5)

        # Ortalama spine
        avg_spine = float(np.mean(series.get("spine_flat", [1.0])))

        # Hata listesi
        errors = []
        if low_reps:
            errors.append({
                "type": "leg_height",
                "count": low_reps,
                "message": f"{low_reps} tekrarda bacak yeterince yükseltilmedi",
                "recommendation": "Kalça ekstansörlerini güçlendirme egzersizleri ekleyin"
            })
        if short_holds:
            errors.append({
                "type": "hold_duration",
                "count": short_holds,
                "message": f"{short_holds} tekrarda pozisyon yeterince tutulmadı",
                "recommendation": "Her tekrarda 2-3 saniye tutmayı hedefleyin"
            })
        if avg_spine < 0.65:
            errors.append({
                "type": "spine_alignment",
                "count": total_reps,
                "message": "Sırt düzlüğü yetersiz — lomber hiperekstensiyon riski",
                "recommendation": "Core aktivasyonuna odaklanın, transversus abdominis kasını aktive edin"
            })

        # Genel skor
        quality_score = round(
            (good_reps / total_reps * 50) +
            (min(avg_spine, 1.0) * 30) +
            (dtw_result.get("similarity_pct", 70) / 100 * 20),
            1
        )

        return {
            "exercise": self.exercise.get("name_tr", "Bilinmiyor"),
            "total_reps": total_reps,
            "good_reps":  good_reps,
            "avg_peak_extension": round(float(np.mean(extensions)), 3),
            "max_peak_extension": round(float(np.max(extensions)), 3),
            "avg_hold_duration":  round(float(np.mean(holds)), 2),
            "spine_alignment":    round(avg_spine * 100, 1),
            "dtw_similarity":     dtw_result.get("similarity_pct", 0),
            "quality_score":      quality_score,
            "errors":             errors,
            "rep_details":        phases,
            "grade": (
                "Mükemmel" if quality_score >= 85 else
                "İyi"       if quality_score >= 70 else
                "Orta"      if quality_score >= 55 else "Geliştirilmeli"
            )
        }


class ReferenceTemplateBuilder:
    """Video(lar)dan referans şablon oluşturur"""

    def __init__(self, exercise_type: str):
        self.tracker = BodyTracker()
        self.analyzer = MovementAnalyzer(exercise_type)
        self.exercise_type = exercise_type

    def build_from_video(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            kp = self.tracker.extract_keypoints(frame)
            entry = {"frame": idx, "time": round(idx/fps, 3), "detected": kp is not None}
            if kp:
                entry.update(kp)
            frames.append(entry)
            idx += 1
        cap.release()

        series = self.analyzer.extract_time_series(frames)
        phases = self.analyzer.detect_phases(series)

        template = {
            "exercise_type": self.exercise_type,
            "source_video":  video_path,
            "fps":           fps,
            "total_frames":  idx,
            "series":        series,
            "phases":        phases,
            "stats": {
                "avg_peak_extension": float(np.mean([p["peak_extension"] for p in phases])) if phases else 0,
                "avg_hold_duration":  float(np.mean([p["hold_duration"]  for p in phases])) if phases else 0,
                "total_reps":         len(phases),
            }
        }
        return template


# ─── CLI: Video'dan şablon oluştur ────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json

    video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/1.mp4"
    exercise   = sys.argv[2] if len(sys.argv) > 2 else "quadruped_hip_extension"

    print(f"Video analiz ediliyor: {video_path}")
    builder = ReferenceTemplateBuilder(exercise)
    template = builder.build_from_video(video_path)

    out_path = f"data/templates/{exercise}_template.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Şablon oluşturuldu: {out_path}")
    print(f"   Tespit edilen tekrar: {template['stats']['total_reps']}")
    print(f"   Ort. uzanım yüksekliği: {template['stats']['avg_peak_extension']:.3f}")
    print(f"   Ort. tutma süresi: {template['stats']['avg_hold_duration']:.2f}s")
