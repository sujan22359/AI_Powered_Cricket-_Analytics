#!/usr/bin/env python3
"""
AthleteRise – Real-Time Cover Drive Analysis (UI-only version)

Functions:
  - download_youtube_video(url) -> str
  - analyze_video(video_path) -> dict

Outputs:
  - output/annotated_video.mp4
  - output/evaluation.json
  - output/report.pdf
  - output/bat_path.png
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import json
import math
from typing import Optional
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Config
CONFIG_FILE = "config.json"
OUTPUT_DIR = "output"

# Load config
with open(CONFIG_FILE, "r") as f:
    CONFIG = json.load(f)

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ---------- Utility functions ----------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def safe_acos(x: float) -> float:
    return math.acos(clamp(x, -1.0, 1.0))

def calculate_angle(a, b, c) -> Optional[float]:
    try:
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        nba = np.linalg.norm(ba)
        nbc = np.linalg.norm(bc)
        if nba < 1e-6 or nbc < 1e-6:
            return None
        cosang = np.dot(ba, bc) / (nba * nbc)
        return math.degrees(safe_acos(cosang))
    except:
        return None

def euclid(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

# ---------- Bat detection ----------
def detect_bat(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(CONFIG["bat_detection"]["hsv_low"])
    upper = np.array(CONFIG["bat_detection"]["hsv_high"])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        return rect  # ((x,y),(w,h),angle)
    return None

# ---------- PDF report ----------
def generate_pdf_report(evaluation, skill_grade, bat_path_img):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUTPUT_DIR, "report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "AthleteRise Cricket Analysis Report")

    c.setFont("Helvetica", 12)
    y = height - 100
    c.drawString(50, y, f"Skill Grade: {skill_grade}")
    y -= 30
    for k,v in evaluation.items():
        c.drawString(50, y, f"{k}: {v['score']}  -  {v['feedback']}")
        y -= 20

    if os.path.exists(bat_path_img):
        c.drawImage(bat_path_img, 50, y-200, width=300, height=200)

    c.showPage()
    c.save()
    print(f"[INFO] PDF report saved to {pdf_path}")

# ---------- Scoring ----------
def score_metric(mean_val, target_range, invert=False, hard_max=None):
    if mean_val is None or not math.isfinite(mean_val):
        return 4
    if not invert:
        lo, hi = target_range
        if lo <= mean_val <= hi:
            return 10
        return max(1, 7 - int(abs(mean_val - (lo+hi)/2)))
    else:
        if hard_max is None:
            hard_max = target_range[1]
        norm = clamp(1.0 - (mean_val / max(1e-6, hard_max)), 0.0, 1.0)
        return int(round(4 + norm * 6))

def skill_grade_from_scores(scores):
    avg = sum(scores.values()) / len(scores)
    if avg >= 8: return "Advanced"
    elif avg >= 5: return "Intermediate"
    else: return "Beginner"

# ---------- YouTube download ----------
def download_youtube_video(url: str, output_dir="tmp") -> str:
    """Download YouTube video and return local file path."""
    import yt_dlp
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {'outtmpl': os.path.join(output_dir, 'video.%(ext)s'), 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)

# ---------- Main analysis ----------
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    out_path = os.path.join(OUTPUT_DIR, "annotated_video.mp4")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))

    stats = { 'elbow': [], 'spine': [], 'head': [], 'foot': [] }
    bat_points = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                elbow_angle = calculate_angle(
                    (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h),
                    (lm[mp_pose.PoseLandmark.LEFT_ELBOW].x * w, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y * h),
                    (lm[mp_pose.PoseLandmark.LEFT_WRIST].x * w, lm[mp_pose.PoseLandmark.LEFT_WRIST].y * h),
                )
                stats['elbow'].append(elbow_angle)

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            bat_rect = detect_bat(frame)
            if bat_rect:
                (x,y),_,_ = bat_rect
                bat_points.append((int(x), int(y)))
                cv2.circle(frame, (int(x), int(y)), 5, (0,255,255), -1)

            out_writer.write(frame)

    cap.release()
    out_writer.release()

    evaluation = {
        "Footwork": {"score": score_metric(np.nanmean(stats['foot']), (0, CONFIG['thresholds']['foot_angle']), False),
                     "feedback": "Good footwork" },
        "Head Position": {"score": score_metric(np.nanmean(stats['head']), (0, CONFIG['thresholds']['head_knee_dx']), True),
                          "feedback": "Head positioning"},
        "Swing Control": {"score": score_metric(np.nanmean(stats['elbow']), tuple(CONFIG['thresholds']['elbow']), False),
                          "feedback": "Swing control"},
        "Balance": {"score": score_metric(np.nanmean(stats['spine']), tuple(CONFIG['thresholds']['spine']), False),
                    "feedback": "Balance control"},
        "Follow-through": {"score": 7, "feedback": "Work on follow-through"}
    }
    skill_grade = skill_grade_from_scores({k:v['score'] for k,v in evaluation.items()})

    json_path = os.path.join(OUTPUT_DIR, "evaluation.json")
    with open(json_path, "w") as f:
        json.dump({"evaluation": evaluation, "skill_grade": skill_grade}, f, indent=2)

    bat_img_path = os.path.join(OUTPUT_DIR, "bat_path.png")
    if bat_points:
        bat_canvas = np.zeros((h,w,3), dtype=np.uint8)
        for pt in bat_points:
            cv2.circle(bat_canvas, pt, 3, (0,255,255), -1)
        cv2.imwrite(bat_img_path, bat_canvas)

    generate_pdf_report(evaluation, skill_grade, bat_img_path)

    return {"evaluation": evaluation, "skill_grade": skill_grade}
