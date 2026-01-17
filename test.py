# identify_movement_3.py
# 3-class: pushup, situp, squat
# OpenCV reads frames
# MediaPipe Tasks PoseLandmarker extracts landmarks
# Heuristics classify clip + map to a note

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def angle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def smooth(x, w=7):
    if len(x) < w:
        return np.array(x, dtype=np.float32)
    k = np.ones(w, dtype=np.float32) / w
    return np.convolve(np.array(x, dtype=np.float32), k, mode="same")


@dataclass
class Series:
    knee_angle: list
    elbow_angle: list
    body_horizontal: list
    ankle_spread: list
    hip_drop: list
    torso_height: list   # mid_hip_y - mid_shoulder_y (bigger change for situps)


def extract_feature_series(
    video_path,
    model_path="models/pose_landmarker_lite.task",
    target_fps=15,
    max_seconds=12,
):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps / target_fps)), 1)
    max_frames = int(target_fps * max_seconds)

    s = Series([], [], [], [], [], [])
    frame_idx = 0

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened() and len(s.knee_angle) < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % step != 0:
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            ts_ms = int((frame_idx / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)
            if not result.pose_landmarks:
                frame_idx += 1
                continue

            lms = result.pose_landmarks[0]

            def P(i):
                return (lms[i].x, lms[i].y)

            L_SH, R_SH = 11, 12
            L_EL, R_EL = 13, 14
            L_WR, R_WR = 15, 16
            L_HIP, R_HIP = 23, 24
            L_KNE, R_KNE = 25, 26
            L_ANK, R_ANK = 27, 28

            mid_sh = ((P(L_SH)[0] + P(R_SH)[0]) / 2, (P(L_SH)[1] + P(R_SH)[1]) / 2)
            mid_hip = ((P(L_HIP)[0] + P(R_HIP)[0]) / 2, (P(L_HIP)[1] + P(R_HIP)[1]) / 2)
            mid_ank = ((P(L_ANK)[0] + P(R_ANK)[0]) / 2, (P(L_ANK)[1] + P(R_ANK)[1]) / 2)

            knee = (angle(P(L_HIP), P(L_KNE), P(L_ANK)) + angle(P(R_HIP), P(R_KNE), P(R_ANK))) / 2
            elbow = (angle(P(L_SH), P(L_EL), P(L_WR)) + angle(P(R_SH), P(R_EL), P(R_WR))) / 2

            # "horizontal body" proxy: torso + hips not too separated vertically (works for pushup/situp)
            sh_to_hip = abs(mid_sh[1] - mid_hip[1])
            hip_to_ank = abs(mid_hip[1] - mid_ank[1])
            horizontal = 1.0 if (sh_to_hip < 0.10 and hip_to_ank < 0.15) else 0.0

            ankle_spread = abs(P(L_ANK)[0] - P(R_ANK)[0])

            # squat indicator: hips go lower relative to knees (y increases downward)
            knee_y = (P(L_KNE)[1] + P(R_KNE)[1]) / 2
            hip_y = (P(L_HIP)[1] + P(R_HIP)[1]) / 2
            hip_drop = hip_y - knee_y

            # situp indicator: torso height changes a lot (shoulders move up/down relative to hips)
            torso_height = mid_hip[1] - mid_sh[1]

            s.knee_angle.append(knee)
            s.elbow_angle.append(elbow)
            s.body_horizontal.append(horizontal)
            s.ankle_spread.append(ankle_spread)
            s.hip_drop.append(hip_drop)
            s.torso_height.append(torso_height)

            frame_idx += 1

    cap.release()
    return s


def classify_clip(s: Series):
    if len(s.knee_angle) < 10:
        return "unknown", 0.0, {}

    knee = smooth(s.knee_angle, 7)
    elbow = smooth(s.elbow_angle, 7)
    horizontal = np.array(s.body_horizontal, dtype=np.float32)
    ankle_spread = smooth(s.ankle_spread, 7)
    hip_drop = smooth(s.hip_drop, 7)
    torso_h = smooth(s.torso_height, 7)

    def rng(x):
        x = np.array(x, dtype=np.float32)
        return float(np.percentile(x, 90) - np.percentile(x, 10))

    knee_range = rng(knee)
    elbow_range = rng(elbow)
    hipdrop_range = rng(hip_drop)
    torso_range = rng(torso_h)
    ankle_range = rng(ankle_spread)

    horiz_mean = float(np.mean(horizontal))

    scores = {}

    # squat: strong knee bend + hip drop oscillation, not necessarily horizontal
    scores["squat"] = 2.2 * knee_range + 2.8 * hipdrop_range + 0.2 * float(horiz_mean < 0.30)

    # pushup: body horizontal + elbow bending
    scores["pushup"] = 3.0 * float(horiz_mean > 0.35) + 2.2 * elbow_range + 0.2 * float(ankle_range < 0.06)

    # situp: often horizontal-ish baseline + torso height changes a lot; elbows may change less than pushup
    scores["situp"] = 2.8 * float(horiz_mean > 0.25) + 3.0 * torso_range + 0.3 * float(elbow_range < 35)

    best = max(scores, key=scores.get)
    vals = np.array(list(scores.values()), dtype=np.float32)
    conf = float((scores[best] - float(vals.min())) / (float(vals.max() - vals.min()) + 1e-6))
    return best, conf, scores


NOTE_MAP = {
    "squat": "C4",
    "pushup": "E4",
    "situp": "G4",
}


def identify(video_path):
    s = extract_feature_series(video_path)
    label, conf, scores = classify_clip(s)
    note = NOTE_MAP.get(label, "NA")
    return label, conf, note, scores


if __name__ == "__main__":
    import sys

    video = sys.argv[1]
    label, conf, note, scores = identify(video)
    print("predicted_exercise:", label)
    print("confidence_like:", round(conf, 3))
    print("mapped_note:", note)
    print("scores:", {k: round(v, 3) for k, v in scores.items()})
