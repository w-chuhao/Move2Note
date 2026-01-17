import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass

# MediaPipe Tasks PoseLandmarker (works with mediapipe versions that removed mp.solutions)
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
    wrist_above_head: list
    body_horizontal: list
    ankle_spread: list
    wrist_height: list
    hip_drop: list


def extract_feature_series(video_path, model_path="models/pose_landmarker_lite.task",
                           target_fps=15, max_seconds=12):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps / target_fps)), 1)
    max_frames = int(target_fps * max_seconds)

    s = Series([], [], [], [], [], [], [])
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

            # OpenCV -> RGB -> MediaPipe Image
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Timestamp required for VIDEO mode (ms)
            ts_ms = int((frame_idx / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, ts_ms)
            if not result.pose_landmarks:
                frame_idx += 1
                continue

            # result.pose_landmarks[0] is a list of 33 landmarks (x,y,z,visibility)
            lms = result.pose_landmarks[0]

            def P(i):
                return (lms[i].x, lms[i].y)

            L_SH, R_SH = 11, 12
            L_EL, R_EL = 13, 14
            L_WR, R_WR = 15, 16
            L_HIP, R_HIP = 23, 24
            L_KNE, R_KNE = 25, 26
            L_ANK, R_ANK = 27, 28
            NOSE = 0

            mid_sh = ((P(L_SH)[0] + P(R_SH)[0]) / 2, (P(L_SH)[1] + P(R_SH)[1]) / 2)
            mid_hip = ((P(L_HIP)[0] + P(R_HIP)[0]) / 2, (P(L_HIP)[1] + P(R_HIP)[1]) / 2)
            mid_ank = ((P(L_ANK)[0] + P(R_ANK)[0]) / 2, (P(L_ANK)[1] + P(R_ANK)[1]) / 2)

            knee = (angle(P(L_HIP), P(L_KNE), P(L_ANK)) + angle(P(R_HIP), P(R_KNE), P(R_ANK))) / 2
            elbow = (angle(P(L_SH), P(L_EL), P(L_WR)) + angle(P(R_SH), P(R_EL), P(R_WR))) / 2

            head_y = P(NOSE)[1]
            lw_y, rw_y = P(L_WR)[1], P(R_WR)[1]
            wrists_high = 1.0 if (lw_y < head_y and rw_y < head_y) else 0.0

            sh_to_hip = abs(mid_sh[1] - mid_hip[1])
            hip_to_ank = abs(mid_hip[1] - mid_ank[1])
            horizontal = 1.0 if (sh_to_hip < 0.10 and hip_to_ank < 0.15) else 0.0

            ankle_spread = abs(P(L_ANK)[0] - P(R_ANK)[0])

            sh_y = (P(L_SH)[1] + P(R_SH)[1]) / 2
            wrist_y = (lw_y + rw_y) / 2
            wrist_height = sh_y - wrist_y

            knee_y = (P(L_KNE)[1] + P(R_KNE)[1]) / 2
            hip_y = (P(L_HIP)[1] + P(R_HIP)[1]) / 2
            hip_drop = hip_y - knee_y

            s.knee_angle.append(knee)
            s.elbow_angle.append(elbow)
            s.wrist_above_head.append(wrists_high)
            s.body_horizontal.append(horizontal)
            s.ankle_spread.append(ankle_spread)
            s.wrist_height.append(wrist_height)
            s.hip_drop.append(hip_drop)

            frame_idx += 1

    cap.release()
    return s


def classify_clip(s: Series):
    if len(s.knee_angle) < 10:
        return "unknown", 0.0, {}

    knee = np.array(s.knee_angle, dtype=np.float32)
    elbow = np.array(s.elbow_angle, dtype=np.float32)
    wrists_high = np.array(s.wrist_above_head, dtype=np.float32)
    horizontal = np.array(s.body_horizontal, dtype=np.float32)
    ankle_spread = np.array(s.ankle_spread, dtype=np.float32)
    wrist_height = np.array(s.wrist_height, dtype=np.float32)
    hip_drop = np.array(s.hip_drop, dtype=np.float32)

    ankle_spread_s = smooth(ankle_spread, 7)
    hip_drop_s = smooth(hip_drop, 7)
    elbow_s = smooth(elbow, 7)
    knee_s = smooth(knee, 7)

    knee_range = float(np.percentile(knee_s, 90) - np.percentile(knee_s, 10))
    elbow_range = float(np.percentile(elbow_s, 90) - np.percentile(elbow_s, 10))
    ankle_range = float(np.percentile(ankle_spread_s, 90) - np.percentile(ankle_spread_s, 10))
    hipdrop_range = float(np.percentile(hip_drop_s, 90) - np.percentile(hip_drop_s, 10))

    scores = {}
    scores["jumpingjack"] = 2.5 * ankle_range + 0.5 * float(np.mean(ankle_spread_s) > 0.20)
    scores["squat"] = 1.8 * knee_range + 2.2 * hipdrop_range + 0.5 * float(np.mean(hip_drop_s) > -0.02)
    scores["pushup"] = 3.0 * float(np.mean(horizontal) > 0.40) + 1.8 * elbow_range
    scores["curl"] = 2.2 * elbow_range + 1.0 * float(ankle_range < 0.05) + 1.0 * float(hipdrop_range < 0.05)
    scores["press"] = 2.5 * float(np.mean(wrist_height) > 0.05) + 1.0 * float(np.mean(wrists_high) > 0.15) + 0.8 * elbow_range

    best = max(scores, key=scores.get)
    vals = np.array(list(scores.values()), dtype=np.float32)
    conf = float((scores[best] - float(vals.min())) / (float(vals.max() - vals.min()) + 1e-6))
    return best, conf, scores


NOTE_MAP = {
    "squat": "C4",
    "pushup": "E4",
    "jumpingjack": "G4",
    "curl": "A4",
    "press": "D4",
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
