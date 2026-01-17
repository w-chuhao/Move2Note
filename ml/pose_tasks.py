import cv2
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def _normalize_xy(xy: np.ndarray) -> np.ndarray:
    # Center at mid-hip, scale by torso length (reduces camera distance / body size effects)
    L_HIP, R_HIP = 23, 24
    L_SH, R_SH = 11, 12
    hip = (xy[L_HIP] + xy[R_HIP]) / 2.0
    sh = (xy[L_SH] + xy[R_SH]) / 2.0
    scale = np.linalg.norm(sh - hip) + 1e-6
    return (xy - hip) / scale


def video_to_keypoints(
    video_path: str,
    model_path: str,
    target_fps: int = 15,
    max_seconds: int = 30,
) -> np.ndarray:
    """
    Returns pose sequence [T, 33, 3] where last dim is (x, y, visibility-like).
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps / target_fps)), 1)
    max_frames = int(target_fps * max_seconds)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    seq = []
    frame_idx = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened() and len(seq) < max_frames:
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

            lms = result.pose_landmarks[0]  # 33 landmarks
            xy = np.array([[p.x, p.y] for p in lms], dtype=np.float32)
            vis = np.array([getattr(p, "visibility", 1.0) for p in lms], dtype=np.float32)

            xy = _normalize_xy(xy)
            feat = np.concatenate([xy, vis[:, None]], axis=1)  # [33,3]
            seq.append(feat)

            frame_idx += 1

    cap.release()

    if len(seq) == 0:
        return np.zeros((1, 33, 3), dtype=np.float32)

    return np.stack(seq, axis=0).astype(np.float32)
