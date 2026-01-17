from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.pose_tasks import video_to_keypoints  # noqa: E402

MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"
CLASSIFIER_PATH = ROOT / "models" / "exercise.pt"
LABEL_MAP_PATH = ROOT / "data" / "label_map.npy"
PHASE_CLASSIFIER_PATH = ROOT / "models" / "phase.pt"
PHASE_LABEL_MAP_PATH = ROOT / "data" / "phase_label_map.npy"

T = 60
D = 33 * 3

NOTE_MAP = {
    "push_ups": "D4",
    "sit_ups": "C4",
    "squats": "E4",
    "pushup": "D4",
    "situp": "C4",
    "squat": "E4",
}


class GRUClassifier(nn.Module):
    def __init__(self, d_in: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.fc(h[0])


class GRUPhase(nn.Module):
    def __init__(self, d_in: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out)


def pad_trim_flatten(seq: np.ndarray) -> np.ndarray:
    flat = seq.reshape(seq.shape[0], D).astype(np.float32)
    if flat.shape[0] >= T:
        x = flat[:T]
    else:
        pad = np.repeat(flat[-1:], T - flat.shape[0], axis=0)
        x = np.concatenate([flat, pad], axis=0)
    return x.reshape(1, T, D)


def window_sequence(flat: np.ndarray, stride: int) -> list[tuple[np.ndarray, int, int]]:
    total = flat.shape[0]
    windows = []
    if total < T:
        pad = np.repeat(flat[-1:], T - total, axis=0)
        x = np.concatenate([flat, pad], axis=0)
        windows.append((x, 0, total))
        return windows

    for start in range(0, total - T + 1, stride):
        end = start + T
        windows.append((flat[start:end], start, end))

    last_end = windows[-1][2] if windows else 0
    if last_end < total:
        start = max(total - T, 0)
        windows.append((flat[start:start + T], start, total))

    return windows


def merge_predictions(
    preds: list[dict[str, float | int | str]],
    fps: float,
) -> list[dict[str, float | str]]:
    if not preds:
        return []

    merged = [preds[0].copy()]
    for item in preds[1:]:
        last = merged[-1]
        if item["label"] == last["label"]:
            last["end_frame"] = item["end_frame"]
            last["confidence"] = (float(last["confidence"]) + float(item["confidence"])) / 2.0
        else:
            merged.append(item.copy())

    for item in merged:
        item["start_s"] = round(float(item["start_frame"]) / fps, 2)
        item["end_s"] = round(float(item["end_frame"]) / fps, 2)

    return merged


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosang = float(np.dot(ba, bc) / denom)
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def fill_nans(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    if np.all(np.isnan(out)):
        return out
    for i in range(1, len(out)):
        if np.isnan(out[i]):
            out[i] = out[i - 1]
    for i in range(len(out) - 2, -1, -1):
        if np.isnan(out[i]):
            out[i] = out[i + 1]
    return out


def smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return arr
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(padded, kernel, mode="valid")


def angle_series(seq_xy: np.ndarray, label: str) -> np.ndarray:
    # Mediapipe pose indices.
    L_SH, R_SH = 11, 12
    L_EL, R_EL = 13, 14
    L_WR, R_WR = 15, 16
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANK, R_ANK = 27, 28

    angles = []
    for frame in seq_xy:
        if label == "push_ups":
            a_l = angle_deg(frame[L_SH], frame[L_EL], frame[L_WR])
            a_r = angle_deg(frame[R_SH], frame[R_EL], frame[R_WR])
        elif label == "squats":
            a_l = angle_deg(frame[L_HIP], frame[L_KNEE], frame[L_ANK])
            a_r = angle_deg(frame[R_HIP], frame[R_KNEE], frame[R_ANK])
        elif label == "sit_ups":
            a_l = angle_deg(frame[L_SH], frame[L_HIP], frame[L_KNEE])
            a_r = angle_deg(frame[R_SH], frame[R_HIP], frame[R_KNEE])
        else:
            angles.append(np.nan)
            continue

        angles.append((a_l + a_r) / 2.0)

    ang = np.array(angles, dtype=np.float32)
    ang = fill_nans(ang)
    if np.any(~np.isnan(ang)):
        ang = smooth(ang, window=5)
    return ang


def count_reps(
    seq_xy: np.ndarray,
    label: str,
    high_thr: float,
    low_thr: float,
    min_frames: int = 8,
) -> list[int]:
    ang = angle_series(seq_xy, label)
    if np.all(np.isnan(ang)):
        return []

    state = "unknown"
    reps = []
    last_rep = -min_frames
    for i, val in enumerate(ang):
        if np.isnan(val):
            continue
        if state == "unknown":
            if val > high_thr:
                state = "up"
            elif val < low_thr:
                state = "down"
            continue
        if state == "up" and val < low_thr:
            state = "down"
        elif state == "down" and val > high_thr and (i - last_rep) >= min_frames:
            reps.append(i)
            last_rep = i
            state = "up"

    return reps


def smooth_labels(labels: list[str], window: int = 3) -> list[str]:
    if window <= 1:
        return labels
    out = []
    for i in range(len(labels)):
        start = max(0, i - window // 2)
        end = min(len(labels), i + window // 2 + 1)
        slice_labels = labels[start:end]
        counts = {}
        for lab in slice_labels:
            counts[lab] = counts.get(lab, 0) + 1
        out.append(max(counts, key=counts.get))
    return out


def phase_to_rep_events(
    phase_labels: list[str],
    fps: float,
    min_frames: int = 8,
) -> list[dict[str, float | str]]:
    events = []
    state = {}
    last_rep = {}

    for i, phase in enumerate(phase_labels):
        if not phase or "_" not in phase:
            continue
        if phase.endswith("_up"):
            exercise = phase[:-3]
            phase_state = "up"
        elif phase.endswith("_down"):
            exercise = phase[:-5]
            phase_state = "down"
        else:
            continue

        if exercise not in state:
            state[exercise] = None
            last_rep[exercise] = -min_frames

        if phase_state == "down":
            state[exercise] = "down"
            continue

        if phase_state == "up":
            if state[exercise] == "down" and (i - last_rep[exercise]) >= min_frames:
                t_s = round(i / fps, 2)
                events.append(
                    {
                        "label": exercise,
                        "note": NOTE_MAP.get(exercise, "NA"),
                        "confidence": 1.0,
                        "start_s": t_s,
                        "end_s": t_s,
                    }
                )
                last_rep[exercise] = i
            state[exercise] = "up"

    return events


def load_classifier() -> tuple[GRUClassifier, dict[str, int]]:
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Missing label map at {LABEL_MAP_PATH}")
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Missing classifier at {CLASSIFIER_PATH}")

    label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
    model = GRUClassifier(D, 128, len(label_map))
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location="cpu"))
    model.eval()
    return model, label_map


def load_phase_classifier() -> tuple[GRUPhase, dict[str, int]] | tuple[None, None]:
    if not PHASE_LABEL_MAP_PATH.exists() or not PHASE_CLASSIFIER_PATH.exists():
        return None, None
    label_map = np.load(PHASE_LABEL_MAP_PATH, allow_pickle=True).item()
    model = GRUPhase(D, 128, len(label_map))
    model.load_state_dict(torch.load(PHASE_CLASSIFIER_PATH, map_location="cpu"))
    model.eval()
    return model, label_map


app = FastAPI(title="Exercise Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL, LABEL_MAP = load_classifier()
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
PHASE_MODEL, PHASE_LABEL_MAP = load_phase_classifier()
PHASE_ID_TO_LABEL = {v: k for k, v in PHASE_LABEL_MAP.items()} if PHASE_LABEL_MAP else {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Please upload an .mp4 file.")

    storage_dir = ROOT / "backend" / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        suffix=".mp4",
        dir=storage_dir,
        delete=False,
    ) as tmp:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty upload.")
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        target_fps = 15
        seq = video_to_keypoints(
            video_path=str(tmp_path),
            model_path=str(MODEL_PATH),
            target_fps=target_fps,
            max_seconds=30,
        )
        flat = seq.reshape(seq.shape[0], D).astype(np.float32)
        seq_xy = seq[:, :, :2]

        rep_events = []
        segments = []
        if PHASE_MODEL is not None:
            with torch.no_grad():
                logits = PHASE_MODEL(torch.from_numpy(flat).unsqueeze(0))
                pred_ids = logits[0].argmax(-1).cpu().numpy().tolist()
            raw_labels = [PHASE_ID_TO_LABEL.get(i, "") for i in pred_ids]
            smooth_phase = smooth_labels(raw_labels, window=3)
            rep_events = phase_to_rep_events(smooth_phase, fps=float(target_fps), min_frames=8)

        if not rep_events:
            preds = []
            stride = 15  # ~1 second at 15 fps
            windows = window_sequence(flat, stride=stride)
            with torch.no_grad():
                for window, start, end in windows:
                    x = window.reshape(1, T, D)
                    logits = MODEL(torch.from_numpy(x))
                    probs = torch.softmax(logits[0], dim=0).numpy()
                    pred_id = int(probs.argmax())
                    pred_label = ID_TO_LABEL[pred_id]
                    conf = float(probs[pred_id])
                    preds.append(
                        {
                            "label": pred_label,
                            "note": NOTE_MAP.get(pred_label, "NA"),
                            "confidence": round(conf, 4),
                            "start_frame": int(start),
                            "end_frame": int(end),
                        }
                    )

            sequence = merge_predictions(preds, fps=float(target_fps))
            for seg in sequence:
                label = str(seg["label"])
                start_frame = int(seg.get("start_frame", 0))
                end_frame = int(seg.get("end_frame", start_frame + 1))
                end_frame = min(end_frame, seq_xy.shape[0])
                start_frame = max(0, min(start_frame, end_frame - 1))

                thresholds = {
                    "push_ups": (160.0, 95.0),
                    "sit_ups": (160.0, 95.0),
                    "squats": (165.0, 100.0),
                }
                high_thr, low_thr = thresholds.get(label, (160.0, 100.0))
                reps = count_reps(
                    seq_xy[start_frame:end_frame],
                    label,
                    high_thr=high_thr,
                    low_thr=low_thr,
                    min_frames=8,
                )

                if reps:
                    for rep in reps:
                        frame_idx = start_frame + rep
                        t_s = round(frame_idx / float(target_fps), 2)
                        rep_events.append(
                            {
                                "label": label,
                                "note": NOTE_MAP.get(label, "NA"),
                                "confidence": seg["confidence"],
                                "start_s": t_s,
                                "end_s": t_s,
                            }
                        )
                else:
                    mid_s = round((float(seg["start_s"]) + float(seg["end_s"])) / 2.0, 2)
                    rep_events.append(
                        {
                            "label": label,
                            "note": NOTE_MAP.get(label, "NA"),
                            "confidence": seg["confidence"],
                            "start_s": mid_s,
                            "end_s": mid_s,
                        }
                    )

                segments.append(
                    {
                        "label": label,
                        "note": NOTE_MAP.get(label, "NA"),
                        "confidence": seg["confidence"],
                        "start_s": seg["start_s"],
                        "end_s": seg["end_s"],
                        "reps": len(reps),
                    }
                )

        rep_events.sort(key=lambda x: float(x["start_s"]))
        primary = rep_events[0] if rep_events else {"label": "NA", "note": "NA", "confidence": 0.0}
        pred_label = primary["label"]
        conf = float(primary["confidence"])
        note = primary["note"]
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    return {
        "label": pred_label,
        "confidence": round(conf, 4),
        "note": note,
        "sequence": rep_events,
        "segments": segments,
        "frames": int(seq.shape[0]),
    }
