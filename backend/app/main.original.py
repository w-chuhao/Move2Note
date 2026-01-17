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

T = 60
D = 33 * 3

NOTE_MAP = {
    "push_ups": "E4",
    "sit_ups": "G4",
    "squats": "C4",
    "pushup": "E4",
    "situp": "G4",
    "squat": "C4",
}


class GRUClassifier(nn.Module):
    def __init__(self, d_in: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.fc(h[0])


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
        item.pop("start_frame", None)
        item.pop("end_frame", None)

    return merged


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
        primary = sequence[0] if sequence else {"label": "NA", "note": "NA", "confidence": 0.0}
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
        "sequence": sequence,
        "frames": int(seq.shape[0]),
    }
