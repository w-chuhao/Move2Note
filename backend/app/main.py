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
        seq = video_to_keypoints(
            video_path=str(tmp_path),
            model_path=str(MODEL_PATH),
            target_fps=15,
            max_seconds=30,
        )
        x = pad_trim_flatten(seq)

        with torch.no_grad():
            logits = MODEL(torch.from_numpy(x))
            probs = torch.softmax(logits[0], dim=0).numpy()

        pred_id = int(probs.argmax())
        pred_label = ID_TO_LABEL[pred_id]
        conf = float(probs[pred_id])
        note = NOTE_MAP.get(pred_label, "NA")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    return {
        "label": pred_label,
        "confidence": round(conf, 4),
        "note": note,
        "frames": int(seq.shape[0]),
    }
