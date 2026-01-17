from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from ml.pose_tasks import video_to_keypoints

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"
label_map = np.load(ROOT / "data" / "label_map.npy", allow_pickle=True).item()
labels_sorted = [lbl for lbl, idx in sorted(label_map.items(), key=lambda kv: kv[1])]

NOTE_MAP = {
    "push_ups": "E4",
    "sit_ups": "G4",
    "squats": "C4",
}

T = 60
D = 33 * 3


class GRUClassifier(nn.Module):
    def __init__(self, d_in, hidden, num_classes):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[0])


def pad_trim_flatten(seq):
    flat = seq.reshape(seq.shape[0], D).astype(np.float32)
    if flat.shape[0] >= T:
        x = flat[:T]
    else:
        pad = np.repeat(flat[-1:], T - flat.shape[0], axis=0)
        x = np.concatenate([flat, pad], axis=0)
    return x.reshape(1, T, D)


def main(video_path: str):
    seq = video_to_keypoints(
        video_path=video_path,
        model_path=str(MODEL_PATH),
        target_fps=15,
        max_seconds=30,
    )
    x = pad_trim_flatten(seq)

    state = torch.load(ROOT / "models" / "exercise.pt", map_location="cpu")
    num_classes = state["fc.weight"].shape[0]

    # Align labels with the checkpoint output dimensionality (drop extra labels like "test").
    use_labels = labels_sorted[:num_classes]
    id_to_label = {i: lbl for i, lbl in enumerate(use_labels)}

    model = GRUClassifier(D, 128, num_classes)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        probs = torch.softmax(logits[0], dim=0).numpy()

    pred_id = int(probs.argmax())
    pred_label = id_to_label.get(pred_id, "unknown")
    conf = float(probs[pred_id])
    note = NOTE_MAP.get(pred_label, "NA")

    print("exercise:", pred_label)
    print("confidence:", round(conf, 4))
    print("note:", note)

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
