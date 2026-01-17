import numpy as np
import torch
import torch.nn as nn
from ml.pose_tasks import video_to_keypoints

NOTE_MAP = {
    "pushup": "E4",
    "situp": "G4",
    "squat": "C4",
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

def pad_trim_flatten(x):
    if x.shape[0] >= T:
        x = x[:T]
    else:
        pad = np.repeat(x[-1:], T - x.shape[0], axis=0)
        x = np.concatenate([x, pad], axis=0)
    return x.reshape(1, T, D).astype(np.float32)

def main(video_path):
    label_map = np.load("data/label_map.npy", allow_pickle=True).item()
    id_to_label = {v: k for k, v in label_map.items()}

    model = GRUClassifier(D, 128, len(label_map))
    model.load_state_dict(torch.load("models/exercise.pt", map_location="cpu"))
    model.eval()

    kp = video_to_keypoints(video_path)
    x = pad_trim_flatten(kp)

    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        probs = torch.softmax(logits[0], dim=0).numpy()

    pred_id = int(probs.argmax())
    pred_label = id_to_label[pred_id]
    conf = float(probs[pred_id])
    note = NOTE_MAP.get(pred_label, "NA")

    print("exercise:", pred_label)
    print("confidence:", round(conf, 4))
    print("note:", note)

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
