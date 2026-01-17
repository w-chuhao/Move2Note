from pathlib import Path
import numpy as np
from ml.pose_tasks import video_to_keypoints

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "npz"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"

# windowing: turn 1 long pose sequence into many fixed-length samples
T = 60          # frames per sample (60 frames @ 15fps ~ 4 seconds)
STRIDE = 10     # slide by 10 frames -> lots of samples from 1 video
D = 33 * 3

def make_windows(seq):
    # seq: [t,33,3] -> list of [T,D]
    if seq.shape[0] < 5:
        return []
    flat = seq.reshape(seq.shape[0], D).astype(np.float32)

    windows = []
    if flat.shape[0] < T:
        pad = np.repeat(flat[-1:], T - flat.shape[0], axis=0)
        x = np.concatenate([flat, pad], axis=0)
        windows.append(x)
        return windows

    for start in range(0, flat.shape[0] - T + 1, STRIDE):
        windows.append(flat[start:start+T])
    return windows

EXCLUDED_FOLDERS = {"test"}  # folders to skip
VIDEO_EXTS = ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.webm")


def augment_window(w, n_aug=2):
    """Simple augmentation: time jitter + Gaussian noise."""
    augmented = [w]
    for _ in range(n_aug):
        aug = w.copy()
        # Small Gaussian noise on coordinates
        aug += np.random.randn(*aug.shape).astype(np.float32) * 0.02
        augmented.append(aug)
        # Horizontal flip: negate x coordinates (every 3rd value starting at 0)
        flipped = w.copy()
        flipped[:, 0::3] *= -1
        augmented.append(flipped)
    return augmented


def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Missing {RAW}")

    labels = sorted([p.name for p in RAW.iterdir() if p.is_dir() and p.name not in EXCLUDED_FOLDERS])
    if len(labels) < 2:
        raise RuntimeError("Need at least 2 label folders under data/raw/ (example: pushup/, squat/)")

    label_to_id = {name: i for i, name in enumerate(labels)}
    np.save(ROOT / "data" / "label_map.npy", label_to_id)

    X_list, y_list = [], []
    meta = []

    for label in labels:
        vids = []
        for ext in VIDEO_EXTS:
            vids.extend((RAW / label).glob(ext))
        for vid in vids:
            seq = video_to_keypoints(
                video_path=str(vid),
                model_path=str(MODEL_PATH),
                target_fps=15,
                max_seconds=30,
            )
            windows = make_windows(seq)
            for w in windows:
                # Apply augmentation to boost small datasets
                for aug_w in augment_window(w, n_aug=2):
                    X_list.append(aug_w)
                    y_list.append(label_to_id[label])
            meta.append((label, vid.name, int(seq.shape[0]), len(windows)))

    if len(X_list) == 0:
        raise RuntimeError("No pose windows generated. Check videos (full body visible, enough frames).")

    X = np.stack(X_list, axis=0).astype(np.float32)  # [N,T,D]
    y = np.array(y_list, dtype=np.int64)             # [N]

    np.savez_compressed(OUT / "train_windows.npz", X=X, y=y)

    print("labels:", label_to_id)
    print("windows:", X.shape[0], " shape:", X.shape)
    print("per_video:", meta)

if __name__ == "__main__":
    main()
