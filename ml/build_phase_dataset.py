from pathlib import Path
import csv
import numpy as np

from ml.pose_tasks import video_to_keypoints

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"

LABELS_CSV = ROOT / "data" / "phase_labels.csv"
OUT = ROOT / "data" / "npz"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_FPS = 15
T = 60
STRIDE = 15
D = 33 * 3


def load_labels():
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing {LABELS_CSV}")

    rows = []
    with LABELS_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "video": row["video"].strip(),
                    "start": int(row["start_frame"]),
                    "end": int(row["end_frame"]),
                    "phase": row["phase"].strip(),
                }
            )
    return rows


def build_label_map(rows):
    phases = sorted({r["phase"] for r in rows})
    label_map = {name: i for i, name in enumerate(phases)}
    np.save(ROOT / "data" / "phase_label_map.npy", label_map)
    return label_map


def resolve_video(path_str):
    p = Path(path_str)
    if p.exists():
        return p
    candidate = RAW / path_str
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Video not found: {path_str}")


def main():
    rows = load_labels()
    label_map = build_label_map(rows)

    by_video = {}
    for r in rows:
        by_video.setdefault(r["video"], []).append(r)

    X_list, y_list = [], []
    for video_name, segments in by_video.items():
        video_path = resolve_video(video_name)
        seq = video_to_keypoints(
            video_path=str(video_path),
            model_path=str(MODEL_PATH),
            target_fps=TARGET_FPS,
            max_seconds=120,
        )
        flat = seq.reshape(seq.shape[0], D).astype(np.float32)

        labels = np.full((flat.shape[0],), -1, dtype=np.int64)
        for seg in segments:
            start = max(0, seg["start"])
            end = min(flat.shape[0], seg["end"])
            if start >= end:
                continue
            labels[start:end] = label_map[seg["phase"]]

        if flat.shape[0] < T:
            pad = np.repeat(flat[-1:], T - flat.shape[0], axis=0)
            flat = np.concatenate([flat, pad], axis=0)
            labels = np.concatenate([labels, np.full((T - labels.shape[0],), -1)], axis=0)

        for start in range(0, flat.shape[0] - T + 1, STRIDE):
            x = flat[start:start + T]
            y = labels[start:start + T]
            if np.all(y == -1):
                continue
            X_list.append(x)
            y_list.append(y)

    if not X_list:
        raise RuntimeError("No labeled windows built. Check phase_labels.csv.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.int64)

    np.savez_compressed(OUT / "phase_windows.npz", X=X, y=y)
    print("saved:", OUT / "phase_windows.npz")


if __name__ == "__main__":
    main()
