from pathlib import Path
import csv
import numpy as np

from ml.pose_tasks import video_to_keypoints

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"

LABELS_CSV = ROOT / "data" / "phase_labels.csv"
SECONDS_CSV = ROOT / "data" / "phase_labels_seconds.csv"
OUT = ROOT / "data" / "npz"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_FPS = 15
T = 60
STRIDE = 15
D = 33 * 3


def load_labels_frames():
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing {LABELS_CSV}")

    rows = []
    with LABELS_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"video", "start_frame", "end_frame", "phase"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{LABELS_CSV} missing required columns {sorted(required)}")
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


def load_labels_seconds():
    if not SECONDS_CSV.exists():
        raise FileNotFoundError(f"Missing {SECONDS_CSV}")

    by_video: dict[str, list[tuple[float, str]]] = {}
    phases = set()
    with SECONDS_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"video", "time_s", "phase"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{SECONDS_CSV} missing required columns {sorted(required)}")
        for row in reader:
            video = row["video"].strip()
            t_s = float(row["time_s"])
            phase = row["phase"].strip()
            by_video.setdefault(video, []).append((t_s, phase))
            phases.add(phase)

    if not by_video:
        raise RuntimeError(f"No labeled events found in {SECONDS_CSV}")

    return by_video, sorted(phases)


def load_labels():
    if LABELS_CSV.exists():
        try:
            rows = load_labels_frames()
            return "frames", rows, sorted({r["phase"] for r in rows})
        except (ValueError, KeyError):
            pass
    events_by_video, phases = load_labels_seconds()
    return "seconds", events_by_video, phases


def build_label_map(phases):
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
    mode, payload, phases = load_labels()
    label_map = build_label_map(phases)

    X_list, y_list = [], []
    if mode == "frames":
        by_video = {}
        for r in payload:
            by_video.setdefault(r["video"], []).append(r)
    else:
        by_video = payload

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
        if mode == "frames":
            for seg in segments:
                start = max(0, seg["start"])
                end = min(flat.shape[0], seg["end"])
                if start >= end:
                    continue
                labels[start:end] = label_map[seg["phase"]]
        else:
            events = sorted(segments, key=lambda x: x[0])
            for i in range(len(events)):
                start_s, phase = events[i]
                end_s = events[i + 1][0] if i + 1 < len(events) else flat.shape[0] / TARGET_FPS
                start = int(round(start_s * TARGET_FPS))
                end = int(round(end_s * TARGET_FPS))
                start = max(0, start)
                end = min(flat.shape[0], end)
                if start >= end:
                    continue
                labels[start:end] = label_map[phase]

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
