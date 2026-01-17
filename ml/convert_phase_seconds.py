from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "data" / "phase_labels_seconds.csv"
OUT_CSV = ROOT / "data" / "phase_labels.csv"

TARGET_FPS = 15


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing {IN_CSV}")

    with IN_CSV.open("r", newline="") as fin, OUT_CSV.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = ["video", "start_frame", "end_frame", "phase"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        by_video = {}
        for row in reader:
            video = row["video"].strip()
            t_s = float(row["time_s"])
            phase = row["phase"].strip()
            by_video.setdefault(video, []).append((t_s, phase))

        for video, events in by_video.items():
            events.sort(key=lambda x: x[0])
            for i in range(len(events) - 1):
                start_s, phase = events[i]
                end_s, _ = events[i + 1]
                start_frame = int(round(start_s * TARGET_FPS))
                end_frame = int(round(end_s * TARGET_FPS))
                if end_frame <= start_frame:
                    continue
                writer.writerow(
                    {
                        "video": video,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "phase": phase,
                    }
                )

    print("saved:", OUT_CSV)


if __name__ == "__main__":
    main()
