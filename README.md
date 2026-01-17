# Hack-n-roll: Move2Note

Turn workout videos into music. Upload an exercise clip and the app predicts the exercise, maps it to a musical note, and plays it back. It also supports a simple song mode (e.g., "Baa Baa Black Sheep") that shows the note/exercise sequence to perform.

## Features
- Upload MP4 videos and get an exercise prediction.
- Plays a note per detected rep (heuristic rep counting).
- Song picker UI with a note/exercise "sheet" view.

## Project structure
- `backend/app/main.py`: FastAPI server and ML inference.
- `ml/pose_tasks.py`: MediaPipe pose extraction.
- `models/`: trained model files (`exercise.pt`, `pose_landmarker_lite.task`).
- `frontend/`: static UI (HTML/CSS/JS).

## Requirements
- Python 3.11+ recommended
- `requirements.txt` for backend dependencies
- Dependencies:
  - fastapi
  - uvicorn[standard]
  - python-multipart
  - numpy
  - torch
  - opencv-python
  - mediapipe

## Setup
Create/activate your environment and install dependencies (make sure you are inside the correct environment):

```bash
conda create -n hacknroll python=3.11
conda activate hacknroll
python -m pip install -r requirements.txt
```

## Run the backend
From repo root:

```bash
uvicorn backend.app.main:app --reload
```

## Run the frontend
From repo root:

```bash
cd frontend
python -m http.server 5173 --bind 127.0.0.1
```

Open `http://127.0.0.1:5173`.

## Usage
1. Start backend and frontend.
2. Upload a short video of a workout.
3. The UI shows the note and exercise sequence and plays the notes.
4. Select a song (e.g., Baa Baa Black Sheep) to see the required exercise order.

## Notes
- The note mapping is currently:
  - sit_ups → C4
  - push_ups → D4
  - squats → E4
- Rep counting uses a simple pose angle heuristic. For higher accuracy, consider retraining with rep-level labels.

## Troubleshooting
- If the frontend shows a directory listing, make sure you run the server with `--directory frontend`.
- If the backend cannot load models, ensure `models/exercise.pt` and `models/pose_landmarker_lite.task` exist.
