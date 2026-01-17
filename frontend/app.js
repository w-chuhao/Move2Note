const form = document.getElementById("upload-form");
const input = document.getElementById("video-input");
const statusEl = document.getElementById("status");
const noteEl = document.getElementById("note-output");
const labelEl = document.getElementById("label-output");
const confEl = document.getElementById("confidence-output");

const API_URL = "http://localhost:8000/predict";

const NOTE_FREQ = {
  C4: 261.63,
  E4: 329.63,
  G4: 392.0,
};

const NOTE_LABEL = {
  pushup: "E4",
  situp: "G4",
  squat: "C4",
};

const setStatus = (msg) => {
  statusEl.textContent = msg;
};

const playNote = async (note) => {
  const freq = NOTE_FREQ[note];
  if (!freq) {
    return;
  }
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioCtx();
  await ctx.resume();
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = "sine";
  osc.frequency.value = freq;
  gain.gain.value = 0.15;
  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.start();
  osc.stop(ctx.currentTime + 0.8);
};

const setResults = (data) => {
  noteEl.textContent = data.note || "—";
  labelEl.textContent = `Label: ${data.label ?? "—"}`;
  confEl.textContent = `Confidence: ${data.confidence ?? "—"}`;

  const resolvedNote = data.note || NOTE_LABEL[data.label];
  if (resolvedNote) {
    playNote(resolvedNote);
  }
};

input.addEventListener("change", () => {
  if (input.files.length) {
    setStatus("Video uploaded. Ready to analyze.");
  } else {
    setStatus("Waiting for a clip.");
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!input.files.length) {
    setStatus("Please select an MP4 file.");
    return;
  }

  const file = input.files[0];
  if (!file.name.toLowerCase().endsWith(".mp4")) {
    setStatus("Only MP4 files are supported.");
    return;
  }

  setStatus("Uploading and analyzing...");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Prediction failed.");
    }
    const data = await res.json();
    setResults(data);
    setStatus(`Done. Frames processed: ${data.frames ?? "-"}`);
  } catch (err) {
    setStatus(err.message || "Something went wrong.");
  }
});
