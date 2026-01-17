const form = document.getElementById("upload-form");
const input = document.getElementById("video-input");
const statusEl = document.getElementById("status");
const noteEl = document.getElementById("note-output");
const labelEl = document.getElementById("label-output");
const confEl = document.getElementById("confidence-output");

const API_URL = "http://localhost:8000/predict";

const NOTE_FREQ = {
  C4: 261.63,
  D4: 293.66,
  E4: 329.63,
  G4: 392.0,
  A4: 440.0,
};

const NOTE_LABEL = {
  push_ups: "E4",
  sit_ups: "G4",
  squats: "C4",
};

const setStatus = (msg) => {
  statusEl.textContent = msg;
};

const playNote = async (note) => {
  const freq = NOTE_FREQ[note];
  if (!freq) {
    console.warn(`Unknown note: ${note}`);
    return;
  }
  
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const ctx = new AudioCtx();
    await ctx.resume();
    
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    
    osc.type = "sine";
    osc.frequency.value = freq;
    
    // Smooth envelope to avoid clicks
    gain.gain.setValueAtTime(0, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0.2, ctx.currentTime + 0.02);
    gain.gain.linearRampToValueAtTime(0.15, ctx.currentTime + 0.3);
    gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.8);
    
    osc.connect(gain);
    gain.connect(ctx.destination);
    
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.8);
    
    console.log(`Playing note: ${note} at ${freq}Hz`);
  } catch (err) {
    console.error("Audio playback failed:", err);
    setStatus(`Note identified: ${note} (playback failed)`);
  }
};

const setResults = (data) => {
  noteEl.textContent = data.note || "—";
  labelEl.textContent = `Label: ${data.label ?? "—"}`;
  confEl.textContent = `Confidence: ${data.confidence ?? "—"}`;

  const resolvedNote = data.note !== "NA" ? data.note : NOTE_LABEL[data.label];
  if (resolvedNote && resolvedNote !== "NA") {
    playNote(resolvedNote);
  } else {
    console.warn("No valid note to play");
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
    setStatus("Please select a video file.");
    return;
  }

  const file = input.files[0];
  const validExts = [".mp4", ".mov", ".avi", ".webm"];
  const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
  if (!validExts.includes(ext)) {
    setStatus("Only MP4, MOV, AVI, and WEBM files are supported.");
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
