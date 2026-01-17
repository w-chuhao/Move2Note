const form = document.getElementById("upload-form");
const input = document.getElementById("video-input");
const statusEl = document.getElementById("status");
const noteEl = document.getElementById("note-output");
const labelEl = document.getElementById("label-output");
const confEl = document.getElementById("confidence-output");
const sequenceEl = document.getElementById("sequence-output");

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
  pushup: "E4",
  situp: "G4",
  squat: "C4",
};

const setStatus = (msg) => {
  statusEl.textContent = msg;
};

const playSequence = async (sequence) => {
  if (!sequence || !sequence.length) {
    return;
  }

  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioCtx();
  await ctx.resume();

  let t = ctx.currentTime + 0.1;
  for (const item of sequence) {
    const resolvedNote = item.note !== "NA" ? item.note : NOTE_LABEL[item.label];
    const freq = NOTE_FREQ[resolvedNote];
    if (!freq) {
      t += 0.5;
      continue;
    }

    const duration = Math.max(0.45, Math.min(1.0, (item.end_s ?? 0) - (item.start_s ?? 0)));
    const delayMs = Math.max(0, (t - ctx.currentTime) * 1000);
    setTimeout(() => {
      noteEl.textContent = resolvedNote;
      labelEl.textContent = `Label: ${item.label ?? "--"}`;
      confEl.textContent = `Confidence: ${item.confidence ?? "--"}`;
    }, delayMs);

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = freq;

    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(0.18, t + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + duration);

    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t);
    osc.stop(t + duration + 0.05);

    t += duration + 0.12;
  }
};

const setResults = (data) => {
  noteEl.textContent = data.note || "--";
  labelEl.textContent = `Label: ${data.label ?? "--"}`;
  confEl.textContent = `Confidence: ${data.confidence ?? "--"}`;

  const sequence = Array.isArray(data.sequence) ? data.sequence : [];
  if (sequence.length) {
    const summary = sequence
      .map((item) => `${item.label} (${item.note})`)
      .join(" -> ");
    sequenceEl.textContent = `Sequence: ${summary}`;
    playSequence(sequence);
    return;
  }

  sequenceEl.textContent = "Sequence: --";
  const resolvedNote = data.note !== "NA" ? data.note : NOTE_LABEL[data.label];
  if (resolvedNote && resolvedNote !== "NA") {
    playSequence([{ label: data.label, note: resolvedNote, start_s: 0, end_s: 0.8 }]);
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
