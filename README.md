# Amharic Speech-to-Text — Two Offline Models, One Web App

Local Amharic speech recognition with **two interchangeable models**, a live microphone waveform, and a transcription history — all running on your own machine.

**CPU only · No GPU · No CUDA · No cloud API · Offline inference**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20only-ee4c2c?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/Models-Apache--2.0-green)

---

## Overview

This project provides a local web interface and CLI for transcribing Amharic speech using two public Hugging Face models. After a one-time setup (dependencies + model download), transcription runs entirely offline.

**Key features**
- Switch between **Shook Tiny** (fast) and **Wav2Vec2 XLSR** (larger) from the UI
- Record from the microphone with a live waveform, or upload an audio file
- Preview and replay the captured clip before sending
- Keep the same clip after transcription to re-send it with the other model for comparison
- Edit the transcription text and copy it (auto-copied on completion)
- Browse, filter, play back, and delete past transcriptions in History
- Runs locally at `http://127.0.0.1:5000` (not exposed to your network)

---

## Models

| UI label | Hugging Face model | Architecture | Parameters | Approx. local size | Notes |
|---|---|---|---:|---:|---|
| **Shook Tiny** | [`b1n1yam/shook-tiny-amharic-stage2-polish`](https://huggingface.co/b1n1yam/shook-tiny-amharic-stage2-polish) | Whisper Tiny fine-tuned for Amharic | 37.8M | 146 MB | Fast default; model card reports ~20.76% WER |
| **Wav2Vec2 XLSR** | [`agkphysics/wav2vec2-large-xlsr-53-amharic`](https://huggingface.co/agkphysics/wav2vec2-large-xlsr-53-amharic) | Wav2Vec2-Large-XLSR-53 (CTC) | 315.7M | 1.2 GB | Model card reports ~26% WER / 7% CER (ALFFA val); output is morpheme-segmented (extra spaces inside words are expected) |

Both models are public checkpoints and are downloaded anonymously during setup — no Hugging Face account, token, or paid service required.

> **Note on Wav2Vec2 spacing:** this checkpoint was trained on morpheme-segmented transcripts, so spaces may appear inside words. This is expected behavior, not a decoding bug.

---

## Requirements

- **Python 3.10+** (64-bit recommended; tested with 3.12)
- **Git** (to clone) or a copy of this repository
- **ffmpeg** on `PATH` — recommended for M4A/WebM/OGG and some other containers  
  - Windows: `winget install Gyan.FFmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: install via your package manager (e.g. `sudo apt install ffmpeg`)
- **RAM:** 8 GB minimum, 16 GB recommended (using both models in one session can reach ~2 GB RSS)
- **Disk:** ~4 GB free (virtual environment + CPU PyTorch + ~1.4 GB model weights)
- **Browser:** Chrome/Edge recommended for microphone and waveform features
- **No Node.js required** to run the app

---

## Quick start

### 1. Get the code

```bash
git clone https://github.com/JoHaile/Amharic-Speech-to-Text.git
cd Amharic-Speech-to-Text
```

> **Model weights are not stored in Git.** `setup` will download them (~1.4 GB) on first run, or you can copy an existing `models/` folder into the repo to skip downloading.

### 2. Run setup (one time)

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

**macOS / Linux (manual):**
```bash
python -m venv .venv
# activate: source .venv/bin/activate  (or .venv\Scripts\activate on Windows)
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0+cpu"
pip install -r requirements.txt
```
Then download both model snapshots into `models/shook-tiny-stage2/` and `models/wav2vec2-large-xlsr-53-amharic/` (the Windows `setup.ps1` does this automatically and verifies offline loading).

`setup.ps1` will:
1. Create `.venv`
2. Install **CPU-only** PyTorch
3. Install pinned dependencies
4. Download both models anonymously into `models/` (skips files that already exist)
5. Verify both models load with networking disabled

It is safe to re-run.

### 3. Start the web app

**Windows:**
```powershell
powershell -ExecutionPolicy Bypass -File .\start-web.ps1
# or
.\.venv\Scripts\python.exe app.py
```

**macOS / Linux:**
```bash
.venv/bin/python app.py
```

Open **http://127.0.0.1:5000** (the Windows start script opens it automatically). The default model (Shook Tiny) loads at startup; Wav2Vec2 loads the first time you select it.

---

## Web interface

- **New recording** — choose a model tab (Shook Tiny / Wav2Vec2), record from the mic, preview the clip, and press **Send** to transcribe.
- **Live waveform** — a real-time visualization while recording (visual only; does not affect audio or transcription).
- **Clip stays available** — after recording or after transcription, the clip remains loaded so you can switch models and press **Send** again to compare outputs. **Record again** starts fresh; **Clear** removes the clip and result.
- **Upload** — drop or browse an audio file, then review and send it from the New recording pane.
- **Editable result** — transcription text can be edited before copying; it is also auto-copied to the clipboard after each transcription.
- **History** — saved transcriptions with model badge, metrics, audio playback, delete, and filter (**All / Shook / Wav2Vec2**). Stored locally in `history.json` and `web_uploads/` (git-ignored).

The server binds to `127.0.0.1` only.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `ASR_PORT` | `5000` | Port for the local web server |
| `ASR_NO_BROWSER` | unset | Set to `1` to prevent auto-opening the browser |

---

## Command-line usage

```bash
# Default model (Shook Tiny)
python transcribe.py "path/to/audio.flac"

# Use Wav2Vec2
python transcribe.py "path/to/audio.flac" --model wav2vec2

# Interactive mode (keeps the model in memory)
python transcribe.py --model wav2vec2
```

### CLI options

| Flag | Description |
|---|---|
| `-m, --model {shook,wav2vec2}` | Model to use (default: `shook`) |
| `-o, --output PATH` | Write transcript to a specific file (default: `output/<name>.txt` or `output/<name>.wav2vec2.txt`) |
| `-i, --interactive` | Force interactive mode |
| `--json` | Print metrics as JSON |

Transcripts are written as UTF-8.

---

## How it works

- **Model registry:** `transcribe.py` defines model specs and loads `AutoModelForSpeechSeq2Seq` (Whisper/Shook Tiny) or `AutoModelForCTC` (Wav2Vec2) with `local_files_only=True`.
- **Audio pipeline:** decode/resample to mono 16 kHz float, average stereo, sanitize NaN/Inf, and fall back to **ffmpeg** for containers that libsndfile cannot open (e.g. AAC-in-M4A).
- **Long audio:** split into ~30 s chunks, snap boundaries near low-energy frames (±0.75 s window) with 0.5 s overlap, and never truncate the end.
- **Wav2Vec2 decoding:** uses plain CTC batch decoding (without stripping word-delimiter as a special token removal side effect) to preserve the model’s expected spacing.
- **Offline by default:** `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, and `HF_DATASETS_OFFLINE` are set before importing Transformers; inference does not use the network.

---

## Reference benchmark (CPU, FP32)

Example measurements from a CPU-only run (greedy decoding):

| Hardware | Intel Core i7-8665U (4C/8T), 16 GB RAM, no GPU |
|---|---|
| OS / Runtime | Windows 10, Python 3.12, PyTorch 2.15.0+cpu, Transformers 5.22.2 |

| Model | Audio | Duration | Processing | RTF | Peak RSS | CPU use |
|---|---|---:|---:|---:|---:|---:|
| Shook Tiny | sample 1 | 3.91 s | 3.37 s | 0.860 | 572 MB | ~320% |
| Shook Tiny | sample 2 | 10.37 s | 7.42 s | 0.715 | 567 MB | ~358% |
| Shook Tiny | sample 3 | 27.49 s | 8.07 s | 0.293 | 570 MB | ~354% |
| Wav2Vec2 XLSR | sample 1 | 3.91 s | 1.96 s | 0.500 | 1604 MB | ~352% |
| Wav2Vec2 XLSR | sample 2 | 10.37 s | 5.85 s | 0.564 | 1455 MB | ~342% |
| Wav2Vec2 XLSR | sample 3 | 27.49 s | 16.84 s | 0.612 | 1442 MB | ~333% |

- **RTF** = processing time / audio duration (below 1.0 is faster than real time).
- Model load times vary with disk/cache (roughly seconds to tens of seconds). The second model loads on first use and stays in memory; using both in one process can reach ~1.8 GB RSS.
- Your results will vary by CPU, audio, and system load.

To reproduce with your own audio:
```bash
python benchmark.py                  # runs models configured in the script
python benchmark.py --models shook
python benchmark.py --models wav2vec2
```
Results are written to `output/benchmark.json`. The benchmark runner reads clip metadata from `audio/_references.json` (audio files are not included in the repo).

---

## Supported audio formats

| Format | Status |
|---|---|
| WAV | Supported (resampled/mixed to mono as needed) |
| FLAC | Supported |
| MP3 | Supported |
| M4A (AAC), WebM, OGG/Opus, AAC, WMA | Accepted; may require **ffmpeg** depending on container/codec |

If a format fails without ffmpeg installed, install ffmpeg or convert the file to WAV/FLAC/MP3.

---

## Project layout

```text
.
├── app.py                  # Local Flask web server
├── transcribe.py           # Inference engine, model registry, CLI
├── benchmark.py            # Benchmark runner
├── setup.ps1               # Windows one-time setup (venv, deps, models)
├── start-web.ps1           # Windows launcher for the web app
├── requirements.txt
├── models/                 # Model weights (downloaded locally, git-ignored)
│   ├── shook-tiny-stage2/
│   └── wav2vec2-large-xlsr-53-amharic/
├── web/
│   ├── templates/index.html
│   └── static/{style.css,app.js}
├── audio/                  # Optional local samples (git-ignored)
├── output/                 # Transcripts/benchmark output (git-ignored)
├── web_uploads/            # Uploaded audio from UI (git-ignored)
└── history.json            # Local transcription history (git-ignored)
```

---

## Limitations

- **Amharic-focused models:** optimized for Amharic (`am`); other languages are not expected to perform well.
- **Accuracy varies:** clear, close-mic speech works best; names, acronyms, numbers, noise, and overlap can degrade results.
- **Wav2Vec2 spacing:** morpheme-segmented output may include spaces inside words.
- **No timestamps or speaker diarization:** plain text output only.
- **Punctuation/normalization:** limited and model-dependent (numbers may be spoken words).
- **Chunk boundaries:** long audio is chunked with overlap to reduce loss, but boundary artifacts are still possible.
- **CPU performance:** longer recordings take proportionally longer; no streaming output.
- **Memory:** loading both models in one server process can use ~2 GB RAM.

---

## Troubleshooting

| Symptom | What to try |
|---|---|
| Model directory not found / incomplete | Run `setup.ps1` (Windows) or ensure both folders exist under `models/` with `model.safetensors` |
| Amharic text shows as boxes/`?` in terminal | Use a UTF-8-capable terminal (Windows Terminal/PowerShell 7) |
| `.m4a` / `.webm` fails to decode | Install ffmpeg and ensure it’s on `PATH` |
| Microphone blocked in browser | Allow microphone permission for `127.0.0.1`, or use the Upload tab |
| No waveform while recording | Browser may lack `AudioContext` support; recording can still work |
| Port 5000 already in use | Set `ASR_PORT` (e.g. `ASR_PORT=5001`) before starting |
| CUDA-related errors | Ensure the venv has CPU-only PyTorch installed from the CPU index |
| First Wav2Vec2 request is slow | The 1.2 GB model is loading from disk; subsequent requests are faster |
| Out of memory | Close other apps; running both models needs extra headroom (~2 GB) |

---

## License & acknowledgements

- This repository contains application code and UI for local inference. Model weights are downloaded separately and are subject to their respective licenses/terms on Hugging Face.
- Models: [b1n1yam/shook-tiny-amharic-stage2-polish](https://huggingface.co/b1n1yam/shook-tiny-amharic-stage2-polish), [agkphysics/wav2vec2-large-xlsr-53-amharic](https://huggingface.co/agkphysics/wav2vec2-large-xlsr-53-amharic).
- Built with [Transformers](https://github.com/huggingface/transformers), [PyTorch](https://pytorch.org/), [librosa](https://librosa.org/), and [Flask](https://flask.palletsprojects.com/).
