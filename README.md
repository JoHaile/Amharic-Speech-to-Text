# Amharic Speech-to-Text — two offline models, one web app

Local Amharic speech recognition with **two interchangeable models**, a live
microphone waveform, and a history of every transcription.

Everything runs on this computer: **CPU only, no GPU, no CUDA, no cloud API, no
internet connection during inference**.

---

## Models

| Tab | Hugging Face model | Architecture | Parameters | Local size |
|---|---|---|---|---|
| **Shook Tiny** | [`b1n1yam/shook-tiny-amharic-stage2-polish`](https://huggingface.co/b1n1yam/shook-tiny-amharic-stage2-polish) | Whisper Tiny fine-tuned for Amharic | 37.8M | 146 MB |
| **Wav2Vec2 XLSR** | [`agkphysics/wav2vec2-large-xlsr-53-amharic`](https://huggingface.co/agkphysics/wav2vec2-large-xlsr-53-amharic) | Wav2Vec2-Large-XLSR-53 CTC | 315.7M | 1.2 GB |

Both are public, Apache-2.0-compatible checkpoints downloaded anonymously — no
Hugging Face token, login, API key or paid service is used.

**Shook Tiny** is the fast, compact default. It produces normally spaced Amharic
text. The model card reports **20.7634 % WER** on its evaluation set.

**Wav2Vec2 XLSR** is the larger alternative. Its model card reports **26 % WER /
7 % CER** on the ALFFA validation set. That model was trained on
morpheme-segmented transcripts, so its output contains spaces inside words
(for example `ማእከሎ ችን` instead of `ማእከሎችን`). This is expected behaviour, not a
decoding bug.

The two models are different checkpoints, not different copies of the same
model. Choose the tab that gives the better result for your recording.

---

## Quick start

The virtual environment and both models are already set up in this folder.

### Web interface (recommended)

```powershell
cd "C:\Users\JO\Documents\Opencode Desktop\STT Amharic Shook Tiny"
powershell -ExecutionPolicy Bypass -File .\start-web.ps1
```

or directly:

```powershell
.\.venv\Scripts\python.exe app.py
```

It preloads Shook Tiny, prints a link and opens your browser automatically:

```text
>>>  Open this in your browser:  http://127.0.0.1:5000
```

Wav2Vec2 XLSR is loaded the first time you select its tab or send audio with it.

### Command line

```powershell
# Default model (Shook Tiny)
.\.venv\Scripts\python.exe transcribe.py "audio\sample1.flac"

# The larger Wav2Vec2 model
.\.venv\Scripts\python.exe transcribe.py "audio\sample1.flac" --model wav2vec2
```

Interactive mode keeps the chosen model in memory:

```powershell
.\.venv\Scripts\python.exe transcribe.py --model wav2vec2
```

```text
audio> audio\meeting.m4a
...
audio> q
```

### CLI options

| Flag | Meaning |
|---|---|
| `-m, --model {shook,wav2vec2}` | Model to use (default: `shook`) |
| `-o, --output PATH` | Write the transcript somewhere other than `output\<name>.txt` |
| `-i, --interactive` | Force interactive mode |
| `--json` | Also print the metrics as JSON |

Default output names:

* Shook Tiny: `output\<audio name>.txt`
* Wav2Vec2: `output\<audio name>.wav2vec2.txt`

Both are written as **UTF-8**, so Amharic (Ethiopic) characters stay intact.

---

## The web page

### Model tabs

A pill-tab selector at the top of the recorder switches between the two models.
Each tab has its own status dot:

* grey — not loaded yet; it loads when selected
* amber — loading
* green — ready
* red — unavailable (the tooltip shows the error)

The selected model is sent with the audio request, so the transcript and the
history entry always say which model produced them.

### Live microphone waveform

While recording, the page draws a live waveform using the browser's Web Audio
API:

* the mic stream goes through an `AnalyserNode`
* the canvas samples the time-domain signal about 60 times per second
* quiet speech gives short bars; speaking close to the mic gives tall bars
* the bars decay smoothly instead of flickering

The waveform is a visual indicator only. It is not added to the recorded audio
and does not affect transcription. If the browser cannot create an
`AudioContext`, recording still works normally without the visualization.

### Recording preview and clip state

When you stop recording, the clip stays on screen in a compact player above
**Send**:

* the browser capture is decoded and rewritten to a small 16-bit WAV copy, so the
  player always has a real duration and plays even where the browser rejects
  Opus/WebM;
* play, pause, seek and the time display are drawn by the app itself, matching
  the rest of the page — History uses the same player for every past clip;
* the clip, its playback position and the result card are **kept when you switch
  model tabs**. Switching only changes the model shown on the badge and in
  *“Press Send to transcribe with …”*;
* the clip is also **kept after a transcription finishes**, so if the text is not
  good you can switch to the other model and press **Send** again with the same
  recording — each send adds its own History entry;
* **Record again** starts a fresh recording; **Clear** drops the clip and its
  result and returns the panel to *Ready to record*;
* if a file format truly cannot be previewed (some rare uploads), the player says
  *Preview unavailable* while **Send** still works, because the server decodes
  with ffmpeg.

### Everything else

* **New recording** tab — press the microphone, speak Amharic, press it again
  to stop. Listen back, then press **Send**.
* **File upload** — drop a WAV/MP3/M4A/FLAC/OGG/WEBM file on the box, or browse.
* **Result card** — the Amharic text, the model badge, audio duration, processing
  time, RTF and peak RAM, plus copy-to-clipboard.
* **History** tab — every past transcription, newest first, with its model badge,
  audio player and its own delete button. Filter with **All / Shook / Wav2Vec2**.
* **Clear all** removes every history entry and its uploaded audio.

The page binds to `127.0.0.1` only, so nothing on your network can reach it.
History and uploaded audio stay local in `history.json` and `web_uploads/`; both
are git-ignored.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `ASR_PORT` | `5000` | Port for the local page |
| `ASR_NO_BROWSER` | unset | `1` stops it opening the browser automatically |

---

## How it works

### One engine, two models

`transcribe.py` contains a small registry:

* `MODEL_SPECS` describes each model: label, local folder, architecture,
  chunk length and required files.
* `load_model(model_key)` loads the matching Transformers class:
  * `AutoModelForSpeechSeq2Seq` for Shook Tiny (Whisper)
  * `AutoModelForCTC` for Wav2Vec2 XLSR
* `transcribe_file(..., model_key=...)` returns the text plus timing and memory
  metrics, and records which model produced it.

Both models use the same audio pipeline:

* decode to **mono, 32-bit float, 16 kHz**
* average stereo to mono
* clean NaNs/infs
* fall back to **ffmpeg** for containers libsndfile cannot open (notably
  AAC-in-M4A)
* split long audio into 30-second chunks, moving each cut to the quietest 50 ms
  frame within ±0.75 s of the boundary and carrying a 0.5 s overlap
* never silently truncate the end of a recording

### Wav2Vec2 decoding detail

The Wav2Vec2 tokenizer treats its word-delimiter token as a special token.
Passing `skip_special_tokens=True` removes every space between Amharic
morphemes. The engine therefore uses a plain `batch_decode()` for CTC output and
keeps the model's natural spacing.

### Offline operation

`transcribe.py` and `app.py` set the offline switches **before** importing
Transformers:

```python
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
```

Models are loaded with `local_files_only=True` from:

```text
models\shook-tiny-stage2\
models\wav2vec2-large-xlsr-53-amharic\
```

Neither folder is committed to Git (`.gitignore` excludes `models/`). Setup
downloads them anonymously; inference never touches the network.

---

## Measured performance

Measured on this machine — **Intel Core i7-8665U (4 cores / 8 threads), 16 GB RAM,
no GPU**, Windows 10, Python 3.12.10, PyTorch 2.15.0+cpu, Transformers 5.22.2,
FP32, greedy decoding.

| Model | Audio | Duration | Processing | RTF | Peak RSS | CPU use |
|---|---|---:|---:|---:|---:|---:|
| Shook Tiny | `sample1.flac` | 3.91 s | 3.37 s | 0.860 | 572 MB | 320 % |
| Shook Tiny | `sample2.flac` | 10.37 s | 7.42 s | 0.715 | 567 MB | 358 % |
| Shook Tiny | `sample3.flac` | 27.49 s | 8.07 s | 0.293 | 570 MB | 354 % |
| Wav2Vec2 XLSR | `sample1.flac` | 3.91 s | 1.96 s | 0.500 | 1604 MB | 352 % |
| Wav2Vec2 XLSR | `sample2.flac` | 10.37 s | 5.85 s | 0.564 | 1455 MB | 342 % |
| Wav2Vec2 XLSR | `sample3.flac` | 27.49 s | 16.84 s | 0.612 | 1442 MB | 333 % |

`RTF` is processing time divided by audio duration; below `1.0` means faster
than playback. Short clips have a fixed per-call overhead, so their RTF sits
higher. Longer files benefit from chunking.

A separate 37.87 s Wav2Vec2 run produced **RTF 0.418** and **1612 MB** peak RSS
over two chunks.

Model load time depends on disk cache: roughly **1–10 s** for Shook Tiny and
**3–20 s** for Wav2Vec2 XLSR. The default model is loaded at server startup; the
second one loads on first use and is kept in memory for the session. With both
models loaded the process reached about **1.8 GB RSS**.

Reproduce with:

```powershell
.\.venv\Scripts\python.exe benchmark.py
.\.venv\Scripts\python.exe benchmark.py --models shook
.\.venv\Scripts\python.exe benchmark.py --models wav2vec2
```

Results are written to `output\benchmark.json`. The runner reads the clip list in
`audio\_references.json`; the clips themselves are not committed, so put your own
files there (or edit the JSON) before running it.

---

## Sample output

**sample1 — 3.91 s**

```text
REF: በጣም ጣፋጭ በሆኑ መጠጦችና ምግቦች ጾምን ከመፍታት መቆጠብ፤
Shook Tiny: መጠን ጠፋጭ በሆኑ መጠጦችና ምግቡ ጾምን ከመፍጠት መቆጠብ
Wav2Vec2:   በጣም ጣፏጭ በሆኑ መጠጦ ችና ምግብዎች ጾም ን ከ መፍታት መት ቆደብ
```

**sample2 — 10.37 s**

```text
REF: የኢኖቬሽንና ቴክኖሎጂ ሚኒስትር በለጠ ሞላ (ዶ/ር) እንደገለጹት፥ በዲጅታል ኢትዮጵያ 2025 ስትራቴጂ
     ተጨባጭ ለውጥ ከመጣባቸው ዘርፎች ትራንስፖርትና ሎጂስቲክስ አንዱ ነው።
Shook Tiny: ንቤሽና ቲክኖቨጂ ሚስቴር ዶክተር ምለጠሙላ እንደገለጹ ወዲህ ስተኔት ወዲህ ኢትዮጵያ
     ምስጢራቴጂ ተጨሻጭ ለውጥህ ከመጣባቸው ዘርፎ ይትራንስፎርትና ሉቪስ ቲክስ አንዱ ነው
Wav2Vec2:   ኑቬሽን ና ቴክኖሎጂ ሚኒስቴር ዶክተር በለጠ ሙላ እንደ ገለጹት በ ዲሽተር የ ኢትዮጵያ
     ሀያ አምስት ትራቴጂ ተጨፋ ጭ ለውጥ ከ መጣ ባቸው ዘርፎች ትራንስፖርት ና ኦሎጂ ስቲክስ አንዱ ነው
```

Common Amharic words come through; errors cluster on proper nouns, loanwords and
numbers. That is expected for models of this size.

---

## Audio support

| Format | Status |
|---|---|
| WAV | works, including 44.1 kHz / stereo (resampled and mixed to mono) |
| FLAC | works |
| MP3 | works |
| M4A (AAC) | works via ffmpeg |
| OGG / OPUS / AAC / WMA | accepted; some containers need ffmpeg |

`librosa`/`soundfile` decode WAV, FLAC, MP3 and OGG directly. AAC-in-M4A and
similar containers are piped through **ffmpeg** automatically. If ffmpeg is
missing, the error explains how to install it or convert the file.

Long recordings are never truncated. Whisper's 30-second window is enforced per
chunk; the Wav2Vec2 model uses the same chunker. Boundaries snap to quiet
frames, overlap by 0.5 s within the budget, and the final chunk always reaches
the end of the file.

---

## Installation on a fresh machine

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

`setup.ps1` is safe to re-run and will:

1. Create `.venv` (Python 3.10+ required).
2. Install **CPU-only** PyTorch from `https://download.pytorch.org/whl/cpu`.
3. Install the pinned dependencies (Transformers, librosa, Flask, etc.).
4. Download **both** models anonymously into `models\`, skipping files that
   already exist.
5. Verify both models load with the network disabled.

Manual equivalent:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0+cpu"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then download both model folders into `models\` as described above.

---

## Project layout

```text
Amharic-Speech-to-Text/
├── models/
│   ├── shook-tiny-stage2/                  # Whisper Tiny, 146 MB (git-ignored)
│   └── wav2vec2-large-xlsr-53-amharic/     # Wav2Vec2, 1.2 GB (git-ignored)
├── audio/                                  # sample recordings (git-ignored)
├── output/                                 # transcripts + benchmark.json (git-ignored)
├── web/
│   ├── templates/index.html                # model tabs, recorder, history
│   └── static/
│       ├── style.css                       # design system + waveform styles
│       └── app.js                          # recording, live wave, tabs, history
├── web_uploads/                            # browser uploads (git-ignored)
├── history.json                            # transcription history (git-ignored)
├── transcribe.py                           # engine + model registry + CLI
├── app.py                                  # local Flask web server
├── benchmark.py                            # both-model benchmark runner
├── setup.ps1                               # one-time setup
├── start-web.ps1                           # starts the web page
├── requirements.txt
├── README.md
└── .gitignore
```

Model weights are never committed.

---

## Limitations

* **Amharic only.** Both checkpoints are fine-tuned for `am`; other languages
  will not work well.
* **Tiny/small-model accuracy.** Clear, close-mic speech works best. Names,
  acronyms, numbers and noisy or overlapping speech degrade noticeably.
* **Wav2Vec2 spacing.** Its output is morpheme-segmented and contains extra
  spaces inside words. That is the model's training convention, not a decoder
  bug.
* **No timestamps or speaker labels.** Plain text only.
* **No digit normalization.** Numbers are usually rendered as spoken words
  (for example `179` → `አንድ መቶ ሰባ ዘጠኝ`).
* **Punctuation** is only what the model emits — sparse and inconsistent.
* **Long-file seams.** Chunks are cut at low-energy points and overlap by 0.5 s,
  but a word landing exactly on a boundary can still be affected. Audio is
  never dropped.
* **CPU speed.** Roughly 0.3–0.9× real time for longer files. Very long
  recordings take correspondingly long, with no streaming output.
* **Memory when both models are loaded.** About 1.8 GB RSS after both have been
  used in the same server process.
* **FFmpeg for M4A.** AAC-in-M4A needs `ffmpeg` on `PATH`; WAV/FLAC/MP3 work
  without it.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Model directory not found` / incomplete | Run `setup.ps1` |
| Amharic shows as `?` or boxes in the terminal | Use Windows Terminal or PowerShell; the scripts force UTF-8 output |
| `.m4a` fails with “Format not recognised” | Install ffmpeg and make sure it is on `PATH` |
| Microphone blocked in the browser | Allow mic permission for `127.0.0.1` in the address bar, or use the file upload box |
| No waveform while recording | The browser may not support `AudioContext`; recording still works. Check the browser console. |
| Port 5000 already in use | Start with another port: `$env:ASR_PORT=5001; .\start-web.ps1` |
| `torch.cuda` errors | The venv must hold the CPU build: reinstall with the `--index-url https://download.pytorch.org/whl/cpu` flag |
| Wav2Vec2 tab takes a long time the first time | It is a 1.2 GB model loading from disk; later requests are fast |
| Out of memory | Close other heavy apps. Using both models together needs about 2 GB free |
