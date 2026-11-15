# Amharic Speech-to-Text (Shook Tiny) — offline, CPU-only

Local Amharic speech recognition using **[b1n1yam/shook-tiny-amharic-stage2-polish](https://huggingface.co/b1n1yam/shook-tiny-amharic-stage2-polish)**.

Give it an Amharic audio recording, get Amharic text. Everything runs on your CPU, with no internet connection, no GPU, no CUDA and no cloud API.

---

## Model

| | |
|---|---|
| Repository | `b1n1yam/shook-tiny-amharic-stage2-polish` |
| Architecture | Whisper Tiny, fine-tuned for Amharic |
| Parameters | 37.8 million (4 encoder / 4 decoder layers, d_model 384) |
| Format | Safetensors, FP32 |
| License | Apache 2.0 |
| Reported eval WER | 20.7634 % (as published by the model author) |
| Sample rate | 16 kHz |
| Window | 30 s per chunk (480 000 samples) |

This is the exact checkpoint named in the task. No other model is substituted.

---

## Quick start

Already set up on this machine — just run:

```powershell
cd "C:\Users\JO\Documents\Opencode Desktop\STT Amharic Shook Tiny"
.\.venv\Scripts\python.exe transcribe.py "audio\your_file.wav"
```

Or activate the environment first, then use plain `python`:

```powershell
.\.venv\Scripts\Activate.ps1
python transcribe.py "audio\your_file.wav"
```

### Interactive mode

Keeps the model in memory so you can transcribe file after file without restarting:

```powershell
python transcribe.py
```

```text
audio> audio\meeting.m4a
...
audio> audio\interview.flac
...
audio> q
```

### Options

| Flag | Meaning |
|---|---|
| `-o, --output PATH` | Write the transcript somewhere other than `output\<name>.txt` |
| `-i, --interactive` | Force interactive mode |
| `--json` | Also print the metrics as JSON |

---

## Web interface (recommended)

Prefer a page over the terminal? Start the local web UI:

```powershell
powershell -ExecutionPolicy Bypass -File .\start-web.ps1
```

or directly:

```powershell
.\.venv\Scripts\python.exe app.py
```

It loads the model once, prints a link and opens your browser automatically:

```text
  >>>  Open this in your browser:  http://127.0.0.1:5000
```

### What you get

**New recording tab** — press the big microphone button, speak Amharic, press it again to
stop. Listen back, then press **Send**. The transcription appears in a large readable
Amharic card with copy-to-clipboard and quick stats.

No microphone? Drop an audio file onto the box at the bottom, or click to browse.

**History tab** — every past transcription, newest first, each with its text, timestamp,
duration, speed, and its own audio player, so you can compare what the model heard. Delete
entries one at a time or clear the whole list.

### How it works

* Recording uses the browser's `MediaRecorder`, which produces WebM/Opus audio; the server
  decodes it through ffmpeg before transcription.
* The model is loaded once on startup and reused, so each Send costs only inference time
  (about 1–2 s for a short clip).
* The page binds to `127.0.0.1` only — nothing on your network can reach it.
* History and uploaded audio are stored locally: `history.json` and `web_uploads/`.
  Both are git-ignored. The Amharic text is also written to `output/<id>.txt`.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `ASR_PORT` | `5000` | Port for the local page |
| `ASR_NO_BROWSER` | unset | `1` stops it opening the browser automatically |

---

## Command line

```text
Loading Shook Tiny (Amharic, stage2-polish) from:
  C:\...\models\shook-tiny-stage2

Model loaded in 0.88s (CPU, FP32)

Loading audio...
Processing audio... (3.91s, ~1 chunk(s) of max 30s)

Amharic transcription:
መጠን ጠፋጭ በሆኑ መጠጦችና ምግቡ ጾምን ከመፍታት መቆጠብ

Saved to: C:\...\output\sample1.txt
Performance: audio 3.91s | process 4.34s | RTF 1.108 | peak RSS 564 MB | CPU 364% of 8 threads
```

A transcript is always written to `output\<audio name>.txt` as **UTF-8** so the Amharic
(Ethiopic) characters stay intact. `RTF` is processing time divided by audio duration;
below 1.0 means faster than playback.

---

## Installation (fresh machine)

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

`setup.ps1` is safe to re-run and will:

1. Create `.venv` (Python 3.10+ required).
2. Install **CPU-only** PyTorch from `https://download.pytorch.org/whl/cpu` (no CUDA wheels).
3. Install the pinned dependencies.
4. Download the model **anonymously** (no token, no login) into `models\shook-tiny-stage2\`,
   skipping the download if the weights are already there.
5. Verify the model loads with the network disabled.

Manual equivalent:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0+cpu"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Audio support

| Format | Status |
|---|---|
| WAV | works (also 44.1 kHz / stereo — resampled and mixed to mono) |
| FLAC | works |
| MP3 | works |
| M4A (AAC) | works via ffmpeg |
| OGG / OPUS / AAC / WMA | accepted; needs ffmpeg for the container formats libsndfile cannot open |

Handling, in `load_audio()`:

* Everything is decoded to **mono, 32-bit float, 16 kHz** — the rate the model expects.
* Stereo is averaged to mono; NaNs/infs are cleaned.
* Formats libsndfile cannot decode (notably AAC-in-M4A) are piped through **ffmpeg**
  automatically. If ffmpeg is missing, you get a clear message telling you to install it
  or convert to WAV/FLAC/MP3.

### Long recordings are never truncated

Whisper only "sees" 30 seconds at a time, and its feature extractor silently cuts anything
longer. `split_chunks()` therefore:

* hard-caps every chunk at exactly 30 s (480 000 samples);
* moves each cut to the quietest 50 ms frame within ±0.75 s of the boundary, so words are
  not sliced in half;
* carries a 0.5 s overlap into the next chunk, taken out of the budget so the window is
  still never exceeded;
* always keeps the final chunk, so the end of the file is transcribed.

Verified: a 113.6 s file becomes 4 chunks, is fully covered, and its last words appear in
the output.

---

## Offline operation

No token, login, API key or paid service is used anywhere — the repository is public and
`gated: false`, so `setup.ps1` downloads it anonymously with `token=False`.

`transcribe.py` sets the offline switches **before** importing transformers, and loads the
model from disk only:

```python
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
...
AutoProcessor.from_pretrained(model_dir, local_files_only=True)
AutoModelForSpeechSeq2Seq.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32)
```

Inference never touches the network. This was tested by pointing the Hub endpoint at an
unreachable address (`HF_ENDPOINT=http://127.0.0.1:9`) with both offline flags set —
transcription still succeeded.

**Model location:** `models\shook-tiny-stage2\` — 146 MB total, 11 files
(`model.safetensors` 144 MB, plus `config.json`, `preprocessor_config.json`,
`generation_config.json`, `tokenizer_config.json`, `special_tokens_map.json`,
`added_tokens.json`, `vocab.json`, `merges.txt`, `normalizer.json`).

---

## Measured performance

Measured on this machine — **Intel Core i7-8665U (4 cores / 8 threads), 16 GB RAM, no GPU**,
Windows 10, Python 3.12.10, PyTorch 2.15.0+cpu, FP32, greedy decoding.

Model load time: **0.68 – 1.22 s** (warm page cache).

| Audio | Duration | Processing | RTF | Peak RSS | CPU use |
|---|---|---|---|---|---|
| `sample1.flac` | 3.91 s | 4.34 s | 1.108 | 564 MB | 364 % (≈3.6 of 8 threads) |
| `sample2.flac` | 10.37 s | 10.48 s | 1.010 | 569 MB | 374 % |
| `sample3.flac` | 27.49 s | 11.15 s | **0.406** | 570 MB | 366 % |
| `long_test.wav` | 37.87 s | 25.08 s | **0.662** | 566 MB | 376 % |
| `long_113s.wav` | 113.60 s | 63.91 s | **0.563** | 578 MB | 371 % |

Short files have a fixed per-call overhead, so their RTF sits around or slightly above 1.
From ~30 s upward it drops to ≈0.4–0.7 — comfortably faster than real time. Peak memory is
flat at ~570 MB, so long files do not grow memory use.

Reproduce with `python benchmark.py` (writes `output\benchmark.json`).

---

## Sample transcription

Reference (ground truth from the source dataset) versus what this model produced:

**sample1 (3.91 s)**
```text
REF: በጣም ጣፋጭ በሆኑ መጠጦችና ምግቦች ጾምን ከመፍታት መቆጠብ፤
HYP: መጠን ጠፋጭ በሆኑ መጠጦችና ምግቡ ጾምን ከመፍጠት መቆጠብ
```

**sample2 (10.37 s)**
```text
REF: የኢኖቬሽንና ቴክኖሎጂ ሚኒስትር በለጠ ሞላ (ዶ/ር) እንደገለጹት፥ በዲጅታል ኢትዮጵያ 2025 ስትራቴጂ ተጨባጭ ለውጥ ከመጣባቸው ዘርፎች ትራንስፖርትና ሎጂስቲክስ አንዱ ነው።
HYP: ንቤሽና ቲክኖቨጂ ሚስቴር ዶክተር ምለጠሙላ እንደገለጹ ወዲህ ስተኔት ወዲህ ኢትዮጵያ ምስጢራቴጂ ተጨሻጭ ለውጥህ ከመጣባቸው ዘርፎ ይትራንስፎርትና ሉቪስ ቲክስ አንዱ ነው
```

Common Amharic words come through correctly; the errors cluster on proper nouns,
loanwords and numbers — expected for a 37.8M-parameter model.

---

## Project layout

```text
STT Amharic Shook Tiny/
├── models/
│   └── shook-tiny-stage2/       # downloaded weights + tokenizer (git-ignored)
├── audio/                       # your recordings / test clips
├── output/                      # transcripts (.txt, UTF-8) + benchmark.json
├── web/
│   ├── templates/index.html     # the page
│   └── static/                  # style.css, app.js
├── web_uploads/                 # audio sent from the browser (git-ignored)
├── history.json                 # transcription history (git-ignored)
├── transcribe.py                # the ASR engine (used by both CLI and web)
├── app.py                       # local web server
├── benchmark.py                 # reproduces the performance table
├── start-web.ps1                # launches the web page
├── requirements.txt
├── setup.ps1
├── README.md
└── .gitignore
```

Model weights are in `.gitignore` and are never committed.

---

## Limitations

* **Amharic only.** The checkpoint is fine-tuned for `am`; other languages will not work well.
* **Tiny model accuracy.** 37.8M parameters. Clear, close-mic speech transcribes well;
  names, acronyms, numbers and noisy or overlapping speech degrade noticeably. Numbers are
  often rendered as spoken words (e.g. `179` → `አንድ መቶ ሰባ ዘጠኝ`).
* **No timestamps or speaker labels.** Plain text only.
* **Long-file seam quality.** Chunks are cut at low-energy points and overlap by 0.5 s, but a
  word landing exactly on a boundary can still be affected. Audio is never dropped.
* **CPU speed.** Roughly 0.4–0.7× real time for longer files. Multi-threaded (≈4 threads
  used). Very long recordings take correspondingly long, with no streaming output.
* **FFmpeg for M4A.** AAC-in-M4A needs `ffmpeg` on `PATH`; WAV/FLAC/MP3 work without it.
* **Numbers written in words.** No digit normalisation or inverse text normalisation.
* **Punctuation** is only what the model emits — it is sparse and inconsistent.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Model directory not found` / incomplete | Run `setup.ps1` |
| Amharic shows as `?` or boxes in the terminal | Use Windows Terminal or PowerShell; the script already forces UTF-8 output |
| `.m4a` fails with “Format not recognised” | Install ffmpeg and make sure it is on `PATH` |
| Microphone blocked in the browser | Allow mic permission for `127.0.0.1` in the address bar, or use the file upload box |
| Port 5000 already in use | Start with another port: `$env:ASR_PORT=5001; .\start-web.ps1` |
| `torch.cuda` errors | The venv must hold the CPU build: reinstall with the `--index-url https://download.pytorch.org/whl/cpu` flag |
| Out of memory | Not expected (≈570 MB); close other heavy apps |
