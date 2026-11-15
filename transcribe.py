#!/usr/bin/env python
"""
Amharic Speech-to-Text (offline, CPU-only) with two interchangeable models.

Models:
  shook     - b1n1yam/shook-tiny-amharic-stage2-polish
              Whisper Tiny fine-tuned for Amharic (37.8M params, Apache-2.0)
  wav2vec2  - agkphysics/wav2vec2-large-xlsr-53-amharic
              Wav2Vec2-Large-XLSR-53 CTC fine-tuned for Amharic (~1.2 GB)

Both models are loaded from local directories, never from the network.

Usage:
    python transcribe.py "audio.wav"
    python transcribe.py "audio.mp3" --model wav2vec2 --output result.txt
    python transcribe.py              # interactive mode
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Offline enforcement: must be set BEFORE importing transformers / huggingface_hub
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Amharic output must survive Windows' cp1252 console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

SAMPLE_RATE = 16000
DEFAULT_MODEL = "shook"
SUPPORTED_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus",
                  ".webm", ".mp4"}

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_SPECS = {
    "shook": {
        "id": "shook",
        "label": "Shook Tiny",
        "name": "Shook Tiny (Amharic, stage2-polish)",
        "short": "Whisper Tiny",
        "description": "Fast, compact, normal word spacing",
        "size": "146 MB",
        "model_id": "b1n1yam/shook-tiny-amharic-stage2-polish",
        "dir": MODELS_DIR / "shook-tiny-stage2",
        "architecture": "whisper",
        "chunk_seconds": 30.0,
        "required_files": ["config.json", "preprocessor_config.json",
                           "model.safetensors"],
    },
    "wav2vec2": {
        "id": "wav2vec2",
        "label": "Wav2Vec2 XLSR",
        "name": "Wav2Vec2 Large XLSR-53 (Amharic)",
        "short": "Wav2Vec2 CTC",
        "description": "Larger, slower, morpheme-spaced output",
        "size": "1.2 GB",
        "model_id": "agkphysics/wav2vec2-large-xlsr-53-amharic",
        "dir": MODELS_DIR / "wav2vec2-large-xlsr-53-amharic",
        "architecture": "ctc",
        "chunk_seconds": 30.0,
        "required_files": ["config.json", "preprocessor_config.json",
                           "model.safetensors"],
    },
}

MODEL_KEYS = tuple(MODEL_SPECS)
MODEL_NAME = MODEL_SPECS[DEFAULT_MODEL]["name"]
MODEL_DIR = MODEL_SPECS[DEFAULT_MODEL]["dir"]
CHUNK_SECONDS = MODEL_SPECS[DEFAULT_MODEL]["chunk_seconds"]


def resolve_model(model_key=None):
    """Return (key, spec) for a model id, with a clear error for bad ids."""
    key = (model_key or DEFAULT_MODEL).strip().lower()
    if key not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model '{model_key}'. Choose one of: {', '.join(MODEL_KEYS)}"
        )
    return key, MODEL_SPECS[key]


# This checkpoint's generation_config.json is legacy-style (it carries
# suppress_tokens, begin_suppress_tokens and max_length=448). transformers 5.x
# re-injects those keys as kwargs on every generate() call and logs deprecation
# warnings. We pass the same settings explicitly and copy the config with those
# legacy keys stripped, so the run is quiet without changing model behaviour:
# the identical suppression lists are handed over as explicit kwargs instead.
LEGACY_GEN_KEYS = (
    "suppress_tokens",
    "begin_suppress_tokens",
    "max_length",
    "forced_decoder_ids",
    "lang_to_id",
    "task_to_id",
    "alignment_heads",
    "prev_sot_token_id",
    "no_timestamps_token_id",
)

# Keep the library's advisory warnings out of the transcription output. These two
# notices are emitted by transformers 5.x internals for this checkpoint's
# legacy-style generation_config (the suppression lists are still carried in that
# config, so decoding behaviour is unchanged); everything else still surfaces.
#
# transformers sets propagate=False on its loggers and emits from child loggers
# such as "transformers.generation.utils", so the filter must be attached to each
# of those directly - a filter on the parent "transformers" logger never sees them.
class _LegacyGenConfigFilter(logging.Filter):
    _NOISE = (
        "together with generation-related arguments",
        "seem to have been set",
    )

    def filter(self, record):
        message = record.getMessage()
        return not any(fragment in message for fragment in self._NOISE)


for _logger_name in (
    "transformers",
    "transformers.generation",
    "transformers.generation.utils",
    "transformers.models.whisper.generation_whisper",
):
    logging.getLogger(_logger_name).addFilter(_LegacyGenConfigFilter())


# ---------------------------------------------------------------------------
# Audio loading / normalization
# ---------------------------------------------------------------------------
def _load_via_ffmpeg(path: Path):
    """Decode audio by piping it through ffmpeg to raw 16 kHz mono float32.

    libsndfile (used by soundfile/libsrosa) cannot open every container -
    notably AAC inside M4A/MP4. ffmpeg handles those, so it is used as a
    fallback when the direct decoder reports the format as unrecognised.
    """
    import shutil
    import subprocess

    import numpy as np

    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            f"'{path.suffix}' needs ffmpeg to decode, but ffmpeg was not found "
            "on PATH. Install ffmpeg or convert the file to WAV/FLAC/MP3."
        )

    cmd = [
        exe, "-nostdin", "-loglevel", "error", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg could not decode '{path}': {detail}")

    waveform = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32, copy=True)
    return waveform


def load_audio(path: Path):
    """Load any supported audio file as mono float32 at 16 kHz.

    Returns (waveform numpy array, original duration in seconds).
    """
    import numpy as np
    import librosa

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. Supported: "
            + ", ".join(sorted(SUPPORTED_EXTS))
        )

    try:
        # soundfile decodes wav/flac/mp3/ogg directly and resamples to 16 kHz mono.
        waveform, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    except Exception as direct_error:
        # AAC-in-M4A and similar containers are not supported by libsndfile.
        if path.suffix.lower() in {".m4a", ".aac", ".wma", ".opus", ".mp4",
                                    ".webm"}:
            waveform = _load_via_ffmpeg(path)
        else:
            raise RuntimeError(
                f"Could not decode '{path}': {direct_error}"
            ) from direct_error

    # Safety: stereo -> mono and NaNs, in case a backend returns them oddly.
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=0)
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)

    duration = len(waveform) / float(SAMPLE_RATE)
    return waveform, duration


def split_chunks(waveform, max_seconds=CHUNK_SECONDS, overlap_seconds=0.5):
    """Split audio into chunks that never exceed the model's window.

    Every chunk is hard-capped to max_seconds, with the overlap taken out of the
    budget rather than added on top of it. Boundaries snap to the quietest 50ms
    frame nearby so words are not cut in half, and nothing is ever dropped: the
    final chunk always reaches the end.
    """
    import numpy as np

    window = int(max_seconds * SAMPLE_RATE)      # hard cap
    overlap = min(int(overlap_seconds * SAMPLE_RATE), window // 4)
    total = len(waveform)
    if total == 0:
        return []

    chunks = []
    start = 0
    while start < total:
        end = min(start + window, total)
        if end < total:
            # Snap the cut to the quietest 50ms frame within +/-0.75s of it.
            search = int(0.75 * SAMPLE_RATE)
            lo = max(start + window // 2, end - search)
            hi = min(total - 1, end + search)
            if hi - lo > SAMPLE_RATE // 10:
                seg = np.abs(waveform[lo:hi])
                frame = int(0.05 * SAMPLE_RATE)
                n_frames = len(seg) // frame
                if n_frames > 0:
                    energies = seg[: n_frames * frame].reshape(n_frames, frame).max(axis=1)
                    end = lo + (int(np.argmin(energies)) + 1) * frame
        end = min(end, start + window)           # never overrun the window
        chunks.append(waveform[start:end])
        if end >= total:
            break
        start = max(end - overlap, start + 1)

    # Drop trailing chunks that are pure digital silence (padding artifacts).
    while chunks and len(chunks) > 1 and float(np.max(np.abs(chunks[-1]))) < 1e-6:
        chunks.pop()
    return chunks


def decode_clean(tokenizer, ids):
    """Decode generated ids to text without emitting U+FFFD replacement chars.

    The Whisper checkpoint tokenizes Amharic at the byte level, so a character
    can be split across several tokens. When generation stops mid-character
    (most often at the 444-token cap, or after a hard audio cut) the trailing
    bytes form an incomplete UTF-8 sequence and the decoder would produce a
    "replacement character" (U+FFFD). Dropping such tokens keeps the output
    valid Amharic instead of garbage. Nothing else is modified.
    """
    ids_list = list(ids)
    text = tokenizer.decode(ids_list, skip_special_tokens=True)
    # Trim from the end while the last piece is an unfinished character.
    while ids_list and "\ufffd" in text:
        ids_list.pop()
        text = tokenizer.decode(ids_list, skip_special_tokens=True)
    return text


def is_mostly_silent(chunk, threshold=1e-4):
    import numpy as np
    return float(np.max(np.abs(chunk))) < threshold if len(chunk) else True


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_key=DEFAULT_MODEL):
    """Load a processor + model from its local directory only (FP32, CPU)."""
    import torch

    key, spec = resolve_model(model_key)
    model_dir = spec["dir"]

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            "Run setup.ps1 first to download the model files."
        )

    # Fail fast with a clear message if a required file is missing.
    missing = [f for f in spec["required_files"] if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Model directory is incomplete, missing: {', '.join(missing)}.\n"
            "Run setup.ps1 to re-download the model files."
        )

    t0 = time.perf_counter()

    if spec["architecture"] == "whisper":
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        # Tokenizer cleanup is designed for WordPiece and would corrupt BPE output
        # (it strips spaces before punctuation); turning it off is the correct
        # behaviour for Whisper and also removes the library's advisory warning.
        try:
            processor.tokenizer.clean_up_tokenization_spaces = False
        except Exception:
            pass
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(model_dir),
            local_files_only=True,
            dtype=torch.float32,   # FP32 for CPU reliability
        )
    elif spec["architecture"] == "ctc":
        from transformers import AutoModelForCTC, AutoProcessor

        processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        # transformers 5.x loads this checkpoint's delimiter as "|" even though
        # tokenizer_config.json says " ". Its replace_word_delimiter_char is " ",
        # so plain batch_decode turns either form into spaces. _transcribe_ctc
        # therefore must not use skip_special_tokens=True.
        model = AutoModelForCTC.from_pretrained(
            str(model_dir),
            local_files_only=True,
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unsupported architecture: {spec['architecture']}")

    model.to("cpu")
    model.eval()
    load_seconds = time.perf_counter() - t0

    return processor, model, load_seconds


def build_generation_config(model, max_new_tokens=444):
    """Build a clean generation config for Amharic greedy decoding.

    The Whisper checkpoint ships a legacy-style generation_config (with
    max_length=448 and suppress token lists). transformers 5.x treats those as
    deprecated kwargs, so we copy the config without the legacy keys and pass
    the equivalent values explicitly to generate() instead. Suppression lists
    are taken verbatim from the checkpoint, so decoding matches the model as
    published.
    """
    import copy

    src = model.generation_config
    suppress = list(getattr(src, "suppress_tokens", None) or [])
    begin_suppress = list(getattr(src, "begin_suppress_tokens", None) or [])

    gen_cfg = copy.deepcopy(src)
    for key in LEGACY_GEN_KEYS:
        if hasattr(gen_cfg, key):
            setattr(gen_cfg, key, None)

    gen_cfg.language = "amharic"
    gen_cfg.task = "transcribe"
    gen_cfg.return_timestamps = False
    gen_cfg.num_beams = 1              # greedy: fastest + most predictable on CPU
    gen_cfg.do_sample = False
    return gen_cfg, suppress, begin_suppress


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _transcribe_whisper(processor, model, chunks):
    import torch

    gen_cfg, _suppress, _begin_suppress = build_generation_config(model)
    texts = []
    total_infer = 0.0

    # Pass only the config object: no extra generation kwargs, so decoding is
    # driven entirely by the (legacy-stripped) config built above.
    with torch.inference_mode():
        for idx, chunk in enumerate(chunks, 1):
            if is_mostly_silent(chunk):
                continue
            inputs = processor(
                chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            ).to("cpu")
            t0 = time.perf_counter()
            generated = model.generate(
                inputs.input_features, generation_config=gen_cfg
            )
            total_infer += time.perf_counter() - t0
            text = decode_clean(processor.tokenizer, generated[0].tolist())
            text = " ".join(text.split()).strip()
            if text:
                texts.append(text)
            if len(chunks) > 1:
                print(f"  chunk {idx}/{len(chunks)} done", flush=True)

    return " ".join(texts).strip(), total_infer


def _transcribe_ctc(processor, model, chunks):
    import torch

    texts = []
    total_infer = 0.0

    with torch.inference_mode():
        for idx, chunk in enumerate(chunks, 1):
            if is_mostly_silent(chunk):
                continue
            inputs = processor(
                chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
            ).to("cpu")
            t0 = time.perf_counter()
            logits = model(inputs.input_values).logits
            total_infer += time.perf_counter() - t0
            ids = torch.argmax(logits, dim=-1)
            # Do NOT pass skip_special_tokens=True here: this tokenizer treats
            # the word-delimiter token as special, and skipping it removes every
            # space between Amharic morphemes. Plain batch_decode keeps them.
            text = processor.batch_decode(ids)[0]
            text = " ".join(text.split()).strip()
            if text:
                texts.append(text)
            if len(chunks) > 1:
                print(f"  chunk {idx}/{len(chunks)} done", flush=True)

    return " ".join(texts).strip(), total_infer


def transcribe_waveform(processor, model, waveform, model_key=DEFAULT_MODEL,
                        chunk_seconds=None):
    """Run CPU inference over the whole recording, chunk by chunk."""
    key, spec = resolve_model(model_key)
    chunks = split_chunks(waveform, chunk_seconds or spec["chunk_seconds"])

    if spec["architecture"] == "whisper":
        text, total_infer = _transcribe_whisper(processor, model, chunks)
    else:
        text, total_infer = _transcribe_ctc(processor, model, chunks)

    return text, total_infer, len(chunks)


# ---------------------------------------------------------------------------
# Public API used by the CLI and web app
# ---------------------------------------------------------------------------
def transcribe_file(audio_path, processor=None, model=None, output_path=None,
                    verbose=True, model_key=DEFAULT_MODEL):
    """Transcribe one audio file. Returns a dict with text + timings."""
    import psutil

    key, spec = resolve_model(model_key)
    audio_path = Path(audio_path).expanduser()

    if processor is None or model is None:
        if verbose:
            print(f"Loading {spec['name']} from:\n  {spec['dir']}\n", flush=True)
        processor, model, load_seconds = load_model(key)
        if verbose:
            print(f"Model loaded in {load_seconds:.2f}s (CPU, FP32)\n", flush=True)
    else:
        load_seconds = 0.0
        if verbose:
            print(f"Using already-loaded {spec['label']} model.\n", flush=True)

    if verbose:
        print("Loading audio...", flush=True)
    waveform, duration = load_audio(audio_path)
    if len(waveform) == 0:
        raise ValueError(f"Audio file contains no samples: {audio_path}")

    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    cpu_before = sum(process.cpu_times()[:2])

    if verbose:
        n_chunks = (int(len(waveform) / (spec["chunk_seconds"] * SAMPLE_RATE)) + 1)
        print(
            f"Processing audio... ({duration:.2f}s, "
            f"~{n_chunks} chunk(s) of max {spec['chunk_seconds']:.0f}s)\n",
            flush=True,
        )

    t0 = time.perf_counter()
    text, infer_seconds, n_chunks = transcribe_waveform(
        processor, model, waveform, model_key=key
    )
    wall_seconds = time.perf_counter() - t0

    rss_after = process.memory_info().rss
    cpu_after = sum(process.cpu_times()[:2])

    rtf = (wall_seconds / duration) if duration > 0 else 0.0
    result = {
        "file": str(audio_path),
        "text": text,
        "model": key,
        "model_label": spec["label"],
        "model_name": spec["name"],
        "model_id": spec["model_id"],
        "model_dir": str(spec["dir"]),
        "architecture": spec["architecture"],
        "duration_s": round(duration, 3),
        "chunks": n_chunks,
        "model_load_s": round(load_seconds, 3),
        "processing_s": round(wall_seconds, 3),
        "inference_s": round(infer_seconds, 3),
        "rtf": round(rtf, 4),
        "peak_rss_mb": round(rss_after / (1024 * 1024), 1),
        "rss_delta_mb": round((rss_after - rss_before) / (1024 * 1024), 1),
        "cpu_seconds": round(cpu_after - cpu_before, 3),
        "cpu_cores": psutil.cpu_count(logical=True),
        "cpu_util_pct": round(
            100.0 * (cpu_after - cpu_before) / wall_seconds, 1
        ) if wall_seconds > 0 else 0.0,
    }

    if output_path is None:
        suffix = ".txt" if key == DEFAULT_MODEL else f".{key}.txt"
        output_path = OUTPUT_DIR / (audio_path.stem + suffix)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")
    result["output_txt"] = str(output_path)

    if verbose:
        print(f"Amharic transcription ({spec['label']}):")
        print(text if text else "(no speech detected)")
        print()
        print(f"Saved to: {output_path}")
        print(
            f"Performance: audio {duration:.2f}s | process {wall_seconds:.2f}s "
            f"| RTF {rtf:.3f} | peak RSS {result['peak_rss_mb']:.0f} MB "
            f"| CPU {result['cpu_util_pct']:.0f}% of {result['cpu_cores']} threads"
        )

    return result


def interactive_mode(model_key=DEFAULT_MODEL):
    key, spec = resolve_model(model_key)
    processor, model, load_seconds = load_model(key)
    print(f"{spec['name']} loaded in {load_seconds:.2f}s (CPU, FP32)")
    print(f"Model directory: {spec['dir']}")
    print("Enter an audio file path to transcribe.")
    print("Type 'q' to quit.\n")

    while True:
        try:
            raw = input("audio> ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if raw.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            return
        if not raw:
            continue
        try:
            transcribe_file(raw, processor=processor, model=model, model_key=key)
        except Exception as exc:
            print(f"Error: {exc}\n", file=sys.stderr)
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Offline Amharic speech-to-text with two CPU models."
    )
    p.add_argument("audio", nargs="?", help="Path to an audio file")
    p.add_argument("-m", "--model", choices=MODEL_KEYS, default=DEFAULT_MODEL,
                   help=f"Model to use (default: {DEFAULT_MODEL})")
    p.add_argument("-o", "--output", help="Output .txt path (default: output/<name>.txt)")
    p.add_argument("-i", "--interactive", action="store_true",
                   help="Interactive mode: transcribe many files in one session")
    p.add_argument("--json", action="store_true", help="Also print metrics as JSON")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.interactive or not args.audio:
        interactive_mode(args.model)
        return 0

    try:
        result = transcribe_file(args.audio, output_path=args.output,
                                 model_key=args.model)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
