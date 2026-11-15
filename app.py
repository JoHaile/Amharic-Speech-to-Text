#!/usr/bin/env python
"""
Local web UI for the two offline Amharic speech-to-text models.

Run it and open the link it prints:

    .\\.venv\\Scripts\\python.exe app.py

What you get in the browser
  * choose Shook Tiny or Wav2Vec2 XLSR with the model tabs
  * record audio with a live microphone waveform, or upload a file
  * press Send to transcribe on the local CPU
  * a History tab with every past transcription, its text, and playable audio

Everything runs on this machine. The server binds to 127.0.0.1 only, so it is
not reachable from other machines on your network.
"""

import json
import os
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path

# Offline enforcement must happen BEFORE transformers / huggingface_hub import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from flask import (Flask, abort, jsonify, render_template, request,
                   send_from_directory)

import transcribe as asr

PROJECT_ROOT = Path(__file__).resolve().parent
WEB_DIR = PROJECT_ROOT / "web"
UPLOAD_DIR = PROJECT_ROOT / "web_uploads"
HISTORY_FILE = PROJECT_ROOT / "history.json"

UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(WEB_DIR / "templates"),
    static_folder=str(WEB_DIR / "static"),
)
# Long recordings are allowed; the browser sends short clips normally.
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# ---------------------------------------------------------------------------
# Models: each one loads lazily, behind its own lock. CPU inference is the
# bottleneck, so serialising each load keeps things predictable. The default
# model is preloaded in main(); the second model loads when its tab is used.
# ---------------------------------------------------------------------------
_states = {
    key: {"processor": None, "model": None, "load_seconds": None, "error": None}
    for key in asr.MODEL_KEYS
}
_locks = {key: threading.Lock() for key in asr.MODEL_KEYS}


def ensure_model(model_key):
    model_key, _spec = asr.resolve_model(model_key)
    state = _states[model_key]
    if state["model"] is not None:
        return state
    with _locks[model_key]:
        if state["model"] is not None:
            return state
        if state["error"] is not None:
            raise RuntimeError(state["error"])
        try:
            processor, model, load_seconds = asr.load_model(model_key)
        except Exception as exc:
            state["error"] = str(exc)
            raise
        state.update(processor=processor, model=model, load_seconds=load_seconds)
        return state


def model_status(model_key):
    model_key, spec = asr.resolve_model(model_key)
    state = _states[model_key]
    return {
        "id": model_key,
        "label": spec["label"],
        "name": spec["name"],
        "short": spec["short"],
        "description": spec["description"],
        "size": spec["size"],
        "model_id": spec["model_id"],
        "architecture": spec["architecture"],
        "model_dir": str(spec["dir"]),
        "ready": state["model"] is not None,
        "load_seconds": state["load_seconds"],
        "error": state["error"],
    }


# ---------------------------------------------------------------------------
# History: a plain JSON list in the project root, newest entry first.
# ---------------------------------------------------------------------------
_history_lock = threading.Lock()


def _normalise_item(item):
    """Give older history entries a model id so the UI can label them."""
    if not isinstance(item, dict):
        return item
    if item.get("model") not in asr.MODEL_KEYS:
        item["model"] = asr.DEFAULT_MODEL
    spec = asr.MODEL_SPECS[item["model"]]
    item.setdefault("model_label", spec["label"])
    return item


def load_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return [_normalise_item(item) for item in data] if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_history(items):
    tmp = HISTORY_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(HISTORY_FILE)


def add_history_entry(entry):
    with _history_lock:
        items = load_history()
        items.insert(0, entry)
        save_history(items)
    return entry


def _delete_audio(name):
    if not name:
        return
    try:
        (UPLOAD_DIR / name).unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.route("/api/status")
def api_status():
    return jsonify({
        "default_model": asr.DEFAULT_MODEL,
        "models": [model_status(key) for key in asr.MODEL_KEYS],
        "sample_rate": asr.SAMPLE_RATE,
    })


@app.route("/api/models/<model_key>/load", methods=["POST"])
def api_model_load(model_key):
    try:
        model_key, _spec = asr.resolve_model(model_key)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        ensure_model(model_key)
    except Exception as exc:
        return jsonify({"error": f"Could not load the model: {exc}"}), 500
    return jsonify(model_status(model_key))


@app.route("/api/history")
def api_history():
    return jsonify(load_history())


@app.route("/api/history/<entry_id>/audio")
def api_history_audio(entry_id):
    for item in load_history():
        if item.get("id") == entry_id:
            name = item.get("audio_file")
            if not name:
                abort(404)
            return send_from_directory(UPLOAD_DIR, name, as_attachment=False)
    abort(404)


@app.route("/api/history/<entry_id>", methods=["DELETE"])
def api_history_delete(entry_id):
    with _history_lock:
        items = load_history()
        kept, removed = [], None
        for item in items:
            if item.get("id") == entry_id:
                removed = item
            else:
                kept.append(item)
        if removed is None:
            abort(404)
        save_history(kept)
    _delete_audio(removed.get("audio_file"))
    return jsonify({"ok": True})


@app.route("/api/history", methods=["DELETE"])
def api_history_clear():
    with _history_lock:
        items = load_history()
        save_history([])
    for item in items:
        _delete_audio(item.get("audio_file"))
    return jsonify({"ok": True})


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    """Transcribe an uploaded clip with the selected model."""
    model_key = request.form.get("model", asr.DEFAULT_MODEL)
    try:
        model_key, spec = asr.resolve_model(model_key)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    upload = request.files.get("audio")
    if upload is None or upload.filename == "":
        return jsonify({"error": "No audio was received."}), 400

    suffix = Path(upload.filename).suffix.lower() or ".webm"
    if suffix not in asr.SUPPORTED_EXTS:
        return jsonify({"error": f"Unsupported audio format '{suffix}'."}), 400

    entry_id = uuid.uuid4().hex[:12]
    stored_name = f"{entry_id}{suffix}"
    stored_path = UPLOAD_DIR / stored_name
    upload.save(stored_path)

    if stored_path.stat().st_size == 0:
        stored_path.unlink(missing_ok=True)
        return jsonify({"error": "The recording was empty. Please try again."}), 400

    try:
        state = ensure_model(model_key)
        result = asr.transcribe_file(
            stored_path,
            processor=state["processor"],
            model=state["model"],
            output_path=asr.OUTPUT_DIR / f"{entry_id}.txt",
            verbose=False,
            model_key=model_key,
        )
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        return jsonify({"error": f"Transcription failed: {exc}"}), 500

    entry = {
        "id": entry_id,
        "text": result["text"],
        "model": model_key,
        "model_label": spec["label"],
        "audio_file": stored_name,
        "original_name": upload.filename,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_s": result["duration_s"],
        "processing_s": result["processing_s"],
        "rtf": result["rtf"],
        "peak_rss_mb": result["peak_rss_mb"],
        "cpu_util_pct": result["cpu_util_pct"],
        "source": request.form.get("source", "recording"),
    }
    add_history_entry(entry)
    return jsonify(entry)


@app.errorhandler(413)
def too_large(_exc):
    return jsonify({"error": "That audio file is too large (200 MB limit)."}), 413


@app.errorhandler(404)
def not_found(_exc):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found."}), 404
    return "Not found", 404


def main():
    try:
        port = int(os.environ.get("ASR_PORT", "5000"))
    except ValueError:
        port = 5000
    url = f"http://127.0.0.1:{port}"

    default_spec = asr.MODEL_SPECS[asr.DEFAULT_MODEL]
    other_keys = [key for key in asr.MODEL_KEYS if key != asr.DEFAULT_MODEL]

    print("=" * 64)
    print("  Amharic Speech-to-Text  --  local web interface")
    print("=" * 64)
    print(f"  Default model : {default_spec['name']}")
    print(f"  Model folder  : {asr.MODELS_DIR}")
    for key in other_keys:
        spec = asr.MODEL_SPECS[key]
        print(f"  Also available: {spec['label']} (loads when selected)")
    print()
    print("  Loading the default model (first load takes a moment)...", flush=True)

    try:
        state = ensure_model(asr.DEFAULT_MODEL)
        print(f"  Ready in {state['load_seconds']:.2f}s  (CPU, FP32)", flush=True)
    except Exception as exc:
        print(f"  WARNING: the default model failed to load: {exc}", flush=True)
        print("  The page still opens and will show this error.", flush=True)

    print()
    print(f"  >>>  Open this in your browser:  {url}")
    print("       Press Ctrl+C to stop the server.")
    print("=" * 64)

    if os.environ.get("ASR_NO_BROWSER") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(host="127.0.0.1", port=port, debug=False,
            threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
