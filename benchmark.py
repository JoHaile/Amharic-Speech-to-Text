"""Benchmark runner: transcribes the reference clips and scores them.

Not part of the app; used to produce the measured numbers in README.md.
"""
import json
import subprocess
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import transcribe  # noqa: E402

refs = json.loads((ROOT / "audio" / "_references.json").read_text(encoding="utf-8"))

processor, model, load_s = transcribe.load_model()
print(f"model_load_s={load_s:.3f}", flush=True)

results = []
for entry in refs:
    audio_path = ROOT / entry["file"]
    before = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-Counter '\\Processor(_Total)\\% Processor Time' -MaxSamples 1)"
         ".CounterSamples[0].CookedValue"],
        capture_output=True, text=True,
    )
    r = transcribe.transcribe_file(audio_path, processor=processor, model=model,
                                   verbose=False)
    ref = entry["reference"]
    hyp = r["text"]
    ratio = SequenceMatcher(None, ref, hyp).ratio()
    ref_words = ref.split()
    hyp_words = hyp.split()
    matched = sum(1 for w in ref_words if w in hyp_words)
    r.update({
        "reference": ref,
        "char_similarity": round(ratio, 4),
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
        "words_matched": matched,
        "word_match_pct": round(100.0 * matched / len(ref_words), 1) if ref_words else 0.0,
    })
    results.append(r)

out = ROOT / "output" / "benchmark.json"
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {out}")

print("\n--- summary ---")
print(f"{'file':<18}{'dur':>7}{'proc':>8}{'RTF':>8}{'RSS MB':>9}{'CPU%':>7}{'sim':>8}{'wmatch':>8}")
for r in results:
    print(f"{Path(r['file']).name:<18}{r['duration_s']:>7.2f}{r['processing_s']:>8.2f}"
          f"{r['rtf']:>8.3f}{r['peak_rss_mb']:>9.0f}{r['cpu_util_pct']:>7.0f}"
          f"{r['char_similarity']:>8.3f}{r['word_match_pct']:>7.1f}%")
