"""Benchmark runner: transcribes the reference clips with both models and scores them.

Not part of the app; used to produce the measured numbers in README.md.

Usage:
    python benchmark.py                 # both models, three reference clips
    python benchmark.py --models shook  # one model only
"""
import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import transcribe  # noqa: E402


def score(entry, result):
    ref = entry["reference"]
    hyp = result["text"]
    ratio = SequenceMatcher(None, ref, hyp).ratio()
    ref_words = ref.split()
    hyp_words = hyp.split()
    matched = sum(1 for w in ref_words if w in hyp_words)
    result.update({
        "reference": ref,
        "char_similarity": round(ratio, 4),
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
        "words_matched": matched,
        "word_match_pct": round(100.0 * matched / len(ref_words), 1) if ref_words else 0.0,
    })
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark both Amharic ASR models.")
    parser.add_argument("--models", nargs="+", choices=transcribe.MODEL_KEYS,
                        default=list(transcribe.MODEL_KEYS),
                        help="Models to benchmark (default: both)")
    parser.add_argument("--output", default=str(ROOT / "output" / "benchmark.json"),
                        help="Where to write the JSON results")
    args = parser.parse_args()

    refs = json.loads((ROOT / "audio" / "_references.json").read_text(encoding="utf-8"))
    results = []

    for model_key in args.models:
        processor, model, load_s = transcribe.load_model(model_key)
        print(f"{model_key}: model_load_s={load_s:.3f}", flush=True)

        for entry in refs:
            audio_path = ROOT / entry["file"]
            r = transcribe.transcribe_file(
                audio_path,
                processor=processor,
                model=model,
                verbose=False,
                model_key=model_key,
            )
            results.append(score(entry, r))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")

    print("\n--- summary ---")
    print(f"{'model':<10}{'file':<18}{'dur':>7}{'proc':>8}{'RTF':>8}"
          f"{'RSS MB':>9}{'CPU%':>7}{'sim':>8}{'wmatch':>8}")
    for r in results:
        print(f"{r['model']:<10}{Path(r['file']).name:<18}{r['duration_s']:>7.2f}"
              f"{r['processing_s']:>8.2f}{r['rtf']:>8.3f}{r['peak_rss_mb']:>9.0f}"
              f"{r['cpu_util_pct']:>7.0f}{r['char_similarity']:>8.3f}"
              f"{r['word_match_pct']:>7.1f}%")


if __name__ == "__main__":
    main()
