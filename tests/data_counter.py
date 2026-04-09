"""
Step 2 of the DataComp word-frequency pipeline.

Reads the token counter JSON produced by read_parquet.py, computes a
subsampling discard probability for each token using the formula:

    discard_prob = max(0, 1 - sqrt(threshold / freq))

and writes one output JSON per threshold value.

Usage:
    python data_counter.py --input  /path/to/DataComp_word_counter.json \
                           --output_dir /path/to/DataComp/
"""

import argparse
import json
from pathlib import Path

import numpy as np


def compute_word_frequency(counter: dict, threshold: float) -> dict:
    total_count = sum(counter.values())
    freqs = {word: count / total_count for word, count in counter.items()}
    return {
        word: max(0.0, 1 - round(np.sqrt(threshold / freqs[word]), 6))
        for word in counter
    }


def main():
    parser = argparse.ArgumentParser(description="Compute subsampling discard probabilities from a token counter JSON.")
    parser.add_argument("--input", required=True, help="Path to the token counter JSON (from read_parquet.py).")
    parser.add_argument("--output_dir", required=True, help="Directory to write the word-frequency JSON files.")
    parser.add_argument("--min_count", type=int, default=5, help="Discard tokens with fewer occurrences before computing frequencies (default: 5).")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        counter = json.load(f)

    counter = {k: v for k, v in sorted(counter.items(), key=lambda x: x[1], reverse=True) if v >= args.min_count}
    total_count = sum(counter.values())
    print(f"Total tokens: {total_count:,}  |  Unique: {len(counter):,}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for threshold in (1e-7):
        word_frequency = compute_word_frequency(counter, threshold)
        suffix = f"1e{int(round(np.log10(threshold)))}"
        out_path = output_dir / f"DataComp_word_frequency_{suffix}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(word_frequency, f, ensure_ascii=False, indent=4)
        print(f"Saved threshold={threshold:.0e} → {out_path}")


if __name__ == "__main__":
    main()
