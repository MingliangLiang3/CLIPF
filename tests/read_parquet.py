"""
Step 1 of the DataComp word-frequency pipeline.

Reads all parquet shards, tokenises captions with open_clip's SimpleTokenizer,
counts token occurrences in parallel, and writes a filtered token-count JSON.

Usage:
    python read_parquet.py --shards_dir /path/to/DataComp/shards \
                           --output     /path/to/DataComp_word_counter.json \
                           --min_count  5
"""

import argparse
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool, cpu_count

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))


def process_file(path):
    from open_clip.tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer()
    counter = Counter()
    df = pd.read_parquet(path, columns=["caption"])
    for caption in df["caption"].dropna():
        counter.update(tokenizer.encode_text(caption))
    return counter


def main():
    parser = argparse.ArgumentParser(description="Count token frequencies across DataComp parquet shards.")
    parser.add_argument("--shards_dir", required=True, help="Directory containing .parquet shard files.")
    parser.add_argument("--output", required=True, help="Path to write the output JSON token counter.")
    parser.add_argument("--min_count", type=int, default=5, help="Discard tokens with fewer than this many occurrences (default: 5).")
    args = parser.parse_args()

    parquet_files = sorted([
        os.path.join(args.shards_dir, f)
        for f in os.listdir(args.shards_dir)
        if f.endswith(".parquet")
    ])
    print(f"Found {len(parquet_files)} parquet files")

    num_workers = min(cpu_count(), len(parquet_files))
    print(f"Using {num_workers} workers")

    word_counter = Counter()
    with Pool(processes=num_workers) as pool:
        for i, partial_counter in enumerate(pool.imap_unordered(process_file, parquet_files), 1):
            word_counter.update(partial_counter)
            if i % 10 == 0 or i == len(parquet_files):
                print(f"  Processed {i}/{len(parquet_files)} files", flush=True)

    print(f"\nTotal unique tokens: {len(word_counter)}")
    print("Top 20 tokens:")
    for token, count in word_counter.most_common(20):
        print(f"  {token}: {count}")

    word_counter = dict(
        (k, v) for k, v in word_counter.most_common() if v >= args.min_count
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(word_counter, f, ensure_ascii=False, indent=4)

    total_count = sum(word_counter.values())
    print(f"\nTotal tokens after filtering (min_count={args.min_count}): {total_count}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
