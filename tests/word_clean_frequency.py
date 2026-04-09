"""
Step 3 of the DataComp word-frequency pipeline.

Cleans the word-frequency JSON by removing noisy tokens:
  - pure symbols / punctuation
  - numbers
  - single-character tokens
  - non-ASCII tokens (mojibake artefacts)
  - tokens exceeding MAX_LEN characters (concatenated URLs / junk)
  - URL / domain fragments (.com, www, http, …)
  - English stopwords (via NLTK) and common contraction suffixes

Usage:
    python word_clean_frequency.py --input  /path/to/DataComp_word_frequency_1e7.json \
                                   --output /path/to/DataComp_word_frequency_1e7_cleaned.json
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

from nltk.corpus import stopwords
from tqdm import tqdm

SYMBOL_CATS = {"S", "P"}  # Unicode "Symbol" or "Punctuation"
STOPS = set(stopwords.words("english"))
CONTRACTION_SUFFIXES = {"'s", "'t", "'re", "'ve", "'ll", "'d", "'m"}
URL_FRAGMENTS = {".com", ".net", ".org", ".edu", ".gov", ".io", "blogspot", "wordpress", "http", "www", "://"}
MAX_LEN = 25

NUMBER_RE = re.compile(r"^\d+([.,]\d+)?$")  # 42, 3.14, 1,000, …


def is_symbol(token: str) -> bool:
    return all(unicodedata.category(ch)[0] in SYMBOL_CATS for ch in token)


def is_url_fragment(token: str) -> bool:
    t = token.lower()
    return any(frag in t for frag in URL_FRAGMENTS)


def classify(token: str) -> str:
    """Return the rejection category for *token*, or 'kept' if it passes."""
    if is_symbol(token):
        return "symbol"
    if NUMBER_RE.match(token):
        return "number"
    if len(token) < 2:
        return "short"
    if not token.isascii():
        return "non_ascii"
    if len(token) > MAX_LEN:
        return "too_long"
    if is_url_fragment(token):
        return "url_fragment"
    if token in CONTRACTION_SUFFIXES:
        return "stopword"
    if token.isalpha() and token.lower() in STOPS:
        return "stopword"
    return "kept"


def main():
    parser = argparse.ArgumentParser(description="Remove noisy tokens from a word-frequency JSON.")
    parser.add_argument("--input", required=True, help="Path to the input word-frequency JSON.")
    parser.add_argument("--output", required=True, help="Path for the cleaned output JSON.")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Unique tokens in file: {len(data):,}")

    stats: dict[str, int] = {}
    new_data = {}
    for token, freq in tqdm(data.items(), desc="Filtering tokens"):
        category = classify(token)
        stats[category] = stats.get(category, 0) + 1
        if category == "kept":
            new_data[token] = freq

    print("\nRemoved by category:")
    for category, count in stats.items():
        if category != "kept":
            print(f"  {category:<15}: {count:>10,}")
    print(f"\nUnique tokens after filtering: {len(new_data):,}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
