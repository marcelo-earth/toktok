"""Download Spanish Wikipedia text for tokenizer training."""

import os
import argparse
from datasets import load_dataset


def download_spanish_wiki(output_dir="data", max_samples=100_000):
    """Download spanish wikipedia articles and save as plain text."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "es_wiki.txt")

    if os.path.exists(output_path):
        print(f"Data already exists at {output_path}")
        return output_path

    print("Downloading Spanish Wikipedia from HuggingFace...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)

    with open(output_path, "w", encoding="utf-8") as f:
        count = 0
        for article in ds:
            text = article["text"].strip()
            if len(text) < 100:
                continue
            f.write(text + "\n\n")
            count += 1
            if count % 10_000 == 0:
                print(f"  Downloaded {count:,} articles...")
            if count >= max_samples:
                break

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done. Saved {count:,} articles ({size_mb:.1f} MB) to {output_path}")
    return output_path


def download_english_sample(output_dir="data", max_samples=20_000):
    """Download a smaller english sample for bilingual comparison."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "en_wiki.txt")

    if os.path.exists(output_path):
        print(f"Data already exists at {output_path}")
        return output_path

    print("Downloading English Wikipedia sample...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    with open(output_path, "w", encoding="utf-8") as f:
        count = 0
        for article in ds:
            text = article["text"].strip()
            if len(text) < 100:
                continue
            f.write(text + "\n\n")
            count += 1
            if count % 5_000 == 0:
                print(f"  Downloaded {count:,} articles...")
            if count >= max_samples:
                break

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done. Saved {count:,} articles ({size_mb:.1f} MB) to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--english", action="store_true", help="Also download english sample")
    args = parser.parse_args()

    download_spanish_wiki(args.output_dir, args.max_samples)
    if args.english:
        download_english_sample(args.output_dir)
