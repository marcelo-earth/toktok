"""Upload trained tokenizer to HuggingFace Hub."""

import os
import argparse
import sentencepiece as spm
from huggingface_hub import HfApi, create_repo


def upload_tokenizer(model_path, repo_name, token=None):
    """Upload a SentencePiece model to HuggingFace Hub.

    Args:
        model_path: Path to .model file (without extension)
        repo_name: HuggingFace repo name (e.g., "username/toktok-es-32k")
        token: HuggingFace API token (or set HF_TOKEN env var)
    """
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN env var or pass --token")
        return

    model_file = f"{model_path}.model"
    vocab_file = f"{model_path}.vocab"

    if not os.path.exists(model_file):
        print(f"Model not found: {model_file}")
        return

    api = HfApi(token=token)

    # create repo if it doesnt exist
    try:
        create_repo(repo_name, token=token, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Repo creation: {e}")

    # upload files
    api.upload_file(path_or_fileobj=model_file, path_in_repo="tokenizer.model", repo_id=repo_name)
    print(f"Uploaded {model_file}")

    if os.path.exists(vocab_file):
        api.upload_file(path_or_fileobj=vocab_file, path_in_repo="tokenizer.vocab", repo_id=repo_name)
        print(f"Uploaded {vocab_file}")

    # create a simple README
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)

    readme = f"""# TokTok Spanish Tokenizer

BPE tokenizer trained on Spanish Wikipedia using SentencePiece.

## Details

- **Vocab size**: {sp.get_piece_size():,}
- **Model type**: BPE
- **Training data**: Spanish Wikipedia (100K articles)
- **Character coverage**: 99.95%
- **Byte fallback**: enabled

## Usage

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

text = "El procesamiento de lenguaje natural es fascinante."
tokens = sp.encode_as_pieces(text)
print(tokens)
```

## Why?

English-centric tokenizers (GPT-4, Llama) use 20-40% more tokens on Spanish text.
This tokenizer is optimized for Spanish, giving better compression and cheaper inference.
"""

    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_name,
    )
    print(f"Uploaded README.md")
    print(f"\nDone! See: https://huggingface.co/{repo_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .model file (without extension)")
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace repo name")
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    upload_tokenizer(args.model, args.repo, args.token)
