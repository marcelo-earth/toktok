"""Train BPE tokenizer with SentencePiece on Spanish text."""

import os
import argparse
import sentencepiece as spm


def train_tokenizer(
    input_file,
    vocab_size=32_000,
    model_prefix=None,
    output_dir="models",
    model_type="bpe",
    character_coverage=0.9995,
    num_threads=4,
):
    """Train a SentencePiece BPE tokenizer.

    Args:
        input_file: Path to training text file
        vocab_size: Target vocabulary size
        model_prefix: Name prefix for output files (default: sp_{vocab_size})
        output_dir: Directory to save model files
        model_type: "bpe" or "unigram"
        character_coverage: How much of the character set to cover (0.9995 good for latin scripts)
        num_threads: Parallel threads for training
    """
    os.makedirs(output_dir, exist_ok=True)

    if model_prefix is None:
        model_prefix = f"sp_{model_type}_{vocab_size // 1000}k"

    model_path = os.path.join(output_dir, model_prefix)

    if os.path.exists(f"{model_path}.model"):
        print(f"Model already exists at {model_path}.model")
        return model_path

    print(f"Training {model_type.upper()} tokenizer with vocab_size={vocab_size:,}")
    print(f"  Input: {input_file}")
    print(f"  Output: {model_path}.model")

    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_path,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        num_threads=num_threads,
        # special tokens
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # training params
        input_sentence_size=1_000_000,
        shuffle_input_sentence=True,
        byte_fallback=True,
    )

    print(f"Done. Model saved to {model_path}.model")
    return model_path


def load_and_test(model_path, test_texts=None):
    """Load a trained model and run quick tests."""
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_path}.model")

    print(f"\nModel: {model_path}")
    print(f"Vocab size: {sp.get_piece_size():,}")

    if test_texts is None:
        test_texts = [
            "El procesamiento de lenguaje natural es una rama de la inteligencia artificial.",
            "The quick brown fox jumps over the lazy dog.",
            "Los transformers revolucionaron el campo del NLP en 2017.",
        ]

    for text in test_texts:
        pieces = sp.encode_as_pieces(text)
        ids = sp.encode_as_ids(text)
        compression = len(text) / len(ids)
        print(f"\n  Text: {text}")
        print(f"  Tokens ({len(ids)}): {pieces}")
        print(f"  Compression: {compression:.1f} chars/tok")

    return sp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/es_wiki.txt")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--model-type", type=str, default="bpe", choices=["bpe", "unigram"])
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--test", action="store_true", help="Run quick test after training")
    args = parser.parse_args()

    model_path = train_tokenizer(
        input_file=args.input,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        output_dir=args.output_dir,
    )

    if args.test:
        load_and_test(model_path)
