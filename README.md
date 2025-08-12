# TokTok

Train a custom BPE tokenizer from scratch on a Spanish corpus using SentencePiece.

## What is this?

Most LLM tokenizers were trained on english-heavy data. This means spanish text gets split into more tokens than necessary, making inference slower and more expensive. This project trains a tokenizer optimized for spanish and compares it against GPT-4 and Llama tokenizers.

## What we do

1. Download a spanish text corpus from Wikipedia
2. Train BPE tokenizers at multiple vocab sizes (8K, 32K, 64K)
3. Compare compression ratios against tiktoken (GPT-4) and Llama
4. Check if our tokenizer follows Zipf's law
5. Find the sweet spot vocab size for bilingual EN/ES

## Setup

```bash
pip install -r requirements.txt
python download_data.py
```

## Train

```bash
python train_tokenizer.py --vocab-size 32000
```

## Results

See `toktok.ipynb` for the full comparison and visualizations.
