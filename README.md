<p align="center">
  <img
    src=".github/logo.png"
    align="center"
    width="100"
    alt="TokTok"
    title="TokTok"
  />
  <h1 align="center">TokTok</h1>
</p>

<p align="center">
  📊 Train a custom BPE tokenizer from scratch on a Spanish corpus using SentencePiece. 📖
</p>

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

## Upload to HuggingFace

```bash
export HF_TOKEN=your_token
python upload_to_hf.py --model models/sp_bpe_32k --repo username/toktok-es-32k
```

## Key findings

- English tokenizers (GPT-4, Llama) use 20-40% more tokens on spanish text
- 32K vocab is the sweet spot for spanish (diminishing returns after that)
- Our tokenizer follows Zipf's law, meaning the vocab distribution is healthy
- Llama 3 handles spanish better than GPT-4 thanks to more multilingual training data

## Results

### Compression comparison

![Compression comparison](plots/compression_comparison.png)

### Zipf's law

![Zipf's law](plots/zipf.png)

### Vocab sweet spot

![Vocab sweet spot](plots/vocab_sweet_spot.png)

See `toktok.ipynb` for the full comparison and visualizations.
