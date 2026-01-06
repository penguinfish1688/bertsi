# Zhen Transformer

A Transformer model for Chinese-to-English machine translation.

## Overview

This is a complete implementation of the Transformer architecture for translating Chinese text into English. The model follows the standard encoder-decoder pattern with multi-head self-attention and cross-attention mechanisms.

## Architecture

- **Encoder**: Stacked encoder layers with self-attention and feed-forward networks
- **Decoder**: Stacked decoder layers with self-attention, cross-attention over encoder output, and feed-forward networks
- **Multi-Head Attention**: 8 attention heads by default for parallel representation learning
- **Pre-Layer Normalization (Pre-LN)**: Layer normalization applied before each sub-layer
- **Activation**: GELU activation function in feed-forward networks
- **Positional Encoding**: Sinusoidal positional embeddings for sequence ordering

## Dataset
- WMT dataset, consisting of roughly 10M English-Simplified Chinese sentence pairs.
- 80% for training, 10% for validation, 10% reserved for testing.


## Model Configuration

Default parameters:
- Embedding dimension: 512
- Number of attention heads: 8
- Number of encoder/decoder layers: 8
- Maximum sequence length: 512
- Dropout rate: 0.1
- Total parameters: 84M

## Usage
```python
from transformer.models.transformer import Transformer

# Initialize model
model = Transformer(
    src_vocab_size=5000,  # Chinese vocabulary
    tgt_vocab_size=5000,  # English vocabulary
    embed_dim=512,
    num_heads=8,
    num_layers=6
)

# Forward pass (training)
logits = model(src_tokens, tgt_tokens)

# Encoding for inference
memory = model.encode(src_tokens)
output = model.decode(tgt_tokens, memory, src_tokens)
```
- Or to run the train script
```python
cd zhen-transformer
python -m transformer.train.train_translation
```
## Results
- 4 Epochs
- Trained on one H200 for 3.5 hours
- BLEU score ~21

### Checkpoint Loss

| Filename                | Epoch | Train Loss | Val Loss |
|------------------------:|:-----:|-----------:|---------:|
| checkpoint_epoch_1.pt   | 1     | 2.2865     | 1.5992   |
| checkpoint_epoch_2.pt   | 2     | 1.6085     | 1.4466   |
| checkpoint_epoch_3.pt   | 3     | 1.4842     | 1.3754   |
| checkpoint_epoch_4.pt   | 4     | 1.4127     | 1.3254   |
| best_model.pt           | 4     | 1.4127     | 1.3254   |