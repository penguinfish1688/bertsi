# Dataset Caching System

## Overview

The dataset caching system saves the **fully processed and tokenized dataset** to disk, allowing you to skip the time-consuming tokenization step (jieba for Chinese, word tokenization for English) on subsequent runs.

## What Gets Cached

When you process a dataset, the following are saved to a `.pkl` file:

1. **Source vocabulary** (Chinese) - built with jieba tokenization
2. **Target vocabulary** (English) - built with word tokenization  
3. **Tokenized and encoded training data** - split and ready to use
4. **Tokenized and encoded validation data**
5. **Tokenized and encoded test data**

## Cache Location

All cache files are stored in: `data/translation/cache/`

Files are named based on a hash of the configuration:
- Sample vs full dataset
- Minimum frequency setting
- Train/val/test split ratios

Example: `data/translation/cache/dataset_a1b2c3d4.pkl`

## Configuration Options

In `transformer/config.yaml`:

```yaml
data:
  use_cached_dataset: true   # Load from cache if available
  download_new: false        # Always download fresh data (ignores cache)
  cache_dir: "data/translation/cache"
```

### Usage Scenarios

| use_cached_dataset | download_new | Behavior |
|-------------------|--------------|----------|
| `false` | `false` | Download → Tokenize → Cache → Train |
| `true` | `false` | Use cache if exists, else download → tokenize → cache |
| `false` | `true` | Always download → tokenize → save new cache |
| `true` | `true` | Always download → tokenize → save new cache |

## Workflow

### First Run (No Cache)
```
1. Download raw sentences
2. Tokenize with jieba (Chinese) and word tokenization (English)
3. Build vocabularies
4. Encode sentences to indices
5. Split into train/val/test
6. Save everything to cache
7. Start training
```

### Subsequent Runs (With Cache)
```
1. Load cached vocabularies
2. Load cached encoded datasets
3. Start training immediately ✨
```

This can save **minutes to hours** depending on dataset size!

## Cache Invalidation

Cache will be regenerated if you change:
- `use_sample` (sample vs full dataset)
- `min_freq` (vocabulary minimum frequency)
- Train/val/test split ratios
- Set `download_new: true`

## Git Ignore

The following are automatically ignored by git:
- `data/translation/cache/` - cached datasets
- `*.pkl` - all pickle files
- `checkpoints/` - model checkpoints
- `*.pt` - PyTorch model files

See [.gitignore](/.gitignore) for full list.

## Manual Cache Management

```bash
# View cache directory
ls -lh data/translation/cache/

# Clear all caches
rm -rf data/translation/cache/*

# Clear specific cache
rm data/translation/cache/dataset_*.pkl
```

## Benefits

✅ **Speed**: Skip tokenization on every run  
✅ **Consistency**: Same vocabulary and splits across runs  
✅ **Disk Space**: Small overhead (~MB for sample data, ~GB for large datasets)  
✅ **Reproducibility**: Cached splits ensure consistent train/val/test sets  
