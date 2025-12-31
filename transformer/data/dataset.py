"""
Dataset and DataLoader for Chinese-English Translation

This module provides:
1. Dataset download and preprocessing
2. PyTorch Dataset class
3. Collate function for batching with padding
4. DataLoader creation
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import random


class TranslationDataset(Dataset):
    """
    PyTorch Dataset for parallel translation data
    """
    
    def __init__(self, src_data: List[List[int]], tgt_data: List[List[int]]):
        """
        Args:
            src_data: List of tokenized source sequences (as indices)
            tgt_data: List of tokenized target sequences (as indices)
        """
        assert len(src_data) == len(tgt_data), "Source and target must have same length"
        self.src_data = src_data
        self.tgt_data = tgt_data
    
    def __len__(self) -> int:
        return len(self.src_data)
    
    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.src_data[idx], self.tgt_data[idx]


def collate_fn(batch: List[Tuple[List[int], List[int]]], 
               src_pad_idx: int, tgt_pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function to pad sequences in a batch
    
    Args:
        batch: List of (source, target) pairs
        src_pad_idx: Padding index for source
        tgt_pad_idx: Padding index for target
    
    Returns:
        Tuple of padded (source_batch, target_batch) tensors
    """
    src_batch, tgt_batch = zip(*batch)
    
    # Find max lengths
    src_max_len = max(len(s) for s in src_batch)
    tgt_max_len = max(len(t) for t in tgt_batch)
    
    # Pad sequences
    src_padded = []
    tgt_padded = []
    
    for src, tgt in zip(src_batch, tgt_batch):
        src_padded.append(src + [src_pad_idx] * (src_max_len - len(src)))
        tgt_padded.append(tgt + [tgt_pad_idx] * (tgt_max_len - len(tgt)))
    
    return torch.tensor(src_padded, dtype=torch.long), torch.tensor(tgt_padded, dtype=torch.long)


def download_sample_data() -> Tuple[List[str], List[str]]:
    """
    Provide sample Chinese-English parallel data for testing
    This is a small sample dataset. For production, use larger datasets.
    
    Returns:
        Tuple of (chinese_sentences, english_sentences)
    """
    # Sample parallel sentences for testing
    parallel_data = [
        ("你好", "hello"),
        ("早上好", "good morning"),
        ("晚上好", "good evening"),
        ("谢谢你", "thank you"),
        ("不客气", "you are welcome"),
        ("我是学生", "i am a student"),
        ("他是老师", "he is a teacher"),
        ("她很漂亮", "she is beautiful"),
        ("今天天气很好", "the weather is nice today"),
        ("我喜欢学习中文", "i like learning chinese"),
        ("这本书很有趣", "this book is interesting"),
        ("我们去吃饭吧", "let us go eat"),
        ("你叫什么名字", "what is your name"),
        ("我住在北京", "i live in beijing"),
        ("中国是一个大国", "china is a big country"),
        ("我爱我的家人", "i love my family"),
        ("明天见", "see you tomorrow"),
        ("祝你好运", "good luck to you"),
        ("这个多少钱", "how much is this"),
        ("请问洗手间在哪里", "where is the restroom please"),
        ("我不明白", "i do not understand"),
        ("你能帮我吗", "can you help me"),
        ("我饿了", "i am hungry"),
        ("水在哪里", "where is the water"),
        ("现在几点了", "what time is it now"),
        ("我需要休息", "i need to rest"),
        ("这个很好吃", "this is delicious"),
        ("我喜欢音乐", "i like music"),
        ("他喜欢运动", "he likes sports"),
        ("她在看书", "she is reading a book"),
        ("我们是朋友", "we are friends"),
        ("请坐", "please sit down"),
        ("请说慢一点", "please speak slowly"),
        ("我会说一点中文", "i can speak a little chinese"),
        ("你会说英文吗", "can you speak english"),
        ("我正在学习", "i am studying"),
        ("这是我的书", "this is my book"),
        ("那是你的笔", "that is your pen"),
        ("他们在工作", "they are working"),
        ("我昨天去了商店", "i went to the store yesterday"),
        ("我每天早上跑步", "i run every morning"),
        ("她在厨房做饭", "she is cooking in the kitchen"),
        ("我的朋友来自美国", "my friend is from america"),
        ("我想喝咖啡", "i want to drink coffee"),
        ("你喜欢什么颜色", "what color do you like"),
        ("我最喜欢蓝色", "my favorite is blue"),
        ("这件衣服太贵了", "this clothes is too expensive"),
        ("你今天看起来很开心", "you look happy today"),
        ("我下周去旅行", "i will travel next week"),
        ("他在大学学习计算机", "he studies computer at university"),
    ]
    
    chinese_sentences = [pair[0] for pair in parallel_data]
    english_sentences = [pair[1] for pair in parallel_data]
    
    return chinese_sentences, english_sentences


def download_wmt_sample(num_samples: int = 10000000) -> Tuple[List[str], List[str]]:
    """
    Download a sample from WMT translation datasets using HuggingFace datasets.
    
    For a real application, you would use datasets like:
    - WMT (Workshop on Machine Translation)
    - UN Parallel Corpus
    - OpenSubtitles
    - OPUS (Open Parallel Corpus)
    
    This function downloads from OPUS-100 dataset (en-zh pairs).
    Falls back to sample data if download fails.
    
    Args:
        num_samples: Number of samples to download (default: 1000)
    
    Returns:
        Tuple of (chinese_sentences, english_sentences)
    """
    print("\n" + "="*60)
    print("📥 DOWNLOADING WMT DATASET")
    print("="*60)
    
    try:
        # Try to import datasets library
        try:
            from datasets import load_dataset
        except ImportError:
            print("⚠️  'datasets' library not found. Installing...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
            from datasets import load_dataset
        
        print("📦 Loading OPUS-100 Chinese-English dataset...")
        print(f"   Requesting {num_samples} samples...")
        
        # Load OPUS-100 dataset for Chinese-English
        # This is a high-quality parallel corpus from OPUS
        dataset = load_dataset("opus100", "en-zh", split="train", trust_remote_code=True)
        
        print(f"✅ Loaded dataset with {len(dataset)} total pairs")
        
        # Extract samples
        num_samples = min(num_samples, len(dataset))
        
        chinese_sentences = []
        english_sentences = []
        
        for i in range(num_samples):
            item = dataset[i]
            translation = item['translation']
            
            # OPUS-100 format: {'en': '...', 'zh': '...'}
            en_text = translation['en']
            zh_text = translation['zh']
            
            # Clean up sentences
            en_text = en_text.strip()
            zh_text = zh_text.strip()
            
            # Skip empty sentences
            if en_text and zh_text:
                chinese_sentences.append(zh_text)
                english_sentences.append(en_text)
        
        print(f"✅ Extracted {len(chinese_sentences)} valid sentence pairs")
        print(f"\n📊 Sample pairs:")
        for i in range(min(3, len(chinese_sentences))):
            print(f"   [{i+1}] ZH: {chinese_sentences[i]}")
            print(f"       EN: {english_sentences[i]}")
        
        print("="*60)
        return chinese_sentences, english_sentences
        
    except Exception as e:
        print(f"\n⚠️  Failed to download WMT dataset: {e}")
        print("📝 Falling back to sample data...")
        print("="*60)
        return download_sample_data()


class TranslationDataPipeline:
    """
    Complete data pipeline for Chinese-English translation
    """
    
    def __init__(self, tokenizer, config):
        """
        Args:
            tokenizer: TranslationTokenizer instance
            config: TranslationConfig instance
        """
        self.tokenizer = tokenizer
        self.config = config
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def load_data(self, use_sample: bool = True) -> Tuple[List[str], List[str]]:
        """
        Load parallel corpus data
        
        Args:
            use_sample: If True, use sample data; otherwise download larger dataset
        """
        print("\n" + "="*60)
        print("📂 LOADING DATA")
        print("="*60)
        
        if use_sample:
            print("📝 Using sample dataset for testing...")
            src_sentences, tgt_sentences = download_sample_data()
        else:
            src_sentences, tgt_sentences = download_wmt_sample()
        
        print(f"✅ Loaded {len(src_sentences)} parallel sentence pairs")
        print(f"\n📊 Sample pairs:")
        for i in range(min(3, len(src_sentences))):
            print(f"   [{i+1}] ZH: {src_sentences[i]}")
            print(f"       EN: {tgt_sentences[i]}")
        
        return src_sentences, tgt_sentences
    
    def prepare_data(self, src_sentences: List[str], tgt_sentences: List[str],
                    train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict:
        """
        Prepare data: tokenize, build vocab, split, and create datasets
        
        Args:
            src_sentences: Chinese sentences
            tgt_sentences: English sentences
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
        
        Returns:
            Dictionary with datasets and info
        """
        print("\n" + "="*60)
        print("⚙️  PREPARING DATA")
        print("="*60)
        
        # Step 1: Build vocabularies
        src_tokenized, tgt_tokenized = self.tokenizer.build_vocabularies(
            src_sentences, tgt_sentences, 
            min_freq=self.config.min_freq
        )
        
        # Step 2: Encode all sentences
        print("\n📊 Step 5: Encoding sentences to indices...")
        src_encoded = [self.tokenizer.src_vocab.encode(tokens) for tokens in src_tokenized]
        tgt_encoded = [self.tokenizer.tgt_vocab.encode(tokens) for tokens in tgt_tokenized]
        print(f"   ✅ Encoded {len(src_encoded)} sentence pairs")
        
        # Show sample encoding
        print(f"\n   📊 Sample encoding:")
        print(f"   ZH tokens: {src_tokenized[0]}")
        print(f"   ZH indices: {src_encoded[0]}")
        print(f"   EN tokens: {tgt_tokenized[0]}")
        print(f"   EN indices: {tgt_encoded[0]}")
        
        # Step 3: Split data
        print(f"\n📊 Step 6: Splitting data (train={train_ratio}, val={val_ratio})...")
        n = len(src_encoded)
        indices = list(range(n))
        random.seed(42)
        random.shuffle(indices)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        print(f"   ✅ Train: {len(train_indices)} samples")
        print(f"   ✅ Val: {len(val_indices)} samples")
        print(f"   ✅ Test: {len(test_indices)} samples")
        
        # Step 4: Create datasets
        print(f"\n📊 Step 7: Creating PyTorch datasets...")
        
        self.train_dataset = TranslationDataset(
            [src_encoded[i] for i in train_indices],
            [tgt_encoded[i] for i in train_indices]
        )
        self.val_dataset = TranslationDataset(
            [src_encoded[i] for i in val_indices],
            [tgt_encoded[i] for i in val_indices]
        )
        self.test_dataset = TranslationDataset(
            [src_encoded[i] for i in test_indices],
            [tgt_encoded[i] for i in test_indices]
        )
        
        print(f"   ✅ Datasets created successfully!")
        
        print("\n" + "="*60)
        print("✅ DATA PREPARATION COMPLETE")
        print("="*60)
        
        return {
            "train_size": len(train_indices),
            "val_size": len(val_indices),
            "test_size": len(test_indices),
            "src_vocab_size": len(self.tokenizer.src_vocab),
            "tgt_vocab_size": len(self.tokenizer.tgt_vocab),
        }
    
    def get_dataloader(self, split: str = "train", shuffle: bool = True) -> DataLoader:
        """
        Get DataLoader for specified split
        
        Args:
            split: One of "train", "val", "test"
            shuffle: Whether to shuffle data
        
        Returns:
            PyTorch DataLoader
        """
        dataset_map = {
            "train": self.train_dataset,
            "val": self.val_dataset,
            "test": self.test_dataset
        }
        
        dataset = dataset_map.get(split)
        if dataset is None:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'")
        
        # Create collate function with vocab padding indices
        from functools import partial
        collate = partial(
            collate_fn,
            src_pad_idx=self.tokenizer.src_vocab.pad_idx,
            tgt_pad_idx=self.tokenizer.tgt_vocab.pad_idx
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=0
        )
    
    def get_sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample batch for testing"""
        dataloader = self.get_dataloader("train", shuffle=False)
        src_batch, tgt_batch = next(iter(dataloader))
        return src_batch, tgt_batch


def create_pipeline(use_sample: bool = True):
    """
    Convenience function to create complete data pipeline
    
    Args:
        use_sample: Use sample data (True) or download full dataset (False)
    
    Returns:
        Tuple of (pipeline, tokenizer, config)
    """
    from transformer.data.config import TranslationConfig
    from transformer.data.tokenizer import TranslationTokenizer
    
    print("\n" + "🚀"*30)
    print("  CHINESE-ENGLISH TRANSLATION DATA PIPELINE")
    print("🚀"*30)
    
    # Create config and tokenizer
    config = TranslationConfig()
    tokenizer = TranslationTokenizer(config)
    
    # Create pipeline
    pipeline = TranslationDataPipeline(tokenizer, config)
    
    # Load and prepare data
    src_sentences, tgt_sentences = pipeline.load_data(use_sample=use_sample)
    info = pipeline.prepare_data(src_sentences, tgt_sentences)
    
    print("\n" + "="*60)
    print("📋 PIPELINE SUMMARY")
    print("="*60)
    print(f"   Source vocabulary: {info['src_vocab_size']} tokens")
    print(f"   Target vocabulary: {info['tgt_vocab_size']} tokens")
    print(f"   Training samples: {info['train_size']}")
    print(f"   Validation samples: {info['val_size']}")
    print(f"   Test samples: {info['test_size']}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Device: {config.device}")
    print("="*60)
    
    return pipeline, tokenizer, config


if __name__ == "__main__":
    # Run pipeline demo
    pipeline, tokenizer, config = create_pipeline(use_sample=True)
    
    # Get sample batch
    print("\n📦 Getting sample batch...")
    src_batch, tgt_batch = pipeline.get_sample_batch()
    print(f"   Source batch shape: {src_batch.shape}")
    print(f"   Target batch shape: {tgt_batch.shape}")
    print(f"   Source sample: {src_batch[0].tolist()}")
    print(f"   Target sample: {tgt_batch[0].tolist()}")
