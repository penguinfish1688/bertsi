"""
Dataset and DataLoader for Chinese-English Translation

This module provides:
1. Dataset download and preprocessing
2. PyTorch Dataset class
3. Collate function for batching with padding
4. DataLoader creation
5. Dataset caching for faster loading
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import random
import pickle
import hashlib


class TranslationDataset(Dataset):
    """
    PyTorch Dataset for parallel translation data with teacher forcing
    """
    
    def __init__(self, src_data: List[List[int]], tgt_input_data: List[List[int]], tgt_output_data: List[List[int]]):
        """
        Args:
            src_data: List of tokenized source sequences (as indices)
            tgt_input_data: List of target input sequences for teacher forcing (with BOS, without last token)
            tgt_output_data: List of target output sequences for training (without BOS, with EOS)
        """
        assert len(src_data) == len(tgt_input_data) == len(tgt_output_data), "All data must have same length"
        self.src_data = src_data
        self.tgt_input_data = tgt_input_data
        self.tgt_output_data = tgt_output_data
    
    def __len__(self) -> int:
        return len(self.src_data)
    
    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], List[int]]:
        return self.src_data[idx], self.tgt_input_data[idx], self.tgt_output_data[idx]


def collate_fn(batch: List[Tuple[List[int], List[int], List[int]]], 
               src_pad_idx: int, tgt_pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function to pad sequences in a batch with teacher forcing
    
    Args:
        batch: List of (source, target_input, target_output) tuples
        src_pad_idx: Padding index for source
        tgt_pad_idx: Padding index for target
    
    Returns:
        Tuple of padded (source_batch, target_input_batch, target_output_batch) tensors
    """
    src_batch, tgt_input_batch, tgt_output_batch = zip(*batch)
    
    # Find max lengths
    src_max_len = max(len(s) for s in src_batch)
    tgt_input_max_len = max(len(t) for t in tgt_input_batch)
    tgt_output_max_len = max(len(t) for t in tgt_output_batch)
    
    # Pad sequences
    src_padded = []
    tgt_input_padded = []
    tgt_output_padded = []
    
    for src, tgt_in, tgt_out in zip(src_batch, tgt_input_batch, tgt_output_batch):
        src_padded.append(src + [src_pad_idx] * (src_max_len - len(src)))
        tgt_input_padded.append(tgt_in + [tgt_pad_idx] * (tgt_input_max_len - len(tgt_in)))
        tgt_output_padded.append(tgt_out + [tgt_pad_idx] * (tgt_output_max_len - len(tgt_out)))
    
    return (torch.tensor(src_padded, dtype=torch.long), 
            torch.tensor(tgt_input_padded, dtype=torch.long),
            torch.tensor(tgt_output_padded, dtype=torch.long))


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


def download_wmt_sample(num_samples: int = 1000) -> Tuple[List[str], List[str]]:
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
    
    def _get_cache_path(self, use_sample: bool, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
        """Generate cache file path based on config using real dataset name"""
        dataset_type = "sample" if use_sample else f"wmt_{self.config.max_samples or 'full'}"
        cache_filename = f"dataset_{dataset_type}_freq{self.config.min_freq}_train{train_ratio}_val{val_ratio}.pkl"
        return os.path.join(self.config.cache_dir, cache_filename)
    
    def _save_cache(self, data: Dict, cache_path: str):
        """Save processed dataset to cache"""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Saved dataset cache to {cache_path}")
    
    def _load_cache(self, cache_path: str) -> Optional[Dict]:
        """Load processed dataset from cache"""
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"📦 Loaded dataset from cache: {cache_path}")
                return data
            except Exception as e:
                print(f"⚠️  Failed to load cache: {e}")
                return None
        return None
    
    def load_data(self, use_sample: bool = True) -> Tuple[List[str], List[str]]:
        """
        Load parallel corpus data
        
        Args:
            use_sample: If True, use sample data; otherwise download larger dataset
        """
        print("\n" + "="*60)
        print("📂 LOADING DATA")
        print("="*60)
        
        # Download data (caching happens in prepare_data after tokenization)
        if use_sample:
            print("📝 Using sample dataset for testing...")
            src_sentences, tgt_sentences = download_sample_data()
        else:
            src_sentences, tgt_sentences = download_wmt_sample(num_samples=self.config.max_samples or 1000)
        
        print(f"✅ Loaded {len(src_sentences)} parallel sentence pairs")
        print(f"\n📊 Sample pairs:")
        for i in range(min(3, len(src_sentences))):
            print(f"   [{i+1}] ZH: {src_sentences[i]}")
            print(f"       EN: {tgt_sentences[i]}")
        
        return src_sentences, tgt_sentences
    
    def prepare_data(self, train_ratio: float = 0.8, val_ratio: float = 0.1, use_sample: bool = True) -> Dict:
        """
        Prepare data: tokenize, build vocab, split, and create datasets
        
        Args:
            src_sentences: Chinese sentences
            tgt_sentences: English sentences
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            use_sample: Whether using sample data (for cache key)
        
        Returns:
            Dictionary with datasets and info
        """
        print("\n" + "="*60)
        print("⚙️  PREPARING DATA")
        print("="*60)
        
        # Check if we should use cached dataset
        cache_path = self._get_cache_path(use_sample, train_ratio, val_ratio)
        
        # If download_new is False, try to load existing cache
        if not self.config.download_new:
            cached_data = self._load_cache(cache_path)
            if cached_data is not None:
                # Restore vocabularies
                self.tokenizer.src_vocab = cached_data['src_vocab']
                self.tokenizer.tgt_vocab = cached_data['tgt_vocab']
                
                # Restore datasets with teacher forcing
                self.train_dataset = TranslationDataset(
                    cached_data['train_src_encoded'],
                    cached_data['train_tgt_input'],
                    cached_data['train_tgt_output']
                )
                self.val_dataset = TranslationDataset(
                    cached_data['val_src_encoded'],
                    cached_data['val_tgt_input'],
                    cached_data['val_tgt_output']
                )
                self.test_dataset = TranslationDataset(
                    cached_data['test_src_encoded'],
                    cached_data['test_tgt_input'],
                    cached_data['test_tgt_output']
                )
                
                print("\n" + "="*60)
                print(f"✅ DATA LOADED FROM CACHE: {os.path.basename(cache_path)}")
                print("="*60)
                
                return {
                    "train_size": len(self.train_dataset),
                    "val_size": len(self.val_dataset),
                    "test_size": len(self.test_dataset),
                    "src_vocab_size": len(self.tokenizer.src_vocab),
                    "tgt_vocab_size": len(self.tokenizer.tgt_vocab),
                }
            else:
                print(f"\n📥 No cached dataset found at {cache_path}")
                print(f"   Downloading and creating new dataset...")
        
        # Process data from scratch
        # Step 0: Download data
        print("Downloading data...")
        src_sentences, tgt_sentences = self.load_data(use_sample=use_sample)
        print(f"   ✅ Loaded {len(src_sentences)} sentence pairs")

        # Step 1: Build vocabularies (includes tokenization with jieba)
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
        
        # Step 4: Create datasets with teacher forcing
        print(f"\n📊 Step 7: Creating PyTorch datasets with teacher forcing...")
        
        train_src_encoded = [src_encoded[i] for i in train_indices]
        train_tgt_encoded = [tgt_encoded[i] for i in train_indices]
        val_src_encoded = [src_encoded[i] for i in val_indices]
        val_tgt_encoded = [tgt_encoded[i] for i in val_indices]
        test_src_encoded = [src_encoded[i] for i in test_indices]
        test_tgt_encoded = [tgt_encoded[i] for i in test_indices]
        
        # Apply teacher forcing: split target into input (:-1) and output (1:)
        train_tgt_input = [seq[:-1] for seq in train_tgt_encoded]
        train_tgt_output = [seq[1:] for seq in train_tgt_encoded]
        val_tgt_input = [seq[:-1] for seq in val_tgt_encoded]
        val_tgt_output = [seq[1:] for seq in val_tgt_encoded]
        test_tgt_input = [seq[:-1] for seq in test_tgt_encoded]
        test_tgt_output = [seq[1:] for seq in test_tgt_encoded]
        
        self.train_dataset = TranslationDataset(train_src_encoded, train_tgt_input, train_tgt_output)
        self.val_dataset = TranslationDataset(val_src_encoded, val_tgt_input, val_tgt_output)
        self.test_dataset = TranslationDataset(test_src_encoded, test_tgt_input, test_tgt_output)
        
        print(f"   ✅ Datasets created successfully!")
        
        # Save complete processed dataset to cache
        cache_data = {
            'src_vocab': self.tokenizer.src_vocab,
            'tgt_vocab': self.tokenizer.tgt_vocab,
            'train_src_encoded': train_src_encoded,
            'train_tgt_input': train_tgt_input,
            'train_tgt_output': train_tgt_output,
            'val_src_encoded': val_src_encoded,
            'val_tgt_input': val_tgt_input,
            'val_tgt_output': val_tgt_output,
            'test_src_encoded': test_src_encoded,
            'test_tgt_input': test_tgt_input,
            'test_tgt_output': test_tgt_output,
        }
        self._save_cache(cache_data, cache_path)
        
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
    
    def get_sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample batch for testing"""
        dataloader = self.get_dataloader("train", shuffle=False)
        src_batch, tgt_input, tgt_output = next(iter(dataloader))
        return src_batch, tgt_input, tgt_output


def create_pipeline(use_sample: bool = True, config=None):
    """
    Convenience function to create complete data pipeline
    
    Args:
        use_sample: Use sample data (True) or download full dataset (False)
        config: TranslationConfig instance (if None, will create default)
    
    Returns:
        Tuple of (pipeline, tokenizer, config)
    """
    from transformer.data.config import TranslationConfig
    from transformer.data.tokenizer import TranslationTokenizer
    
    print("\n" + "🚀"*30)
    print("  CHINESE-ENGLISH TRANSLATION DATA PIPELINE")
    print("🚀"*30)
    
    # Create config and tokenizer
    if config is None:
        config = TranslationConfig()
    tokenizer = TranslationTokenizer(config)
    
    # Create pipeline
    pipeline = TranslationDataPipeline(tokenizer, config)
    
    # Load and prepare data
    info = pipeline.prepare_data(use_sample=use_sample)
    
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
    src_batch, tgt_input, tgt_output = pipeline.get_sample_batch()
    print(f"   Source batch shape: {src_batch.shape}")
    print(f"   Target input batch shape: {tgt_input.shape}")
    print(f"   Target output batch shape: {tgt_output.shape}")
    print(f"   Source sample: {src_batch[0].tolist()}")
    print(f"   Target input sample: {tgt_input[0].tolist()}")
    print(f"   Target output sample: {tgt_output[0].tolist()}")
