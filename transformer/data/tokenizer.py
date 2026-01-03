"""
Tokenizer for Chinese-English Translation

This module provides tokenizers for both Chinese and English text.
- Chinese: Uses SentencePiece for subword tokenization
- English: Uses Byte Pair Encoding (BPE) via tokenizers library
"""
import re
from typing import List, Dict, Optional, Tuple
from collections import Counter
import sentencepiece as spm
from transformer.data.config import TranslationConfig
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, Sequence
import json
import os
import tempfile


class ChineseTokenizer:
    """
    Chinese tokenizer using SentencePiece for subword tokenization
    """
    
    def __init__(self, model_path: Optional[str] = None, vocab_size: int = 16000):
        """
        Initialize SentencePiece tokenizer for Chinese
        
        Args:
            model_path: Path to trained SentencePiece model (optional)
            vocab_size: Vocabulary size for training (default: 16000)
        """
        try:
            self.spm = spm
            self.sp = None
            self.model_path = model_path
            self.vocab_size = vocab_size
            
            if model_path and os.path.exists(model_path):
                self.sp = spm.SentencePieceProcessor()
                self.sp.load(model_path) # type: ignore[reportAttributeAccessIssue]
                print(f"✅ Loaded SentencePiece model from {model_path}")
        except ImportError:
            print("⚠️  sentencepiece not installed. Install with: pip install sentencepiece")
            self.spm = None
            self.sp = None
    def load_model(self, model_path: str):
        """
        Load a pre-trained SentencePiece model
        
        Args:
            model_path: Path to the SentencePiece model file
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path) # type: ignore[reportAttributeAccessIssue]
        self.model_path = model_path
        print(f"✅ Loaded SentencePiece model from {model_path}")

    def train(self, sentences: List[str], model_prefix: str = "chinese_sp"):
        """
        Train SentencePiece model on Chinese sentences
        
        Args:
            sentences: List of Chinese sentences
            model_prefix: Prefix for output model files
        """
        if self.spm is None:
            raise ImportError("sentencepiece not installed")
        
        # Write sentences to temporary file
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            temp_file = f.name
            for sent in sentences:
                f.write(sent + '\n')
        
        try:
            # Train SentencePiece model
            print(f"🔧 Training SentencePiece model with vocab_size={self.vocab_size}...")
            self.spm.SentencePieceTrainer.train( # type: ignore[reportAttributeAccessIssue]
                input=temp_file,
                model_prefix=model_prefix,
                vocab_size=self.vocab_size,
                character_coverage=0.9995,  # High coverage for Chinese
                model_type='unigram',  # Unigram model works well for Chinese
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                pad_piece='<pad>',
                unk_piece='<unk>',
                bos_piece='<bos>',
                eos_piece='<eos>'
            )
            # Load the trained model
            self.model_path = f"{model_prefix}.model"
            self.sp = self.spm.SentencePieceProcessor()
            self.sp.load(self.model_path) # type: ignore[reportAttributeAccessIssue]
            print(f"✅ SentencePiece model trained and saved to {self.model_path}")
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Chinese text using SentencePiece
        
        Args:
            text: Chinese text string
        Returns:
            List of Chinese subword tokens
        """
        if self.sp is None:
            # Fallback: character-level tokenization
            print("⚠️  SentencePiece model not loaded, using character-level tokenization")
            raise NotImplementedError("Character-level tokenization not implemented")
        
        # Use SentencePiece for subword tokenization
        tokens = self.sp.encode_as_pieces(text) # type: ignore[reportAttributeAccessIssue]
        return tokens


class EnglishTokenizer:
    """
    English tokenizer using Byte Pair Encoding (BPE)
    """
    
    def __init__(self, model_path: Optional[str] = None, vocab_size: int = 16000):
        """
        Initialize BPE tokenizer for English
        
        Args:
            vocab_size: Vocabulary size for BPE (default: 16000)
            model_path: Path to saved tokenizer (optional)
        """
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.model_path = model_path
        
        try:
            self.Tokenizer = Tokenizer
            self.BPE = BPE
            self.BpeTrainer = BpeTrainer
            self.Whitespace = Whitespace
            self.Lowercase = Lowercase
            self.Sequence = Sequence
            
            if model_path and os.path.exists(model_path):
                self.tokenizer = Tokenizer.from_file(model_path)
                print(f"✅ Loaded BPE tokenizer from {model_path}")
        except ImportError:
            print("⚠️  tokenizers not installed. Install with: pip install tokenizers")
    
    def load_model(self, model_path: str):
        """
        Load a pre-trained BPE tokenizer
        
        Args:
            model_path: Path to the BPE tokenizer file
        """
        self.tokenizer = Tokenizer.from_file(model_path)
        self.model_path = model_path
        print(f"✅ Loaded BPE tokenizer from {model_path}")
    
    def train(self, sentences: List[str], save_path: Optional[str] = None):
        """
        Train BPE tokenizer on English sentences
        
        Args:
            sentences: List of English sentences
            save_path: Path to save the trained tokenizer
        """
        if not hasattr(self, 'Tokenizer') or self.Tokenizer is None:
            raise ImportError("tokenizers library not installed")

        # Initialize BPE tokenizer
        self.tokenizer = self.Tokenizer(self.BPE(unk_token="<unk>"))
        
        # Set up normalizer (lowercase) and pre-tokenizer (whitespace)
        self.tokenizer.normalizer = self.Lowercase() # type: ignore[reportAttributeAccessIssue]
        self.tokenizer.pre_tokenizer = self.Whitespace() # type: ignore[reportAttributeAccessIssue]
        
        # Set up trainer
        trainer = self.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
        )
        
        # Write sentences to temporary file
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            temp_file = f.name
            for sent in sentences:
                f.write(sent + '\n')
        
        try:
            print(f"🔧 Training BPE tokenizer with vocab_size={self.vocab_size}...")
            self.tokenizer.train([temp_file], trainer)
            print(f"✅ BPE tokenizer trained successfully")
            
            if save_path:
                self.tokenizer.save(save_path)
                self.model_path = save_path
                print(f"💾 BPE tokenizer saved to {save_path}")
        finally:
            os.unlink(temp_file)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize English text using BPE
        
        Args:
            text: English text string
        Returns:
            List of BPE tokens
        """
        if self.tokenizer is None:
            # Fallback: basic tokenization
            print("⚠️  BPE tokenizer not trained, using basic tokenization")
            raise NotImplementedError("Basic tokenization not implemented")
        
        # Use BPE tokenizer
        encoding = self.tokenizer.encode(text)
        return encoding.tokens


class Vocabulary:
    """
    Vocabulary class to map tokens to indices and vice versa
    """
    
    def __init__(self, pad_token: str = "<pad>", unk_token: str = "<unk>",
                 bos_token: str = "<bos>", eos_token: str = "<eos>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        # Initialize with special tokens
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self.token_freq: Counter = Counter()
        
        # Add special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary"""
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for token in special_tokens:
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
    
    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.pad_token]
    
    @property
    def unk_idx(self) -> int:
        return self.token2idx[self.unk_token]
    
    @property
    def bos_idx(self) -> int:
        return self.token2idx[self.bos_token]
    
    @property
    def eos_idx(self) -> int:
        return self.token2idx[self.eos_token]
    
    def __len__(self) -> int:
        return len(self.token2idx)
    
    def build_from_corpus(self, corpus: List[List[str]], min_freq: int = 1):
        """
        Build vocabulary from tokenized corpus
        
        Args:
            corpus: List of tokenized sentences (list of token lists)
            min_freq: Minimum frequency to include token
        """
        print(f"📊 Building vocabulary from {len(corpus)} sentences...")
        
        # Count all tokens
        for tokens in corpus:
            self.token_freq.update(tokens)
        
        # Add tokens meeting minimum frequency
        for token, freq in self.token_freq.items():
            if freq >= min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
        
        print(f"✅ Vocabulary built: {len(self)} tokens (min_freq={min_freq})")
    
    def encode(self, tokens: List[str], add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Convert tokens to indices
        
        Args:
            tokens: List of tokens
            add_bos: Whether to add BOS token at start
            add_eos: Whether to add EOS token at end
        Returns:
            List of token indices
        """
        indices = []
        if add_bos:
            indices.append(self.bos_idx)
        
        for token in tokens:
            indices.append(self.token2idx.get(token, self.unk_idx))
        
        if add_eos:
            indices.append(self.eos_idx)
        
        return indices
    
    def decode(self, indices: List[int], remove_special: bool = True) -> List[str]:
        """
        Convert indices back to tokens
        
        Args:
            indices: List of token indices
            remove_special: Whether to remove special tokens
        Returns:
            List of tokens
        """
        special_indices = {self.pad_idx, self.bos_idx, self.eos_idx}
        tokens = []
        
        for idx in indices:
            if remove_special and idx in special_indices:
                continue
            token = self.idx2token.get(idx, self.unk_token)
            tokens.append(token)
        
        return tokens
    
    def save(self, path: str):
        """Save vocabulary to JSON file"""
        data = {
            "token2idx": self.token2idx,
            "special_tokens": {
                "pad": self.pad_token,
                "unk": self.unk_token,
                "bos": self.bos_token,
                "eos": self.eos_token
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Vocabulary saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load vocabulary from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls(
            pad_token=data["special_tokens"]["pad"],
            unk_token=data["special_tokens"]["unk"],
            bos_token=data["special_tokens"]["bos"],
            eos_token=data["special_tokens"]["eos"]
        )
        vocab.token2idx = data["token2idx"]
        vocab.idx2token = {int(v): k for k, v in vocab.token2idx.items()}
        print(f"📂 Vocabulary loaded from {path}: {len(vocab)} tokens")
        return vocab


class TranslationTokenizer:
    """
    Combined tokenizer for Chinese-English translation
    Uses SentencePiece for Chinese and BPE for English
    """
    
    def __init__(self, config: "TranslationConfig",
                 chinese_vocab_size: int = 16000,
                 english_vocab_size: int = 16000):
        """
        Initialize translation tokenizer
        
        Args:
            config: Optional configuration object
            chinese_model_path: Path to trained SentencePiece model
            english_model_path: Path to trained BPE tokenizer
            chinese_vocab_size: Vocabulary size for Chinese (default: 8000)
            english_vocab_size: Vocabulary size for English (default: 10000)
        """
        self.config = config
        self.chinese_model_path = config.chinese_model_path
        self.english_model_path = config.english_model_path
        self.chinese_tokenizer = ChineseTokenizer(
            model_path=config.chinese_model_path,
            vocab_size=chinese_vocab_size
        )
        self.english_tokenizer = EnglishTokenizer(
            model_path=config.english_model_path,
            vocab_size=english_vocab_size
        )

        self.src_vocab = Vocabulary(
            config.pad_token, config.unk_token,
            config.bos_token, config.eos_token
        )
        self.tgt_vocab = Vocabulary(
            config.pad_token, config.unk_token,
            config.bos_token, config.eos_token
        )
    
    def tokenize_source(self, text: str) -> List[str]:
        """Tokenize Chinese source text"""
        return self.chinese_tokenizer.tokenize(text)
    
    def tokenize_target(self, text: str) -> List[str]:
        """Tokenize English target text"""
        return self.english_tokenizer.tokenize(text)
    
    def build_vocabularies(self, src_sentences: List[str], tgt_sentences: List[str], 
                          min_freq: int = 2,
                          chinese_model_prefix: str = "chinese_sp",
                          rebuild_chinese_model: bool = False,
                          english_model_path: str = "english_bpe.json",
                          rebuild_english_model: bool = False) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Build vocabularies from parallel corpus
        
        Args:
            src_sentences: List of Chinese sentences
            tgt_sentences: List of English sentences
            min_freq: Minimum word frequency
            chinese_model_prefix: Prefix for SentencePiece model files
            english_model_path: Path to save BPE tokenizer
        """
        print("\n" + "="*60)
        print("🔤 TRAINING TOKENIZERS AND BUILDING VOCABULARIES")
        print("="*60)
        
        # Train tokenizers
        
        if rebuild_chinese_model:
            print("\n🎓 Step 1: Training Chinese SentencePiece tokenizer...")
            self.chinese_tokenizer.train(src_sentences, model_prefix=chinese_model_prefix)
        else:
            if self.chinese_model_path is None:
                raise ValueError("chinese_model_path must be provided to load existing model")
            print(f"\nℹ️ Step 1: Skipping Chinese tokenizer training (using existing model {self.chinese_model_path})")
            self.chinese_tokenizer.load_model(self.chinese_model_path)
        
        
        if rebuild_english_model:
            print("\n🎓 Step 2: Training English BPE tokenizer...")
            self.english_tokenizer.train(tgt_sentences, save_path=english_model_path)
        else:
            if self.english_model_path is None:
                raise ValueError("english_model_path must be provided to load existing model")
            print(f"\nℹ️ Step 2: Skipping English tokenizer training (using existing model {self.english_model_path})")
            self.english_tokenizer.load_model(self.english_model_path)
        
        # Tokenize all sentences
        print("\n📝 Step 3: Tokenizing source sentences (Chinese) with SentencePiece...")
        src_tokenized = [self.tokenize_source(s) for s in src_sentences]
        print(f"   ✅ Tokenized {len(src_tokenized)} Chinese sentences")
        print(f"   📊 Sample: {src_sentences[0][:50]}... → {src_tokenized[0][:10]}")
        
        print("\n📝 Step 4: Tokenizing target sentences (English) with BPE...")
        tgt_tokenized = [self.tokenize_target(s) for s in tgt_sentences]
        print(f"   ✅ Tokenized {len(tgt_tokenized)} English sentences")
        print(f"   📊 Sample: {tgt_sentences[0][:50]}... → {tgt_tokenized[0][:10]}")
        
        # Build vocabularies
        print("\n📚 Step 5: Building source vocabulary...")
        self.src_vocab.build_from_corpus(src_tokenized, min_freq)
        
        print("\n📚 Step 6: Building target vocabulary...")
        self.tgt_vocab.build_from_corpus(tgt_tokenized, min_freq)
        
        print("\n" + "="*60)
        print(f"✅ VOCABULARY BUILDING COMPLETE")
        print(f"   Source vocab size: {len(self.src_vocab)}")
        print(f"   Target vocab size: {len(self.tgt_vocab)}")
        print("="*60)
        
        return src_tokenized, tgt_tokenized
    
    def encode_pair(self, src_text: str, tgt_text: str) -> Tuple[List[int], List[int]]:
        """
        Encode a source-target pair
        
        Returns:
            Tuple of (source_ids, target_ids)
        """
        src_tokens = self.tokenize_source(src_text)
        tgt_tokens = self.tokenize_target(tgt_text)
        
        src_ids = self.src_vocab.encode(src_tokens)
        tgt_ids = self.tgt_vocab.encode(tgt_tokens)
        
        return src_ids, tgt_ids
    
    def decode_target(self, indices: List[int]) -> str:
        """Decode target indices back to text"""
        tokens = self.tgt_vocab.decode(indices)
        return " ".join(tokens)
    
    def save(self, directory: str):
        """Save both vocabularies"""
        os.makedirs(directory, exist_ok=True)
        self.src_vocab.save(os.path.join(directory, "src_vocab.json"))
        self.tgt_vocab.save(os.path.join(directory, "tgt_vocab.json"))
    
    @classmethod
    def load(cls, directory: str, config: "TranslationConfig") -> "TranslationTokenizer":
        """Load tokenizer with saved vocabularies"""
        tokenizer = cls(config)
        tokenizer.src_vocab = Vocabulary.load(os.path.join(directory, "src_vocab.json"))
        tokenizer.tgt_vocab = Vocabulary.load(os.path.join(directory, "tgt_vocab.json"))
        return tokenizer
