"""
Tokenizer for Chinese-English Translation

This module provides tokenizers for both Chinese and English text.
- Chinese: Uses SentencePiece for subword tokenization
- English: Uses GPT-2 style BPE via transformers library (fully reversible)
"""
import re
from typing import List, Dict, Optional, Tuple
from collections import Counter
import sentencepiece as spm
from transformer.data.config import TranslationConfig
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
    English tokenizer using GPT-2 style BPE (fully reversible).
    
    Uses HuggingFace transformers GPT2TokenizerFast which:
    - Preserves spaces and capitalization
    - Is fully reversible (tokenize -> detokenize gives original text)
    - Uses byte-level BPE for handling any text
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize GPT-2 style BPE tokenizer for English.
        
        By default, loads the official GPT-2 tokenizer from HuggingFace.
        
        Args:
            model_path: Path to saved tokenizer directory, or 'gpt2' for official.
                        If None, loads official GPT-2 tokenizer.
        """
        self.tokenizer = None
        self.model_path = model_path
        
        try:
            from transformers import GPT2TokenizerFast
            self.GPT2TokenizerFast = GPT2TokenizerFast
            
            # Load tokenizer
            load_path = model_path if model_path else 'gpt2'
            self.tokenizer = GPT2TokenizerFast.from_pretrained(load_path)
            self._add_special_tokens()
            self.model_path = load_path
            print(f"✅ Loaded GPT-2 tokenizer from {load_path} (vocab_size={self.vocab_size})")
        except ImportError:
            print("⚠️  transformers not installed. Install with: pip install transformers")
            self.GPT2TokenizerFast = None
    
    @property
    def vocab_size(self) -> int:
        """Return the actual vocabulary size of the loaded tokenizer."""
        if self.tokenizer is None:
            return 0
        return len(self.tokenizer)
    
    def _add_special_tokens(self):
        """Add special tokens if not present."""
        if self.tokenizer is None:
            return
        
        special_tokens = {
            'pad_token': '<pad>',
            'bos_token': '<bos>',
            'eos_token': '<eos>',
            'unk_token': '<unk>'
        }
        
        # Only add tokens that aren't already present
        tokens_to_add = {}
        for key, value in special_tokens.items():
            if getattr(self.tokenizer, key, None) is None:
                tokens_to_add[key] = value
        
        if tokens_to_add:
            self.tokenizer.add_special_tokens(tokens_to_add)
    
    def load_model(self, model_path: str='gpt2'):
        """
        Load a pre-trained GPT-2 style tokenizer.
        
        Use 'gpt2' as model_path to load the official GPT-2 tokenizer.
        
        Args:
            model_path: Path to the tokenizer directory, or 'gpt2' for official tokenizer
        """
        if self.GPT2TokenizerFast is None:
            raise ImportError("transformers library not installed")
        
        self.tokenizer = self.GPT2TokenizerFast.from_pretrained(model_path)
        self._add_special_tokens()
        self.model_path = model_path
        print(f"✅ Loaded GPT-2 tokenizer from {model_path} (vocab_size={self.vocab_size})")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize English text using GPT-2 style BPE.
        
        Preserves spaces and capitalization for full reversibility.
        
        Args:
            text: English text string
        Returns:
            List of BPE tokens
        """
        if self.tokenizer is None:
            raise NotImplementedError("GPT-2 tokenizer not loaded")
        
        return self.tokenizer.tokenize(text)
    
    def detokenize(self, tokens: List[str]) -> str:
        """
        Reconstruct original text from BPE tokens.
        
        This is fully reversible - returns exact original text including
        spaces and capitalization.
        
        Args:
            tokens: List of BPE tokens from tokenize()
        Returns:
            Original text string
        """
        if not tokens:
            return ""
        
        if self.tokenizer is None:
            raise NotImplementedError("GPT-2 tokenizer not loaded")
        
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: English text string
        Returns:
            List of token IDs
        """
        if self.tokenizer is None:
            raise NotImplementedError("GPT-2 tokenizer not loaded")
        
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
        Returns:
            Decoded text string
        """
        if self.tokenizer is None:
            raise NotImplementedError("GPT-2 tokenizer not loaded")
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


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
                 chinese_vocab_size: int = 16000):
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
        self.chinese_tokenizer = ChineseTokenizer(
            model_path=config.chinese_model_path,
            vocab_size=chinese_vocab_size
        )
        self.english_tokenizer = EnglishTokenizer()

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
                          rebuild_chinese_model: bool = False) -> Tuple[List[List[str]], List[List[str]]]:
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
        
        # Train Chinese tokenizers
        if rebuild_chinese_model:
            print("\n🎓 Step 1: Training Chinese SentencePiece tokenizer...")
            self.chinese_tokenizer.train(src_sentences, model_prefix=chinese_model_prefix)
        else:
            if self.chinese_model_path is None:
                raise ValueError("chinese_model_path must be provided to load existing model")
            print(f"\nℹ️ Step 1: Skipping Chinese tokenizer training (using existing model {self.chinese_model_path})")
            self.chinese_tokenizer.load_model(self.chinese_model_path)

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
