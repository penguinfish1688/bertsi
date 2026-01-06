#!/usr/bin/env python3
"""
Quick test script to verify translation decoding works correctly
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformer.data.config import TranslationConfig
from transformer.data.dataset import create_pipeline
from transformer.models.transformer import Transformer
from transformer.train.train_translation import greedy_decode

def test_decode():
    """Test that decoding works correctly"""
    print("\n" + "="*60)
    print("🧪 TESTING TRANSLATION DECODING")
    print("="*60)
    
    # Load config
    config = TranslationConfig.from_yaml("transformer/config.yaml")
    
    # Create pipeline and tokenizer
    print("\n📚 Loading data and tokenizer...")
    pipeline, tokenizer, _ = create_pipeline(use_sample=True, config=config)
    
    # Test English tokenizer decode
    print("\n🔤 Testing English tokenizer:")
    test_text = "hello world this is a test"
    tokens = tokenizer.tokenize_target(test_text)
    print(f"   Original: {test_text}")
    print(f"   Tokens: {tokens}")
    
    # Encode to IDs
    token_ids = tokenizer.english_tokenizer.tokenizer.convert_tokens_to_ids(tokens)
    print(f"   Token IDs: {token_ids}")
    
    # Decode back
    decoded = tokenizer.english_tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"   Decoded: {decoded}")
    print(f"   Match: {decoded == test_text}")
    
    # Test with vocab encode/decode
    print("\n🔤 Testing vocabulary encode/decode:")
    vocab_ids = tokenizer.tgt_vocab.encode(tokens, add_bos=True, add_eos=True)
    print(f"   Vocab IDs (with BOS/EOS): {vocab_ids}")
    
    vocab_tokens = tokenizer.tgt_vocab.decode(vocab_ids, remove_special=False)
    print(f"   Vocab tokens: {vocab_tokens}")
    
    # Decode using English tokenizer
    # Need to remove special token IDs first
    bos_idx, eos_idx, pad_idx = tokenizer.tgt_vocab.bos_idx, tokenizer.tgt_vocab.eos_idx, tokenizer.tgt_vocab.pad_idx
    clean_ids = [idx for idx in vocab_ids if idx not in {bos_idx, eos_idx, pad_idx}]
    decoded_clean = tokenizer.english_tokenizer.decode(clean_ids, skip_special_tokens=True)
    print(f"   Decoded (clean): {decoded_clean}")
    
    # Test with actual model if checkpoint exists
    checkpoint_path = "checkpoints/best_model.pt"
    if os.path.exists(checkpoint_path):
        print("\n🤖 Testing with actual model:")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        model = Transformer(
            src_vocab_size=len(tokenizer.src_vocab),
            tgt_vocab_size=len(tokenizer.tgt_vocab),
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_len=config.max_len,
            dropout=config.dropout,
            src_pad_idx=tokenizer.src_vocab.pad_idx,
            tgt_pad_idx=tokenizer.tgt_vocab.pad_idx
        ).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Test translation
        test_sentences = ["你好", "谢谢你"]
        for src in test_sentences:
            translation, tokens = greedy_decode(model, src, tokenizer, config.max_len, device=device)
            print(f"\n   🇨🇳 Chinese: {src}")
            print(f"   🔤 Tokens: {tokens[:10]}...")
            print(f"   🇺🇸 English: {translation}")
    else:
        print(f"\n⚠️  Checkpoint not found at {checkpoint_path}, skipping model test")
    
    print("\n" + "="*60)
    print("✅ TESTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_decode()
