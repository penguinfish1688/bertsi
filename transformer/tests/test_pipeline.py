"""
Tests for Chinese-English Translation Pipeline

Run with: python -m pytest transformer/tests/test_pipeline.py -v
Or: python transformer/tests/test_pipeline.py
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import unittest


class TestTokenizer(unittest.TestCase):
    """Tests for tokenizers"""
    
    def setUp(self):
        from transformer.data.tokenizer import ChineseTokenizer, EnglishTokenizer, Vocabulary
        self.zh_tokenizer = ChineseTokenizer()
        self.en_tokenizer = EnglishTokenizer()
    
    def test_chinese_tokenizer(self):
        """Test Chinese tokenization"""
        text = "我喜欢学习中文"
        tokens = self.zh_tokenizer.tokenize(text)
        
        print(f"\n📝 Chinese tokenization test:")
        print(f"   Input: {text}")
        print(f"   Tokens: {tokens}")
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        # All tokens should be non-empty strings
        for token in tokens:
            self.assertIsInstance(token, str)
            self.assertGreater(len(token), 0)
    
    def test_english_tokenizer(self):
        """Test English tokenization"""
        text = "I like learning Chinese."
        tokens = self.en_tokenizer.tokenize(text)
        
        print(f"\n📝 English tokenization test:")
        print(f"   Input: {text}")
        print(f"   Tokens: {tokens}")
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        # Should be lowercase
        for token in tokens:
            self.assertEqual(token, token.lower())
    
    def test_vocabulary(self):
        """Test vocabulary building and encoding"""
        from transformer.data.tokenizer import Vocabulary
        
        vocab = Vocabulary()
        corpus = [
            ["hello", "world"],
            ["hello", "python"],
            ["world", "is", "great"]
        ]
        vocab.build_from_corpus(corpus, min_freq=1)
        
        print(f"\n📚 Vocabulary test:")
        print(f"   Vocab size: {len(vocab)}")
        print(f"   PAD idx: {vocab.pad_idx}")
        print(f"   UNK idx: {vocab.unk_idx}")
        print(f"   BOS idx: {vocab.bos_idx}")
        print(f"   EOS idx: {vocab.eos_idx}")
        
        # Test special tokens
        self.assertEqual(vocab.pad_idx, 0)
        self.assertEqual(vocab.unk_idx, 1)
        
        # Test encoding
        encoded = vocab.encode(["hello", "world"])
        print(f"   Encoded ['hello', 'world']: {encoded}")
        
        self.assertIsInstance(encoded, list)
        self.assertEqual(encoded[0], vocab.bos_idx)  # BOS
        self.assertEqual(encoded[-1], vocab.eos_idx)  # EOS
        
        # Test decoding
        decoded = vocab.decode(encoded, remove_special=True)
        print(f"   Decoded back: {decoded}")
        self.assertEqual(decoded, ["hello", "world"])
        
        # Test unknown token
        encoded_unk = vocab.encode(["unknown_token"])
        self.assertIn(vocab.unk_idx, encoded_unk)


class TestTranslationTokenizer(unittest.TestCase):
    """Tests for combined translation tokenizer"""
    
    def test_build_vocabularies(self):
        """Test vocabulary building from parallel corpus"""
        from transformer.data.tokenizer import TranslationTokenizer
        
        tokenizer = TranslationTokenizer()
        
        src_sentences = ["你好", "早上好", "谢谢你"]
        tgt_sentences = ["hello", "good morning", "thank you"]
        
        print("\n📚 Building vocabularies from parallel corpus...")
        src_tok, tgt_tok = tokenizer.build_vocabularies(src_sentences, tgt_sentences, min_freq=1)
        
        self.assertGreater(len(tokenizer.src_vocab), 4)  # Special tokens + words
        self.assertGreater(len(tokenizer.tgt_vocab), 4)
        
        print(f"   ✅ Source vocab size: {len(tokenizer.src_vocab)}")
        print(f"   ✅ Target vocab size: {len(tokenizer.tgt_vocab)}")
    
    def test_encode_pair(self):
        """Test encoding a source-target pair"""
        from transformer.data.tokenizer import TranslationTokenizer
        
        tokenizer = TranslationTokenizer()
        
        src_sentences = ["你好世界", "早上好"]
        tgt_sentences = ["hello world", "good morning"]
        tokenizer.build_vocabularies(src_sentences, tgt_sentences, min_freq=1)
        
        src_ids, tgt_ids = tokenizer.encode_pair("你好世界", "hello world")
        
        print(f"\n📝 Encode pair test:")
        print(f"   Source IDs: {src_ids}")
        print(f"   Target IDs: {tgt_ids}")
        
        self.assertIsInstance(src_ids, list)
        self.assertIsInstance(tgt_ids, list)
        self.assertGreater(len(src_ids), 2)  # At least BOS, token, EOS
        self.assertGreater(len(tgt_ids), 2)


class TestDataset(unittest.TestCase):
    """Tests for dataset and dataloader"""
    
    def test_translation_dataset(self):
        """Test TranslationDataset"""
        from transformer.data.dataset import TranslationDataset
        
        src_data = [[1, 2, 3], [4, 5, 6, 7]]
        tgt_input_data = [[10, 11], [13, 14, 15]]  # Target input (with BOS, without last token)
        tgt_output_data = [[11, 12], [14, 15, 16]]  # Target output (without BOS, with EOS)
        
        dataset = TranslationDataset(src_data, tgt_input_data, tgt_output_data)
        
        print(f"\n📦 Dataset test:")
        print(f"   Dataset size: {len(dataset)}")
        
        self.assertEqual(len(dataset), 2)
        
        src, tgt_in, tgt_out = dataset[0]
        self.assertEqual(src, [1, 2, 3])
        self.assertEqual(tgt_in, [10, 11])
        self.assertEqual(tgt_out, [11, 12])
    
    def test_collate_fn(self):
        """Test batch collation with padding"""
        from transformer.data.dataset import collate_fn
        
        batch = [
            ([1, 2, 3], [10, 11], [11, 12]),
            ([4, 5], [12, 13], [13, 14, 15]),
            ([6, 7, 8, 9], [15], [16, 17])
        ]
        
        src_batch, tgt_input_batch, tgt_output_batch = collate_fn(batch, src_pad_idx=0, tgt_pad_idx=0)
        
        print(f"\n📦 Collate function test:")
        print(f"   Source batch shape: {src_batch.shape}")
        print(f"   Target input batch shape: {tgt_input_batch.shape}")
        print(f"   Target output batch shape: {tgt_output_batch.shape}")
        print(f"   Source batch:\n{src_batch}")
        print(f"   Target input batch:\n{tgt_input_batch}")
        print(f"   Target output batch:\n{tgt_output_batch}")
        
        # Check shapes
        self.assertEqual(src_batch.shape[0], 3)  # Batch size
        self.assertEqual(src_batch.shape[1], 4)  # Max src length
        self.assertEqual(tgt_input_batch.shape[0], 3)
        self.assertEqual(tgt_output_batch.shape[0], 3)
        
        # Check padding
        self.assertEqual(src_batch[1, 2].item(), 0)  # Padded
        self.assertEqual(tgt_output_batch[0, 2].item(), 0)  # Padded


class TestDataPipeline(unittest.TestCase):
    """Tests for complete data pipeline"""
    
    def test_create_pipeline(self):
        """Test pipeline creation"""
        from transformer.data.dataset import create_pipeline
        
        print("\n🚀 Testing complete pipeline...")
        pipeline, tokenizer, config = create_pipeline(use_sample=True)
        
        self.assertIsNotNone(pipeline.train_dataset)
        self.assertIsNotNone(pipeline.val_dataset)
        self.assertIsNotNone(pipeline.test_dataset)
        
        print(f"   ✅ Train dataset size: {len(pipeline.train_dataset)}")
        print(f"   ✅ Val dataset size: {len(pipeline.val_dataset)}")
        print(f"   ✅ Test dataset size: {len(pipeline.test_dataset)}")
    
    def test_get_dataloader(self):
        """Test dataloader creation"""
        from transformer.data.dataset import create_pipeline
        
        pipeline, tokenizer, config = create_pipeline(use_sample=True)
        
        train_loader = pipeline.get_dataloader("train", shuffle=True)
        
        print(f"\n📦 DataLoader test:")
        
        batch_count = 0
        for src_batch, tgt_input_batch, tgt_output_batch in train_loader:
            print(f"   Batch {batch_count + 1}:")
            print(f"      Source shape: {src_batch.shape}")
            print(f"      Target input shape: {tgt_input_batch.shape}")
            print(f"      Target output shape: {tgt_output_batch.shape}")
            
            self.assertEqual(src_batch.dim(), 2)  # (batch, seq_len)
            self.assertEqual(tgt_input_batch.dim(), 2)
            self.assertEqual(tgt_output_batch.dim(), 2)
            
            batch_count += 1
            if batch_count >= 2:
                break
        
        print(f"   ✅ Successfully iterated through {batch_count} batches")


class TestModelIntegration(unittest.TestCase):
    """Tests for model integration with data pipeline"""
    
    def test_model_forward(self):
        """Test that model can process batches from pipeline"""
        from transformer.data.dataset import create_pipeline
        from transformer.models.transformer import Transformer
        
        print("\n🔧 Testing model integration...")
        
        # Create pipeline
        pipeline, tokenizer, config = create_pipeline(use_sample=True)
        
        # Create model
        model = Transformer(
            src_vocab_size=len(tokenizer.src_vocab),
            tgt_vocab_size=len(tokenizer.tgt_vocab),
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=2,  # Use fewer layers for testing
            max_len=config.max_len,
            dropout=config.dropout,
            src_pad_idx=tokenizer.src_vocab.pad_idx,
            tgt_pad_idx=tokenizer.tgt_vocab.pad_idx
        )
        
        print(f"   Model created with:")
        print(f"      Source vocab: {len(tokenizer.src_vocab)}")
        print(f"      Target vocab: {len(tokenizer.tgt_vocab)}")
        print(f"      Embed dim: {config.embed_dim}")
        print(f"      Num heads: {config.num_heads}")
        
        # Get a batch
        src_batch, tgt_input_batch, tgt_output_batch = pipeline.get_sample_batch()
        
        print(f"\n   Input shapes:")
        print(f"      Source: {src_batch.shape}")
        print(f"      Target input: {tgt_input_batch.shape}")
        print(f"      Target output: {tgt_output_batch.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(src_batch, tgt_input_batch)
        
        print(f"\n   Output shape: {output.shape}")
        print(f"   Expected: (batch_size, tgt_len, tgt_vocab_size)")
        print(f"   Actual: ({output.shape[0]}, {output.shape[1]}, {output.shape[2]})")
        
        # Verify output shape
        self.assertEqual(output.shape[0], src_batch.shape[0])  # Batch size
        self.assertEqual(output.shape[1], tgt_input_batch.shape[1])  # Target sequence length
        self.assertEqual(output.shape[2], len(tokenizer.tgt_vocab))  # Vocab size
        
        print(f"\n   ✅ Model forward pass successful!")
    
    def test_model_loss_computation(self):
        """Test that loss can be computed"""
        from transformer.data.dataset import create_pipeline
        from transformer.models.transformer import Transformer
        import torch.nn as nn
        
        print("\n📊 Testing loss computation...")
        
        # Create pipeline
        pipeline, tokenizer, config = create_pipeline(use_sample=True)
        
        # Create model
        model = Transformer(
            src_vocab_size=len(tokenizer.src_vocab),
            tgt_vocab_size=len(tokenizer.tgt_vocab),
            embed_dim=256,  # Smaller for testing
            num_heads=4,
            num_layers=2,
            max_len=config.max_len,
            dropout=0.1,
            src_pad_idx=tokenizer.src_vocab.pad_idx,
            tgt_pad_idx=tokenizer.tgt_vocab.pad_idx
        )
        
        # Get a batch
        src_batch, tgt_input_batch, tgt_output_batch = pipeline.get_sample_batch()
        
        print(f"   Source shape: {src_batch.shape}")
        print(f"   Target input shape: {tgt_input_batch.shape}")
        print(f"   Target output shape: {tgt_output_batch.shape}")
        
        # Forward pass
        model.train()
        logits = model(src_batch, tgt_input_batch)
        
        # Compute loss
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.tgt_vocab.pad_idx)
        
        # Reshape for loss: (batch * seq_len, vocab_size) vs (batch * seq_len)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = tgt_output_batch.reshape(-1)
        
        loss = criterion(logits_flat, targets_flat)
        
        print(f"   Loss: {loss.item():.4f}")
        
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        
        print(f"   ✅ Loss computation successful!")
    
    def test_model_inference(self):
        """Test inference (greedy decoding)"""
        from transformer.data.dataset import create_pipeline
        from transformer.models.transformer import Transformer
        
        print("\n🔮 Testing inference (greedy decode)...")
        
        # Create pipeline
        pipeline, tokenizer, config = create_pipeline(use_sample=True)
        
        # Create model
        model = Transformer(
            src_vocab_size=len(tokenizer.src_vocab),
            tgt_vocab_size=len(tokenizer.tgt_vocab),
            embed_dim=256,
            num_heads=4,
            num_layers=2,
            max_len=config.max_len,
            dropout=0.1,
            src_pad_idx=tokenizer.src_vocab.pad_idx,
            tgt_pad_idx=tokenizer.tgt_vocab.pad_idx
        )
        model.eval()
        
        # Sample source sentence
        test_sentence = "你好"
        src_tokens = tokenizer.tokenize_source(test_sentence)
        src_ids = tokenizer.src_vocab.encode(src_tokens)
        src_tensor = torch.tensor([src_ids])
        
        print(f"   Source: {test_sentence}")
        print(f"   Source tokens: {src_tokens}")
        print(f"   Source IDs: {src_ids}")
        
        # Greedy decoding
        max_len = 20
        bos_idx = tokenizer.tgt_vocab.bos_idx
        eos_idx = tokenizer.tgt_vocab.eos_idx
        
        tgt_ids = [bos_idx]
        
        with torch.no_grad():
            memory = model.encode(src_tensor)
            
            for _ in range(max_len):
                tgt_tensor = torch.tensor([tgt_ids])
                output = model.decode(tgt_tensor, memory, src_tensor)
                
                # Get last token prediction
                next_token_logits = output[0, -1, :]
                next_token = next_token_logits.argmax().item()
                
                tgt_ids.append(next_token)
                
                if next_token == eos_idx:
                    break
        
        # Decode result
        result_tokens = tokenizer.tgt_vocab.decode(tgt_ids)
        result_text = " ".join(result_tokens)
        
        print(f"   Output IDs: {tgt_ids}")
        print(f"   Output tokens: {result_tokens}")
        print(f"   Output text: {result_text}")
        
        self.assertIsInstance(result_text, str)
        print(f"\n   ✅ Inference test successful!")


def run_tests():
    """Run all tests with verbose output"""
    print("\n" + "🧪"*30)
    print("  RUNNING PIPELINE TESTS")
    print("🧪"*30)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests in order
    suite.addTests(loader.loadTestsFromTestCase(TestTokenizer))
    suite.addTests(loader.loadTestsFromTestCase(TestTranslationTokenizer))
    suite.addTests(loader.loadTestsFromTestCase(TestDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestDataPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestModelIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("   ✅ ALL TESTS PASSED!")
    else:
        print("   ❌ SOME TESTS FAILED")
        for test, traceback in result.failures + result.errors:
            print(f"\n   Failed: {test}")
            print(f"   {traceback}")
    
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
