"""
Training Script for Chinese-English Translation

This script demonstrates how to train the Transformer model
on Chinese-English translation data.

Usage:
    python -m transformer.train.train_translation
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from tqdm import tqdm
import time
from accelerate import Accelerator

from transformer.data.dataset import create_pipeline
from transformer.models.transformer import Transformer


class DummyAccelerator:
    """Dummy accelerator for non-accelerate mode"""
    def __init__(self, device):
        self.device = device
        self.is_local_main_process = True
        self.is_main_process = True
        self.sync_gradients = True
    
    def backward(self, loss):
        loss.backward()
    
    def clip_grad_norm_(self, parameters, max_norm):
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    
    def unwrap_model(self, model):
        return model
    
    def wait_for_everyone(self):
        pass


def train_epoch(model, dataloader, optimizer, criterion, accelerator, epoch, max_len):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)
    
    for src_batch, tgt_input, tgt_output in progress_bar:
        # No need to manually move to device - accelerator handles this

        # Truncate long sentences
        src_len = src_batch.size(1)
        if src_len > max_len:
            src_batch = src_batch[:, :max_len]
        tgt_input_len = tgt_input.size(1)
        if tgt_input_len > max_len:
            tgt_input = tgt_input[:, :max_len]
            tgt_output = tgt_output[:, :max_len]

        # Forward pass (teacher forcing already applied in dataset)
        optimizer.zero_grad()
        logits = model(src_batch, tgt_input) # logits: (batch_size, tgt_len, tgt_vocab_size)

        # Compute loss
        logits_flat = logits.reshape(-1, logits.shape[-1]) # (batch_size * (tgt_len - 1), tgt_vocab_size)
        targets_flat = tgt_output.reshape(-1) # (batch_size * (tgt_len - 1))
        loss = criterion(logits_flat, targets_flat)
        
        # Backward pass with accelerator
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / num_batches


def evaluate(model, dataloader, criterion, accelerator, max_len):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for src_batch, tgt_input, tgt_output in dataloader:
            # No need to manually move to device - accelerator handles this
            # Truncate long sentences
            src_len = src_batch.size(1)
            if src_len > max_len:
                src_batch = src_batch[:, :max_len]
            tgt_input_len = tgt_input.size(1)
            if tgt_input_len > max_len:
                tgt_input = tgt_input[:, :max_len]
                tgt_output = tgt_output[:, :max_len]

            logits = model(src_batch, tgt_input)
            
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = tgt_output.reshape(-1)
            loss = criterion(logits_flat, targets_flat)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


def greedy_decode(model, src_sentence, tokenizer, max_len, device="cpu", unwrap=False):
    """Decode using greedy search
    
    Args:
        model: Transformer model (might be wrapped by Accelerator)
        src_sentence: Source sentence string
        tokenizer: Tokenizer instance
        max_len: Maximum decoding length
        device: Target device
        unwrap: Whether to unwrap model (if using Accelerator)
    """
    # Unwrap model if needed (for Accelerator compatibility)
    if unwrap and hasattr(model, 'module'):
        model = model.module
    
    model.eval()
    
    # Tokenize and encode source
    src_tokens = tokenizer.tokenize_source(src_sentence)
    
    if len(src_tokens) > max_len:
        src_tokens = src_tokens[:max_len]

    src_ids = tokenizer.src_vocab.encode(src_tokens)
    src_tensor = torch.tensor([src_ids], device=device) # (1, src_len)
    
    # Get encoder output
    with torch.no_grad():
        memory = model.encode(src_tensor)
    
    # Start decoding
    bos_idx = tokenizer.tgt_vocab.bos_idx
    eos_idx = tokenizer.tgt_vocab.eos_idx
    
    tgt_ids = [bos_idx]
    
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids], device=device)
            output = model.decode(tgt_tensor, memory, src_tensor)
            
            next_token_logits = output[0, -1, :]
            next_token = next_token_logits.argmax().item()
            
            tgt_ids.append(next_token)
            
            if next_token == eos_idx:
                break
    
    # Decode to text
    result_tokens = tokenizer.tgt_vocab.decode(tgt_ids)
    return " ".join(result_tokens)


def load_model_and_translate(checkpoint_path, config_path="transformer/config.yaml", 
                             test_sentences=None, use_sample=True):
    """
    Load a saved model checkpoint and run translation examples
    
    Args:
        checkpoint_path: Path to the checkpoint file (e.g., 'checkpoints/best_model.pt')
        config_path: Path to the YAML config file
        test_sentences: List of Chinese sentences to translate (default: predefined examples)
        use_sample: Whether to use sample data for loading vocabularies
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print("\n" + "="*60)
    print("📥 LOADING MODEL AND RUNNING TRANSLATIONS")
    print("="*60)
    
    # Load config from YAML
    from transformer.data.config import TranslationConfig
    config = TranslationConfig.from_yaml(config_path)
    
    print(f"\n📋 Configuration loaded from: {config_path}")
    print(f"   - Model: {config.num_layers} layers, {config.embed_dim} dim, {config.num_heads} heads")
    print(f"   - Max length: {config.max_len}")
    
    # Create pipeline to get tokenizer and vocabularies
    # We need this to load the vocabulary for encoding/decoding
    print(f"\n📚 Loading vocabularies...")
    pipeline, tokenizer, _ = create_pipeline(use_sample=use_sample, config=config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📱 Device: {device}")
    
    # Create model with same architecture
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
    print(f"\n💾 Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
        return None, None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded successfully!")
        print(f"   - Trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"   - Training loss: {checkpoint.get('train_loss', 'unknown'):.4f}")
        print(f"   - Validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    else:
        # Old format - just model state dict
        model.load_state_dict(checkpoint)
        print(f"✅ Model loaded successfully (old format)")
    
    model.eval()
    
    # Default test sentences if none provided
    if test_sentences is None:
        test_sentences = [
            "你好",
            "谢谢你",
            "我喜欢学习中文",
            "今天天气很好",
            "我住在北京",
            "这本书很有趣",
            "明天见",
            "祝你好运"
        ]
    
    # Run translations
    print("\n" + "="*60)
    print("🔮 TRANSLATION EXAMPLES")
    print("="*60)
    
    for i, src in enumerate(test_sentences, 1):
        translation = greedy_decode(model, src, tokenizer, config.max_len, device=device)
        print(f"\n[{i}]")
        print(f"   🇨🇳 Chinese: {src}")
        print(f"   🇺🇸 English: {translation}")
    
    print("\n" + "="*60)
    print("✅ Translation complete!")
    print("="*60)
    
    return model, tokenizer


def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the checkpoint directory
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Path to the latest checkpoint file, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Get all .pt files in the checkpoint directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    if not checkpoint_files:
        return None
    
    # Get full paths and modification times
    checkpoint_paths = [os.path.join(checkpoint_dir, f) for f in checkpoint_files]
    
    # Sort by modification time (most recent first)
    latest_checkpoint = max(checkpoint_paths, key=os.path.getmtime)
    
    return latest_checkpoint


def train(config_path="transformer/config.yaml", use_sample=True, resume_from=None, use_accelerate=True, mixed_precision="no"):
    """Main training function
    
    Args:
        config_path: Path to YAML config file
        use_sample: Whether to use sample dataset
        resume_from: Path to checkpoint to resume training from (optional). 
                    Use 'latest' to automatically load the most recent checkpoint from the checkpoints directory.
        use_accelerate: Whether to use Accelerate for distributed training (default: True)
        mixed_precision: Mixed precision mode: 'no', 'fp16', 'bf16' (default: 'no')
    """
    
    print("\n" + "="*60)
    print("🎓 CHINESE-ENGLISH TRANSLATION TRAINING")
    print("="*60)
    
    # Initialize Accelerator
    if use_accelerate:
        accelerator = Accelerator(mixed_precision=mixed_precision)
        device = accelerator.device
        print(f"\n🚀 Using Accelerate library")
        print(f"   ✅ Distributed type: {accelerator.distributed_type}")
        print(f"   ✅ Number of processes: {accelerator.num_processes}")
        print(f"   ✅ Mixed precision: {mixed_precision}")
        print(f"   📱 Device: {device}")
    else:
        accelerator = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n📱 Device: {device} (not using Accelerate)")
    
    # Load config from YAML
    from transformer.data.config import TranslationConfig
    config = TranslationConfig.from_yaml(config_path)
    
    print(f"\n📋 Configuration loaded from: {config_path}")
    print(f"   - Epochs: {config.num_epochs}")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Learning rate: {config.learning_rate}")
    print(f"   - Checkpoint frequency: {config.checkpoint_frequency}")
    print(f"   - Download new data: {config.download_new}")
    if resume_from:
        print(f"   - Resume from: {resume_from}")
    
    # Create pipeline
    pipeline, tokenizer, _ = create_pipeline(use_sample=use_sample, config=config)
    
    # Create model (don't move to device yet if using Accelerator)
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
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {num_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.tgt_vocab.pad_idx)
    
    # DataLoaders
    train_loader = pipeline.get_dataloader("train", shuffle=True)
    val_loader = pipeline.get_dataloader("val", shuffle=False)
    
    # Prepare with Accelerator (or move to device manually)
    if use_accelerate:
        assert accelerator is not None, "Accelerator must be initialized"
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
        print(f"\n✅ Model, optimizer, and dataloaders prepared with Accelerator")
    else:
        model = model.to(device)
        
        # Apply FP16/BF16 if requested (only on CUDA)
        if mixed_precision == "fp16":
            if device.type == "cuda":
                model = model.half()
                print(f"\n✅ Model converted to FP16")
            else:
                print(f"\n⚠️  FP16 is only supported on CUDA devices, ignoring mixed_precision setting")
                mixed_precision = "no"
        elif mixed_precision == "bf16":
            if device.type == "cuda":
                model = model.bfloat16()
                print(f"\n✅ Model converted to BF16")
            else:
                print(f"\n⚠️  BF16 is only supported on CUDA devices, ignoring mixed_precision setting")
                mixed_precision = "no"
        
        # Create dummy accelerator for compatibility
        accelerator = DummyAccelerator(device)
        
        if mixed_precision != "no":
            print(f"   📱 Mixed precision: {mixed_precision}")
    
    assert accelerator is not None, "Either Accelerator or DummyAccelerator must be initialized"

    # Load checkpoint if resuming
    start_epoch = 1
    best_val_loss = float('inf')
    
    if resume_from:
        # Handle 'latest' keyword for resume_from
        if resume_from == "latest":
            latest_checkpoint = get_latest_checkpoint(config.checkpoint_dir)
            if latest_checkpoint:
                resume_from = latest_checkpoint
                print(f"\n🔍 'latest' keyword detected - found: {resume_from}")
            else:
                print(f"\n⚠️  'latest' keyword used but no checkpoints found in {config.checkpoint_dir}")
                print(f"   Starting training from scratch...")
                resume_from = ""

        if os.path.exists(resume_from):
            print(f"\n📥 Loading checkpoint from: {resume_from}")
            checkpoint = torch.load(resume_from, map_location="cpu")

            # Get unwrapped model for loading state dict
            unwrapped_model = accelerator.unwrap_model(model)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"   ✅ Model state loaded")
            else:
                unwrapped_model.load_state_dict(checkpoint)
                print(f"   ✅ Model state loaded (old format)")
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"   ✅ Optimizer state loaded")
            
            # Get start epoch and best loss
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"   ✅ Resuming from epoch {start_epoch}")
            
            if 'val_loss' in checkpoint:
                best_val_loss = checkpoint['val_loss']
                print(f"   ✅ Previous best val_loss: {best_val_loss:.4f}")
            
            print(f"   📊 Previous training loss: {checkpoint.get('train_loss', 'N/A')}")
        else:
            print(f"\n⚠️  Warning: Checkpoint file not found at {resume_from}")
            print(f"   Starting training from scratch...")
    
    # Training loop
    if resume_from and start_epoch > 1:
        print(f"\n🚀 Resuming training from epoch {start_epoch} to {config.num_epochs}...")
    else:
        print(f"\n🚀 Starting training for {config.num_epochs} epochs...")
    print(f"   Checkpoints will be saved every {config.checkpoint_frequency} epochs")
    print("-" * 60)
    
    for epoch in range(start_epoch, config.num_epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, accelerator, epoch, config.max_len)
        
        # Validate
        val_loss = evaluate(model, val_loader, criterion, accelerator, config.max_len)
        
        elapsed = time.time() - start_time
        
        # Save best model (only on main process)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config.checkpoint_dir, "best_model.pt")
            
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, best_model_path)
                print(f"   💾 Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every N epochs (only on main process)
        if epoch % config.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, checkpoint_path)
                print(f"   💾 Saved checkpoint at epoch {epoch}")
        
        print(f"   Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, time={elapsed:.1f}s")
    
    print("-" * 60)
    print(f"✅ Training complete! Best val_loss: {best_val_loss:.4f}")
    
    # Test some translations
    print("\n" + "="*60)
    print("🔮 SAMPLE TRANSLATIONS")
    print("="*60)
    
    test_sentences = ["你好", "谢谢你", "我喜欢学习中文"]
    
    # Get unwrapped model and correct device for inference
    unwrapped_model = accelerator.unwrap_model(model)
    inference_device = accelerator.device if use_accelerate else device
    
    for src in test_sentences:
        translation = greedy_decode(unwrapped_model, src, tokenizer, config.max_len, device=inference_device)
        print(f"   🇨🇳 {src}")
        print(f"   🇺🇸 {translation}")
        print()
    
    return model, tokenizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or test Chinese-English translation model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: train a new model or test existing model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to checkpoint file (for test mode or resume training)')
    parser.add_argument('--config', type=str, default='transformer/config.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--use_sample', action='store_true',
                        help='Use sample data instead of full dataset')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint (use with --checkpoint)')
    parser.add_argument('--no_accelerate', action='store_true',
                        help='Disable Accelerate (use single GPU only)')
    parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['no', 'fp16', 'bf16'],
                        help='Mixed precision mode: no, fp16, or bf16')
    parser.add_argument('--sentences', type=str, nargs='+',
                        help='Custom sentences to translate (for test mode)')

    args = parser.parse_args()
    
    if args.mode == 'train':
        # Train using YAML config
        resume_checkpoint = args.checkpoint if args.resume else None
        model, tokenizer = train(
            config_path=args.config,
            use_sample=args.use_sample,
            resume_from=resume_checkpoint,
            use_accelerate=not args.no_accelerate,
            mixed_precision=args.mixed_precision
        )
    else:
        # Test mode - load and translate
        model, tokenizer = load_model_and_translate(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            test_sentences=args.sentences,
            use_sample=args.use_sample
        )
