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

from transformer.data.dataset import create_pipeline
from transformer.models.transformer import Transformer


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for src_batch, tgt_batch in progress_bar:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        
        # Teacher forcing: use all but last target token as input
        tgt_input = tgt_batch[:, :-1] # (batch_size, tgt_len - 1)
        tgt_output = tgt_batch[:, 1:] # (batch_size, tgt_len - 1)

        # Forward pass
        optimizer.zero_grad()
        logits = model(src_batch, tgt_input) # logits: (batch_size, tgt_len - 1, tgt_vocab_size)

        # Compute loss
        logits_flat = logits.reshape(-1, logits.shape[-1]) # (batch_size * (tgt_len - 1), tgt_vocab_size)
        targets_flat = tgt_output.reshape(-1) # (batch_size * (tgt_len - 1))
        loss = criterion(logits_flat, targets_flat)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]
            
            logits = model(src_batch, tgt_input)
            
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = tgt_output.reshape(-1)
            loss = criterion(logits_flat, targets_flat)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


def greedy_decode(model, src_sentence, tokenizer, max_len=50, device="cpu"):
    """Decode using greedy search"""
    model.eval()
    
    # Tokenize and encode source
    src_tokens = tokenizer.tokenize_source(src_sentence)
    src_ids = tokenizer.src_vocab.encode(src_tokens)
    src_tensor = torch.tensor([src_ids]).to(device)
    
    # Get encoder output
    with torch.no_grad():
        memory = model.encode(src_tensor)
    
    # Start decoding
    bos_idx = tokenizer.tgt_vocab.bos_idx
    eos_idx = tokenizer.tgt_vocab.eos_idx
    
    tgt_ids = [bos_idx]
    
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids]).to(device)
            output = model.decode(tgt_tensor, memory, src_tensor)
            
            next_token_logits = output[0, -1, :]
            next_token = next_token_logits.argmax().item()
            
            tgt_ids.append(next_token)
            
            if next_token == eos_idx:
                break
    
    # Decode to text
    result_tokens = tokenizer.tgt_vocab.decode(tgt_ids)
    return " ".join(result_tokens)


def train(config_path="transformer/config.yaml", use_sample=True):
    """Main training function"""
    
    print("\n" + "="*60)
    print("🎓 CHINESE-ENGLISH TRANSLATION TRAINING")
    print("="*60)
    
    # Load config from YAML
    from transformer.data.config import TranslationConfig
    config = TranslationConfig.from_yaml(config_path)
    
    print(f"\n📋 Configuration loaded from: {config_path}")
    print(f"   - Epochs: {config.num_epochs}")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Learning rate: {config.learning_rate}")
    print(f"   - Checkpoint frequency: {config.checkpoint_frequency}")
    print(f"   - Use cached dataset: {config.use_cached_dataset}")
    print(f"   - Download new data: {config.download_new}")
    
    # Create pipeline
    pipeline, tokenizer, _ = create_pipeline(use_sample=use_sample, config=config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📱 Device: {device}")
    
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
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {num_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.tgt_vocab.pad_idx)
    
    # DataLoaders
    train_loader = pipeline.get_dataloader("train", shuffle=True)
    val_loader = pipeline.get_dataloader("val", shuffle=False)
    
    # Training loop
    print(f"\n🚀 Starting training for {config.num_epochs} epochs...")
    print(f"   Checkpoints will be saved every {config.checkpoint_frequency} epochs")
    print("-" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, config.num_epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)
        
        elapsed = time.time() - start_time
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config.checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, best_model_path)
            print(f"   💾 Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every N epochs
        if epoch % config.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
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
    print("🔮 SAMPLE TRANSLATIONS (untrained model produces random output)")
    print("="*60)
    
    test_sentences = ["你好", "谢谢你", "我喜欢学习中文"]
    
    for src in test_sentences:
        translation = greedy_decode(model, src, tokenizer, device=device)
        print(f"   🇨🇳 {src}")
        print(f"   🇺🇸 {translation}")
        print()
    
    return model, tokenizer


if __name__ == "__main__":
    # Train using YAML config
    model, tokenizer = train(
        config_path="transformer/config.yaml",
        use_sample=False
    )
