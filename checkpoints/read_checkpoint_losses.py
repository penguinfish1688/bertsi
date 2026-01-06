"""
Script to read loss values from checkpoint files.

This script scans a directory for PyTorch checkpoint files (.pt, .pth)
and extracts training and validation loss information.
"""
import os
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def read_checkpoint_info(checkpoint_path: str) -> Dict:
    """
    Read information from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary containing checkpoint information
    """
    try:
        # Load checkpoint on CPU to avoid GPU memory issues
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'path': checkpoint_path,
            'filename': os.path.basename(checkpoint_path),
            'epoch': checkpoint.get('epoch', 'N/A'),
            'train_loss': checkpoint.get('train_loss', 'N/A'),
            'val_loss': checkpoint.get('val_loss', 'N/A'),
        }
        
        # Try to get file modification time
        info['modified'] = os.path.getmtime(checkpoint_path)
        
        return info
    except Exception as e:
        return {
            'path': checkpoint_path,
            'filename': os.path.basename(checkpoint_path),
            'error': str(e)
        }


def find_checkpoints(directory: str, extensions: List[str] = ['.pt', '.pth']) -> List[str]:
    """
    Find all checkpoint files in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to look for
        
    Returns:
        List of checkpoint file paths
    """
    checkpoint_files = []
    
    if not os.path.exists(directory):
        print(f"⚠️  Directory not found: {directory}")
        return checkpoint_files
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                checkpoint_files.append(os.path.join(root, file))
    
    return sorted(checkpoint_files)


def display_checkpoint_summary(checkpoints_info: List[Dict], sort_by: str = 'epoch'):
    """
    Display a summary table of checkpoint information.
    
    Args:
        checkpoints_info: List of checkpoint information dictionaries
        sort_by: Field to sort by ('epoch', 'train_loss', 'val_loss', 'modified')
    """
    if not checkpoints_info:
        print("No checkpoints found.")
        return
    
    # Filter out checkpoints with errors
    valid_checkpoints = [c for c in checkpoints_info if 'error' not in c]
    error_checkpoints = [c for c in checkpoints_info if 'error' in c]
    
    if valid_checkpoints:
        # Sort checkpoints
        if sort_by in ['epoch', 'train_loss', 'val_loss', 'modified']:
            # Handle 'N/A' values
            def sort_key(c):
                val = c.get(sort_by, float('inf'))
                if val == 'N/A':
                    return float('inf')
                return val
            valid_checkpoints = sorted(valid_checkpoints, key=sort_key)
        
        print("\n" + "="*100)
        print("📊 CHECKPOINT LOSS SUMMARY")
        print("="*100)
        print(f"{'Filename':<40} {'Epoch':>8} {'Train Loss':>12} {'Val Loss':>12}")
        print("-"*100)
        
        for info in valid_checkpoints:
            epoch = info['epoch'] if info['epoch'] != 'N/A' else 'N/A'
            train_loss = f"{info['train_loss']:.4f}" if info['train_loss'] != 'N/A' else 'N/A'
            val_loss = f"{info['val_loss']:.4f}" if info['val_loss'] != 'N/A' else 'N/A'
            
            print(f"{info['filename']:<40} {epoch:>8} {train_loss:>12} {val_loss:>12}")
        
        print("="*100)
        
        # Summary statistics
        valid_train_losses = [c['train_loss'] for c in valid_checkpoints if c['train_loss'] != 'N/A']
        valid_val_losses = [c['val_loss'] for c in valid_checkpoints if c['val_loss'] != 'N/A']
        
        if valid_train_losses:
            print(f"\n📈 Training Loss Statistics:")
            print(f"   Best:  {min(valid_train_losses):.4f}")
            print(f"   Worst: {max(valid_train_losses):.4f}")
            print(f"   Mean:  {sum(valid_train_losses)/len(valid_train_losses):.4f}")
        
        if valid_val_losses:
            print(f"\n📉 Validation Loss Statistics:")
            print(f"   Best:  {min(valid_val_losses):.4f}")
            print(f"   Worst: {max(valid_val_losses):.4f}")
            print(f"   Mean:  {sum(valid_val_losses)/len(valid_val_losses):.4f}")
        
        # Find best models
        if valid_val_losses:
            best_val_checkpoint = min(valid_checkpoints, key=lambda c: c['val_loss'] if c['val_loss'] != 'N/A' else float('inf'))
            print(f"\n🏆 Best Model (by validation loss):")
            print(f"   File: {best_val_checkpoint['filename']}")
            print(f"   Epoch: {best_val_checkpoint['epoch']}")
            print(f"   Val Loss: {best_val_checkpoint['val_loss']:.4f}")
    
    if error_checkpoints:
        print(f"\n❌ Failed to read {len(error_checkpoints)} checkpoint(s):")
        for info in error_checkpoints:
            print(f"   {info['filename']}: {info['error']}")


def main():
    parser = argparse.ArgumentParser(description='Read loss values from checkpoint files')
    parser.add_argument('--dir', '-d', type=str, default='checkpoints',
                      help='Directory containing checkpoint files (default: checkpoints)')
    parser.add_argument('--sort', '-s', type=str, default='epoch',
                      choices=['epoch', 'train_loss', 'val_loss', 'modified'],
                      help='Sort checkpoints by field (default: epoch)')
    parser.add_argument('--extensions', '-e', type=str, nargs='+', default=['.pt', '.pth'],
                      help='File extensions to search for (default: .pt .pth)')
    parser.add_argument('--recursive', '-r', action='store_true',
                      help='Search subdirectories recursively')
    
    args = parser.parse_args()
    
    # Find checkpoint files
    print(f"\n🔍 Searching for checkpoints in: {args.dir}")
    checkpoint_files = find_checkpoints(args.dir, args.extensions)
    
    if not checkpoint_files:
        print(f"No checkpoint files found with extensions: {args.extensions}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint file(s)")
    
    # Read checkpoint information
    checkpoints_info = []
    for ckpt_path in checkpoint_files:
        info = read_checkpoint_info(ckpt_path)
        checkpoints_info.append(info)
    
    # Display summary
    display_checkpoint_summary(checkpoints_info, sort_by=args.sort)


if __name__ == "__main__":
    main()
