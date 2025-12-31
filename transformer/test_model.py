#!/usr/bin/env python
"""
Test a trained Chinese-English translation model

Usage examples:
    # Test with default sentences
    python -m transformer.test_model

    # Test with custom checkpoint
    python -m transformer.test_model --checkpoint checkpoints/checkpoint_epoch_50.pt

    # Test with custom sentences
    python -m transformer.test_model --sentences "你好世界" "我爱中国" "早上好"
    
    # Interactive mode
    python -m transformer.test_model --interactive
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer.train.train_translation import load_model_and_translate, greedy_decode
import torch


def interactive_translate(model, tokenizer, config, device):
    """
    Interactive translation mode - translate sentences as user types
    
    Args:
        model: Trained Transformer model
        tokenizer: TranslationTokenizer instance
        config: TranslationConfig instance
        device: torch device
    """
    print("\n" + "="*60)
    print("🌐 INTERACTIVE TRANSLATION MODE")
    print("="*60)
    print("Type Chinese sentences to translate (Ctrl+C or 'quit' to exit)")
    print("-" * 60)
    
    try:
        while True:
            # Get user input
            chinese_text = input("\n🇨🇳 Chinese: ").strip()
            
            # Check for exit
            if chinese_text.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not chinese_text:
                continue
            
            # Translate
            try:
                translation = greedy_decode(model, chinese_text, tokenizer, config.max_len, device=device)
                print(f"🇺🇸 English: {translation}")
            except Exception as e:
                print(f"❌ Error during translation: {e}")
                
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    import argparse
    from transformer.data.config import TranslationConfig
    
    parser = argparse.ArgumentParser(description='Test Chinese-English translation model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='transformer/config.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--use_sample', action='store_true',
                        help='Use sample data for vocabulary')
    parser.add_argument('--sentences', type=str, nargs='+',
                        help='Custom sentences to translate')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Load model and run translations
    model, tokenizer = load_model_and_translate(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        test_sentences=args.sentences,
        use_sample=args.use_sample
    )
    
    # If interactive mode requested and model loaded successfully
    if args.interactive and model is not None:
        config = TranslationConfig.from_yaml(args.config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        interactive_translate(model, tokenizer, config, device)
