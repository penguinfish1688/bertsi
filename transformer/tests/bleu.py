"""Evaluate the BLEU score by streaming 100 sentences from WMT19."""
from transformer.data.config import TranslationConfig
from transformer.data.tokenizer import TranslationTokenizer
from transformer.models.transformer import Transformer
from sacrebleu import corpus_bleu
import torch
import os


def greedy_decode(model, src_sentence, tokenizer, max_len, device):
    """
    Greedy decoding: generate translation one token at a time.
    
    Args:
        model: Trained Transformer model
        src_sentence: Chinese sentence (string)
        tokenizer: TranslationTokenizer instance
        max_len: Maximum generation length
        device: torch device
    
    Returns:
        translation: Translated English sentence
        output_tokens: List of generated token strings
    """
    model.eval()
    
    # Tokenize and encode source
    src_tokens = tokenizer.tokenize_source(src_sentence)
    
    if len(src_tokens) > max_len:
        src_tokens = src_tokens[:max_len]

    src_ids = tokenizer.src_vocab.encode(src_tokens)
    src_tensor = torch.tensor([src_ids], device=device)  # (1, src_len)
    
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
    
    # Get token strings from vocabulary
    result_tokens = tokenizer.tgt_vocab.decode(tgt_ids, remove_special=False)
    
    # For proper English text reconstruction:
    # Filter out special tokens and join
    filtered_tokens = [t for t in result_tokens if t not in ['<bos>', '<eos>', '<pad>', '<unk>']]
    
    # Use English tokenizer's detokenize method which properly handles BPE
    if filtered_tokens:
        decoded_text = tokenizer.english_tokenizer.detokenize(filtered_tokens)
    else:
        decoded_text = ""
    
    return decoded_text, filtered_tokens


def load_wmt19_stream(num_samples=100):
    """
    Stream load sentences from WMT19 dataset.
    
    Args:
        num_samples: Number of samples to load
        
    Returns:
        Tuple of (chinese_sentences, english_sentences)
    """
    print("\n" + "="*60)
    print(f"📥 STREAMING {num_samples} SAMPLES FROM WMT19")
    print("="*60)
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("⚠️  'datasets' library not found. Installing...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
        from datasets import load_dataset
    
    print("📦 Loading WMT19 Chinese-English dataset (streaming mode)...")
    
    # Load WMT19 dataset with streaming
    dataset = load_dataset("wmt/wmt19", "zh-en", split="train", streaming=True)
    
    chinese_sentences = []
    english_sentences = []
    
    for idx, item in enumerate(dataset):
        if idx >= num_samples:
            break
        
        translation = item['translation']
        en_text = translation['en'].strip()
        zh_text = translation['zh'].strip()
        
        # Skip empty sentences
        if en_text and zh_text:
            chinese_sentences.append(zh_text)
            english_sentences.append(en_text)
            
        if (idx + 1) % 20 == 0:
            print(f"   Loaded {idx + 1}/{num_samples} samples...")
    
    print(f"✅ Loaded {len(chinese_sentences)} valid sentence pairs")
    print("\n📊 Sample pairs:")
    for i in range(min(3, len(chinese_sentences))):
        print(f"   [{i+1}] ZH: {chinese_sentences[i][:50]}...")
        print(f"       EN: {english_sentences[i][:50]}...")
    print("="*60)
    
    return chinese_sentences, english_sentences


def evaluate_bleu(
    checkpoint_path,
    num_samples=100,
    config_path="transformer/config.yaml"
):
    """
    Evaluate BLEU score by streaming samples from WMT19.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of samples to evaluate (default: 100)
        config_path: Path to config YAML file
        
    Returns:
        BLEU score
    """
    print("\n" + "="*60)
    print("📥 LOADING MODEL")
    print("="*60)
    
    # Load config
    config = TranslationConfig.from_yaml(config_path)
    print(f"\n📋 Configuration loaded from: {config_path}")
    print(f"   - Model: {config.num_layers} layers, {config.embed_dim} dim, {config.num_heads} heads")
    print(f"   - Max length: {config.max_len}")
    
    # Create tokenizer (need to load vocabularies from cache)
    print(f"\n📚 Loading tokenizer and vocabularies...")
    tokenizer = TranslationTokenizer(config)
    
    # We need to load vocabularies from a cached dataset
    # Find any cached dataset to load vocabularies
    import pickle
    cache_files = [f for f in os.listdir(config.cache_dir) if f.endswith('.pkl')]
    if not cache_files:
        raise ValueError("No cached dataset found! Please train the model first to create vocabularies.")
    
    cache_path = os.path.join(config.cache_dir, cache_files[0])
    print(f"   Loading vocabularies from: {cache_path}")
    
    with open(cache_path, 'rb') as f:
        cached_data = pickle.load(f)
    
    tokenizer.src_vocab = cached_data['src_vocab']
    tokenizer.tgt_vocab = cached_data['tgt_vocab']
    
    print(f"   ✅ Vocabularies loaded: {len(tokenizer.src_vocab)} src tokens, {len(tokenizer.tgt_vocab)} tgt tokens")

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
    
    # Load checkpoint
    print(f"\n💾 Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded successfully!")
        print(f"   - Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   - Training loss: {checkpoint.get('train_loss', 'unknown'):.4f}")
        print(f"   - Validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ Model loaded successfully (old format)")
    
    model.eval()

    # Stream load WMT19 data
    print("\n" + "="*60)
    print(f"🔮 LOADING {num_samples} samples from wmt19")
    print("="*60)
    chinese_sentences, english_references = load_wmt19_stream(num_samples)
    
    # Run translations
    print("\n" + "="*60)
    print("🔮 TRANSLATING SENTENCES")
    print("="*60)
    references = []
    hypotheses = []
    
    for i, zh_sentence in enumerate(chinese_sentences):
        try:
            translation, _ = greedy_decode(model, zh_sentence, tokenizer, config.max_len, device=device)
        except Exception as e:
            print(f"❌ Error during translation of sample {i+1}: {e}")
            continue
        hypotheses.append(translation)
        references.append(english_references[i])
        
        if (i + 1) % 20 == 0:
            print(f"   Translated {i + 1}/{len(chinese_sentences)} sentences...")
        
        # Show first 3 examples
        if i < 3:
            print(f"\n[{i+1}]")
            print(f"   🇨🇳 Chinese:   {zh_sentence[:60]}...")
            print(f"   🇺🇸 Reference: {english_references[i][:60]}...")
            print(f"   🤖 Translated: {translation[:60]}...")
    
    print(f"\n✅ Translation complete!")
    print("="*60)

    # Compute BLEU score
    print("\n" + "="*60)
    print("📊 COMPUTING BLEU SCORE")
    print("="*60)
    
    bleu = corpus_bleu(hypotheses, [references])
    
    print(f"\n📈 BLEU Score: {bleu.score:.2f}")
    print(f"   Evaluated on {len(hypotheses)} samples from WMT19")
    print(f"   - Precision scores: {bleu.precisions}")
    print(f"   - Brevity penalty: {bleu.bp:.4f}")
    
    # Show some sample comparisons
    print(f"\n🔍 Sample translations:")
    for i in range(min(5, len(hypotheses))):
        print(f"\n[{i+1}]")
        print(f"   Chinese:    {chinese_sentences[i][:50]}...")
        print(f"   Reference:  {references[i][:50]}...")
        print(f"   Hypothesis: {hypotheses[i][:50]}...")
    
    print("\n" + "="*60)
    return bleu.score

