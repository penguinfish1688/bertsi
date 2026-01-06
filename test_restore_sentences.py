"""Test the restore_dataset_sentences function."""
from transformer.tests.bleu import restore_dataset_sentences
from transformer.data.config import TranslationConfig
from transformer.data.dataset import create_pipeline

def main():
    print("\n" + "="*60)
    print("🧪 TESTING RESTORE_DATASET_SENTENCES")
    print("="*60)
    
    # Load config
    config_path = "transformer/config.yaml"
    config = TranslationConfig.from_yaml(config_path)
    config.download_new = False  # Use cached data - DO NOT download new data
    config.min_freq = 2  # Use the same min_freq as the cached dataset
    config.max_samples = 5000000  # Match the cached dataset size
    
    print(f"\n📋 Loading pipeline from cached WMT dataset...")
    print(f"    Expected cache: dataset_wmt_5000000_freq2_train0.8_val0.1.pkl")
    pipeline, tokenizer, _ = create_pipeline(use_sample=False, config=config)
    
    # Test restoring sentences
    num_samples = 20
    print(f"\n🔄 Restoring {num_samples} sentences...")
    chinese_sentences, english_sentences = restore_dataset_sentences(
        pipeline, tokenizer, num_samples
    )
    
    # Display results
    print("\n" + "="*60)
    print("📊 RESTORED SENTENCES")
    print("="*60)
    
    for i, (zh, en) in enumerate(zip(chinese_sentences, english_sentences)):
        print(f"\n[{i+1}]")
        print(f"   🇨🇳 Chinese:  {zh}")
        print(f"   🇺🇸 English:  {en}")
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETE")
    print("="*60)
    print(f"\nTotal sentences restored: {len(chinese_sentences)}")

if __name__ == "__main__":
    main()
