"""
Configuration for Chinese-English Translation
"""
from dataclasses import dataclass
from typing import Optional
import torch
import yaml
import os


@dataclass
class TranslationConfig:
    """Configuration for translation model and training"""
    
    # Model architecture
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 8
    max_len: int = 512
    dropout: float = 0.1
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 4000
    checkpoint_frequency: int = 5  # Save checkpoint every N epochs
    
    # Data
    min_freq: int = 2  # Minimum word frequency to include in vocabulary
    max_samples: Optional[int] = None  # Limit samples for testing (None = all)
    download_new: bool = False  # If true, download and replace dataset; if false, use existing if found
    cache_dir: str = "data/translation/cache"
    
    # Tokenization
    src_lang: str = "zh"  # Chinese
    tgt_lang: str = "en"  # English
    
    # Special tokens
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"  # Beginning of sentence
    eos_token: str = "<eos>"  # End of sentence
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    data_dir: str = "data/translation"
    checkpoint_dir: str = "checkpoints"
    
    @classmethod
    def from_yaml(cls, yaml_path: str = "transformer/config.yaml"):
        """
        Load configuration from YAML file
        
        Args:
            yaml_path: Path to YAML configuration file
        
        Returns:
            TranslationConfig instance
        """
        if not os.path.exists(yaml_path):
            print(f"⚠️  Config file not found at {yaml_path}, using defaults")
            return cls()
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Flatten nested structure
        flat_config = {}
        
        if 'model' in config_dict:
            flat_config.update(config_dict['model'])
        
        if 'training' in config_dict:
            flat_config.update(config_dict['training'])
        
        if 'data' in config_dict:
            data_config = config_dict['data']
            flat_config['min_freq'] = data_config.get('min_freq', 2)
            flat_config['max_samples'] = data_config.get('max_samples')
            flat_config['download_new'] = data_config.get('download_new', False)
            flat_config['cache_dir'] = data_config.get('cache_dir', 'data/translation/cache')
        
        if 'tokenization' in config_dict:
            flat_config.update(config_dict['tokenization'])
        
        if 'paths' in config_dict:
            flat_config['data_dir'] = config_dict['paths'].get('data_dir', 'data/translation')
            flat_config['checkpoint_dir'] = config_dict['paths'].get('checkpoint_dir', 'checkpoints')
        
        # Create instance with loaded values
        return cls(**{k: v for k, v in flat_config.items() if v is not None})
    
    def __post_init__(self):
        """Ensure directories exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
