import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.models.attention import MultiheadAttention
from transformer.models.position_encode import TransformerEmbedding

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """forward
        Args:
            src: (batch_size, seq_len, embed_dim)
            src_mask: attention mask (batch, seq_len, seq_len)
        """
        # Self-attention (Pre-LN)
        _src = src # _src for residual connection
        src = self.norm1(src)
        attn_output = self.self_attn(query=src, key=src, value=src, mask=src_mask)
        src = self.dropout1(attn_output)
        src = src + _src

        # Feedforward (Pre-LN)
        _src = src
        src = self.norm2(src)
        ff_output = self.linear2(F.relu(self.linear1(src)))
        src = self.dropout2(ff_output)
        src = src + _src

        return src

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout)
        self.cross_attn = MultiheadAttention(embed_dim, num_heads, dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm_mem = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """forward
        Args:
            tgt: (batch_size, tgt_seq_len, embed_dim)
            memory: (batch_size, src_seq_len, embed_dim)
            tgt_mask: target attention mask (batch, tgt_seq_len, tgt_seq_len)
            memory_mask: memory attention mask (batch, tgt_seq_len, src_seq_len)
        """
        # Self-attention
        _tgt = tgt
        tgt = self.norm1(tgt)
        attn_output = self.self_attn(query=tgt, key=tgt, value=tgt, mask=tgt_mask)
        tgt = self.dropout1(attn_output)
        tgt = tgt + _tgt

        # Cross-attention with encoder output
        _tgt = tgt
        tgt = self.norm2(tgt)
        memory = self.norm_mem(memory)
        attn_output = self.cross_attn(query=tgt, key=memory, value=memory, mask=memory_mask)
        tgt = self.dropout2(attn_output)
        tgt = tgt + _tgt

        # Feedforward (Pre-LN)
        _tgt = tgt
        tgt = self.norm3(tgt)
        ff_output = self.linear2(F.relu(self.linear1(tgt)))
        tgt = self.dropout3(ff_output)
        tgt = tgt + _tgt

        return tgt

class Encoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
    
    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src
    
class Decoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return tgt

class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks (e.g., translation)
    
    Args:
        src_vocab_size: Source vocabulary size (e.g., Chinese)
        tgt_vocab_size: Target vocabulary size (e.g., English)
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        num_layers: Number of encoder/decoder layers
        max_len: Maximum sequence length
        dropout: Dropout rate
        src_pad_idx: Source padding token index
        tgt_pad_idx: Target padding token index
    Output:
        Logits over target vocabulary for each position (B)
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embed_dim: int = 512, 
                 num_heads: int = 8, num_layers: int = 6, max_len: int = 256,
                 dropout: float = 0.1, src_pad_idx: int = 0, tgt_pad_idx: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.max_len = max_len
        
        # Embedding layers for source and target
        self.src_embedding = TransformerEmbedding(src_vocab_size, embed_dim, max_len, dropout, src_pad_idx)
        self.tgt_embedding = TransformerEmbedding(tgt_vocab_size, embed_dim, max_len, dropout, tgt_pad_idx)
        
        # Encoder and Decoder
        self.encoder = Encoder(num_layers=num_layers, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.decoder = Decoder(num_layers=num_layers, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(embed_dim, tgt_vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Create source mask for padding tokens
        Args:
            src: Source token IDs (batch_size, src_len)
        Returns:
            Mask of shape (batch_size, 1, 1, src_len)
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Create target mask combining padding mask and causal mask
        Args:
            tgt: Target token IDs (batch_size, tgt_len)
        Returns:
            Mask of shape (batch_size, 1, tgt_len, tgt_len)
        """
        batch_size, tgt_len = tgt.shape
        
        # Padding mask: (batch_size, 1, 1, tgt_len)
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Causal mask: prevent attending to future tokens
        # Shape: (1, 1, tgt_len, tgt_len)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Combine masks
        tgt_mask = tgt_pad_mask & causal_mask
        return tgt_mask
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            src: Source token IDs (batch_size, src_len)
            tgt: Target token IDs (batch_size, tgt_len)
        Returns:
            Logits of shape (batch_size, tgt_len, tgt_vocab_size)
        """
        # Create masks
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # Embed source and target
        src_emb = self.src_embedding(src)  # (batch_size, src_len, embed_dim)
        tgt_emb = self.tgt_embedding(tgt)  # (batch_size, tgt_len, embed_dim)
        
        # Encode source
        memory = self.encoder(src_emb, src_mask)
        
        # Decode target with attention to memory
        output = self.decoder(tgt_emb, memory, tgt_mask, src_mask)
        
        # Project to vocabulary
        logits = self.output_projection(output)

        return logits
    
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """Encode source sequence only (for inference)"""
        src_mask = self.make_src_mask(src)
        src_emb = self.src_embedding(src)
        return self.encoder(src_emb, src_mask)
    
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """Decode target sequence given memory (for inference)"""
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        tgt_emb = self.tgt_embedding(tgt)
        output = self.decoder(tgt_emb, memory, tgt_mask, src_mask)
        return self.output_projection(output)