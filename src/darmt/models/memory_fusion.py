"""
Learned Memory Fusion Layer for integrating coprocessor latents with base memory.

This implements attention-based memory fusion instead of naive concatenation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryFusionLayer(nn.Module):
    """
    Learned fusion layer for integrating coprocessor latents with ARMT memory.
    
    Uses cross-attention where latent embeddings attend to current memory,
    then combines via learned projection to maintain memory coherence.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize memory fusion layer.
        
        Args:
            hidden_size: Dimension of memory and latent embeddings
            num_attention_heads: Number of attention heads for fusion
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        
        assert (
            self.head_dim * num_attention_heads == hidden_size
        ), "hidden_size must be divisible by num_attention_heads"
        
        # Cross-attention: latents (query) attend to memory (key/value)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norm
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Projection to combine memory and attended latents
        self.fusion_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        
        # Gate to control how much latent information to integrate
        self.integration_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        memory_tokens: torch.Tensor,
        latent_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse latent embeddings with memory tokens.
        
        Args:
            memory_tokens: Base memory from ARMT [batch, mem_len, hidden_size]
            latent_embeddings: Latents from coprocessor [batch, num_latents, hidden_size]
        
        Returns:
            Fused memory tokens [batch, mem_len, hidden_size]
        """
        batch_size, mem_len, hidden_size = memory_tokens.shape
        
        # Step 1: Cross-attention - latents attend to memory
        # This extracts relevant information from memory for each latent
        attended_latents, _ = self.cross_attention(
            query=latent_embeddings,  # [batch, num_latents, hidden]
            key=memory_tokens,        # [batch, mem_len, hidden]
            value=memory_tokens,      # [batch, mem_len, hidden]
        )
        attended_latents = self.norm1(attended_latents + latent_embeddings)
        
        # Step 2: Pool latents into single representation per batch
        # Use mean pooling to get fixed-size representation
        pooled_latents = attended_latents.mean(dim=1, keepdim=True)  # [batch, 1, hidden]
        pooled_latents = pooled_latents.expand(-1, mem_len, -1)  # [batch, mem_len, hidden]
        
        # Step 3: Compute integration gate
        # Controls how much of the latent information to integrate
        combined = torch.cat([memory_tokens, pooled_latents], dim=-1)  # [batch, mem_len, 2*hidden]
        gate = self.integration_gate(combined)  # [batch, mem_len, hidden]
        
        # Step 4: Fuse memory with gated latent information
        fusion_input = torch.cat([memory_tokens, pooled_latents], dim=-1)
        fusion_output = self.fusion_projection(fusion_input)  # [batch, mem_len, hidden]
        fusion_output = self.norm2(fusion_output)
        
        # Step 5: Apply gate to blend original and fused memory
        fused_memory = gate * fusion_output + (1 - gate) * memory_tokens
        
        return fused_memory
