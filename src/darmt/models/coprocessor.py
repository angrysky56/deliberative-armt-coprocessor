"""
Coprocessor module for deliberative reasoning.

The coprocessor processes the ARMT's memory state and generates
latent embeddings for augmenting the memory.
"""

import torch
import torch.nn as nn


class SimpleCoprocessor(nn.Module):
    """
    Simplified coprocessor for deliberative reasoning.

    The coprocessor:
    1. Takes the ARMT's KV-cache/memory state as input
    2. Processes it through transformer layers
    3. Generates latent embeddings to augment memory
    """

    def __init__(
        self, num_layers: int = 6, hidden_size: int = 768, num_heads: int = 12, dropout: float = 0.1
    ) -> None:
        """
        Initialize SimpleCoprocessor.

        Args:
            num_layers: Number of transformer layers
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size

        # Transformer encoder for processing memory
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm for output
        self.output_norm = nn.LayerNorm(hidden_size)

        print(f"[Init] SimpleCoprocessor created: {num_layers}L, {hidden_size}H")

    def forward(self, kv_cache: torch.Tensor, num_latents: int = 32) -> torch.Tensor:
        """
        Process KV-cache and generate latent embeddings.

        Args:
            kv_cache: KV-cache or memory state [batch, seq_len, hidden_size]
            num_latents: Number of latent embeddings to generate

        Returns:
            Latent embeddings [batch, num_latents, hidden_size]
        """
        # Process through transformer
        processed_memory = self.transformer(kv_cache)

        # Extract latent embeddings
        # Strategy: Take the last num_latents tokens
        # In a real implementation, you might use:
        # - Learnable query tokens
        # - Cross-attention
        # - More sophisticated selection mechanisms
        latent_embeddings = processed_memory[:, -num_latents:, :]

        # Normalize output
        latent_embeddings = self.output_norm(latent_embeddings)

        return latent_embeddings

    def generate_latent_embeddings(
        self, kv_cache: torch.Tensor, num_embeddings: int = 32
    ) -> torch.Tensor:
        """
        Generate latent embeddings (alias for forward pass).

        This method name matches the interface expected by the dual architecture.

        Args:
            kv_cache: KV-cache or memory state
            num_embeddings: Number of latent embeddings

        Returns:
            Latent embeddings
        """
        return self.forward(kv_cache, num_embeddings)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
