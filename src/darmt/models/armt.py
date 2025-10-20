"""
Simple ARMT (Associative Recurrent Memory Transformer) implementation.

Based on: Rodkin et al., "Associative Recurrent Memory Transformer" (2024)
https://arxiv.org/abs/2407.04841

This is a simplified implementation for research purposes.
"""

import torch
import torch.nn as nn
from typing import Any

from darmt.utils.memory import MemoryState


class SimpleARMT(nn.Module):
    """
    Simplified ARMT implementation using standard Transformer layers.

    The full ARMT includes:
    - Segment-level recurrence
    - Associative memory at each layer
    - Memory tokens that persist across segments

    This simplified version uses:
    - Standard Transformer Encoder layers
    - Learnable memory tokens
    - Basic memory passing between segments
    """

    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 768,
        num_mem_tokens: int = 32,
        vocab_size: int = 32000,
        num_heads: int = 12,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize SimpleARMT.

        Args:
            num_layers: Number of transformer layers
            hidden_size: Hidden dimension size
            num_mem_tokens: Number of memory tokens
            vocab_size: Size of vocabulary
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_mem_tokens = num_mem_tokens

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Learnable memory tokens (initial state)
        self.mem_tokens = nn.Parameter(torch.randn(1, num_mem_tokens, hidden_size))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Language modeling head
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        # Layer norm for memory tokens
        self.memory_norm = nn.LayerNorm(hidden_size)

        print(f"[Init] SimpleARMT created: {num_layers}L, {hidden_size}H, {num_mem_tokens}M")

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_state: MemoryState | None = None,
        return_hidden_states: bool = False,
        return_attention_weights: bool = False,
    ) -> dict[str, Any]:
        """
        Forward pass through ARMT.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            memory_state: Previous memory state (or None for initial)
            return_hidden_states: Whether to return hidden states
            return_attention_weights: Whether to return attention weights

        Returns:
            Dictionary containing:
                - logits: Output logits [batch, seq_len, vocab_size]
                - memory_state: Updated memory state
                - hidden_states: (optional) Hidden states from each layer
                - attention_weights: (optional) Attention weights
        """
        batch_size, seq_len = input_ids.shape

        # Get input embeddings
        input_embeds = self.embedding(input_ids)

        # Get or create memory tokens
        if memory_state is None:
            mem_tokens_batch = self.mem_tokens.expand(batch_size, -1, -1)
        else:
            mem_tokens_batch = memory_state["memory_tokens"]
            # If memory has been augmented with latents, only use the original memory tokens
            if mem_tokens_batch.shape[1] > self.num_mem_tokens:
                mem_tokens_batch = mem_tokens_batch[:, : self.num_mem_tokens, :]
            mem_tokens_batch = self.memory_norm(mem_tokens_batch)

        # Combine memory tokens with input embeddings
        # Memory tokens come first, then input tokens
        full_input = torch.cat([mem_tokens_batch, input_embeds], dim=1)

        # Process through transformer
        output_embeds = self.transformer(full_input)

        # Split output into memory and sequence tokens
        new_memory = output_embeds[:, : self.num_mem_tokens, :]
        token_embeds = output_embeds[:, self.num_mem_tokens :, :]

        # Generate logits
        logits = self.lm_head(token_embeds)

        # Prepare output dictionary
        output = {
            "logits": logits,
            "memory_state": {"memory_tokens": new_memory, "associative_memory": None},
        }

        if return_hidden_states:
            output["hidden_states"] = token_embeds

        if return_attention_weights:
            # Note: TransformerEncoder doesn't expose attention weights by default
            # In production, you'd need to modify the encoder or use a custom implementation
            output["attention_weights"] = None

        return output

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extract_kv_cache(self, memory_state: MemoryState) -> torch.Tensor:
        """
        Extract KV-cache from memory state.

        In a full implementation, this would extract actual key-value pairs
        from the transformer layers. Here we use memory tokens as a proxy.

        Args:
            memory_state: Current memory state

        Returns:
            KV-cache representation
        """
        return memory_state["memory_tokens"]
