"""
Unified ARMT model with increased capacity.

This is the critical baseline for Experiment 0. According to the
October 2025 research, a unified model with the same total parameter
count often matches or beats the dual architecture.
"""

import torch
import torch.nn as nn
from typing import Any

from darmt.utils.memory import MemoryState


class UnifiedARMT(nn.Module):
    """
    Unified ARMT with deeper architecture.

    This model has the same total parameters as ARMT + Coprocessor
    but uses a single unified architecture. It's the critical baseline
    for validating whether the dual architecture provides genuine benefits.

    From the research:
    "A unified soft-embedding baseline with the same parameter count
    nearly matches the dual architecture performance."
    """

    def __init__(
        self,
        num_layers: int = 18,  # More layers than base ARMT
        hidden_size: int = 768,
        num_mem_tokens: int = 32,
        vocab_size: int = 32000,
        num_heads: int = 12,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize UnifiedARMT.

        Args:
            num_layers: Number of transformer layers (should match total of ARMT + Coprocessor)
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

        # Learnable memory tokens
        self.mem_tokens = nn.Parameter(torch.randn(1, num_mem_tokens, hidden_size))

        # Deeper transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Language modeling head
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        # Layer norm
        self.memory_norm = nn.LayerNorm(hidden_size)

        print(f"[Init] UnifiedARMT created: {num_layers}L, {hidden_size}H, {num_mem_tokens}M")

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_state: MemoryState | None = None,
        return_hidden_states: bool = False,
        return_attention_weights: bool = False,
    ) -> dict[str, Any]:
        """
        Forward pass through Unified ARMT.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            memory_state: Previous memory state
            return_hidden_states: Whether to return hidden states
            return_attention_weights: Whether to return attention weights

        Returns:
            Dictionary containing logits and updated memory state
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get input embeddings
        input_embeds = self.embedding(input_ids)

        # Get or create memory tokens
        if memory_state is None:
            mem_tokens_batch = self.mem_tokens.expand(batch_size, -1, -1)
        else:
            mem_tokens_batch = memory_state["memory_tokens"]
            mem_tokens_batch = self.memory_norm(mem_tokens_batch)

        # Combine memory tokens with input
        full_input = torch.cat([mem_tokens_batch, input_embeds], dim=1)

        # Process through deeper transformer
        output_embeds = self.transformer(full_input)

        # Split output
        new_memory = output_embeds[:, : self.num_mem_tokens, :]
        token_embeds = output_embeds[:, self.num_mem_tokens :, :]

        # Generate logits
        logits = self.lm_head(token_embeds)

        # Prepare output
        output = {
            "logits": logits,
            "memory_state": {"memory_tokens": new_memory, "associative_memory": None},
        }

        if return_hidden_states:
            output["hidden_states"] = token_embeds

        if return_attention_weights:
            output["attention_weights"] = None

        return output

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
