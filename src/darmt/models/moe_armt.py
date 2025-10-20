"""
MoE-Enhanced ARMT model.

Integrates Mixture of Experts layers into the ARMT architecture,
replacing a subset of FFN layers with sparse MoE layers for improved
capacity and specialization.
"""

import torch
import torch.nn as nn
from typing import Any

from darmt.utils.memory import MemoryState
from darmt.models.moe import MoELayer


class MoEARMT(nn.Module):
    """
    ARMT with sparse Mixture of Experts layers.

    Architecture:
    - Standard transformer layers with some FFN layers replaced by MoE
    - Following successful patterns: replace every Nth FFN layer
    - Memory-augmented attention (from ARMT)
    - Sparse expert activation per token

    Based on research showing MoE enables better scaling than
    monolithic architectures or dual-system approaches.
    """

    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 512,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        num_mem_tokens: int = 32,
        vocab_size: int = 50257,
        dropout: float = 0.1,
        num_experts: int = 8,
        expert_top_k: int = 2,
        moe_frequency: int = 4,  # Use MoE every Nth layer
        load_balance_loss_coef: float = 0.01,
    ) -> None:
        """
        Initialize MoE-ARMT.

        Args:
            num_layers: Total number of transformer layers
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            intermediate_size: FFN/Expert intermediate dimension
            num_mem_tokens: Number of memory tokens
            vocab_size: Vocabulary size
            dropout: Dropout probability
            num_experts: Number of experts per MoE layer
            expert_top_k: Number of experts to activate per token
            moe_frequency: Use MoE every Nth layer (e.g., 4 means layers 3,7,11,...)
            load_balance_loss_coef: Weight for load balancing loss
        """
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_mem_tokens = num_mem_tokens
        self.vocab_size = vocab_size
        self.moe_frequency = moe_frequency

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Build transformer layers
        self.layers = nn.ModuleList()
        self.layer_is_moe = []  # Track which layers use MoE

        for layer_idx in range(num_layers):
            # Use MoE for every Nth layer (0-indexed: 3, 7, 11, ...)
            use_moe = (layer_idx + 1) % moe_frequency == 0

            layer = self._build_layer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                use_moe=use_moe,
                num_experts=num_experts if use_moe else 0,
                expert_top_k=expert_top_k if use_moe else 0,
                load_balance_loss_coef=load_balance_loss_coef if use_moe else 0,
            )

            self.layers.append(layer)
            self.layer_is_moe.append(use_moe)

        # Output projection
        self.ln_final = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Initialize memory tokens
        self.memory_tokens = nn.Parameter(torch.randn(1, num_mem_tokens, hidden_size))

        print(f"[Init] MoEARMT created: {num_layers}L, {hidden_size}H")
        print(f"       MoE layers: {sum(self.layer_is_moe)}/{num_layers}")
        print(f"       Experts per MoE layer: {num_experts}, Top-k: {expert_top_k}")

    def _build_layer(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float,
        use_moe: bool,
        num_experts: int,
        expert_top_k: int,
        load_balance_loss_coef: float,
    ) -> nn.Module:
        """Build a single transformer layer (with or without MoE)."""

        class TransformerLayer(nn.Module):
            """Single transformer layer with optional MoE."""

            def __init__(self):
                super().__init__()
                self.use_moe = use_moe

                # Attention
                self.ln1 = nn.LayerNorm(hidden_size)
                self.attn = nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )

                # FFN or MoE
                self.ln2 = nn.LayerNorm(hidden_size)

                if use_moe:
                    # Sparse MoE layer
                    self.ffn = MoELayer(
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        num_experts=num_experts,
                        top_k=expert_top_k,
                        dropout=dropout,
                        load_balance_loss_coef=load_balance_loss_coef,
                    )
                else:
                    # Standard FFN
                    self.ffn = nn.Sequential(
                        nn.Linear(hidden_size, intermediate_size),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(intermediate_size, hidden_size),
                    )

            def forward(self, x, memory=None):
                # Self-attention
                normed = self.ln1(x)
                attn_out, _ = self.attn(normed, normed, normed)
                x = x + attn_out

                # FFN or MoE
                normed = self.ln2(x)

                if self.use_moe:
                    ffn_out, aux_loss = self.ffn(normed)
                    x = x + ffn_out
                    return x, aux_loss
                else:
                    ffn_out = self.ffn(normed)
                    x = x + ffn_out
                    return x, {}

        return TransformerLayer()

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_state: MemoryState | None = None,
        return_hidden_states: bool = False,
        return_aux_losses: bool = True,
    ) -> dict[str, Any]:
        """
        Forward pass through MoE-ARMT.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            memory_state: Previous memory state (for now, simplified)
            return_hidden_states: Whether to return intermediate hidden states
            return_aux_losses: Whether to return MoE auxiliary losses

        Returns:
            Dictionary containing:
                - logits: Output logits [batch, seq_len, vocab_size]
                - aux_losses: Auxiliary losses from MoE layers (if enabled)
                - memory_state: Updated memory state
        """
        batch_size = input_ids.size(0)

        # Token embedding
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        # Add memory tokens (simplified from ARMT)
        memory = self.memory_tokens.expand(batch_size, -1, -1)
        x = torch.cat([memory, x], dim=1)

        # Process through transformer layers
        total_aux_loss = 0.0
        layer_aux_losses = []

        for layer_idx, layer in enumerate(self.layers):
            x, aux_loss = layer(x)

            if aux_loss:  # MoE layer returns aux loss
                layer_aux_losses.append(aux_loss)
                if "load_balance_loss" in aux_loss:
                    total_aux_loss += aux_loss["load_balance_loss"]

        # Remove memory tokens
        x = x[:, self.num_mem_tokens:, :]

        # Final layer norm and projection
        x = self.ln_final(x)
        logits = self.lm_head(x)

        # Prepare output
        output = {
            "logits": logits,
            "memory_state": None,  # Simplified for now
        }

        if return_aux_losses and total_aux_loss > 0:
            output["aux_loss"] = total_aux_loss
            output["layer_aux_losses"] = layer_aux_losses

        return output

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def extract_kv_cache(self, memory_state: MemoryState | None) -> MemoryState | torch.Tensor:
        """Compatibility method with ARMT interface."""
        return self.memory_tokens if memory_state is None else memory_state
