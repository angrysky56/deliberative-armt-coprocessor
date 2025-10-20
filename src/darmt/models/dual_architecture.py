"""
Dual Architecture: ARMT + Coprocessor.

This combines the ARMT base model with a coprocessor for deliberative reasoning.
The ARMT can be frozen (for transfer learning) or trainable (for end-to-end training).
"""

import torch
import torch.nn as nn
from typing import Any

from darmt.models.armt import SimpleARMT
from darmt.models.coprocessor import SimpleCoprocessor
from darmt.models.memory_fusion import MemoryFusionLayer
from darmt.utils.memory import MemoryState, augment_memory_with_latents


class DualArchitectureARMT(nn.Module):
    """
    Dual architecture combining ARMT + Coprocessor.

    Architecture:
    1. ARMT processes input and generates memory state
    2. Coprocessor deliberates on memory and generates latent embeddings
    3. Memory is augmented with latent embeddings
    4. Augmented memory used in next segment

    Design Options:
    - ARMT can be frozen (transfer learning: pretrained ARMT + train coprocessor)
    - ARMT can be trainable (end-to-end training: train both components)
    
    For architectural comparison experiments (like Experiment 0), use trainable ARMT
    to ensure fair comparison with unified models.
    """

    def __init__(
        self,
        armt_model: SimpleARMT,
        coprocessor_model: SimpleCoprocessor,
        num_latents: int = 32,
        freeze_armt: bool = True,
        use_learned_fusion: bool = True,
        num_fusion_heads: int = 8,
    ) -> None:
        """
        Initialize DualArchitectureARMT.

        Args:
            armt_model: Pre-initialized ARMT model
            coprocessor_model: Pre-initialized coprocessor model
            num_latents: Number of latent embeddings to generate
            freeze_armt: Whether to freeze ARMT parameters (recommended)
            use_learned_fusion: Whether to use learned memory fusion (recommended)
            num_fusion_heads: Number of attention heads in fusion layer
        """
        super().__init__()
        self.armt = armt_model
        self.coprocessor = coprocessor_model
        self.num_latents = num_latents
        self.use_learned_fusion = use_learned_fusion

        # Initialize memory fusion layer
        if use_learned_fusion:
            self.fusion_layer = MemoryFusionLayer(
                hidden_size=armt_model.hidden_size,
                num_attention_heads=num_fusion_heads,
                dropout=0.1,
            )
            print("[Init] DualArchitectureARMT: Using LEARNED memory fusion")
        else:
            self.fusion_layer = None
            print("[Init] DualArchitectureARMT: Using NAIVE concatenation (not recommended)")

        # Freeze ARMT if specified
        if freeze_armt:
            for param in self.armt.parameters():
                param.requires_grad = False
            print("[Init] DualArchitectureARMT: ARMT is FROZEN")
        else:
            print("[Init] DualArchitectureARMT: ARMT is TRAINABLE (not recommended)")

        print("[Init] DualArchitectureARMT: Coprocessor is TRAINABLE")

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_state: MemoryState | None = None,
        use_coprocessor: bool = True,
        return_hidden_states: bool = False,
        return_attention_weights: bool = False,
    ) -> dict[str, Any]:
        """
        Forward pass through dual architecture.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            memory_state: Previous memory state
            use_coprocessor: Whether to use coprocessor (for ablation studies)
            return_hidden_states: Whether to return hidden states
            return_attention_weights: Whether to return attention weights

        Returns:
            Dictionary containing logits and updated memory state
        """
        # Step 1: Process through ARMT
        # Use no_grad only if ARMT is frozen AND not in training mode
        armt_frozen = not any(p.requires_grad for p in self.armt.parameters())
        
        if armt_frozen:
            with torch.no_grad():
                armt_output = self.armt(
                    input_ids,
                    memory_state,
                    return_hidden_states=return_hidden_states,
                    return_attention_weights=return_attention_weights,
                )
        else:
            # ARMT is trainable - compute with gradients
            armt_output = self.armt(
                input_ids,
                memory_state,
                return_hidden_states=return_hidden_states,
                return_attention_weights=return_attention_weights,
            )

        base_memory_state = armt_output["memory_state"]
        logits = armt_output["logits"]

        # Step 2: Coprocessor deliberation (if enabled)
        if use_coprocessor:
            # Extract KV-cache for coprocessor
            kv_cache = self.armt.extract_kv_cache(base_memory_state)

            # Generate latent embeddings
            latent_embeddings = self.coprocessor.generate_latent_embeddings(
                kv_cache, self.num_latents
            )

            # Augment memory with latent embeddings using learned fusion
            augmented_memory = augment_memory_with_latents(
                base_memory_state, latent_embeddings, fusion_layer=self.fusion_layer
            )

            # Use base logits but augmented memory for next segment
            final_memory_state = augmented_memory
            final_logits = logits
        else:
            # No coprocessor: use base ARMT output
            final_memory_state = base_memory_state
            final_logits = logits

        # Prepare output
        output = {"logits": final_logits, "memory_state": final_memory_state}

        if return_hidden_states and "hidden_states" in armt_output:
            output["hidden_states"] = armt_output["hidden_states"]

        if return_attention_weights and "attention_weights" in armt_output:
            output["attention_weights"] = armt_output["attention_weights"]

        return output

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count parameters.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Parameter count
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def get_armt_parameters(self) -> int:
        """Count ARMT parameters."""
        return sum(p.numel() for p in self.armt.parameters())

    def get_coprocessor_parameters(self) -> int:
        """Count coprocessor parameters (including fusion layer)."""
        coprocessor_params = sum(p.numel() for p in self.coprocessor.parameters())
        if self.fusion_layer is not None:
            fusion_params = sum(p.numel() for p in self.fusion_layer.parameters())
            return coprocessor_params + fusion_params
        return coprocessor_params
    
    def get_fusion_parameters(self) -> int:
        """Count fusion layer parameters."""
        if self.fusion_layer is not None:
            return sum(p.numel() for p in self.fusion_layer.parameters())
        return 0
