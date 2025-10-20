"""Memory state management utilities for ARMT models."""

import torch
from typing import TypedDict


class MemoryState(TypedDict):
    """Type definition for ARMT memory state."""

    memory_tokens: torch.Tensor  # Shape: [batch, num_mem_tokens, hidden_size]
    associative_memory: dict[str, torch.Tensor] | None  # Optional A-matrices


def create_initial_memory(
    batch_size: int, num_mem_tokens: int, hidden_size: int, device: torch.device
) -> MemoryState:
    """
    Create initial memory state for ARMT.

    Args:
        batch_size: Number of sequences in batch
        num_mem_tokens: Number of memory tokens
        hidden_size: Dimension of hidden states
        device: Device to create tensors on

    Returns:
        Initial memory state dictionary
    """
    return {
        "memory_tokens": torch.randn(
            batch_size, num_mem_tokens, hidden_size, device=device
        ),
        "associative_memory": None,
    }


def augment_memory_with_latents(
    memory_state: MemoryState,
    latent_embeddings: torch.Tensor,
    fusion_layer=None,
) -> MemoryState:
    """
    Augment memory state with latent embeddings from coprocessor.

    Args:
        memory_state: Current memory state
        latent_embeddings: Latent embeddings from coprocessor
                          Shape: [batch, num_latents, hidden_size]
        fusion_layer: Optional learned fusion layer for integration
                     If None, falls back to naive concatenation (not recommended)

    Returns:
        Updated memory state with augmented memory tokens
    """
    current_memory = memory_state["memory_tokens"]
    
    if fusion_layer is not None:
        # Use learned fusion (recommended)
        augmented_memory = fusion_layer(current_memory, latent_embeddings)
    else:
        # Fallback: naive concatenation (causes performance issues)
        augmented_memory = torch.cat([current_memory, latent_embeddings], dim=1)

    return {
        "memory_tokens": augmented_memory,
        "associative_memory": memory_state.get("associative_memory"),
    }


def extract_kv_cache(memory_state: MemoryState) -> torch.Tensor:
    """
    Extract KV-cache representation from memory state.

    In a real ARMT implementation, this would extract the actual key-value
    cache from the transformer layers. Here we use memory tokens as a proxy.

    Args:
        memory_state: Current memory state

    Returns:
        KV-cache tensor [batch, seq_len, hidden_size]
    """
    return memory_state["memory_tokens"]
