"""
Quick test to validate the learned memory fusion layer.
"""

import torch
from darmt.models.memory_fusion import MemoryFusionLayer


def test_memory_fusion():
    """Test that memory fusion layer works correctly."""
    print("Testing MemoryFusionLayer...")
    
    # Create test data
    batch_size = 2
    mem_len = 64
    num_latents = 32
    hidden_size = 512
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize fusion layer
    fusion = MemoryFusionLayer(
        hidden_size=hidden_size,
        num_attention_heads=8,
        dropout=0.1,
    ).to(device)
    
    # Create random memory and latents
    memory_tokens = torch.randn(batch_size, mem_len, hidden_size, device=device)
    latent_embeddings = torch.randn(batch_size, num_latents, hidden_size, device=device)
    
    print(f"Memory tokens shape: {memory_tokens.shape}")
    print(f"Latent embeddings shape: {latent_embeddings.shape}")
    
    # Test forward pass
    fused_memory = fusion(memory_tokens, latent_embeddings)
    
    print(f"Fused memory shape: {fused_memory.shape}")
    
    # Verify output properties
    assert fused_memory.shape == memory_tokens.shape, "Output shape mismatch!"
    assert not torch.isnan(fused_memory).any(), "NaN detected in output!"
    assert not torch.isinf(fused_memory).any(), "Inf detected in output!"
    
    # Test gradient flow
    loss = fused_memory.mean()
    loss.backward()
    
    # Check that fusion layer parameters have gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in fusion.parameters())
    assert has_grad, "No gradients computed for fusion layer!"
    
    # Count parameters
    num_params = sum(p.numel() for p in fusion.parameters())
    print(f"Fusion layer parameters: {num_params / 1e6:.2f}M")
    
    print("✓ All tests passed!")
    return True


if __name__ == "__main__":
    test_memory_fusion()
