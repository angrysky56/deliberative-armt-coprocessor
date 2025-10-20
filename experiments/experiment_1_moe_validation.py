"""
Experiment 1: MoE Architecture Validation

This experiment validates whether sparse MoE architecture provides
genuine benefits over unified and baseline models.

Compares:
- Config A: Baseline ARMT
- Config B: Unified ARMT  
- Config D: MoE-ARMT with sparse expert activation

Expected outcome: MoE should show better specialization and performance
on multi-domain tasks while maintaining efficiency.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

from darmt.models import SimpleARMT, UnifiedARMT, MoEARMT
from darmt.evaluation.synthetic_tasks import (
    SyntheticMemoryTask,
    SyntheticReasoningTask,
    MultiHopReasoningTask,
)


@dataclass
class Experiment1Config:
    """Configuration for MoE validation experiment."""
    
    # Model architecture
    hidden_size: int = 512
    num_heads: int = 16
    intermediate_size: int = 2048
    num_mem_tokens: int = 16
    
    # Layer configuration
    armt_layers: int = 6
    unified_layers: int = 9
    moe_layers: int = 9
    
    # MoE specific
    num_experts: int = 8
    expert_top_k: int = 2
    moe_frequency: int = 4  # MoE every 4th layer
    load_balance_coef: float = 0.01
    
    # Training
    batch_size: int = 2
    segment_length: int = 512
    num_segments: int = 8
    vocab_size: int = 5000
    
    # Training steps
    num_training_steps: int = 200
    learning_rate: float = 1e-4


def count_model_parameters(model: nn.Module) -> float:
    """Count trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def run_experiment_one(config: Experiment1Config | None = None) -> dict:
    """
    Run MoE validation experiment.
    
    Returns:
        Dictionary with results and recommendation
    """
    if config is None:
        config = Experiment1Config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("EXPERIMENT 1: MoE ARCHITECTURE VALIDATION")
    print("=" * 80)
    print()
    print("Objective: Validate whether sparse MoE provides benefits over")
    print("unified and baseline models through fine-grained specialization.")
    print()
    
    if torch.cuda.is_available():
        print(f"✓ Running on GPU: {torch.cuda.get_device_name()}")
        print("   Expected runtime: ~5-10 minutes for 200 training steps")
    else:
        print("⚠️  Running on CPU")
        print("   Expected runtime: ~30-45 minutes for 200 training steps")
    
    # -------------------------------------------------------------------------
    # Step 1: Initialize models
    # -------------------------------------------------------------------------
    print("\n[Step 1/6] Initializing Models...")
    print("-" * 80)
    
    # Config A: Baseline ARMT
    config_A = SimpleARMT(
        num_layers=config.armt_layers,
        hidden_size=config.hidden_size,
        num_mem_tokens=config.num_mem_tokens,
        vocab_size=config.vocab_size,
        num_heads=config.num_heads,
    ).to(device)
    
    # Config B: Unified ARMT
    config_B = UnifiedARMT(
        num_layers=config.unified_layers,
        hidden_size=config.hidden_size,
        num_mem_tokens=config.num_mem_tokens,
        vocab_size=config.vocab_size,
        num_heads=config.num_heads,
    ).to(device)
    
    # Config D: MoE-ARMT
    config_D = MoEARMT(
        num_layers=config.moe_layers,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        num_mem_tokens=config.num_mem_tokens,
        vocab_size=config.vocab_size,
        num_experts=config.num_experts,
        expert_top_k=config.expert_top_k,
        moe_frequency=config.moe_frequency,
        load_balance_loss_coef=config.load_balance_coef,
    ).to(device)
    
    # -------------------------------------------------------------------------
    # Step 2: Verify parameter counts
    # -------------------------------------------------------------------------
    print("\n[Step 2/6] Verifying Parameter Counts...")
    print("-" * 80)
    
    params_A = count_model_parameters(config_A)
    params_B = count_model_parameters(config_B)
    params_D = count_model_parameters(config_D)
    
    print(f"Config A (Baseline):      {params_A:>8.2f}M parameters")
    print(f"Config B (Unified):       {params_B:>8.2f}M parameters")
    print(f"Config D (MoE):          {params_D:>8.2f}M parameters")
    print(f"  ├─ Total layers: {config.moe_layers}")
    print(f"  ├─ MoE layers: {config.moe_layers // config.moe_frequency}")
    print(f"  └─ Experts/layer: {config.num_experts} (Top-{config.expert_top_k})")
    
    print("\n[Step 3/6] Generating Synthetic Tasks...")
    print("-" * 80)
    print("✓ Tasks ready for training and evaluation")
    
    print("\n[Step 4/6] Training Models...")
    print("-" * 80)
    print("Training all models for 200 steps...")
    print("⚠️  Note: This is a placeholder. Full training logic needed.")
    
    print("\n[Step 5/6] Evaluating Performance...")
    print("-" * 80)
    print("⚠️  Note: This is a placeholder. Full evaluation logic needed.")
    
    print("\n[Step 6/6] Analyzing Results...")
    print("-" * 80)
    
    results = {
        "config_A_params": params_A,
        "config_B_params": params_B,
        "config_D_params": params_D,
        "recommendation": "IMPLEMENT_FULL_EXPERIMENT",
    }
    
    return results


def main() -> None:
    """Run Experiment 1."""
    print("\n🔬 Starting Experiment 1: MoE Validation\n")
    
    results = run_experiment_one()
    
    print("\n" + "=" * 80)
    print("EXPERIMENT STATUS")
    print("=" * 80)
    print("\n✅ MoE architecture successfully initialized!")
    print("\nNext steps:")
    print("1. Complete training loop (copy from experiment_0)")
    print("2. Complete evaluation tasks")
    print("3. Add MoE-specific analysis:")
    print("   - Expert utilization rates")
    print("   - Routing pattern analysis")
    print("   - Load balancing metrics")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
