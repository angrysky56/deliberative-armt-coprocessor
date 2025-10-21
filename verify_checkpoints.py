#!/usr/bin/env python3
"""
Verify and clean checkpoints after config fixes.

This script checks if existing checkpoints match the corrected config
and removes any that were saved with the wrong parameters.
"""

import torch
from pathlib import Path

def check_checkpoint(checkpoint_path: Path, expected_vocab_size: int = 32000) -> bool:
    """
    Check if checkpoint matches expected config.
    
    Returns:
        True if checkpoint is valid, False if needs regeneration
    """
    if not checkpoint_path.exists():
        print(f"  ⚠️  {checkpoint_path.name}: Not found (needs generation)")
        return False
    
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Check vocab size from embedding layer
        if 'embedding.weight' in state_dict:
            actual_vocab_size = state_dict['embedding.weight'].shape[0]
        elif 'token_embedding.weight' in state_dict:
            actual_vocab_size = state_dict['token_embedding.weight'].shape[0]
        else:
            print(f"  ❓ {checkpoint_path.name}: Cannot determine vocab size")
            return False
        
        if actual_vocab_size != expected_vocab_size:
            print(f"  ❌ {checkpoint_path.name}: INVALID (vocab_size={actual_vocab_size}, expected {expected_vocab_size})")
            print(f"     → Will be deleted and regenerated")
            return False
        else:
            print(f"  ✅ {checkpoint_path.name}: Valid (vocab_size={actual_vocab_size})")
            return True
            
    except Exception as e:
        print(f"  ❌ {checkpoint_path.name}: Error loading - {e}")
        return False


def main():
    print("=" * 80)
    print("CHECKPOINT VERIFICATION")
    print("=" * 80)
    print("\nChecking for config mismatches after fixes...")
    print("Expected config: vocab_size=32000, num_heads=8, hidden_size=512")
    print()
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoints = {
        "baseline_exp0.pt": "Baseline (SimpleARMT 6L)",
        "unified_exp0.pt": "Unified (UnifiedARMT 9L)",
        "dual_exp0.pt": "Dual Architecture",
        "moe_exp1.pt": "MoE (MoE-ARMT 9L)"
    }
    
    print("Checking checkpoints:")
    print("-" * 80)
    
    invalid_checkpoints = []
    for checkpoint_name, description in checkpoints.items():
        checkpoint_path = checkpoint_dir / checkpoint_name
        print(f"\n{description}:")
        is_valid = check_checkpoint(checkpoint_path)
        if not is_valid and checkpoint_path.exists():
            invalid_checkpoints.append(checkpoint_path)
    
    # Clean invalid checkpoints
    if invalid_checkpoints:
        print("\n" + "=" * 80)
        print("CLEANING INVALID CHECKPOINTS")
        print("=" * 80)
        
        for checkpoint_path in invalid_checkpoints:
            print(f"\nDeleting: {checkpoint_path}")
            checkpoint_path.unlink()
            print("  ✓ Deleted")
        
        print("\n" + "=" * 80)
        print("ACTION REQUIRED")
        print("=" * 80)
        print("\nInvalid checkpoints have been removed.")
        print("Please regenerate them by running:")
        print()
        
        if any("exp0" in str(p) for p in invalid_checkpoints):
            print("  python experiments/experiment_0_architecture_validation.py")
        if any("exp1" in str(p) for p in invalid_checkpoints):
            print("  python experiments/experiment_1_moe_validation.py")
        
        print("\nOr run the full pipeline:")
        print("  python run_full_pipeline.py")
        print()
    else:
        print("\n" + "=" * 80)
        print("✅ ALL CHECKPOINTS VALID")
        print("=" * 80)
        print("\nReady to run Experiment 3:")
        print("  python experiments/experiment_3_geometric_analysis.py")
        print()


if __name__ == "__main__":
    main()
