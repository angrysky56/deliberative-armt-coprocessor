#!/usr/bin/env python3
"""
Run complete experiment pipeline with trained models.

This script executes Experiments 0, 1, and 3 in sequence to:
1. Train and save baseline/unified models (Exp 0)
2. Train and save MoE model (Exp 1)  
3. Run geometric analysis on trained models (Exp 3)

Total runtime: ~15-25 minutes on GPU, ~1-2 hours on CPU
"""

import sys
from pathlib import Path

def main():
    print("=" * 80)
    print("DARMT EXPERIMENT PIPELINE")
    print("=" * 80)
    print("\nThis will run all three experiments in sequence:")
    print("  1. Experiment 0: Train Baseline + Unified models")
    print("  2. Experiment 1: Train MoE model")
    print("  3. Experiment 3: Geometric analysis on trained models")
    print("\n" + "=" * 80)
    
    # Experiment 0
    print("\n\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "EXPERIMENT 0: ARCHITECTURE VALIDATION" + " " * 21 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    try:
        from darmt.evaluation.experiment_zero import run_experiment_zero
        results_0 = run_experiment_zero()
        print("\n✅ Experiment 0 complete!")
        print(f"   Recommendation: {results_0['recommendation']}")
    except Exception as e:
        print(f"\n❌ Experiment 0 failed: {e}")
        sys.exit(1)
    
    # Experiment 1
    print("\n\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 23 + "EXPERIMENT 1: MOE VALIDATION" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    try:
        # Import and run experiment 1
        sys.path.insert(0, str(Path(__file__).parent / "experiments"))
        from experiment_1_moe_validation import main as exp1_main
        exp1_main()
        print("\n✅ Experiment 1 complete!")
    except Exception as e:
        print(f"\n❌ Experiment 1 failed: {e}")
        sys.exit(1)
    
    # Experiment 3
    print("\n\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 21 + "EXPERIMENT 3: GEOMETRIC ANALYSIS" + " " * 25 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    try:
        from experiment_3_geometric_analysis import main as exp3_main
        exp3_main()
        print("\n✅ Experiment 3 complete!")
    except Exception as e:
        print(f"\n❌ Experiment 3 failed: {e}")
        sys.exit(1)
    
    # Final summary
    print("\n\n")
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("\n✅ All experiments finished successfully!")
    print("\nCheckpoints saved:")
    print("  - checkpoints/baseline_exp0.pt")
    print("  - checkpoints/unified_exp0.pt")
    print("  - checkpoints/dual_exp0.pt")
    print("  - checkpoints/moe_exp1.pt")
    print("\nResults available in:")
    print("  - results/experiment_3_geometric/")
    print("\nKey findings:")
    print("  - Unified vs MoE trajectory comparison")
    print("  - Geometric explanation of performance differences")
    print("  - Task coherence and smoothness metrics")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
