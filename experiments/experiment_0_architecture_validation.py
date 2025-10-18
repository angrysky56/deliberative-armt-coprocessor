"""
Run Experiment 0: Architecture Validation

This script runs the critical architecture validation experiment.
Execute this BEFORE implementing any further features.
"""

from darmt.evaluation.experiment_zero import run_experiment_zero, Experiment0Config


def main() -> None:
    """Run Experiment 0 with default configuration."""
    print("\n🔬 Starting Experiment 0: Architecture Validation\n")
    print("This experiment validates whether the dual architecture")
    print("provides genuine benefits over a unified model.\n")

    # Run the experiment
    results = run_experiment_zero()

    # Print final recommendation
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)

    recommendation = results["recommendation"]

    if recommendation == "PROCEED":
        print("\n✅ You can proceed with:")
        print("   - Implementing adaptive triggers (MeCo, ARS)")
        print("   - Training on real datasets (BABILong, GSM8K)")
        print("   - Optimizing the coprocessor architecture")
        print("\nNext steps:")
        print("   1. Train on BABILong for memory evaluation")
        print("   2. Train on GSM8K for reasoning evaluation")
        print("   3. Implement adaptive compute mechanisms")

    elif recommendation == "PIVOT_TO_UNIFIED":
        print("\n❌ The dual architecture does not provide sufficient benefits.")
        print("\nRecommended pivot:")
        print("   - Focus on deeper unified ARMT architectures")
        print("   - Improve training objectives and data augmentation")
        print("   - Investigate better memory mechanisms")
        print("\nThis finding aligns with recent research (October 2025)")
        print("showing unified models often match dual architectures.")

    else:
        print("\n⚠️  Mixed results - further investigation needed.")
        print("\nSuggested actions:")
        print("   1. Test on more diverse benchmarks")
        print("   2. Try different coprocessor architectures")
        print("   3. Analyze where the coprocessor helps/hurts")
        print("   4. Consider hybrid approaches")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
