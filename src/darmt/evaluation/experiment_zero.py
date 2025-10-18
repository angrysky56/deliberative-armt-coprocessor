"""
Experiment 0: Architecture Validation

This is the MOST CRITICAL experiment. It validates whether the dual
architecture (ARMT + Coprocessor) provides genuine benefits over a
unified model with equivalent parameters.

Based on research finding (October 2025):
'A unified model with the same parameter count nearly matches dual architecture.'

We MUST validate this before proceeding with further development.
"""

import torch
import torch.nn as nn
from typing import Any
from dataclasses import dataclass

from darmt.models.armt import SimpleARMT
from darmt.models.coprocessor import SimpleCoprocessor
from darmt.models.unified import UnifiedARMT
from darmt.models.dual_architecture import DualArchitectureARMT
from darmt.utils.memory import create_initial_memory


@dataclass
class Experiment0Config:
    """Configuration for Experiment 0."""

    # Model architecture
    hidden_size: int = 768
    num_heads: int = 12
    vocab_size: int = 32000

    # ARMT baseline (Config A)
    armt_layers: int = 12
    num_mem_tokens: int = 32

    # Unified model (Config B)
    unified_layers: int = 18  # Should match total of ARMT + Coprocessor

    # Dual architecture (Config C)
    coprocessor_layers: int = 6

    # Training/evaluation
    batch_size: int = 4
    segment_length: int = 1024
    num_segments: int = 10
    num_latents: int = 32

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def count_model_parameters(model: nn.Module) -> int:
    """Count parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def run_experiment_zero(config: Experiment0Config | None = None) -> dict[str, Any]:
    """
    Run Experiment 0: Validate unified vs dual architecture.

    This experiment tests three configurations:
    - Config A: ARMT Baseline (~137M params)
    - Config B: Unified ARMT (~200M params)
    - Config C: Dual (ARMT + Coprocessor) (~200M params)

    Success Criteria:
    1. Config C must beat Config B by >5% on reasoning tasks
    2. Config C must maintain Config A's memory retrieval accuracy
    3. Coprocessor must show emergent specialization

    Args:
        config: Experiment configuration (uses defaults if None)

    Returns:
        Dictionary with experiment results and recommendation
    """
    if config is None:
        config = Experiment0Config()

    print("=" * 80)
    print("EXPERIMENT 0: ARCHITECTURE VALIDATION")
    print("=" * 80)
    print("\nObjective: Validate whether dual architecture provides genuine benefits")
    print("over a unified model with equivalent parameters.\n")

    device = torch.device(config.device)

    # -------------------------------------------------------------------------
    # Step 1: Initialize three configurations
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Initializing Models...")
    print("-" * 80)

    # Config A: ARMT Baseline
    config_A = SimpleARMT(
        num_layers=config.armt_layers,
        hidden_size=config.hidden_size,
        num_mem_tokens=config.num_mem_tokens,
        vocab_size=config.vocab_size,
    ).to(device)

    # Config B: Unified ARMT (more layers)
    config_B = UnifiedARMT(
        num_layers=config.unified_layers,
        hidden_size=config.hidden_size,
        num_mem_tokens=config.num_mem_tokens,
        vocab_size=config.vocab_size,
    ).to(device)

    # Config C: Dual Architecture
    armt_for_dual = SimpleARMT(
        num_layers=config.armt_layers,
        hidden_size=config.hidden_size,
        num_mem_tokens=config.num_mem_tokens,
        vocab_size=config.vocab_size,
    ).to(device)

    coprocessor = SimpleCoprocessor(
        num_layers=config.coprocessor_layers, hidden_size=config.hidden_size
    ).to(device)

    config_C = DualArchitectureARMT(
        armt_model=armt_for_dual, coprocessor_model=coprocessor, num_latents=config.num_latents
    ).to(device)

    # -------------------------------------------------------------------------
    # Step 2: Verify parameter counts
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Verifying Parameter Counts...")
    print("-" * 80)

    params_A = count_model_parameters(config_A)
    params_B = count_model_parameters(config_B)
    params_C_trainable = count_model_parameters(config_C)
    params_C_total = config_C.count_parameters(trainable_only=False) / 1e6

    print(f"Config A (Baseline):      {params_A:>8.2f}M parameters (all trainable)")
    print(f"Config B (Unified):       {params_B:>8.2f}M parameters (all trainable)")
    print(
        f"Config C (Dual):          {params_C_total:>8.2f}M total "
        f"({params_C_trainable:>8.2f}M trainable)"
    )

    # Verify parameter matching
    total_B = params_B
    total_C = params_C_total
    param_diff_pct = abs(total_B - total_C) / total_B * 100

    print(f"\nParameter matching: {param_diff_pct:.2f}% difference")
    if param_diff_pct > 5:
        print("⚠️  WARNING: Parameter counts don't match closely (>5% difference)")
    else:
        print("✓ Parameter counts match well")

    # -------------------------------------------------------------------------
    # Step 3: Generate dummy data
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Generating Test Data...")
    print("-" * 80)

    # Simulate long context (multiple segments)
    context_segments = []
    for i in range(config.num_segments):
        segment = torch.randint(
            0,
            config.vocab_size,
            (config.batch_size, config.segment_length),
            device=device,
        )
        context_segments.append(segment)

    # Query segment (for reasoning task)
    query_segment = torch.randint(
        0, config.vocab_size, (config.batch_size, config.segment_length), device=device
    )

    # Dummy labels
    labels = torch.randint(
        0, config.vocab_size, (config.batch_size, config.segment_length), device=device
    )

    print(f"Context: {config.num_segments} segments × {config.segment_length} tokens")
    print(f"Query: 1 segment × {config.segment_length} tokens")
    print(f"Total context: ~{config.num_segments * config.segment_length} tokens")

    # -------------------------------------------------------------------------
    # Step 4: Run forward passes
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Running Forward Passes...")
    print("-" * 80)

    def process_segments(model, segments, use_coprocessor=None):
        """Process multiple segments through a model."""
        memory = None
        final_output = None

        for i, segment in enumerate(segments):
            if use_coprocessor is not None:
                # Dual architecture
                output = model(segment, memory, use_coprocessor=use_coprocessor)
            else:
                # Single architecture
                output = model(segment, memory)

            memory = output["memory_state"]
            final_output = output

        return final_output, memory

    # Process context through all three configs
    print("\nProcessing context segments...")

    with torch.no_grad():
        _, memory_A = process_segments(config_A, context_segments)
        _, memory_B = process_segments(config_B, context_segments)
        _, memory_C = process_segments(config_C, context_segments, use_coprocessor=True)

    print("✓ Context processing complete")

    # Process query segment
    print("\nProcessing query segment...")

    with torch.no_grad():
        output_A = config_A(query_segment, memory_A)
        output_B = config_B(query_segment, memory_B)
        output_C = config_C(query_segment, memory_C, use_coprocessor=True)

    print("✓ Query processing complete")

    # -------------------------------------------------------------------------
    # Step 5: Evaluate and report results
    # -------------------------------------------------------------------------
    print("\n[Step 5/5] Evaluating Results...")
    print("-" * 80)

    # Compute mock metrics (in real experiment, use actual datasets)
    criterion = nn.CrossEntropyLoss()

    loss_A = criterion(
        output_A["logits"].view(-1, config.vocab_size), labels.view(-1)
    ).item()
    loss_B = criterion(
        output_B["logits"].view(-1, config.vocab_size), labels.view(-1)
    ).item()
    loss_C = criterion(
        output_C["logits"].view(-1, config.vocab_size), labels.view(-1)
    ).item()

    # Mock accuracy scores (replace with real evaluation)
    # These should be from BABILong and GSM8K benchmarks
    results = {
        "config_A": {
            "name": "Config A (Baseline)",
            "memory_accuracy": 75.0 + torch.rand(1).item() * 10,  # Mock
            "reasoning_accuracy": 50.0 + torch.rand(1).item() * 10,  # Mock
            "loss": loss_A,
            "params_M": params_A,
        },
        "config_B": {
            "name": "Config B (Unified)",
            "memory_accuracy": 74.0 + torch.rand(1).item() * 10,  # Mock
            "reasoning_accuracy": 65.0 + torch.rand(1).item() * 15,  # Mock
            "loss": loss_B,
            "params_M": params_B,
        },
        "config_C": {
            "name": "Config C (Dual)",
            "memory_accuracy": 73.0 + torch.rand(1).item() * 10,  # Mock
            "reasoning_accuracy": 68.0 + torch.rand(1).item() * 15,  # Mock
            "loss": loss_C,
            "params_M": params_C_total,
            "params_trainable_M": params_C_trainable,
        },
    }

    # Print results table
    print("\nResults Summary:")
    print("=" * 80)
    print(
        f"{'Configuration':<25} {'Memory (%)':>12} {'Reasoning (%)':>15} {'Params (M)':>12}"
    )
    print("-" * 80)

    for key in ["config_A", "config_B", "config_C"]:
        res = results[key]
        print(
            f"{res['name']:<25} "
            f"{res['memory_accuracy']:>12.2f} "
            f"{res['reasoning_accuracy']:>15.2f} "
            f"{res['params_M']:>12.2f}"
        )

    # -------------------------------------------------------------------------
    # Success Criteria Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 80)

    reasoning_gain = (
        results["config_C"]["reasoning_accuracy"] - results["config_B"]["reasoning_accuracy"]
    )
    memory_delta = (
        results["config_C"]["memory_accuracy"] - results["config_A"]["memory_accuracy"]
    )

    print(f"\n1. Reasoning Performance: Dual vs Unified")
    print(f"   Gain: {reasoning_gain:+.2f}%")
    if reasoning_gain > 5.0:
        print("   ✅ SUCCESS: Dual architecture shows >5% gain over Unified")
        criterion_1 = True
    else:
        print("   ❌ FAILURE: Dual architecture is not significantly better than Unified")
        print("   → Unified model is sufficient; dual architecture not justified")
        criterion_1 = False

    print(f"\n2. Memory Retrieval: Dual vs Baseline")
    print(f"   Delta: {memory_delta:+.2f}%")
    if memory_delta > -2.0:
        print("   ✅ SUCCESS: Dual architecture maintains memory performance")
        criterion_2 = True
    else:
        print("   ❌ FAILURE: Dual architecture harms memory retrieval")
        criterion_2 = False

    print(f"\n3. Coprocessor Specialization")
    print("   ⚠️  Requires post-hoc analysis of weights and activations")
    print("   → Manual inspection needed to verify emergent specialization")

    # -------------------------------------------------------------------------
    # Final Recommendation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if criterion_1 and criterion_2:
        print("\n✅ PROCEED with dual architecture development")
        print("   Both success criteria met. The coprocessor provides genuine benefits.")
        recommendation = "PROCEED"
    elif not criterion_1:
        print("\n❌ PIVOT to improving unified ARMT")
        print("   Unified model matches dual performance. Focus on:")
        print("   - Deeper unified architectures")
        print("   - Better training objectives")
        print("   - More efficient memory mechanisms")
        recommendation = "PIVOT_TO_UNIFIED"
    else:
        print("\n⚠️  INVESTIGATE further")
        print("   Mixed results. Consider:")
        print("   - Adjusting coprocessor architecture")
        print("   - Testing on more diverse benchmarks")
        print("   - Analyzing failure modes")
        recommendation = "INVESTIGATE"

    print("\n" + "=" * 80)

    results["recommendation"] = recommendation
    results["dual_vs_unified_reasoning_gain"] = reasoning_gain
    results["dual_vs_baseline_memory_delta"] = memory_delta

    return results
