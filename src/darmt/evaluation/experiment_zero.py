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
from darmt.evaluation.synthetic_tasks import (
    SyntheticMemoryTask,
    SyntheticReasoningTask,
    MultiHopReasoningTask,
)


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


def count_model_parameters(model: nn.Module) -> float:
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

    # Inform user about device and expected runtime
    if config.device == "cuda":
        print(f"✓ Running on GPU: {torch.cuda.get_device_name(0)}")
        print("   Expected runtime: ~5-10 minutes for 200 training steps")
    else:
        print("⚠️  Running on CPU (no CUDA available)")
        print("   Expected runtime: ~30-45 minutes for 200 training steps")

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
        armt_model=armt_for_dual,
        coprocessor_model=coprocessor,
        num_latents=config.num_latents,
        freeze_armt=False,  # CRITICAL: Enable fair comparison by training all params
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
    # Step 3: Generate synthetic evaluation tasks
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Generating Synthetic Evaluation Tasks...")
    print("-" * 80)

    # Initialize synthetic tasks
    memory_task = SyntheticMemoryTask(vocab_size=config.vocab_size, num_markers=5)
    reasoning_task = SyntheticReasoningTask(vocab_size=config.vocab_size)
    multihop_task = MultiHopReasoningTask(vocab_size=config.vocab_size, num_hops=3)

    # Generate data for each task
    print("Generating memory retrieval task...")
    memory_context, memory_query, memory_metadata = memory_task.generate_data(
        config.batch_size, config.segment_length, config.num_segments, device
    )

    print("Generating pattern reasoning task...")
    reasoning_context, reasoning_query, reasoning_labels = reasoning_task.generate_data(
        config.batch_size, config.segment_length, config.num_segments, device
    )

    print("Generating multi-hop reasoning task...")
    multihop_context, multihop_query, multihop_metadata = multihop_task.generate_data(
        config.batch_size, config.segment_length, config.num_segments, device
    )

    print(f"✓ Tasks generated: {config.num_segments} segments × {config.segment_length} tokens each")
    print(f"  Total context: ~{config.num_segments * config.segment_length} tokens")

    # -------------------------------------------------------------------------
    # Step 4: Train all models on synthetic tasks
    # -------------------------------------------------------------------------
    print("\n[Step 4/7] Training Models on Synthetic Tasks...")
    print("-" * 80)
    print("Training for 200 steps to allow architectural differences to emerge...")

    # Training configuration
    learning_rate = 1e-4
    num_training_steps = 200

    # Initialize optimizers
    optimizer_A = torch.optim.AdamW(config_A.parameters(), lr=learning_rate)
    optimizer_B = torch.optim.AdamW(config_B.parameters(), lr=learning_rate)
    optimizer_C = torch.optim.AdamW(config_C.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training curves storage
    training_curves = {
        "config_A": {"memory": [], "reasoning": [], "multihop": []},
        "config_B": {"memory": [], "reasoning": [], "multihop": []},
        "config_C": {"memory": [], "reasoning": [], "multihop": []},
    }

    def train_on_task(model, context, query, target_metadata, task_name, optimizer, use_coprocessor=None):
        """Train model on one task batch."""
        model.train()
        optimizer.zero_grad()

        memory = None
        # Process context
        for segment in context:
            if use_coprocessor is not None:
                output = model(segment, memory, use_coprocessor=use_coprocessor)
            else:
                output = model(segment, memory)
            memory = output["memory_state"]

        # Process query
        if use_coprocessor is not None:
            final_output = model(query, memory, use_coprocessor=use_coprocessor)
        else:
            final_output = model(query, memory)

        # Compute loss based on task type
        logits = final_output["logits"]
        loss = torch.tensor(0.0, device=logits.device, requires_grad=True)  # Initialize loss

        if task_name == "memory":
            # For memory task, predict the marker positions
            marker_positions = target_metadata["marker_positions"]
            marker_tokens = target_metadata["marker_tokens"]

            # Get predictions at marker positions
            batch_loss = 0.0
            for b in range(logits.size(0)):
                for pos, token in zip(marker_positions[b], marker_tokens[b]):
                    if pos < logits.size(1):
                        token_logits = logits[b, pos:pos+1]  # [1, vocab_size]
                        token_target = token.unsqueeze(0)  # [1]
                        batch_loss += criterion(token_logits, token_target)
            loss = batch_loss / (logits.size(0) * len(marker_positions[0]))

        elif task_name == "reasoning":
            # For pattern reasoning, predict continuation
            loss = criterion(logits.view(-1, logits.size(-1)), target_metadata.view(-1))

        elif task_name == "multihop":
            # For multi-hop, predict answer token
            answer_positions = target_metadata["answer_positions"]
            answer_tokens = target_metadata["answer_tokens"]

            batch_loss = 0.0
            for b in range(logits.size(0)):
                pos = answer_positions[b]
                token = answer_tokens[b]
                if pos < logits.size(1):
                    token_logits = logits[b, pos:pos+1]
                    token_target = token.unsqueeze(0)
                    batch_loss += criterion(token_logits, token_target)
            loss = batch_loss / logits.size(0)
        else:
            # Fallback case for unknown task names
            raise ValueError(f"Unknown task name: {task_name}")

        loss.backward()
        optimizer.step()

        return loss.item()

    print("\nTraining progress:")
    for step in range(num_training_steps):
        # Rotate through tasks
        task_idx = step % 3

        if task_idx == 0:
            # Train on memory task
            loss_A = train_on_task(config_A, memory_context, memory_query, memory_metadata, "memory", optimizer_A)
            loss_B = train_on_task(config_B, memory_context, memory_query, memory_metadata, "memory", optimizer_B)
            loss_C = train_on_task(config_C, memory_context, memory_query, memory_metadata, "memory", optimizer_C, use_coprocessor=True)

            training_curves["config_A"]["memory"].append(loss_A)
            training_curves["config_B"]["memory"].append(loss_B)
            training_curves["config_C"]["memory"].append(loss_C)

        elif task_idx == 1:
            # Train on reasoning task
            loss_A = train_on_task(config_A, reasoning_context, reasoning_query, reasoning_labels, "reasoning", optimizer_A)
            loss_B = train_on_task(config_B, reasoning_context, reasoning_query, reasoning_labels, "reasoning", optimizer_B)
            loss_C = train_on_task(config_C, reasoning_context, reasoning_query, reasoning_labels, "reasoning", optimizer_C, use_coprocessor=True)

            training_curves["config_A"]["reasoning"].append(loss_A)
            training_curves["config_B"]["reasoning"].append(loss_B)
            training_curves["config_C"]["reasoning"].append(loss_C)

        else:
            # Train on multi-hop task
            loss_A = train_on_task(config_A, multihop_context, multihop_query, multihop_metadata, "multihop", optimizer_A)
            loss_B = train_on_task(config_B, multihop_context, multihop_query, multihop_metadata, "multihop", optimizer_B)
            loss_C = train_on_task(config_C, multihop_context, multihop_query, multihop_metadata, "multihop", optimizer_C, use_coprocessor=True)

            training_curves["config_A"]["multihop"].append(loss_A)
            training_curves["config_B"]["multihop"].append(loss_B)
            training_curves["config_C"]["multihop"].append(loss_C)

        # Print progress every 50 steps
        if (step + 1) % 50 == 0:
            avg_loss_A = (training_curves["config_A"]["memory"][-1] +
                         training_curves["config_A"]["reasoning"][-1] +
                         training_curves["config_A"]["multihop"][-1]) / 3
            avg_loss_B = (training_curves["config_B"]["memory"][-1] +
                         training_curves["config_B"]["reasoning"][-1] +
                         training_curves["config_B"]["multihop"][-1]) / 3
            avg_loss_C = (training_curves["config_C"]["memory"][-1] +
                         training_curves["config_C"]["reasoning"][-1] +
                         training_curves["config_C"]["multihop"][-1]) / 3

            print(f"  Step {step + 1:3d}/200: "
                  f"Config A: {avg_loss_A:.3f} | "
                  f"Config B: {avg_loss_B:.3f} | "
                  f"Config C: {avg_loss_C:.3f}")

    print("\n✓ Training complete!")

    # Analyze training curves
    print("\nTraining Curve Analysis:")
    print("-" * 80)

    # Calculate final vs initial loss reduction for each config
    for config_name, curves in training_curves.items():
        config_label = config_name.replace("config_", "Config ")

        # Average across all tasks
        all_losses = curves["memory"] + curves["reasoning"] + curves["multihop"]
        initial_loss = sum(all_losses[:10]) / 10  # First 10 steps
        final_loss = sum(all_losses[-10:]) / 10   # Last 10 steps
        loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100

        print(f"{config_label}: {loss_reduction:.1f}% loss reduction "
              f"(from {initial_loss:.3f} to {final_loss:.3f})")

    # Set models to eval mode
    config_A.eval()
    config_B.eval()
    config_C.eval()

    # -------------------------------------------------------------------------
    # Step 5: Run evaluations on synthetic tasks
    # -------------------------------------------------------------------------
    print("\n[Step 5/7] Running Synthetic Task Evaluations...")
    print("-" * 80)

    def evaluate_model_on_task(model, context, query, use_coprocessor=None):
        """Process segments and return final output."""
        memory = None

        # Process context segments
        for segment in context:
            if use_coprocessor is not None:
                output = model(segment, memory, use_coprocessor=use_coprocessor)
            else:
                output = model(segment, memory)
            memory = output["memory_state"]

        # Process query
        if use_coprocessor is not None:
            final_output = model(query, memory, use_coprocessor=use_coprocessor)
        else:
            final_output = model(query, memory)

        return final_output

    results_dict: dict[str, Any] = {
        "config_A": {"name": "Config A (Baseline)", "params_M": params_A},
        "config_B": {"name": "Config B (Unified)", "params_M": params_B},
        "config_C": {
            "name": "Config C (Dual)",
            "params_M": params_C_total,
            "params_trainable_M": params_C_trainable,
        },
    }

    print("\n[Task 1] Memory Retrieval...")
    with torch.no_grad():
        output_A_mem = evaluate_model_on_task(config_A, memory_context, memory_query)
        output_B_mem = evaluate_model_on_task(config_B, memory_context, memory_query)
        output_C_mem = evaluate_model_on_task(
            config_C, memory_context, memory_query, use_coprocessor=True
        )

    mem_acc_A = memory_task.evaluate_memory_retrieval(
        output_A_mem["logits"], memory_metadata, device
    )
    mem_acc_B = memory_task.evaluate_memory_retrieval(
        output_B_mem["logits"], memory_metadata, device
    )
    mem_acc_C = memory_task.evaluate_memory_retrieval(
        output_C_mem["logits"], memory_metadata, device
    )

    results_dict["config_A"]["memory_accuracy"] = mem_acc_A
    results_dict["config_B"]["memory_accuracy"] = mem_acc_B
    results_dict["config_C"]["memory_accuracy"] = mem_acc_C

    print(f"  Config A: {mem_acc_A:.2f}%")
    print(f"  Config B: {mem_acc_B:.2f}%")
    print(f"  Config C: {mem_acc_C:.2f}%")

    print("\n[Task 2] Pattern Reasoning...")
    with torch.no_grad():
        output_A_reason = evaluate_model_on_task(
            config_A, reasoning_context, reasoning_query
        )
        output_B_reason = evaluate_model_on_task(
            config_B, reasoning_context, reasoning_query
        )
        output_C_reason = evaluate_model_on_task(
            config_C, reasoning_context, reasoning_query, use_coprocessor=True
        )

    reason_acc_A = reasoning_task.evaluate_pattern_completion(
        output_A_reason["logits"], reasoning_labels
    )
    reason_acc_B = reasoning_task.evaluate_pattern_completion(
        output_B_reason["logits"], reasoning_labels
    )
    reason_acc_C = reasoning_task.evaluate_pattern_completion(
        output_C_reason["logits"], reasoning_labels
    )

    results_dict["config_A"]["reasoning_accuracy"] = reason_acc_A
    results_dict["config_B"]["reasoning_accuracy"] = reason_acc_B
    results_dict["config_C"]["reasoning_accuracy"] = reason_acc_C

    print(f"  Config A: {reason_acc_A:.2f}%")
    print(f"  Config B: {reason_acc_B:.2f}%")
    print(f"  Config C: {reason_acc_C:.2f}%")

    print("\n[Task 3] Multi-hop Reasoning...")
    with torch.no_grad():
        output_A_multihop = evaluate_model_on_task(
            config_A, multihop_context, multihop_query
        )
        output_B_multihop = evaluate_model_on_task(
            config_B, multihop_context, multihop_query
        )
        output_C_multihop = evaluate_model_on_task(
            config_C, multihop_context, multihop_query, use_coprocessor=True
        )

    multihop_acc_A = multihop_task.evaluate_multi_hop_reasoning(
        output_A_multihop["logits"], multihop_metadata
    )
    multihop_acc_B = multihop_task.evaluate_multi_hop_reasoning(
        output_B_multihop["logits"], multihop_metadata
    )
    multihop_acc_C = multihop_task.evaluate_multi_hop_reasoning(
        output_C_multihop["logits"], multihop_metadata
    )

    results_dict["config_A"]["multihop_accuracy"] = multihop_acc_A
    results_dict["config_B"]["multihop_accuracy"] = multihop_acc_B
    results_dict["config_C"]["multihop_accuracy"] = multihop_acc_C

    print(f"  Config A: {multihop_acc_A:.2f}%")
    print(f"  Config B: {multihop_acc_B:.2f}%")
    print(f"  Config C: {multihop_acc_C:.2f}%")

    # -------------------------------------------------------------------------
    # Step 6: Report results
    # -------------------------------------------------------------------------
    print("\n[Step 6/7] Analyzing Results...")
    print("-" * 80)

    # Print comprehensive results table
    print("\nResults Summary:")
    print("=" * 95)
    print(
        f"{'Configuration':<25} {'Memory (%)':>12} {'Reasoning (%)':>15} "
        f"{'MultiHop (%)':>15} {'Params (M)':>12}"
    )
    print("-" * 95)

    for key in ["config_A", "config_B", "config_C"]:
        res = results_dict[key]
        print(
            f"{res['name']:<25} "
            f"{res['memory_accuracy']:>12.2f} "
            f"{res['reasoning_accuracy']:>15.2f} "
            f"{res['multihop_accuracy']:>15.2f} "
            f"{res['params_M']:>12.2f}"
        )

    # -------------------------------------------------------------------------
    # Success Criteria Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 95)

    # Calculate performance deltas
    reasoning_gain = (
        results_dict["config_C"]["reasoning_accuracy"]
        - results_dict["config_B"]["reasoning_accuracy"]
    )
    memory_delta = (
        results_dict["config_C"]["memory_accuracy"]
        - results_dict["config_A"]["memory_accuracy"]
    )
    multihop_gain = (
        results_dict["config_C"]["multihop_accuracy"]
        - results_dict["config_B"]["multihop_accuracy"]
    )

    # Criterion 1: Reasoning performance
    print("\n1. Reasoning Performance: Dual vs Unified")
    print(f"   Pattern Task Gain: {reasoning_gain:+.2f}%")
    print(f"   Multi-hop Task Gain: {multihop_gain:+.2f}%")

    avg_reasoning_gain = (reasoning_gain + multihop_gain) / 2
    print(f"   Average Reasoning Gain: {avg_reasoning_gain:+.2f}%")

    if avg_reasoning_gain > 5.0:
        print("   ✅ SUCCESS: Dual architecture shows >5% avg gain over Unified")
        criterion_1 = True
    elif avg_reasoning_gain > 2.0:
        print("   ⚠️  MARGINAL: Dual architecture shows small improvement (2-5%)")
        print("   → Consider if coprocessor complexity is justified")
        criterion_1 = "MARGINAL"
    else:
        print("   ❌ FAILURE: Dual architecture is not significantly better than Unified")
        print("   → Unified model is sufficient; dual architecture not justified")
        criterion_1 = False

    # Criterion 2: Memory preservation
    print("\n2. Memory Retrieval: Dual vs Baseline")
    print(f"   Delta: {memory_delta:+.2f}%")
    if memory_delta > -2.0:
        print("   ✅ SUCCESS: Dual architecture maintains memory performance")
        criterion_2 = True
    else:
        print("   ❌ FAILURE: Dual architecture harms memory retrieval")
        print("   → Memory augmentation is interfering with base ARMT")
        criterion_2 = False

    # Additional analysis
    print("\n3. Unified vs Baseline Analysis")
    unified_memory_gain = (
        results_dict["config_B"]["memory_accuracy"]
        - results_dict["config_A"]["memory_accuracy"]
    )
    unified_reasoning_gain = (
        results_dict["config_B"]["reasoning_accuracy"]
        - results_dict["config_A"]["reasoning_accuracy"]
    )
    print(f"   Memory improvement: {unified_memory_gain:+.2f}%")
    print(f"   Reasoning improvement: {unified_reasoning_gain:+.2f}%")

    if unified_reasoning_gain > 10.0:
        print("   ✅ Deeper unified model shows strong gains")
        print("   → Extra parameters are well-utilized")
    else:
        print("   → Marginal gains from extra parameters")

    print("\n4. Coprocessor Specialization")
    print("   ⚠️  Requires post-hoc analysis of weights and activations")
    print("   → Manual inspection needed to verify emergent specialization")

    # -------------------------------------------------------------------------
    # Step 7: Final Recommendation with Training Analysis
    # -------------------------------------------------------------------------
    print("\n[Step 7/7] Final Recommendation with Training Analysis...")
    print("-" * 80)

    # Analyze training efficiency
    all_losses_A = training_curves["config_A"]["memory"] + training_curves["config_A"]["reasoning"] + training_curves["config_A"]["multihop"]
    all_losses_B = training_curves["config_B"]["memory"] + training_curves["config_B"]["reasoning"] + training_curves["config_B"]["multihop"]
    all_losses_C = training_curves["config_C"]["memory"] + training_curves["config_C"]["reasoning"] + training_curves["config_C"]["multihop"]

    initial_loss_A = sum(all_losses_A[:10]) / 10
    initial_loss_B = sum(all_losses_B[:10]) / 10
    initial_loss_C = sum(all_losses_C[:10]) / 10

    final_loss_A = sum(all_losses_A[-10:]) / 10
    final_loss_B = sum(all_losses_B[-10:]) / 10
    final_loss_C = sum(all_losses_C[-10:]) / 10

    learning_efficiency_A = ((initial_loss_A - final_loss_A) / initial_loss_A) * 100
    learning_efficiency_B = ((initial_loss_B - final_loss_B) / initial_loss_B) * 100
    learning_efficiency_C = ((initial_loss_C - final_loss_C) / initial_loss_C) * 100

    print("\nTraining Efficiency:")
    print(f"  Config A (Baseline):  {learning_efficiency_A:.1f}% loss reduction")
    print(f"  Config B (Unified):   {learning_efficiency_B:.1f}% loss reduction")
    print(f"  Config C (Dual):      {learning_efficiency_C:.1f}% loss reduction")

    learning_advantage_C_over_B = learning_efficiency_C - learning_efficiency_B
    print(f"\n  Dual vs Unified learning advantage: {learning_advantage_C_over_B:+.1f}%")

    # -------------------------------------------------------------------------
    # Final Recommendation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("FINAL RECOMMENDATION")
    print("=" * 95)

    if criterion_1 is True and criterion_2:
        print("\n✅ PROCEED with dual architecture development")
        print("   Both success criteria met. The coprocessor provides genuine benefits.")

        if learning_advantage_C_over_B > 5.0:
            print(f"\n   💡 BONUS: Dual architecture also shows {learning_advantage_C_over_B:.1f}% better learning efficiency!")
            print("   → The coprocessor helps the model learn faster")

        print("\n   Recommended next steps:")
        print("   1. Implement adaptive triggers (MeCo, ARS)")
        print("   2. Train on real datasets (BABILong, GSM8K)")
        print("   3. Optimize coprocessor architecture")
        recommendation = "PROCEED"

    elif criterion_1 == "MARGINAL" and criterion_2:
        print("\n⚠️  MARGINAL GAINS - Consider cost/benefit")
        print("   Dual architecture shows small improvements but adds complexity.")

        if learning_advantage_C_over_B > 5.0:
            print(f"\n   💡 However, dual architecture shows {learning_advantage_C_over_B:.1f}% better learning efficiency")
            print("   → This suggests it may scale better with more training")
            recommendation = "PROCEED_CAUTIOUSLY"
        else:
            print("\n   Options:")
            print("   1. Proceed cautiously - implement adaptive triggers to maximize efficiency")
            print("   2. Test on real benchmarks before committing to dual approach")
            print("   3. Consider simpler memory augmentation strategies")
            recommendation = "MARGINAL"

    elif not criterion_1 and criterion_2:
        if learning_advantage_C_over_B > 10.0:
            print("\n⚠️  MIXED SIGNALS")
            print(f"   Dual architecture learns {learning_advantage_C_over_B:.1f}% more efficiently")
            print("   but final performance doesn't surpass unified model.")
            print("\n   Possible interpretations:")
            print("   1. Needs more training steps to realize advantage")
            print("   2. Training tasks don't highlight dual architecture strengths")
            print("   3. Better suited for different types of problems")
            print("\n   Recommended: Test on real benchmarks (BABILong, GSM8K)")
            recommendation = "INVESTIGATE"
        else:
            print("\n❌ PIVOT to improving unified ARMT")
            print("   Unified model matches dual performance. Focus on:")
            print("   - Deeper unified architectures")
            print("   - Better training objectives")
            print("   - More efficient memory mechanisms")
            print("\n   The coprocessor adds complexity without sufficient benefit.")
            recommendation = "PIVOT_TO_UNIFIED"
    else:
        print("\n❌ INVESTIGATE or REDESIGN")
        print("   The dual architecture is harming performance.")
        print("\n   Critical issues to address:")
        if not criterion_2:
            print("   - Memory augmentation is interfering with ARMT")
            print("   - Consider different integration strategies")
        if not criterion_1 and criterion_1 is not False:
            print("   - Coprocessor not improving reasoning")
            print("   - May need different architecture or training")
        recommendation = "INVESTIGATE"

    print("\n" + "=" * 95)

    # Package results
    results_dict["recommendation"] = recommendation
    results_dict["dual_vs_unified_reasoning_gain"] = avg_reasoning_gain
    results_dict["dual_vs_baseline_memory_delta"] = memory_delta
    results_dict["unified_vs_baseline_reasoning"] = unified_reasoning_gain
    results_dict["training_curves"] = training_curves
    results_dict["learning_efficiency"] = {
        "config_A": learning_efficiency_A,
        "config_B": learning_efficiency_B,
        "config_C": learning_efficiency_C,
        "dual_vs_unified_advantage": learning_advantage_C_over_B,
    }

    return results_dict
