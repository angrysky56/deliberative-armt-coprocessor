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
from typing import Any, Sequence

from darmt.models import SimpleARMT, UnifiedARMT, MoEARMT
from darmt.evaluation.synthetic_tasks import (
    SyntheticMemoryTask,
    SyntheticReasoningTask,
    MultiHopReasoningTask,
)


@dataclass
class Experiment1Config:
    """Configuration for MoE validation experiment."""

    # Model architecture (MUST match Experiment 0!)
    hidden_size: int = 512
    num_heads: int = 8  # Match Experiment 0
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
    vocab_size: int = 32000  # Match Experiment 0

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

    # -------------------------------------------------------------------------
    # Step 3: Generate synthetic evaluation tasks
    # -------------------------------------------------------------------------
    print("\n[Step 3/6] Generating Synthetic Evaluation Tasks...")
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
    print("\n[Step 4/6] Training Models on Synthetic Tasks...")
    print("-" * 80)
    print("Training for 200 steps to allow architectural differences to emerge...")

    # Training configuration
    learning_rate = config.learning_rate
    num_training_steps = config.num_training_steps

    # Initialize optimizers
    optimizer_A = torch.optim.AdamW(config_A.parameters(), lr=learning_rate)
    optimizer_B = torch.optim.AdamW(config_B.parameters(), lr=learning_rate)
    optimizer_D = torch.optim.AdamW(config_D.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training curves storage
    training_curves = {
        "config_A": {"memory": [], "reasoning": [], "multihop": []},
        "config_B": {"memory": [], "reasoning": [], "multihop": []},
        "config_D": {"memory": [], "reasoning": [], "multihop": [], "load_balance": []},
    }
    def train_on_task(
        model: nn.Module,
        context: Sequence[torch.Tensor],
        query: torch.Tensor,
        target_metadata: dict | torch.Tensor,
        task_name: str,
        optimizer: torch.optim.Optimizer
    ) -> tuple[float, float]:
        """Train model on one task batch. Returns (task_loss, aux_loss)."""
        model.train()
        optimizer.zero_grad()

        memory = None
        aux_losses = []

        # Process context
        for segment in context:
            output = model(segment, memory)
            memory = output["memory_state"]

            # Collect MoE auxiliary losses if present
            if "auxiliary_loss" in output:
                aux_losses.append(output["auxiliary_loss"])

        # Process query
        final_output = model(query, memory)
        if "auxiliary_loss" in final_output:
            aux_losses.append(final_output["auxiliary_loss"])

        # Compute task loss based on task type
        logits = final_output["logits"]

        if task_name == "memory":
            if not isinstance(target_metadata, dict):
                raise TypeError("Memory task requires metadata dictionary.")
            # For memory task, predict the marker positions
            marker_positions = target_metadata["marker_positions"]
            marker_tokens = target_metadata["marker_tokens"]

            batch_loss = torch.zeros((), device=logits.device)
            valid_targets = 0
            for b in range(logits.size(0)):
                for pos, token in zip(marker_positions[b], marker_tokens[b]):
                    if pos < logits.size(1):
                        token_logits = logits[b, pos:pos+1]
                        token_target = token.unsqueeze(0)
                        batch_loss = batch_loss + criterion(token_logits, token_target)
                        valid_targets += 1
            if valid_targets == 0:
                task_loss = torch.zeros((), device=logits.device)
            else:
                task_loss = batch_loss / valid_targets

        elif task_name == "reasoning":
            # For pattern reasoning, predict continuation
            if not isinstance(target_metadata, torch.Tensor):
                raise TypeError("Reasoning task requires target tensor.")
            target = target_metadata.to(logits.device).long()
            if target.dim() == 1:
                task_loss = criterion(logits[:, -1, :], target)
            else:
                target_length = target.size(1)
                logits_view = logits[:, -target_length:, :].reshape(-1, logits.size(-1))
                task_loss = criterion(logits_view, target.reshape(-1))
        elif task_name == "multihop":
            if not isinstance(target_metadata, dict):
                raise TypeError("Multi-hop task requires metadata dictionary.")
            # For multi-hop, predict answer token
            answer_positions = target_metadata["answer_positions"]
            answer_tokens = target_metadata["answer_tokens"]

            batch_loss = torch.zeros((), device=logits.device)
            valid_answers = 0
            for b in range(logits.size(0)):
                pos = answer_positions[b]
                token = answer_tokens[b]
                if pos < logits.size(1):
                    token_logits = logits[b, pos:pos+1]
                    token_target = token.unsqueeze(0)
                    batch_loss = batch_loss + criterion(token_logits, token_target)
                    valid_answers += 1
            if valid_answers == 0:
                task_loss = torch.zeros((), device=logits.device)
            else:
                task_loss = batch_loss / valid_answers
        else:
            raise ValueError(f"Unsupported task name: {task_name}")

        # Combine task loss with auxiliary loss (for MoE)
        if aux_losses:
            aux_tensors = []
            for loss in aux_losses:
                if not isinstance(loss, torch.Tensor):
                    loss = torch.tensor(loss, device=device)
                aux_tensors.append(loss)
            aux_loss = torch.stack(aux_tensors).mean()
        else:
            aux_loss = torch.zeros((), device=device)
        total_loss = task_loss + aux_loss

        total_loss.backward()
        optimizer.step()

        # Clear GPU cache after each training step to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return task_loss.item(), aux_loss.item()

    print("\nTraining progress:")
    print(f"(MoE config: {config.moe_layers} layers, MoE every {config.moe_frequency}th layer)")

    for step in range(num_training_steps):
        # Rotate through tasks
        task_idx = step % 3

        if task_idx == 0:
            # Train on memory task
            loss_A, _ = train_on_task(config_A, memory_context, memory_query, memory_metadata, "memory", optimizer_A)
            torch.cuda.empty_cache()
            loss_B, _ = train_on_task(config_B, memory_context, memory_query, memory_metadata, "memory", optimizer_B)
            torch.cuda.empty_cache()
            loss_D, aux_D = train_on_task(config_D, memory_context, memory_query, memory_metadata, "memory", optimizer_D)
            torch.cuda.empty_cache()

            training_curves["config_A"]["memory"].append(loss_A)
            training_curves["config_B"]["memory"].append(loss_B)
            training_curves["config_D"]["memory"].append(loss_D)
            training_curves["config_D"]["load_balance"].append(aux_D)

        elif task_idx == 1:
            # Train on reasoning task
            loss_A, _ = train_on_task(config_A, reasoning_context, reasoning_query, reasoning_labels, "reasoning", optimizer_A)
            torch.cuda.empty_cache()
            loss_B, _ = train_on_task(config_B, reasoning_context, reasoning_query, reasoning_labels, "reasoning", optimizer_B)
            torch.cuda.empty_cache()
            loss_D, aux_D = train_on_task(config_D, reasoning_context, reasoning_query, reasoning_labels, "reasoning", optimizer_D)
            torch.cuda.empty_cache()

            training_curves["config_A"]["reasoning"].append(loss_A)
            training_curves["config_B"]["reasoning"].append(loss_B)
            training_curves["config_D"]["reasoning"].append(loss_D)
            training_curves["config_D"]["load_balance"].append(aux_D)

        else:
            # Train on multi-hop task
            loss_A, _ = train_on_task(config_A, multihop_context, multihop_query, multihop_metadata, "multihop", optimizer_A)
            torch.cuda.empty_cache()
            loss_B, _ = train_on_task(config_B, multihop_context, multihop_query, multihop_metadata, "multihop", optimizer_B)
            torch.cuda.empty_cache()
            loss_D, aux_D = train_on_task(config_D, multihop_context, multihop_query, multihop_metadata, "multihop", optimizer_D)
            torch.cuda.empty_cache()

            training_curves["config_A"]["multihop"].append(loss_A)
            training_curves["config_B"]["multihop"].append(loss_B)
            training_curves["config_D"]["multihop"].append(loss_D)
            training_curves["config_D"]["load_balance"].append(aux_D)

        # Print progress every 50 steps
        if (step + 1) % 50 == 0:
            avg_loss_A = (training_curves["config_A"]["memory"][-1] +
                         training_curves["config_A"]["reasoning"][-1] +
                         training_curves["config_A"]["multihop"][-1]) / 3
            avg_loss_B = (training_curves["config_B"]["memory"][-1] +
                         training_curves["config_B"]["reasoning"][-1] +
                         training_curves["config_B"]["multihop"][-1]) / 3
            avg_loss_D = (training_curves["config_D"]["memory"][-1] +
                         training_curves["config_D"]["reasoning"][-1] +
                         training_curves["config_D"]["multihop"][-1]) / 3
            avg_aux_D = training_curves["config_D"]["load_balance"][-1]

            print(f"  Step {step + 1:3d}/{num_training_steps}: "
                  f"A: {avg_loss_A:.3f} | "
                  f"B: {avg_loss_B:.3f} | "
                  f"D: {avg_loss_D:.3f} (aux: {avg_aux_D:.4f})")

    print("\n✓ Training complete!")

    # Set models to eval mode
    config_A.eval()
    config_B.eval()
    config_D.eval()

    # Save trained checkpoints for future analysis (e.g., Experiment 3)
    print("\n[Saving Trained Models]")
    print("-" * 80)
    import os
    from pathlib import Path

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Save baseline (Config A) - only if not already saved by Experiment 0
    baseline_path = checkpoint_dir / "baseline_exp0.pt"
    if not baseline_path.exists():
        torch.save(config_A.state_dict(), baseline_path)
        print(f"✓ Saved baseline model: {baseline_path}")
    else:
        print(f"  (baseline checkpoint already exists from Experiment 0)")

    # Save unified (Config B) - only if not already saved by Experiment 0
    unified_path = checkpoint_dir / "unified_exp0.pt"
    if not unified_path.exists():
        torch.save(config_B.state_dict(), unified_path)
        print(f"✓ Saved unified model: {unified_path}")
    else:
        print(f"  (unified checkpoint already exists from Experiment 0)")

    # Save MoE model (Config D) - this is unique to Experiment 1
    moe_path = checkpoint_dir / "moe_exp1.pt"
    torch.save(config_D.state_dict(), moe_path)
    print(f"✓ Saved MoE model: {moe_path}")

    # -------------------------------------------------------------------------
    # Step 5: Run evaluations on synthetic tasks
    # -------------------------------------------------------------------------
    print("\n[Step 5/6] Running Synthetic Task Evaluations...")
    print("-" * 80)
    def evaluate_model_on_task(model: nn.Module, context: Sequence[torch.Tensor], query: torch.Tensor) -> dict:
        """Process segments and return final output."""
        memory = None

        # Process context segments
        for segment in context:
            output = model(segment, memory)
            memory = output["memory_state"]

        # Process query
        final_output = model(query, memory)
        return final_output

    results_dict: dict[str, Any] = {
        "config_A": {"name": "Config A (Baseline)", "params_M": params_A},
        "config_B": {"name": "Config B (Unified)", "params_M": params_B},
        "config_D": {"name": "Config D (MoE)", "params_M": params_D},
    }

    print("\n[Task 1] Memory Retrieval...")
    with torch.no_grad():
        output_A_mem = evaluate_model_on_task(config_A, memory_context, memory_query)
        output_B_mem = evaluate_model_on_task(config_B, memory_context, memory_query)
        output_D_mem = evaluate_model_on_task(config_D, memory_context, memory_query)

    mem_acc_A = memory_task.evaluate_memory_retrieval(
        output_A_mem["logits"], memory_metadata, device
    )
    mem_acc_B = memory_task.evaluate_memory_retrieval(
        output_B_mem["logits"], memory_metadata, device
    )
    mem_acc_D = memory_task.evaluate_memory_retrieval(
        output_D_mem["logits"], memory_metadata, device
    )

    results_dict["config_A"]["memory_accuracy"] = mem_acc_A
    results_dict["config_B"]["memory_accuracy"] = mem_acc_B
    results_dict["config_D"]["memory_accuracy"] = mem_acc_D

    print(f"  Config A: {mem_acc_A:.2f}%")
    print(f"  Config B: {mem_acc_B:.2f}%")
    print(f"  Config D: {mem_acc_D:.2f}%")

    print("\n[Task 2] Pattern Reasoning...")
    with torch.no_grad():
        output_A_reason = evaluate_model_on_task(config_A, reasoning_context, reasoning_query)
        output_B_reason = evaluate_model_on_task(config_B, reasoning_context, reasoning_query)
        output_D_reason = evaluate_model_on_task(config_D, reasoning_context, reasoning_query)

    reason_acc_A = reasoning_task.evaluate_pattern_completion(
        output_A_reason["logits"], reasoning_labels
    )
    reason_acc_B = reasoning_task.evaluate_pattern_completion(
        output_B_reason["logits"], reasoning_labels
    )
    reason_acc_D = reasoning_task.evaluate_pattern_completion(
        output_D_reason["logits"], reasoning_labels
    )

    results_dict["config_A"]["reasoning_accuracy"] = reason_acc_A
    results_dict["config_B"]["reasoning_accuracy"] = reason_acc_B
    results_dict["config_D"]["reasoning_accuracy"] = reason_acc_D

    print(f"  Config A: {reason_acc_A:.2f}%")
    print(f"  Config B: {reason_acc_B:.2f}%")
    print(f"  Config D: {reason_acc_D:.2f}%")

    print("\n[Task 3] Multi-hop Reasoning...")
    with torch.no_grad():
        output_A_multihop = evaluate_model_on_task(config_A, multihop_context, multihop_query)
        output_B_multihop = evaluate_model_on_task(config_B, multihop_context, multihop_query)
        output_D_multihop = evaluate_model_on_task(config_D, multihop_context, multihop_query)

    multihop_acc_A = multihop_task.evaluate_multi_hop_reasoning(
        output_A_multihop["logits"], multihop_metadata
    )
    multihop_acc_B = multihop_task.evaluate_multi_hop_reasoning(
        output_B_multihop["logits"], multihop_metadata
    )
    multihop_acc_D = multihop_task.evaluate_multi_hop_reasoning(
        output_D_multihop["logits"], multihop_metadata
    )

    results_dict["config_A"]["multihop_accuracy"] = multihop_acc_A
    results_dict["config_B"]["multihop_accuracy"] = multihop_acc_B
    results_dict["config_D"]["multihop_accuracy"] = multihop_acc_D

    print(f"  Config A: {multihop_acc_A:.2f}%")
    print(f"  Config B: {multihop_acc_B:.2f}%")
    print(f"  Config D: {multihop_acc_D:.2f}%")

    # -------------------------------------------------------------------------
    # Step 6: Analyze results with MoE-specific metrics
    # -------------------------------------------------------------------------
    print("\n[Step 6/6] Analyzing Results...")
    print("-" * 80)

    # Print comprehensive results table
    print("\nResults Summary:")
    print("=" * 95)
    print(
        f"{'Configuration':<25} {'Memory (%)':>12} {'Reasoning (%)':>15} "
        f"{'MultiHop (%)':>15} {'Params (M)':>12}"
    )
    print("-" * 95)

    for key in ["config_A", "config_B", "config_D"]:
        res = results_dict[key]
        print(
            f"{res['name']:<25} "
            f"{res['memory_accuracy']:>12.2f} "
            f"{res['reasoning_accuracy']:>15.2f} "
            f"{res['multihop_accuracy']:>15.2f} "
            f"{res['params_M']:>12.2f}"
        )

    # Calculate performance deltas
    moe_vs_unified_reasoning = (
        results_dict["config_D"]["reasoning_accuracy"]
        - results_dict["config_B"]["reasoning_accuracy"]
    )
    moe_vs_unified_multihop = (
        results_dict["config_D"]["multihop_accuracy"]
        - results_dict["config_B"]["multihop_accuracy"]
    )
    moe_vs_baseline_memory = (
        results_dict["config_D"]["memory_accuracy"]
        - results_dict["config_A"]["memory_accuracy"]
    )

    # Training efficiency analysis
    all_losses_A = (training_curves["config_A"]["memory"] +
                   training_curves["config_A"]["reasoning"] +
                   training_curves["config_A"]["multihop"])
    all_losses_B = (training_curves["config_B"]["memory"] +
                   training_curves["config_B"]["reasoning"] +
                   training_curves["config_B"]["multihop"])
    all_losses_D = (training_curves["config_D"]["memory"] +
                   training_curves["config_D"]["reasoning"] +
                   training_curves["config_D"]["multihop"])

    initial_loss_A = sum(all_losses_A[:10]) / 10
    initial_loss_B = sum(all_losses_B[:10]) / 10
    initial_loss_D = sum(all_losses_D[:10]) / 10

    final_loss_A = sum(all_losses_A[-10:]) / 10
    final_loss_B = sum(all_losses_B[-10:]) / 10
    final_loss_D = sum(all_losses_D[-10:]) / 10

    learning_efficiency_A = ((initial_loss_A - final_loss_A) / initial_loss_A) * 100
    learning_efficiency_B = ((initial_loss_B - final_loss_B) / initial_loss_B) * 100
    learning_efficiency_D = ((initial_loss_D - final_loss_D) / initial_loss_D) * 100

    print("\n" + "=" * 95)
    print("MoE ARCHITECTURE ANALYSIS")
    print("=" * 95)

    print("\n1. Performance vs Unified Model")
    print(f"   Pattern Reasoning Gain: {moe_vs_unified_reasoning:+.2f}%")
    print(f"   Multi-hop Reasoning Gain: {moe_vs_unified_multihop:+.2f}%")
    avg_reasoning_gain = (moe_vs_unified_reasoning + moe_vs_unified_multihop) / 2
    print(f"   Average Reasoning Gain: {avg_reasoning_gain:+.2f}%")

    print("\n2. Memory Preservation vs Baseline")
    print(f"   Memory Delta: {moe_vs_baseline_memory:+.2f}%")

    print("\n3. Training Efficiency")
    print(f"   Config A (Baseline):  {learning_efficiency_A:.1f}% loss reduction")
    print(f"   Config B (Unified):   {learning_efficiency_B:.1f}% loss reduction")
    print(f"   Config D (MoE):       {learning_efficiency_D:.1f}% loss reduction")
    learning_advantage_D_over_B = learning_efficiency_D - learning_efficiency_B
    print(f"   MoE vs Unified learning advantage: {learning_advantage_D_over_B:+.1f}%")

    print("\n4. Load Balancing Loss")
    avg_aux_loss = sum(training_curves["config_D"]["load_balance"]) / len(training_curves["config_D"]["load_balance"])
    print(f"   Average auxiliary loss: {avg_aux_loss:.4f}")
    print("   (Lower is better - indicates balanced expert utilization)")

    # Final recommendation
    print("\n" + "=" * 95)
    print("FINAL RECOMMENDATION")
    print("=" * 95)

    if avg_reasoning_gain > 5.0 and moe_vs_baseline_memory > -2.0:
        print("\n✅ SUCCESS: MoE architecture provides genuine benefits!")
        print(f"   - {avg_reasoning_gain:.1f}% improvement in reasoning tasks")
        print("   - Memory performance preserved")
        if learning_advantage_D_over_B > 5.0:
            print(f"   - {learning_advantage_D_over_B:.1f}% better learning efficiency")
        recommendation = "PROCEED_WITH_MOE"
    elif avg_reasoning_gain > 2.0 and moe_vs_baseline_memory > -2.0:
        print("\n⚠️  MARGINAL: MoE shows improvement but may not justify complexity")
        print(f"   - {avg_reasoning_gain:.1f}% improvement in reasoning")
        print("   - Consider if sparse activation benefits outweigh implementation cost")
        recommendation = "MARGINAL"
    else:
        print("\n❌ INSUFFICIENT: MoE does not provide clear advantages")
        print("   - Focus on unified model optimization instead")
        print("   - Or investigate different expert configurations")
        recommendation = "PIVOT_TO_UNIFIED"

    results_dict["recommendation"] = recommendation
    results_dict["moe_vs_unified_reasoning"] = avg_reasoning_gain
    results_dict["moe_vs_baseline_memory"] = moe_vs_baseline_memory
    results_dict["training_curves"] = training_curves
    results_dict["learning_efficiency"] = {
        "config_A": learning_efficiency_A,
        "config_B": learning_efficiency_B,
        "config_D": learning_efficiency_D,
        "moe_vs_unified_advantage": learning_advantage_D_over_B,
    }

    print("\n" + "=" * 95)

    return results_dict


def main() -> None:
    """Run Experiment 1."""
    print("\n🔬 Starting Experiment 1: MoE Validation\n")

    results = run_experiment_one()

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    recommendation = results["recommendation"]

    if recommendation == "PROCEED_WITH_MOE":
        print("\n✅ Proceed with MoE architecture development:")
        print("   - Analyze expert specialization patterns")
        print("   - Optimize expert capacity and routing")
        print("   - Test on real benchmarks (BABILong, GSM8K)")
    elif recommendation == "MARGINAL":
        print("\n⚠️  Marginal results - further investigation needed:")
        print("   - Try different expert configurations")
        print("   - Test with longer training")
        print("   - Evaluate on domain-specific tasks")
    else:
        print("\n❌ Consider alternative approaches:")
        print("   - Focus on unified model optimization")
        print("   - Investigate different architectures")
        print("   - Analyze where MoE underperformed")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
