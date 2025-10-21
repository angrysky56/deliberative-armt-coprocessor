"""
Experiment 3: Geometric Analysis of Unified vs Alternative Architectures

Uses Reasoning-Flow framework to understand WHY unified ARMT outperforms
MoE and dual architectures through geometric trajectory analysis.

Research Questions:
1. Why does unified architecture outperform MoE? (trajectory smoothness)
2. How do memory tokens organize geometrically? (clustering patterns)
3. What causes MoE's catastrophic memory failure? (routing fragmentation)
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# Add analysis module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from darmt.models import SimpleARMT, UnifiedARMT, MoEARMT
from darmt.evaluation.synthetic_tasks import (
    SyntheticMemoryTask,
    SyntheticReasoningTask,
    MultiHopReasoningTask,
)
from darmt.analysis.utils_stat import (
    pairwise_similarity,
    pairwise_menger_curvature_similarity,
    plot_similarity_heatmap,
)


@dataclass
class GeometricAnalysisConfig:
    """Configuration for geometric analysis experiment."""

    # Model architecture
    hidden_size: int = 512
    num_heads: int = 16
    num_mem_tokens: int = 16

    # Task configuration
    batch_size: int = 2
    segment_length: int = 512
    num_segments: int = 8
    vocab_size: int = 5000

    # Analysis configuration
    extract_layers: Optional[List[int]] = None  # Which layers to extract (None = all)
    num_samples: int = 10  # Samples per task type

    # Output
    save_dir: str = "results/experiment_3_geometric"

    def __post_init__(self):
        if self.extract_layers is None:
            self.extract_layers = [0, 3, 6, 9]  # Sample across depth


def load_trained_models(config: GeometricAnalysisConfig, device: torch.device) -> Dict[str, torch.nn.Module]:
    """
    Load trained models from Experiments 0 and 1.

    If trained checkpoints exist, load them. Otherwise, initialize fresh models.
    """
    models = {}

    print("\n[Loading Models]")
    print("-" * 80)

    # Check for trained checkpoints
    checkpoint_dir = Path("checkpoints")

    # Unified ARMT (9 layers)
    unified_checkpoint = checkpoint_dir / "unified_exp0.pt"
    if unified_checkpoint.exists():
        print(f"✓ Loading trained Unified model from {unified_checkpoint}")
        model = UnifiedARMT(
            num_layers=9,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_mem_tokens=config.num_mem_tokens,
            vocab_size=config.vocab_size,
        )
        model.load_state_dict(torch.load(unified_checkpoint, map_location=device))
    else:
        print("⚠️  No trained Unified checkpoint found, using fresh initialization")
        model = UnifiedARMT(
            num_layers=9,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_mem_tokens=config.num_mem_tokens,
            vocab_size=config.vocab_size,
        )
    models["unified"] = model.to(device).eval()

    # MoE ARMT (9 layers, 2 MoE)
    moe_checkpoint = checkpoint_dir / "moe_exp1.pt"
    if moe_checkpoint.exists():
        print(f"✓ Loading trained MoE model from {moe_checkpoint}")
        model = MoEARMT(
            num_layers=9,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_mem_tokens=config.num_mem_tokens,
            vocab_size=config.vocab_size,
            num_experts=8,
            expert_top_k=2,
            moe_frequency=4,
        )
        model.load_state_dict(torch.load(moe_checkpoint, map_location=device))
    else:
        print("⚠️  No trained MoE checkpoint found, using fresh initialization")
        model = MoEARMT(
            num_layers=9,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_mem_tokens=config.num_mem_tokens,
            vocab_size=config.vocab_size,
            num_experts=8,
            expert_top_k=2,
            moe_frequency=4,
        )
    models["moe"] = model.to(device).eval()

    # Baseline ARMT (6 layers)
    baseline_checkpoint = checkpoint_dir / "baseline_exp0.pt"
    if baseline_checkpoint.exists():
        print(f"✓ Loading trained Baseline model from {baseline_checkpoint}")
        model = SimpleARMT(
            num_layers=6,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_mem_tokens=config.num_mem_tokens,
            vocab_size=config.vocab_size,
        )
        model.load_state_dict(torch.load(baseline_checkpoint, map_location=device))
    else:
        print("⚠️  No trained Baseline checkpoint found, using fresh initialization")
        model = SimpleARMT(
            num_layers=6,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_mem_tokens=config.num_mem_tokens,
            vocab_size=config.vocab_size,
        )
    models["baseline"] = model.to(device).eval()

    print(f"\n✓ Loaded {len(models)} models")
    return models


@torch.no_grad()
def extract_memory_trajectories(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    memory_state: Any,
    extract_layers: Optional[List[int]],
    device: torch.device
) -> Dict[int, np.ndarray]:
    """
    Extract memory token hidden states from specific layers.

    Returns:
        Dict mapping layer_idx -> memory_token_embeddings [num_mem_tokens, hidden_size]
    """
    trajectories = {}

    # Hook to capture layer outputs
    layer_outputs = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # Output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Extract memory tokens (first num_mem_tokens)
            if hasattr(model, 'num_mem_tokens'):
                num_mem = model.num_mem_tokens
                memory_tokens = hidden_states[:, :num_mem, :]  # [batch, mem, hidden]
                layer_outputs[layer_idx] = memory_tokens.detach().cpu().numpy()
        return hook

    # Register hooks on specified layers
    handles = []
    
    # Try to get layers - handle both ModuleList and TransformerEncoder
    layers = None
    if hasattr(model, "layers"):
        # MoE-style: self.layers = nn.ModuleList()
        layers = model.layers
    elif hasattr(model, "transformer"):
        # Unified/SimpleARMT style: self.transformer = nn.TransformerEncoder()
        if hasattr(model.transformer, "layers"):
            layers = model.transformer.layers
        else:
            # Access internal layers from TransformerEncoder
            layers = list(model.transformer.children())
    
    # Determine which layers to extract
    layer_indices: List[int] = []
    if layers is not None and len(layers) > 0:
        if extract_layers is None:
            layer_indices = list(range(len(layers)))
        else:
            layer_indices = [idx for idx in extract_layers if idx < len(layers)]
        
        # Register hooks
        for layer_idx in layer_indices:
            handle = layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            handles.append(handle)
    
    # Forward pass
    output = model(input_ids, memory_state)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Average across batch dimension
    for layer_idx, mem_tokens in layer_outputs.items():
        # mem_tokens: [batch, num_mem_tokens, hidden_size]
        # Average over batch, keep memory tokens separate
        trajectories[layer_idx] = mem_tokens.mean(axis=0)  # [num_mem_tokens, hidden_size]

    return trajectories


def generate_task_sequences(config: GeometricAnalysisConfig, device: torch.device) -> Dict[str, List[torch.Tensor]]:
    """
    Generate sequences for each task type.

    Returns:
        Dict mapping task_name -> list of (context, query) tuples
    """
    print("\n[Generating Task Sequences]")
    print("-" * 80)

    # Initialize tasks
    memory_task = SyntheticMemoryTask(vocab_size=config.vocab_size, num_markers=5)
    reasoning_task = SyntheticReasoningTask(vocab_size=config.vocab_size)
    multihop_task = MultiHopReasoningTask(vocab_size=config.vocab_size, num_hops=3)

    sequences = {
        "memory": [],
        "reasoning": [],
        "multihop": []
    }

    # Generate multiple samples per task
    for i in range(config.num_samples):
        # Memory task
        mem_context, mem_query, _ = memory_task.generate_data(
            config.batch_size, config.segment_length, config.num_segments, device
        )
        sequences["memory"].append((mem_context, mem_query))

        # Reasoning task
        reason_context, reason_query, _ = reasoning_task.generate_data(
            config.batch_size, config.segment_length, config.num_segments, device
        )
        sequences["reasoning"].append((reason_context, reason_query))

        # Multihop task
        multi_context, multi_query, _ = multihop_task.generate_data(
            config.batch_size, config.segment_length, config.num_segments, device
        )
        sequences["multihop"].append((multi_context, multi_query))

    print(f"✓ Generated {config.num_samples} samples per task type")
    return sequences


def analyze_model_trajectories(
    model_name: str,
    model: torch.nn.Module,
    task_sequences: Dict[str, List],
    config: GeometricAnalysisConfig,
    device: torch.device
) -> Dict[str, List[np.ndarray]]:
    """
    Extract and organize trajectories for a single model.

    Returns:
        Dict mapping label -> list of trajectory vectors
        Label format: "{task_name}_layer{layer_idx}_sample{sample_idx}"
    """
    print(f"\n[Analyzing {model_name} Trajectories]")
    print("-" * 80)

    all_trajectories = {}

    for task_name, sequences in task_sequences.items():
        for sample_idx, (context, query) in enumerate(sequences):
            # Process context
            memory = None
            for segment in context:
                output = model(segment, memory)
                memory = output["memory_state"]

            # Process query and extract trajectories
            layer_trajectories = extract_memory_trajectories(
                model, query, memory, config.extract_layers, device
            )

            # Store with descriptive labels
            for layer_idx, mem_tokens in layer_trajectories.items():
                label = f"{task_name}_L{layer_idx}_S{sample_idx}"
                # Convert memory tokens to list of vectors (one per memory token)
                all_trajectories[label] = [mem_tokens[i] for i in range(mem_tokens.shape[0])]

    print(f"✓ Extracted {len(all_trajectories)} trajectory sequences")
    return all_trajectories


def compute_geometric_metrics(
    model_trajectories: Dict[str, Dict[str, List[np.ndarray]]],
    config: GeometricAnalysisConfig
) -> Dict[str, Any]:
    """
    Compute geometric similarity metrics for all models.

    Returns metrics for:
    - Order-0 (positions): Semantic clustering
    - Order-1 (velocities): Logical structure
    - Order-2 (curvature): Trajectory smoothness
    """
    print("\n[Computing Geometric Metrics]")
    print("-" * 80)

    results = {}

    for model_name, trajectories in model_trajectories.items():
        print(f"\nAnalyzing {model_name}...")

        model_results = {}

        # Order-0: Positions (semantic clustering)
        try:
            labels_0, sim_0 = pairwise_similarity(
                trajectories, order=0, metric="mean_cos", align="truncate"
            )
            model_results["order_0"] = {
                "labels": labels_0,
                "similarity": sim_0,
                "avg_within_task": compute_within_task_similarity(labels_0, sim_0)
            }
            print(f"  ✓ Order-0 (positions): {sim_0.shape}")
        except Exception as e:
            print(f"  ⚠️  Order-0 failed: {e}")
            model_results["order_0"] = None

        # Order-1: Velocities (logical structure)
        try:
            labels_1, sim_1 = pairwise_similarity(
                trajectories, order=1, metric="mean_cos", align="truncate"
            )
            model_results["order_1"] = {
                "labels": labels_1,
                "similarity": sim_1,
                "avg_within_task": compute_within_task_similarity(labels_1, sim_1)
            }
            print(f"  ✓ Order-1 (velocities): {sim_1.shape}")
        except Exception as e:
            print(f"  ⚠️  Order-1 failed: {e}")
            model_results["order_1"] = None

        # Order-2: Curvature (smoothness)
        try:
            labels_2, sim_2 = pairwise_menger_curvature_similarity(
                trajectories, metric="pearson", align="truncate"
            )
            model_results["order_2"] = {
                "labels": labels_2,
                "similarity": sim_2,
                "avg_curvature": np.mean(np.diag(sim_2))  # Self-similarity = smoothness
            }
            print(f"  ✓ Order-2 (curvature): {sim_2.shape}")
        except Exception as e:
            print(f"  ⚠️  Order-2 failed: {e}")
            model_results["order_2"] = None

        results[model_name] = model_results

    return results


def compute_within_task_similarity(labels: List[str], similarity_matrix: np.ndarray) -> Dict[str, float]:
    """Compute average similarity within each task type."""
    task_sims = {"memory": [], "reasoning": [], "multihop": []}

    for i, label_i in enumerate(labels):
        task_i = label_i.split("_")[0]
        for j, label_j in enumerate(labels):
            task_j = label_j.split("_")[0]
            if task_i == task_j and i != j:
                task_sims[task_i].append(similarity_matrix[i, j])

    return {
        task: float(np.mean(sims)) if sims else 0.0
        for task, sims in task_sims.items()
    }


def visualize_results(
    geometric_results: Dict[str, Any],
    config: GeometricAnalysisConfig
) -> None:
    """Create visualizations comparing architectures."""
    print("\n[Creating Visualizations]")
    print("-" * 80)

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create heatmaps for each model and order
    for model_name, results in geometric_results.items():
        model_dir = save_dir / model_name
        model_dir.mkdir(exist_ok=True)

        for order_name in ["order_0", "order_1", "order_2"]:
            if results[order_name] is None:
                continue

            labels = results[order_name]["labels"]
            sim = results[order_name]["similarity"]

            title = f"{model_name.upper()}: {order_name.replace('_', '-')} Similarity"
            save_path = model_dir / f"{order_name}_heatmap.pdf"

            plot_similarity_heatmap(
                sim,
                labels=labels,
                title=title,
                save_pdf_path=str(save_path),
                show_axis_text=False,  # Too many labels
                color_scale="RdBu_r"
            )

            print(f"  ✓ Saved {order_name} heatmap: {save_path}")


def save_results(
    geometric_results: Dict[str, Any],
    config: GeometricAnalysisConfig
) -> None:
    """Save numerical results to JSON."""
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for model_name, results in geometric_results.items():
        model_results = {}
        for order_name, order_data in results.items():
            if order_data is None:
                model_results[order_name] = None
            else:
                model_results[order_name] = {
                    "avg_within_task": order_data.get("avg_within_task", {}),
                    "avg_curvature": float(order_data.get("avg_curvature", 0.0)),
                    "similarity_shape": list(order_data["similarity"].shape)
                }
        serializable_results[model_name] = model_results

    results_path = save_dir / "geometric_metrics.json"
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n✓ Saved results to {results_path}")


def print_summary(geometric_results: Dict[str, Any]) -> None:
    """Print summary comparing models."""
    print("\n" + "=" * 80)
    print("GEOMETRIC ANALYSIS SUMMARY")
    print("=" * 80)

    # Compare smoothness (order-2 curvature)
    print("\n1. Trajectory Smoothness (Order-2 Curvature)")
    print("-" * 80)
    for model_name in ["baseline", "unified", "moe"]:
        if model_name not in geometric_results:
            continue
        order_2 = geometric_results[model_name].get("order_2")
        if order_2:
            curvature = order_2.get("avg_curvature", 0.0)
            print(f"  {model_name:12s}: {curvature:.4f} (lower = smoother)")

    # Compare task coherence (order-1 within-task similarity)
    print("\n2. Task Coherence (Order-1 Within-Task Similarity)")
    print("-" * 80)
    for model_name in ["baseline", "unified", "moe"]:
        if model_name not in geometric_results:
            continue
        order_1 = geometric_results[model_name].get("order_1")
        if order_1:
            within_task = order_1.get("avg_within_task", {})
            print(f"  {model_name:12s}:")
            for task, sim in within_task.items():
                print(f"    - {task:10s}: {sim:.4f}")

    print("\n" + "=" * 80)


def main():
    """Run Experiment 3: Geometric Analysis."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: GEOMETRIC ANALYSIS")
    print("=" * 80)
    print("\nObjective: Understand WHY unified architecture outperforms MoE")
    print("using geometric trajectory analysis from Reasoning-Flow framework.")
    print()

    # Configuration
    config = GeometricAnalysisConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Extracting from layers: {config.extract_layers}")
    print(f"Samples per task: {config.num_samples}")

    # Load models
    models = load_trained_models(config, device)

    # Generate task sequences
    task_sequences = generate_task_sequences(config, device)

    # Extract trajectories for each model
    model_trajectories = {}
    for model_name, model in models.items():
        trajectories = analyze_model_trajectories(
            model_name, model, task_sequences, config, device
        )
        model_trajectories[model_name] = trajectories

    # Compute geometric metrics
    geometric_results = compute_geometric_metrics(model_trajectories, config)

    # Visualize and save results
    visualize_results(geometric_results, config)
    save_results(geometric_results, config)

    # Print summary
    print_summary(geometric_results)

    print("\n" + "=" * 80)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {config.save_dir}")
    print("\nKey findings should show:")
    print("  ✓ Unified: Smooth trajectories, high task coherence")
    print("  ✗ MoE: Fragmented trajectories, low task coherence")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
