def visualize_memory_manifolds(unified_model, moe_model):
    """
    Create Reasoning-Flow style visualizations:

    Figure 1: Unified ARMT
    - (a) Order-0: Memory tokens cluster by task
    - (b) Order-1: Flows align within task type
    - (c) Order-2: Smooth curvature

    Figure 2: MoE ARMT
    - (a) Order-0: Similar clustering
    - (b) Order-1: FRAGMENTED flows (different experts)
    - (c) Order-2: HIGH curvature (discontinuities)
    """

    # Extract trajectories
    unified_traj = extract_trajectories(unified_model)
    moe_traj = extract_trajectories(moe_model)

    # Create side-by-side comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Unified row
    plot_pca_trajectories(unified_traj, axes[0, 0], title="Unified: Positions")
    plot_velocity_alignment(unified_traj, axes[0, 1], title="Unified: Velocities")
    plot_curvature_heatmap(unified_traj, axes[0, 2], title="Unified: Curvature")

    # MoE row (showing fragmentation)
    plot_pca_trajectories(moe_traj, axes[1, 0], title="MoE: Positions")
    plot_velocity_alignment(moe_traj, axes[1, 1], title="MoE: Fragmented")
    plot_curvature_heatmap(moe_traj, axes[1, 2], title="MoE: High Curvature")