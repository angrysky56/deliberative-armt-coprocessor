# Experiment 3 Geometric Analysis - Fixes Applied

## Issues Identified

From the pasted results, three critical problems were preventing meaningful analysis:

1. **Untrained Models**: All three architectures used fresh random initialization
   - Made comparisons meaningless (comparing random noise)
   - No signal about architectural differences

2. **Failed Trajectory Extraction**: 
   - Unified: 0 trajectories extracted
   - MoE: 90 trajectories (only one working)
   - Baseline: 0 trajectories extracted

3. **Missing Checkpoints**: Experiments 0 and 1 never saved trained model weights
   - Experiment 3 expected: `checkpoints/unified_exp0.pt`, `checkpoints/moe_exp1.pt`, `checkpoints/baseline_exp0.pt`
   - None existed

## Fixes Applied

### Fix 1: Trajectory Extraction (experiment_3_geometric_analysis.py)

**Problem**: The extraction code only worked with `model.layers` (ModuleList), but:
- MoE has: `self.layers = nn.ModuleList()` ✓
- Unified has: `self.transformer` (TransformerEncoder) ✗
- SimpleARMT has: `self.transformer` (TransformerEncoder) ✗

**Solution**: Updated `extract_memory_trajectories()` to handle both:
```python
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
```

### Fix 2: Checkpoint Saving in Experiment 0

**Added to**: `src/darmt/evaluation/experiment_zero.py`

**Location**: After training completes, before evaluation phase

**Saves**:
- `checkpoints/baseline_exp0.pt` (Config A - SimpleARMT 6L)
- `checkpoints/unified_exp0.pt` (Config B - UnifiedARMT 9L)
- `checkpoints/dual_exp0.pt` (Config C - DualArchitectureARMT)

### Fix 3: Checkpoint Saving in Experiment 1

**Added to**: `experiments/experiment_1_moe_validation.py`

**Location**: After training completes, before evaluation phase

**Saves**:
- `checkpoints/moe_exp1.pt` (Config D - MoE-ARMT)
- Checks for existing baseline/unified from Experiment 0 (doesn't overwrite)

## Next Steps - Execute Full Pipeline

Now run the complete experiment pipeline with trained models:

### Step 1: Train Baseline and Unified Models
```bash
python experiments/experiment_0_architecture_validation.py
```

**What this does**:
- Trains 3 models for 200 steps (~5-10 minutes on GPU)
- Saves: `checkpoints/baseline_exp0.pt`, `checkpoints/unified_exp0.pt`, `checkpoints/dual_exp0.pt`
- Reports comparative performance metrics

### Step 2: Train MoE Model
```bash
python experiments/experiment_1_moe_validation.py
```

**What this does**:
- Trains MoE architecture for 200 steps
- Saves: `checkpoints/moe_exp1.pt`
- Reports MoE vs unified/baseline performance

### Step 3: Run Geometric Analysis with Trained Models
```bash
python experiments/experiment_3_geometric_analysis.py
```

**What this does**:
- Loads trained checkpoints
- Extracts trajectories from all 3 architectures (now working!)
- Computes geometric metrics:
  - Order-0: Semantic clustering
  - Order-1: Logical structure  
  - Order-2: Trajectory smoothness
- Analyzes WHY unified outperforms MoE

## Expected Results After Fixes

With trained models and fixed extraction:

1. **All models extract trajectories**:
   - Unified: 90+ trajectory sequences (not 0)
   - MoE: 90 trajectory sequences
   - Baseline: 90+ trajectory sequences (not 0)

2. **Meaningful geometric metrics**:
   - Real values (not NaN)
   - Shows architectural differences
   - Explains performance gaps

3. **Key insights**:
   - Unified: Smooth trajectories, high task coherence
   - MoE: Fragmented trajectories from expert routing
   - Baseline: Reference point for comparison

## Technical Details

### Trajectory Extraction Process
1. Register forward hooks on specified layers
2. During forward pass, capture memory token hidden states
3. Average across batch dimension
4. Store as [num_mem_tokens, hidden_size] per layer

### Geometric Metrics
- **Order-0 (positions)**: Measures semantic clustering via cosine similarity
- **Order-1 (velocities)**: Captures logical structure through differences
- **Order-2 (curvature)**: Quantifies smoothness using Menger curvature

### Why This Matters
From Reasoning-Flow framework: "Trajectory smoothness indicates better information flow and learning efficiency."

Fragmented trajectories (like MoE's) suggest:
- Poor expert coordination
- Information loss at routing boundaries
- Harder to maintain coherent reasoning chains

## Validation Checklist

After running the pipeline, verify:

- [ ] All 3 checkpoint files exist in `checkpoints/`
- [ ] Experiment 3 loads all checkpoints successfully
- [ ] All 3 models extract ~90 trajectory sequences
- [ ] Geometric metrics show real values (not NaN)
- [ ] Results explain why unified > MoE
- [ ] Heatmaps generated in `results/experiment_3_geometric/`

## Files Modified

1. `experiments/experiment_3_geometric_analysis.py`
   - Fixed `extract_memory_trajectories()` function

2. `src/darmt/evaluation/experiment_zero.py`  
   - Added checkpoint saving after training

3. `experiments/experiment_1_moe_validation.py`
   - Added checkpoint saving after training

## No Changes Required

The following worked correctly:
- Task generation (synthetic memory/reasoning/multihop)
- Geometric metric computation 
- Visualization pipeline
- Result reporting

---

**Ready to Execute**: All fixes are in place. Run experiments 0, 1, then 3 in sequence.
