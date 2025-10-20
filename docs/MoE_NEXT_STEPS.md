# MoE-ARMT Implementation: Next Steps

## What Works Now
✅ MoE architecture successfully initializes
✅ Sparse expert routing implemented
✅ Load balancing loss integrated
✅ All components compatible with existing codebase

## Immediate Next Steps

### 1. Complete Experiment 1 Training Loop
Copy the training logic from `experiment_0_architecture_validation.py`:
- Lines 200-350 contain the training loop
- Need to integrate MoE auxiliary loss into optimizer
- Track expert utilization during training

Key modification needed:
```python
# In training loop, add aux loss from MoE layers
output = model(input_ids, memory)
main_loss = criterion(output["logits"], targets)

# Add MoE auxiliary loss
total_loss = main_loss
if "aux_loss" in output:
    total_loss = total_loss + output["aux_loss"]

total_loss.backward()
```

### 2. Add MoE-Specific Analysis

Create `src/darmt/evaluation/moe_analysis.py`:
```python
def analyze_expert_routing(model, data):
    """
    Track which experts are being used for which tokens.
    
    Returns:
        - Expert utilization rates
        - Token-expert affinity patterns  
        - Load balancing effectiveness
    """
```

### 3. Parameter Count Adjustment

Current issue: MoE model has 62.9M vs Unified 33.5M (87% more!).

Options to match parameters:
- Reduce num_experts from 8 to 4
- Reduce intermediate_size  
- Use MoE only on 1 layer instead of 2

Or: Compare on **active parameters** (what's actually used per token):
- MoE: ~37M active (with top-2 of 8 experts)
- Unified: 33.5M active (all parameters)

This is a fairer comparison for efficiency.

### 4. Research-Aligned Experiments

Based on 2024-2025 MoE research, test:

**Experiment 1a: Varying Expert Count**
- 4 experts, top-1
- 8 experts, top-2 (current)
- 16 experts, top-2

**Experiment 1b: Varying MoE Frequency**  
- Every 2nd layer (4 MoE layers)
- Every 4th layer (2 MoE layers) - current
- Every 6th layer (1 MoE layer)

**Experiment 1c: Expert Specialization Analysis**
Per OpenMoE findings:
- Experts DON'T specialize by domain/topic
- They DO specialize by token syntax/context
- Analyze routing patterns to verify

## Timeline Estimate

- **1-2 hours**: Complete training loop + aux loss integration
- **2-3 hours**: Add MoE analysis tools
- **30 min**: Run full experiment (200 steps x 3 models)
- **1 hour**: Analyze results + create visualizations

**Total: 4-6 hours to complete validation**

## Expected Outcomes

If MoE architecture is successful, we should see:

✅ **Better multi-domain performance** (reasoning + memory tasks)
✅ **Diverse expert utilization** (not collapsed to 2-3 experts)
✅ **Faster convergence** due to specialization
✅ **Token-level routing patterns** (not task-level)

If not successful:
- Indicates need for more training data/steps
- May need architectural tweaks (expert capacity, routing strategy)

## Decision Tree

```
Run Experiment 1 (Full)
├─ MoE beats Unified → PROCEED with MoE-ARMT
│  └─ Next: Train on real datasets (BABILong, GSM8K)
│
└─ MoE doesn't beat Unified → INVESTIGATE
   ├─ Check expert collapse (load balancing)
   ├─ Try different expert counts  
   └─ If still fails → Unified model is best path
```

## References

Key papers guiding this implementation:
- Switch Transformer (2021): Top-1 routing, load balancing
- Mixtral (2024): Top-2 routing, 8 experts  
- OpenMoE (2024): Expert specialization analysis
- DeepSeek-MoE (2024): Fine-grained experts, shared experts
