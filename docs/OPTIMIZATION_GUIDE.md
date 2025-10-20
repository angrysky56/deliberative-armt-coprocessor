# Unified Architecture Optimization Guide

**Technical roadmap for optimizing DARMT unified models**

## Overview

This guide provides detailed technical steps for optimizing the unified ARMT architecture based on experimental findings from Experiments 0 and 1.

**Core finding**: Unified models outperform both dual architectures and sparse MoE while using 2× fewer parameters.

---

## Phase 1: Architecture Optimization (Weeks 1-2)

### 1.1 Depth-Width Scaling Experiments

**Goal**: Find optimal model configuration for 50-100M parameter budget

#### Experiment Setup

Create `experiments/experiment_2_unified_optimization.py`:

```python
from darmt.models.unified import UnifiedARMT
from darmt.evaluation.synthetic_tasks import (
    SyntheticMemoryTask,
    SyntheticReasoningTask, 
    MultiHopReasoningTask
)

# Test configurations
CONFIGS = {
    "deep_narrow": {
        "num_layers": 18,
        "hidden_size": 512,
        "num_heads": 16,
        "intermediate_size": 2048,
        "params_M": 60
    },
    "balanced": {
        "num_layers": 12,
        "hidden_size": 768,
        "num_heads": 12,
        "intermediate_size": 3072,
        "params_M": 80
    },
    "wide_shallow": {
        "num_layers": 9,
        "hidden_size": 1024,
        "num_heads": 16,
        "intermediate_size": 4096,
        "params_M": 100
    }
}

def run_scaling_experiment():
    """Test each configuration on synthetic tasks."""
    results = {}
    
    for name, config in CONFIGS.items():
        print(f"\nTesting {name} configuration...")
        
        # Initialize model
        model = UnifiedARMT(**config)
        
        # Train for 500 steps (longer than Exp 0/1)
        # Evaluate on all tasks
        # Record: memory_acc, reasoning_acc, multihop_acc, training_time
        
        results[name] = train_and_evaluate(model, config)
    
    return results
```

#### Success Criteria

| Metric | Current Baseline | Target | Priority |
|--------|------------------|--------|----------|
| Memory Retrieval | 16-33% | **70%+** | CRITICAL |
| Pattern Reasoning | 33% | **50%+** | HIGH |
| Multi-hop Reasoning | 100% | **95%+** | MAINTAIN |
| Training Time | N/A | < 30 min (GPU) | MEDIUM |

#### Analysis Framework

```python
def analyze_scaling_results(results):
    """
    Analyze trade-offs:
    1. Performance vs. parameter count
    2. Performance vs. training time
    3. Memory vs. reasoning trade-offs
    4. Layer depth vs. width effects
    """
    
    # Plot performance curves
    plot_pareto_frontier(results, x="params_M", y="memory_acc")
    plot_pareto_frontier(results, x="params_M", y="reasoning_acc")
    
    # Identify optimal configuration
    best_config = find_best_tradeoff(
        results, 
        weights={"memory": 0.5, "reasoning": 0.3, "efficiency": 0.2}
    )
    
    return best_config
```

---

### 1.2 Memory Token Optimization

**Goal**: Optimize number and initialization of memory tokens

#### Current Status

- **Current**: 16 memory tokens
- **Performance**: 16-33% retrieval accuracy
- **Issue**: May need more capacity or better initialization

#### Experiments

```python
MEMORY_CONFIGS = [
    {"num_tokens": 8, "description": "Minimal memory"},
    {"num_tokens": 16, "description": "Current baseline"},
    {"num_tokens": 32, "description": "2× capacity"},
    {"num_tokens": 64, "description": "4× capacity"},
    {"num_tokens": 128, "description": "Maximum capacity"},
]

def test_memory_capacity():
    """Test memory token count vs. retrieval accuracy."""
    
    # Use best architecture from 1.1
    base_config = load_optimal_config()
    
    results = {}
    for mem_config in MEMORY_CONFIGS:
        config = {**base_config, **mem_config}
        model = UnifiedARMT(**config)
        
        # Train specifically on memory-heavy tasks
        acc = train_memory_tasks(model, num_steps=1000)
        
        results[mem_config["num_tokens"]] = {
            "accuracy": acc,
            "params_M": count_parameters(model) / 1e6,
            "memory_overhead": mem_config["num_tokens"] * config["hidden_size"]
        }
    
    return results
```

#### Memory Initialization Strategies

Test different initialization approaches:

```python
# 1. Random (current)
memory_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_size))

# 2. Orthogonal (better for diversity)
memory_tokens = nn.Parameter(torch.nn.init.orthogonal_(
    torch.empty(1, num_tokens, hidden_size)
))

# 3. Learned from data (pretrain on memory tasks)
memory_tokens = pretrain_memory_tokens(
    dataset=memory_task_dataset,
    num_tokens=num_tokens,
    hidden_size=hidden_size
)

# 4. Cluster-based (initialize with K-means of embeddings)
memory_tokens = initialize_from_clusters(
    embeddings=token_embeddings,
    num_clusters=num_tokens
)
```

#### Expected Outcomes

**Hypothesis**: 
- More tokens → better retrieval (up to a point)
- Diminishing returns after 32-64 tokens
- Optimal: 32 tokens for 512H, 64 tokens for 768H

**Validation metric**: Retrieval accuracy improvement per added memory token

---

### 1.3 Attention Pattern Analysis

**Goal**: Understand memory-input interactions for architectural insights

#### Analysis Tools

```python
def analyze_attention_patterns(model, inputs, memory_state):
    """
    Extract and visualize attention patterns:
    1. Memory → Input attention (what memory attends to)
    2. Input → Memory attention (what inputs attend to)
    3. Layer-wise attention evolution
    4. Head specialization patterns
    """
    
    # Hook attention weights
    attention_weights = {}
    
    def hook_fn(module, input, output):
        attention_weights[module.name] = output[1]  # attention weights
    
    # Register hooks
    for name, layer in model.named_modules():
        if isinstance(layer, nn.MultiheadAttention):
            layer.register_forward_hook(hook_fn)
    
    # Forward pass
    output = model(inputs, memory_state)
    
    # Analyze patterns
    memory_input_attn = extract_memory_input_attention(attention_weights)
    input_memory_attn = extract_input_memory_attention(attention_weights)
    
    return {
        "memory_to_input": memory_input_attn,
        "input_to_memory": input_memory_attn,
        "layer_evolution": compute_layer_evolution(attention_weights),
        "head_specialization": analyze_head_roles(attention_weights)
    }
```

#### Visualizations

```python
def visualize_attention_patterns(analysis_results):
    """
    Create visualizations:
    1. Heatmap: Memory tokens × Input positions
    2. Line plot: Attention entropy across layers
    3. Clustering: Head specialization groups
    4. Animation: Attention evolution during processing
    """
    
    # Memory-Input attention heatmap
    plot_attention_heatmap(
        attention=analysis_results["memory_to_input"],
        title="Memory Token Attention to Input Positions"
    )
    
    # Layer-wise entropy evolution
    plot_entropy_evolution(
        entropy_per_layer=analysis_results["layer_evolution"],
        title="Attention Concentration Across Layers"
    )
    
    # Head role clustering
    plot_head_clusters(
        specialization=analysis_results["head_specialization"],
        title="Attention Head Specialization Patterns"
    )
```

#### Key Questions to Answer

1. **Do memory tokens specialize?**
   - Are some tokens for recent context, others for patterns?
   - Does specialization emerge naturally or need inductive bias?

2. **What do inputs attend to in memory?**
   - Do reasoning tasks attend differently than memory tasks?
   - Can we identify "retrieval" vs "reasoning" attention patterns?

3. **How many attention heads are needed?**
   - Do all heads contribute equally?
   - Can we prune redundant heads?

4. **Layer-wise behavior:**
   - Early layers: More input-focused?
   - Late layers: More memory-focused?

---

## Phase 2: Training Optimization (Weeks 2-3)

### 2.1 Multi-Task Loss Engineering

**Current Problem**: Models plateau at ~33% reasoning accuracy

#### Balanced Multi-Task Loss

```python
class MultiTaskLoss(nn.Module):
    """
    Balanced loss for memory + reasoning tasks.
    
    Addresses:
    1. Task imbalance (memory is harder than reasoning)
    2. Loss scale differences
    3. Curriculum learning
    """
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha  # Memory weight
        self.beta = beta    # Reasoning weight
        self.gamma = gamma  # Multi-hop weight
        
    def forward(self, outputs, targets, task_type):
        if task_type == "memory":
            loss = self.memory_loss(outputs, targets)
            return self.alpha * loss
            
        elif task_type == "reasoning":
            loss = self.reasoning_loss(outputs, targets)
            return self.beta * loss
            
        elif task_type == "multihop":
            loss = self.multihop_loss(outputs, targets)
            return self.gamma * loss
    
    def memory_loss(self, outputs, targets):
        """
        Marker retrieval loss.
        
        Problem: Only a few positions matter (markers)
        Solution: Weighted loss focusing on marker positions
        """
        logits = outputs["logits"]
        marker_positions = targets["marker_positions"]
        marker_tokens = targets["marker_tokens"]
        
        # Compute loss only at marker positions
        loss = 0.0
        for batch_idx in range(logits.size(0)):
            for pos, token in zip(marker_positions[batch_idx], marker_tokens[batch_idx]):
                token_logits = logits[batch_idx, pos]
                loss += F.cross_entropy(token_logits.unsqueeze(0), token.unsqueeze(0))
        
        return loss / (logits.size(0) * len(marker_positions[0]))
```

#### Dynamic Loss Weighting

```python
class DynamicLossWeights:
    """
    Adjust loss weights based on task difficulty.
    
    Idea: Give more weight to tasks the model is struggling with
    """
    
    def __init__(self, initial_weights={"memory": 1.0, "reasoning": 1.0}):
        self.weights = initial_weights
        self.task_losses = {task: [] for task in initial_weights}
    
    def update(self, task, loss_value):
        """Update weights based on recent performance."""
        self.task_losses[task].append(loss_value)
        
        # Recompute weights every 50 steps
        if len(self.task_losses[task]) >= 50:
            avg_losses = {
                task: np.mean(losses[-50:]) 
                for task, losses in self.task_losses.items()
            }
            
            # Weight ∝ relative difficulty
            total = sum(avg_losses.values())
            self.weights = {
                task: loss / total 
                for task, loss in avg_losses.items()
            }
    
    def get_weight(self, task):
        return self.weights[task]
```

---

### 2.2 Curriculum Learning

**Goal**: Start with easy examples, gradually increase difficulty

#### Memory Task Curriculum

```python
class MemoryCurriculum:
    """
    Progressive memory task difficulty:
    1. Close markers (easy retrieval)
    2. Distant markers (harder retrieval)
    3. More distractors
    4. Longer contexts
    """
    
    def __init__(self, num_stages=5):
        self.stage = 0
        self.num_stages = num_stages
        
    def get_task_params(self):
        """Return parameters for current curriculum stage."""
        if self.stage == 0:
            return {
                "num_markers": 3,
                "context_length": 256,
                "marker_separation": 50,
                "distractors": 0
            }
        elif self.stage == 1:
            return {
                "num_markers": 5,
                "context_length": 512,
                "marker_separation": 100,
                "distractors": 10
            }
        # ... more stages
    
    def should_advance(self, accuracy):
        """Advance curriculum when accuracy > 80%."""
        if accuracy > 0.8 and self.stage < self.num_stages - 1:
            self.stage += 1
            return True
        return False
```

#### Reasoning Task Curriculum

```python
class ReasoningCurriculum:
    """
    Progressive reasoning difficulty:
    1. Short patterns (length 3)
    2. Medium patterns (length 5)
    3. Long patterns (length 7)
    4. Complex rules
    """
    
    def get_task_params(self):
        return {
            "pattern_length": 3 + (self.stage * 2),
            "pattern_complexity": ["simple", "medium", "complex"][min(self.stage, 2)],
            "num_examples": 5 + self.stage
        }
```

---

### 2.3 Data Augmentation

**Goal**: Increase task diversity without collecting more data

#### Memory Task Augmentation

```python
def augment_memory_task(batch):
    """
    Augmentations:
    1. Vary marker positions
    2. Add random distractors
    3. Shuffle context segments (test order-invariance)
    4. Inject noise tokens
    """
    
    # Randomize marker positions
    batch = randomize_marker_positions(batch, std=20)
    
    # Add distractors (tokens similar to markers)
    batch = inject_distractors(batch, num_distractors=5)
    
    # Segment shuffling (test if order matters)
    if random.random() < 0.3:
        batch = shuffle_segments(batch)
    
    # Token-level noise
    batch = add_token_noise(batch, noise_rate=0.05)
    
    return batch
```

#### Reasoning Task Augmentation

```python
def augment_reasoning_task(batch):
    """
    Augmentations:
    1. Vary pattern types
    2. Add irrelevant context
    3. Test rule generalization
    4. Negative examples (wrong patterns)
    """
    
    # Multiple pattern types
    pattern_types = ["arithmetic", "fibonacci", "geometric", "custom"]
    batch["pattern_type"] = random.choice(pattern_types)
    
    # Irrelevant context
    batch = prepend_irrelevant_tokens(batch, num_tokens=50)
    
    # Hard negatives
    if random.random() < 0.2:
        batch = create_hard_negative(batch)
    
    return batch
```

---

### 2.4 Training Loop Improvements

#### Learning Rate Schedule

```python
def get_lr_schedule(optimizer, num_training_steps):
    """
    Warmup + Cosine decay schedule.
    
    Better than constant LR:
    - Warmup prevents early instability
    - Cosine decay finds better minima
    """
    
    warmup_steps = int(0.1 * num_training_steps)
    
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < warmup_steps:
            return current_step / warmup_steps
        
        # Cosine decay phase
        progress = (current_step - warmup_steps) / (num_training_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

#### Gradient Clipping

```python
# Prevent gradient explosions
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### Mixed Precision Training

```python
# Faster training on GPU with minimal accuracy loss
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(inputs)
    loss = criterion(output, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Phase 3: Real Benchmark Evaluation (Weeks 4-5)

### 3.1 BABILong Setup

```python
from darmt.evaluation.benchmarks import BABILongBenchmark

# Initialize benchmark
benchmark = BABILongBenchmark(
    tasks=["single_supporting_fact", "two_supporting_facts"],
    context_lengths=[1000, 5000, 10000],
    num_samples=100
)

# Evaluate model
results = benchmark.evaluate(
    model=optimized_unified_model,
    batch_size=4,
    max_length=512
)

# Expected results
"""
Context 1K: > 90% (baseline)
Context 5K: > 85% (good)
Context 10K: > 80% (excellent)
"""
```

### 3.2 GSM8K Setup

```python
from darmt.evaluation.benchmarks import GSM8KBenchmark

# Math reasoning benchmark
benchmark = GSM8KBenchmark(
    split="test",
    num_samples=500
)

results = benchmark.evaluate(
    model=optimized_unified_model,
    use_chain_of_thought=True
)

# Target: > 20% accuracy for small models
```

### 3.3 LongBench v2 Setup

```python
from darmt.evaluation.benchmarks import LongBenchV2

benchmark = LongBenchV2(
    tasks=["multi_document_qa", "summarization"],
    context_lengths=[4000, 8000]
)

results = benchmark.evaluate(model=optimized_unified_model)
```

---

## Phase 4: Production Optimizations (Week 6+)

### 4.1 Inference Speed

#### KV-Cache for Memory Tokens

```python
class OptimizedUnifiedARMT(UnifiedARMT):
    """
    Inference-optimized version with KV-cache.
    
    Idea: Cache memory token keys/values since they don't change
    """
    
    def forward_with_cache(self, input_ids, kv_cache=None):
        if kv_cache is None:
            # First call: compute memory KV
            memory_kv = self.compute_memory_kv()
            kv_cache = {"memory_kv": memory_kv}
        
        # Reuse cached memory KV
        output = self.forward_fast(input_ids, kv_cache["memory_kv"])
        
        return output, kv_cache
```

#### Flash Attention

```python
# Install flash-attn
# pip install flash-attn

from flash_attn import flash_attn_func

# Replace standard attention with Flash Attention
# 2-4× faster, same accuracy
```

---

### 4.2 Model Quantization

```python
# INT8 quantization (minimal accuracy loss)
from torch.quantization import quantize_dynamic

quantized_model = quantize_dynamic(
    model,
    {nn.Linear},  # Quantize linear layers
    dtype=torch.qint8
)

# Results:
# - 4× smaller model size
# - 2-3× faster inference
# - < 1% accuracy loss
```

---

## Success Metrics Summary

### Phase 1 Targets (Weeks 1-2)

- [ ] Memory retrieval: **70%+** (currently 16-33%)
- [ ] Pattern reasoning: **50%+** (currently 33%)
- [ ] Multi-hop: **95%+** (maintain 100%)
- [ ] Optimal architecture identified

### Phase 2 Targets (Weeks 2-3)

- [ ] Training loss convergence: **< 0.3** (currently ~0.75)
- [ ] Curriculum learning: **5 stages** implemented
- [ ] Data augmentation: **3× diversity**

### Phase 3 Targets (Weeks 4-5)

- [ ] BABILong 1K: **> 90%**
- [ ] BABILong 10K: **> 80%**
- [ ] GSM8K: **> 20%**

### Phase 4 Targets (Week 6+)

- [ ] Inference speed: **2× faster**
- [ ] Model size: **4× smaller** (quantized)
- [ ] Production deployment ready

---

## Monitoring & Debugging

### Training Visualization

```python
import wandb

# Initialize Weights & Biases
wandb.init(project="darmt-optimization")

# Log metrics
wandb.log({
    "memory_accuracy": mem_acc,
    "reasoning_accuracy": reason_acc,
    "training_loss": loss.item(),
    "learning_rate": optimizer.param_groups[0]["lr"]
})

# Log attention patterns
wandb.log({"attention_heatmap": wandb.Image(attention_viz)})
```

### Gradient Analysis

```python
def check_gradient_health(model):
    """
    Monitor gradient flow:
    1. Check for vanishing gradients
    2. Check for exploding gradients
    3. Identify dead neurons
    """
    
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append((name, grad_norm))
    
    # Warn about issues
    for name, norm in grad_norms:
        if norm < 1e-6:
            print(f"⚠️  Vanishing gradient in {name}: {norm}")
        elif norm > 100:
            print(f"⚠️  Exploding gradient in {name}: {norm}")
```

---

## Next Steps

1. **Start Phase 1.1**: Run depth-width scaling experiments
2. **Set up monitoring**: Initialize W&B logging
3. **Create experiment branch**: `git checkout -b phase-1-optimization`
4. **Run baseline**: Establish current performance as reference

**Estimated timeline**: 4-6 weeks to production-ready unified model

**Team requirements**:
- 1 researcher: Architecture experiments
- 1 engineer: Training optimization
- GPU resources: 1× RTX 3060 (sufficient for phases 1-3)
