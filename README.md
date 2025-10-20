# DARMT: Deliberative ARMT Co-Processor

**Synergizing Memory and Reasoning with Adaptive Compute**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.9+](https://img.shields.io/badge/PyTorch-2.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

DARMT (Deliberative ARMT Co-Processor) is a research implementation exploring the synergy between **associative memory** (via ARMT) and **deliberative reasoning** (via a coprocessor module). This project is based on cutting-edge research from 2024-2025 and implements adaptive compute mechanisms for efficient long-context processing.

### Key Features

- ✅ **ARMT Integration**: Associative Recurrent Memory Transformer with 50M+ token capacity
- ✅ **Dual Architecture**: Optional coprocessor for deliberative reasoning
- ✅ **Learned Memory Fusion**: Attention-based integration instead of naive concatenation
- ✅ **Adaptive Triggers**: MeCo (metacognitive) and ARS (certainty-based) mechanisms
- ✅ **Experiment 0**: Architecture validation (unified vs. dual) as per latest research
- ✅ **Modern Stack**: PyTorch 2.9+, Transformers 4.57+, Python 3.12+

## Research Background

### The Core Hypothesis

This project tests whether combining:
1. **ARMT's associative memory** (for long-context retrieval)
2. **Coprocessor's deliberative reasoning** (for complex problem-solving)
3. **Learned memory fusion** (for coherent integration)
4. **Adaptive compute triggers** (for efficiency)

...produces better results than a unified model with equivalent parameters.

### Critical Research Finding (October 2025)

**IMPORTANT**: The paper ["Exploring System 1 and 2 communication for latent reasoning in LLMs"](https://arxiv.org/abs/2510.00494) found that:

> *"A unified soft-embedding baseline—a single model with the same forward pass and shared representations—nearly matches the dual architecture performance, suggesting current dual designs mostly add compute rather than qualitatively improving reasoning."*

**Therefore, this project implements Experiment 0 FIRST** to validate whether the dual architecture provides genuine benefits before proceeding with further development.

### Key Papers

1. **ARMT**: [Associative Recurrent Memory Transformer](https://arxiv.org/abs/2407.04841) (Rodkin et al., 2024)
2. **System 1/2**: [Exploring System 1 and 2 communication for latent reasoning](https://arxiv.org/abs/2510.00494) (2025)
3. **MeCo**: [Adaptive Tool Use with Meta-Cognition Trigger](https://arxiv.org/abs/2502.12961) (2025)
4. **ARS**: [Adaptive Reasoning Suppression](https://arxiv.org/abs/2510.00071) (2025)

## Architecture: Learned Memory Fusion

### The Problem with Naive Concatenation

Early experiments revealed that **naive concatenation** of coprocessor latents with ARMT memory causes significant performance degradation:

- ❌ **Memory interference**: Marker retrieval accuracy dropped by 13%
- ❌ **Unbounded growth**: Memory size grows indefinitely (512 → 544 → 576 → ...)
- ❌ **No learned integration**: Coprocessor and ARMT compete rather than cooperate

### The Solution: Learned Memory Fusion Layer

We implement an **attention-based fusion mechanism** that properly integrates coprocessor outputs:

```
┌─────────────────────────────────────────────────────────┐
│                   Dual Architecture                      │
│                                                          │
│  ┌──────────┐  Memory   ┌──────────────┐               │
│  │   ARMT   │─────────▶ │  Coprocessor │               │
│  │ (6 layers)│           │  (3 layers)  │               │
│  └──────────┘           └──────┬───────┘               │
│       │                         │                        │
│       │ Base Memory      Latent Embeddings              │
│       │                         │                        │
│       └────────┬────────────────┘                       │
│                ▼                                         │
│     ┌─────────────────────┐                             │
│     │  Fusion Layer       │                             │
│     │  • Cross-attention  │                             │
│     │  • Gated integration│                             │
│     │  • Layer norm       │                             │
│     └─────────┬───────────┘                             │
│               ▼                                          │
│     Augmented Memory (same size as input!)              │
└─────────────────────────────────────────────────────────┘
```

#### Fusion Layer Architecture

```python
class MemoryFusionLayer:
    """
    1. Cross-Attention: Latents attend to memory
       - Extracts relevant information from base memory
       
    2. Pooling: Aggregate latent information
       - Reduces to fixed-size representation
       
    3. Gated Integration: Learned blending
       - Gate controls how much latent info to integrate
       - Preserves memory coherence
       
    4. Residual Connection: Stability
       - Ensures gradual updates to memory
    """
```

**Benefits:**
- ✅ **Fixed memory size**: No unbounded growth
- ✅ **Learned integration**: Cross-attention finds relevant information
- ✅ **Gated control**: Model learns when to integrate vs. preserve
- ✅ **Gradient flow**: Proper backpropagation to both components

**Parameter Count:** ~4.7M additional parameters for fusion layer

## Installation

### Using uv (recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <your-repo-url>
cd deliberative-armt-coprocessor

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# For development with testing tools
uv pip install -e ".[dev]"
```

### Using pip

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -e .
```

## Project Structure

```
darmt/
├── src/darmt/
│   ├── models/           # Core model implementations
│   │   ├── armt.py                 # Base ARMT architecture
│   │   ├── coprocessor.py          # Deliberative module
│   │   ├── unified.py              # Unified baseline
│   │   ├── dual_architecture.py    # ARMT + Coprocessor
│   │   └── memory_fusion.py        # Learned fusion layer ⭐NEW
│   ├── triggers/         # Adaptive compute mechanisms
│   │   ├── meco.py                 # Metacognitive trigger
│   │   ├── ars.py                  # Adaptive reasoning suppression
│   │   └── base.py                 # Abstract trigger interface
│   ├── evaluation/       # Benchmarking and metrics
│   │   ├── benchmarks.py           # BABILong, GSM8K, etc.
│   │   ├── metrics.py              # Performance metrics
│   │   ├── synthetic_tasks.py      # Synthetic evaluation tasks
│   │   └── experiment_zero.py      # Architecture validation
│   └── utils/            # Utilities
│       ├── memory.py               # Memory state management
│       └── visualization.py        # Result plotting
├── experiments/          # Experiment scripts
│   └── experiment_0_architecture_validation.py
├── tests/                # Unit tests
│   └── test_fusion_layer.py       # Fusion layer tests
└── docs/                 # Documentation
```

## Quick Start

### Step 1: Run Experiment 0 (Architecture Validation)

**This must be run first** to validate the core hypothesis:

```bash
# Run the validation experiment (~15 minutes on RTX 3060)
python experiments/experiment_0_architecture_validation.py
```

Or use the Python API:

```python
from darmt.evaluation.experiment_zero import run_experiment_zero

# Run the validation experiment
results = run_experiment_zero(
    segment_length=1024,
    num_segments=10,
    hidden_size=768,
    num_layers_armt=12,
    num_layers_coprocessor=6
)

# Check if dual architecture is worth pursuing
if results["dual_vs_unified_reasoning_gain"] > 0.05:
    print("✅ Dual architecture validated! Proceed with development.")
else:
    print("❌ Unified architecture performs as well. Consider pivoting.")
```

### Step 2: Train a Model (if Experiment 0 succeeds)

```python
from darmt.models.armt import SimpleARMT
from darmt.models.coprocessor import SimpleCoprocessor
from darmt.models.dual_architecture import DualArchitectureARMT

# Initialize components
armt = SimpleARMT(
    num_layers=12,
    hidden_size=768,
    num_heads=12
)

coprocessor = SimpleCoprocessor(
    num_layers=6,
    hidden_size=768,
    num_heads=12
)

# Create dual architecture with learned fusion
model = DualArchitectureARMT(
    armt_model=armt,
    coprocessor_model=coprocessor,
    num_latents=32,
    freeze_armt=False,          # Set True for transfer learning
    use_learned_fusion=True,    # Use attention-based fusion (recommended!)
    num_fusion_heads=8
)

print(f"Total parameters: {model.count_parameters() / 1e6:.2f}M")
print(f"Fusion layer: {model.get_fusion_parameters() / 1e6:.2f}M")
```

### Step 3: Use Adaptive Compute

```python
from darmt.triggers.meco import MetacognitiveTrigger
from darmt.triggers.ars import AdaptiveReasoningSuppression

# Use metacognitive triggering
meco_trigger = MetacognitiveTrigger(
    uncertainty_threshold=0.7,
    entropy_threshold=2.0
)

# Or use adaptive reasoning suppression
ars_trigger = AdaptiveReasoningSuppression(
    confidence_threshold=0.85,
    max_iterations=10
)
```

## Experiment 0: Architecture Validation

The most critical experiment tests three configurations:

| Config | Description | Params | Purpose |
|--------|-------------|--------|---------|
| **A** | ARMT Baseline | ~52M | Baseline performance |
| **B** | Unified ARMT (deeper) | ~61M | Parameter-matched unified model |
| **C** | ARMT + Coprocessor + Fusion | ~61M | Dual architecture with learned fusion |

### Config C Breakdown

- ARMT: ~52M parameters (6 layers)
- Coprocessor: ~5M parameters (3 layers)
- **Fusion Layer: ~4.7M parameters** ⭐
- **Total: ~61M parameters**

### Success Criteria

✅ Config C must beat Config B by >5% on reasoning tasks  
✅ Config C must maintain Config A's memory retrieval accuracy (no degradation!)  
✅ The coprocessor must show emergent specialization  
✅ Fusion layer must enable coherent memory integration

### What Changed (Recent Fixes)

**Previous Issue (Naive Concatenation):**
- Memory retrieval: 31.82% (Config C) vs 45.10% (Config B) ❌
- **Config C performed WORSE than baseline!**

**With Learned Fusion:**
- Memory size stays constant (no unbounded growth) ✅
- Cross-attention integrates information coherently ✅
- Gated mechanism preserves important information ✅
- Proper gradient flow to all components ✅

## Adaptive Compute Mechanisms

### 1. MeCo (Metacognitive Trigger)

Training-free method that uses the model's internal representation to decide when to invoke the coprocessor.

```python
# Signals used:
# - Prediction entropy (uncertainty)
# - Confidence (max softmax probability)
# - Attention dispersion
# - Hidden state variance
```

### 2. ARS (Adaptive Reasoning Suppression)

Training-free method that dynamically suppresses redundant reasoning steps based on certainty monitoring.

```python
# Features:
# - Multi-checkpoint certainty estimation
# - Progressive suppression thresholds
# - Up to 53% token reduction while maintaining accuracy
```

## Benchmarks

The project includes implementations for:

- **BABILong**: Long-context QA (up to 50M tokens)
- **GSM8K**: Mathematical reasoning
- **seqBench**: Sequential reasoning with configurable depth
- **LongBench v2**: Real-world multi-document QA

## Development

### Running Tests

```bash
# Test fusion layer
python tests/test_fusion_layer.py

# Run full test suite
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
ruff check src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Implementation Details

### Memory Fusion Layer

The fusion layer implements sophisticated memory integration:

1. **Cross-Attention Phase**: Latent embeddings attend to base memory
   - Query: Coprocessor latents
   - Key/Value: ARMT memory tokens
   - Output: Attended latents with relevant memory information

2. **Pooling Phase**: Aggregate attended latents
   - Mean pooling across latent dimension
   - Broadcast to match memory sequence length

3. **Gated Integration Phase**: Learn when to integrate
   - Concatenate memory + pooled latents
   - Compute integration gate (sigmoid)
   - Blend via: `gate * fused + (1 - gate) * original`

4. **Projection Phase**: Final transformation
   - Feed-forward network
   - Layer normalization
   - Residual connection

This ensures:
- ✅ Coherent memory updates
- ✅ Preservation of important information
- ✅ Learnable integration strategy
- ✅ Stable training dynamics

## Citation

If you use this code in your research, please cite the relevant papers:

```bibtex
@misc{rodkin2024associativerecurrentmemorytransformer,
    title={Associative Recurrent Memory Transformer}, 
    author={Ivan Rodkin and Yuri Kuratov and Aydar Bulatov and Mikhail Burtsev},
    year={2024},
    eprint={2407.04841},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Acknowledgments

This project builds upon research from:
- **ARMT team** (Ivan Rodkin, Yuri Kuratov, et al.)
- **System 1/2 communication paper** (October 2025)
- **MeCo** (Wenjun Li, et al., 2025)
- **ARS** (Dongqi Zheng, 2025)

---

**Status**: 🚧 Experimental Research Code

**Recent Update**: Implemented learned memory fusion layer to fix memory interference issues

**Next Steps**: 
1. ✅ Run Experiment 0 with learned fusion
2. Validate that dual architecture now shows benefits
3. Implement adaptive compute triggers if validated
