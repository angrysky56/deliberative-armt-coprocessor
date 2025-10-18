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
- ✅ **Adaptive Triggers**: MeCo (metacognitive) and ARS (certainty-based) mechanisms
- ✅ **Experiment 0**: Architecture validation (unified vs. dual) as per latest research
- ✅ **Modern Stack**: PyTorch 2.9+, Transformers 4.57+, Python 3.12+

## Research Background

### The Core Hypothesis

This project tests whether combining:
1. **ARMT's associative memory** (for long-context retrieval)
2. **Coprocessor's deliberative reasoning** (for complex problem-solving)
3. **Adaptive compute triggers** (for efficiency)

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
│   │   ├── armt.py              # Base ARMT architecture
│   │   ├── coprocessor.py       # Deliberative module
│   │   ├── unified.py           # Unified baseline
│   │   └── dual_architecture.py # ARMT + Coprocessor
│   ├── triggers/         # Adaptive compute mechanisms
│   │   ├── meco.py              # Metacognitive trigger
│   │   ├── ars.py               # Adaptive reasoning suppression
│   │   └── base.py              # Abstract trigger interface
│   ├── evaluation/       # Benchmarking and metrics
│   │   ├── benchmarks.py        # BABILong, GSM8K, etc.
│   │   ├── metrics.py           # Performance metrics
│   │   └── experiment_zero.py   # Architecture validation
│   └── utils/            # Utilities
│       ├── memory.py            # Memory state management
│       └── visualization.py     # Result plotting
├── experiments/          # Experiment scripts
│   └── experiment_0_architecture_validation.py
├── tests/               # Unit tests
└── docs/                # Documentation
```

## Quick Start

### Step 1: Run Experiment 0 (Architecture Validation)

**This must be run first** to validate the core hypothesis:

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
from darmt.models.dual_architecture import DualArchitectureARMT
from darmt.triggers.meco import MetacognitiveTrigger

# Initialize model
model = DualArchitectureARMT(
    armt_layers=12,
    coprocessor_layers=6,
    hidden_size=768,
    trigger=MetacognitiveTrigger()
)

# Train on your data
# ... training code ...
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
| **A** | ARMT Baseline | ~137M | Baseline performance |
| **B** | Unified ARMT (deeper) | ~200M | Parameter-matched unified model |
| **C** | ARMT + Coprocessor | ~200M | Dual architecture |

### Success Criteria

✅ Config C must beat Config B by >5% on reasoning tasks  
✅ Config C must maintain Config A's memory retrieval accuracy  
✅ The coprocessor must show emergent specialization

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

**Next Steps**: Run Experiment 0 to validate core hypothesis before proceeding with full implementation.
