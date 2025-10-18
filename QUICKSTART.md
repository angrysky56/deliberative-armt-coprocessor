# Quick Start Guide

## Installation

```bash
cd /your-path-to/deliberative-armt-coprocessor

# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install the package
uv pip install -e .
```

## Step 1: Run Experiment 0 (CRITICAL - Must Run First!)

**This experiment validates whether the dual architecture is worth pursuing.**

```bash
# Using the experiment script
python experiments/experiment_0_architecture_validation.py

# Or using Python directly
python -c "from darmt.evaluation.experiment_zero import run_experiment_zero; run_experiment_zero()"
```

### What Experiment 0 Tests

- **Config A (Baseline)**: ARMT alone (~137M parameters)
- **Config B (Unified)**: Deeper ARMT (~200M parameters)
- **Config C (Dual)**: ARMT + Coprocessor (~200M parameters)

### Success Criteria

✅ Config C must beat Config B by >5% on reasoning
✅ Config C must maintain Config A's memory accuracy
✅ Coprocessor must show specialization (manual analysis)

## Step 2: Interpret Results

### If "PROCEED" ✅

The dual architecture validated! You can:

1. Implement adaptive triggers (MeCo, ARS)
2. Train on real benchmarks (BABILong, GSM8K)
3. Optimize coprocessor architecture

### If "PIVOT_TO_UNIFIED" ❌

The unified model performs as well. Instead:

1. Focus on deeper unified architectures
2. Improve training objectives
3. Investigate better memory mechanisms

This finding aligns with October 2025 research.

### If "INVESTIGATE" ⚠️

Mixed results. Consider:

1. Testing on more benchmarks
2. Trying different architectures
3. Analyzing failure modes
4. Hybrid approaches

## Step 3: Next Steps (if validated)

### Train on Real Data

```python
from darmt.models.dual_architecture import DualArchitectureARMT
from darmt.models.armt import SimpleARMT
from darmt.models.coprocessor import SimpleCoprocessor

# Initialize models
armt = SimpleARMT(num_layers=12, hidden_size=768)
coprocessor = SimpleCoprocessor(num_layers=6, hidden_size=768)
model = DualArchitectureARMT(armt, coprocessor)

# Train on your data
# ... training loop ...
```

### Implement Adaptive Triggers

See `src/darmt/triggers/` for MeCo and ARS implementations (coming soon).

## Benchmarks

### BABILong

Long-context QA at 50M+ tokens.

### GSM8K

Mathematical reasoning benchmark.

### seqBench

Sequential reasoning with configurable depth.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
black src/ tests/
isort src/ tests/
mypy src/
```

## Important Notes

1. **Always run Experiment 0 first** - Don't skip this!
2. **GPU recommended** - CPU will be very slow
3. **Mock data** - Experiment 0 uses dummy data; replace with real benchmarks
4. **Parameter matching** - Ensure Config B and C have similar total parameters

## Research Papers

1. [ARMT (2024)](https://arxiv.org/abs/2407.04841)
2. [System 1/2 Communication (2025)](https://arxiv.org/abs/2510.00494)
3. [MeCo (2025)](https://arxiv.org/abs/2502.12961)
4. [ARS (2025)](https://arxiv.org/abs/2510.00071)

## Support

- Check the README for detailed documentation
- See `docs/` for additional guides
- Open issues for bugs or questions

---

**Status**: Research code - Experiment 0 validates core hypothesis
