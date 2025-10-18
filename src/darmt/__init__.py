"""
DARMT: Deliberative ARMT Co-Processor

Synergizing Memory and Reasoning with Adaptive Compute.

This package implements:
- ARMT (Associative Recurrent Memory Transformer) for long-context memory
- Coprocessor module for deliberative reasoning
- Adaptive compute triggers (MeCo, ARS)
- Experiment 0 for architecture validation

Based on research from 2024-2025.
"""

__version__ = "0.1.0"

from darmt.models.armt import SimpleARMT
from darmt.models.coprocessor import SimpleCoprocessor
from darmt.models.unified import UnifiedARMT
from darmt.models.dual_architecture import DualArchitectureARMT

__all__ = [
    "SimpleARMT",
    "SimpleCoprocessor",
    "UnifiedARMT",
    "DualArchitectureARMT",
]
