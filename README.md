# NRR Operators

Reference implementation for **Non-Resolution Reasoning (NRR)** operators from:

> Saito, K. (2026). Design Principles for Non-Collapsing Operators: Information Preservation in NRR State Transformations. *arXiv preprint* arXiv:XXXX.XXXXX.

## Overview

This repository provides a working implementation of the eight principle-satisfying operators described in the paper:

- **δ (Dampening)**: Reduces overconfidence
- **σ (Stripping)**: Proportional bias removal
- **ρ (Positioning)**: Temporal coordinate assignment
- **α (Abstraction)**: Relational structure augmentation
- **ι (Invariance)**: Stable structure extraction
- **τ (Deferred)**: Identity operation until output
- **κ (CPP)**: Contradiction-preserving integration
- **π (Persistence)**: Historical state preservation

## Quick Start
```python
from nrr_prototype import NRRState, Interpretation, NRROperators

# Create state
state = NRRState([
    Interpretation("Financial crisis", "economic", 0.7),
    Interpretation("Personal dissolution", "psychological", 0.2),
])

# Apply operator
state_new = NRROperators.dampening(state, lambda_param=0.3)

print(f"Entropy: {state.entropy():.3f} → {state_new.entropy():.3f} bits")
# Output: Entropy increased (no collapse)
```

## Installation
```bash
pip install numpy
```

Then download `nrr_prototype.py` and import it.

## Validation Demo

Run the paper's validation demonstration:
```bash
python demo_for_paper.py
```

This reproduces the results shown in Paper 3, Appendix C, Figure 3.

## Core Components

- `nrr_prototype.py` - Main implementation
  - `NRRState`: State space management
  - `NRROperators`: 8 operators
  - `CollapseDetector`: Entropy validation
  - `LLMtoNRRBridge`: LLM integration

- `demo_for_paper.py` - Validation demonstration

## Key Results

- **Principle-satisfying operators**: 0% collapse rate
- **Principle-violating operators**: 15.6% collapse rate (across 135 states)

## Citation
```bibtex
@article{saito2026operators,
  title={Design Principles for Non-Collapsing Operators},
  author={Saito, Kei},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

CC BY 4.0

## Related Papers

- Paper 1: [arXiv:2512.13478](https://arxiv.org/abs/2512.13478)
- Paper 2: [arXiv:2601.19933](https://arxiv.org/abs/2601.19933)
- Paper 3: arXiv:XXXX.XXXXX (this paper)
