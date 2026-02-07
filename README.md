# Fractal Llama

A Novel Architecture with Mixture of Recursive Experts (MoRE).

## Overview
This architecture combines:
1. Sparse Mixture of Experts routing
2. Recursive "thinking" within each expert
3. Step embeddings to differentiate recursion depths

Key Innovation: Each "expert" is itself a tiny recursive transformer that "thinks" for N steps before producing output. This creates a fractal-like structure where computation depth varies per token based on routing.

## Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the model:
```bash
python fractal_llama.py
```
