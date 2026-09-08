# k-vertices

Reference implementation of hierarchical k-vertices with minimum-volume
regularization, accompanying the paper *Hierarchical K-Vertices for
Simplex-Constrained Soft Representation* (under review).

## Requirements

numpy, scipy

## Usage

```python
import numpy as np
from kvertices import hierarchical_k_vertices, refine_minvol, encode

X = np.random.randn(20, 1000)  # data matrix, (features, samples)

W0, _ = hierarchical_k_vertices(X, k_target=5, splitting_method="max_gap")
W, H = refine_minvol(X, W0)    # H+Ref model: vertices W, memberships H
H_new = encode(W, X)           # simplex memberships for (new) data
```
