# utils/topology_metrics.py
"""
Betti number computation for binary 3D vessel masks, via cubical-complex
persistent homology (gudhi). b0 = connected components, b1 = independent loops
(topological handles) -- both meaningful for vessel networks, which should form
one connected tree/graph (few components) with branching but few spurious loops.
"""

import numpy as np
import gudhi


def compute_betti_numbers(binary_volume):
    """
    Args:
        binary_volume: 3D numpy array, 0/1 (or bool).
    Returns:
        (b0, b1): connected components, independent loops.
    """
    vol = np.asarray(binary_volume).astype(np.uint8)
    # Sublevel-set filtration over occupied voxels: invert so occupied (1) -> 0 (low),
    # background (0) -> 1 (high), then read off topology at threshold 0.5.
    cc = gudhi.CubicalComplex(top_dimensional_cells=1 - vol)
    cc.compute_persistence()
    betti = cc.persistent_betti_numbers(from_value=0.5, to_value=-0.5)
    b0 = betti[0] if len(betti) > 0 else 0
    b1 = betti[1] if len(betti) > 1 else 0
    return b0, b1
