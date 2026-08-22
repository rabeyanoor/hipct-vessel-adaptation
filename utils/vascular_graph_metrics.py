# utils/vascular_graph_metrics.py
"""
Skeleton-graph connectivity/branching stats for binary 3D vessel masks.
Skeletonizes via skimage's Lee algorithm (3D-native, skimage.morphology.skeletonize
auto-selects it for ndim==3 -- skeletonize_3d was merged into this function and no
longer exists as a separate name in the installed skimage version), then builds a
networkx graph over 26-connected skeleton voxels so a predicted mask's vessel-tree
structure (branch points, endpoints, connected components, total physical length)
can be compared against the same stats computed on the real GT skeleton. Betti
numbers (utils/topology_metrics.py) report loop/component counts from persistent
homology directly on the volume; this instead walks the thinned centerline graph,
which is what actually exposes fragmentation (extra components) or spurious
branching (extra branch points) as vessel-tree structure.
"""

import itertools

import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

_NEIGHBOR_OFFSETS = [o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)]


def skeleton_to_graph(binary_skeleton, spacing=(1.0, 1.0, 1.0)):
    """26-connected graph over a 3D binary skeleton. Edge weight is the physical
    distance between voxel centers (um), using per-axis `spacing` -- so total edge
    weight is a real vessel length, not a voxel count."""
    coords = np.argwhere(binary_skeleton)
    voxel_set = {tuple(c) for c in coords}
    spacing = np.asarray(spacing, dtype=np.float64)

    G = nx.Graph()
    G.add_nodes_from(map(tuple, coords))
    for c in coords:
        c = tuple(c)
        for dz, dy, dx in _NEIGHBOR_OFFSETS:
            n = (c[0] + dz, c[1] + dy, c[2] + dx)
            if n > c and n in voxel_set:  # n > c: visit each undirected edge once
                dist = float(np.linalg.norm(np.array([dz, dy, dx]) * spacing))
                G.add_edge(c, n, weight=dist)
    return G


def compute_vascular_graph_stats(binary_volume, spacing=(1.0, 1.0, 1.0)):
    """Skeletonizes `binary_volume` and returns connectivity/branching stats for
    its skeleton graph: component count, branch points (degree>=3), endpoints
    (degree==1), and total physical vessel length in the units of `spacing`."""
    skel = skeletonize(np.asarray(binary_volume).astype(bool))
    n_skeleton_voxels = int(skel.sum())
    if n_skeleton_voxels == 0:
        return {
            "n_skeleton_voxels": 0, "n_components": 0, "n_branch_points": 0,
            "n_endpoints": 0, "total_length_um": 0.0, "mean_degree": 0.0,
        }

    G = skeleton_to_graph(skel, spacing)
    degrees = dict(G.degree())
    return {
        "n_skeleton_voxels": n_skeleton_voxels,
        "n_components": nx.number_connected_components(G),
        "n_branch_points": sum(1 for d in degrees.values() if d >= 3),
        "n_endpoints": sum(1 for d in degrees.values() if d == 1),
        "total_length_um": sum(d["weight"] for _, _, d in G.edges(data=True)),
        "mean_degree": sum(degrees.values()) / len(degrees),
    }
