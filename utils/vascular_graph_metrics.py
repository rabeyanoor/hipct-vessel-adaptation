# utils/vascular_graph_metrics.py
"""
Connectivity/branching stats from the skeleton of a binary 3D vessel mask.
skimage's skeletonize uses the Lee method for 3D input (skeletonize_3d no longer
exists in the installed version). The skeleton voxels are turned into a
26-connected networkx graph, and we count branch points, endpoints, components
and total length, for both prediction and GT. Unlike the Betti numbers
(utils/topology_metrics.py), this works on the centerline, so broken-up vessels
and extra branches show up directly.
"""

import itertools

import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

_NEIGHBOR_OFFSETS = [o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)]


def skeleton_to_graph(binary_skeleton, spacing=(1.0, 1.0, 1.0)):
    """26-connected graph of a 3D skeleton. Edge weight = physical distance
    between voxel centers (um), so the sum of weights is the vessel length."""
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
    """Skeleton graph stats: components, branch points (degree>=3), endpoints
    (degree==1) and total length (in the units of `spacing`)."""
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
