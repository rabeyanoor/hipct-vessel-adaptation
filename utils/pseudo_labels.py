# utils/pseudo_labels.py
"""
Frangi-vesselness pseudo-label generation.

HiP-CT raw downloads from the ESRF portal ship as grayscale tomography volumes only --
there is no expert vessel annotation included. Training/evaluating against a raw
intensity threshold of the same image (as the pipeline previously did) is not a vessel
label at all, so DSC/HD95 computed against it are not meaningful.

This module builds a weak/pseudo vessel-segmentation label from a multiscale Hessian
(Frangi) vesselness filter, which responds to tubular structures rather than raw
intensity. This is a heuristic label, not ground truth -- any paper using it must state
that explicitly (see module docstring in evaluate.py).
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import remove_small_objects, closing, ball


def generate_vessel_pseudolabel(volume, black_ridges=False, sigmas=(1, 2, 3), min_size=30,
                                 denoise_sigma=1.0):
    """
    Args:
        volume: 3D numpy array (D, H, W), raw or normalized intensity.
        black_ridges: True if vessels appear darker than surrounding tissue,
            False if vessels appear brighter (see ORGAN_CONFIG["black_ridges"]).
        sigmas: Gaussian scales (in voxels) spanning the expected vessel radii.
        min_size: connected components smaller than this (in voxels) are dropped
            as vesselness-filter noise rather than real vessel segments.
        denoise_sigma: raw HiP-CT slices are speckle-noisy enough that the
            Hessian eigenvalues Frangi relies on are dominated by noise rather
            than the underlying tubular structure; a light Gaussian
            pre-smoothing pass lets the vesselness response track real
            structure instead.

    Returns:
        float32 binary mask, same shape as `volume`.
    """
    vol = volume.astype(np.float32)
    vmin, vmax = vol.min(), vol.max()
    rng = vmax - vmin
    if rng > 1e-8:
        vol = (vol - vmin) / rng

    vol_smooth = gaussian_filter(vol, sigma=denoise_sigma) if denoise_sigma > 0 else vol
    vesselness = frangi(vol_smooth, sigmas=sigmas, black_ridges=black_ridges)

    if vesselness.max() <= 1e-8:
        return np.zeros_like(vol, dtype=np.float32)

    try:
        thresh = threshold_otsu(vesselness)
    except ValueError:
        thresh = np.percentile(vesselness, 95)

    mask = vesselness > thresh
    mask = remove_small_objects(mask, max_size=min_size)
    mask = closing(mask, ball(1))

    return mask.astype(np.float32)
