# utils/pseudo_labels.py
"""
Frangi-vesselness pseudo-label generation.

The ESRF HiP-CT downloads are grayscale volumes only, with no vessel labels.
The first version of the pipeline used an intensity threshold of the same image
as the target, which is not a vessel label, so Dice/HD95 against it meant nothing.

This makes a pseudo-label with a multiscale Hessian (Frangi) vesselness filter,
which picks up tube-like structures. It is a heuristic, not ground truth.
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
        min_size: components smaller than this (voxels) are removed as noise.
        denoise_sigma: Gaussian smoothing before Frangi; the raw slices are
            noisy enough that the Hessian otherwise mostly responds to noise.

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
