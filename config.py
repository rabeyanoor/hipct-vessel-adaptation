# config.py
"""
Global configuration and metadata for HiP-VesselAdapt multi-organ pipeline.
Registers spacing, paths, and training parameters for Heart, Lung, and Kidney datasets.
"""

ORGAN_CONFIG = {
    "heart": {
        "raw_dir": "data/raw/heart_jp2",
        "path": "data/heart.nii.gz",
        "spacing": [26.04, 26.04, 26.04],  # Voxel spacing in micrometers (μm)
        "black_ridges": True,  # verified visually during development: vessel lumen is darker than tissue
    },
    "lung": {
        "raw_dir": "data/raw/lung_jp2",
        "path": "data/lung.nii.gz",
        "spacing": [169.36, 169.36, 169.36], # Voxel spacing in micrometers (μm)
        "black_ridges": True,  # verified visually during development: a visible airway/vessel cross-section is only traced with black_ridges=True
    },
    "kidney": {
        "raw_dir": "data/raw/kidney_jp2",
        "path": "data/kidney.nii.gz",
        "spacing": [50.00, 50.00, 50.00],    # Voxel spacing in micrometers (μm)
        "black_ridges": True,  # both polarities picked up the same vessel-wall ring at the center patch during development; True kept for consistency
    }
}

TRAIN_CONFIG = {
    "patch_size": (64, 64, 64),
    "batch_size": 1,
    "num_epochs": 10,
    "samples_per_epoch": 6,  # per organ; kept at the size that has completed reliably twice on this memory-flaky machine
    "lr": 1e-4,
    "checkpoint_dir": "checkpoints",
    "best_model_path": "checkpoints/best_model.pth"
}
