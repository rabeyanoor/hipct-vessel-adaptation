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
        "black_ridges": True,  # vessel lumen renders darker than surrounding tissue in this scan
    },
    "lung": {
        "raw_dir": "data/raw/lung_jp2",
        "path": "data/lung.nii.gz",
        "spacing": [169.36, 169.36, 169.36], # Voxel spacing in micrometers (μm)
        "black_ridges": True,  # airway/vessel lumen is the dark phase relative to parenchyma
    },
    "kidney": {
        "raw_dir": "data/raw/kidney_jp2",
        "path": "data/kidney.nii.gz",
        "spacing": [50.00, 50.00, 50.00],    # Voxel spacing in micrometers (μm)
        "black_ridges": True,  # kept consistent with the other two organs' polarity
    }
}

# Extra kidney volumes from the Human Organ Atlas raw archive (HF mirror
# eternalaudrey/hipct_3d) -- same donor+side as kidney_1_dense/kidney_1_voi's
# specimen (LADAF-2021-17-Right), at additional native resolutions. Raw
# grayscale only, no expert GT, so these are Frangi pseudo-labeled via
# MultiOrganHiPCTDataset (utils/dataset.py) the same way ORGAN_CONFIG's
# heart/lung/kidney are -- NOT mixed into RealGTKidneyDataset's real-GT
# training (README.md section 6: real GT and pseudo-labels must not be
# combined quantitatively). Addresses baki.sh item 3 (wider real-spacing
# training data): the existing kidney_1_dense/voi pair only spans 50um/5.2um;
# this adds three more native spacings for the same physical kidney.
HOA_KIDNEY_MULTISPACING_CONFIG = {
    "kidney_hoa17_25um": {
        "spacing": [25.0, 25.0, 25.0],
        "black_ridges": True,
        "source": "eternalaudrey/hipct_3d (HF), LADAF-2021-17_kidney_right_complete-organ_25.0um_bm05",
    },
    "kidney_hoa17_6um": {
        "spacing": [6.5, 6.5, 6.5],
        "black_ridges": True,
        "source": "eternalaudrey/hipct_3d (HF), LADAF-2021-17_kidney_right_VOI-02_6.5um_bm05",
    },
    "kidney_hoa17_2um": {
        "spacing": [2.6, 2.6, 2.6],
        "black_ridges": True,
        "source": "eternalaudrey/hipct_3d (HF), LADAF-2021-17_kidney_right_VOI-01.1_2.6um_bm05",
    },
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
