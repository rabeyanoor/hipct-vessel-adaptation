# baseline_zeroshot.py
"""
Zero-shot SAM-Med3D baseline: pretrained image_encoder + pretrained mask_decoder,
with the PhysicsAwareAdapter and CollapseDetectionHead left at their freshly
initialized (untrained) state -- i.e. no HiP-VesselAdapt fine-tuning at all.

This is Table 1's baseline row: it establishes how well the generic foundation
model does on HiP-CT vessel pseudo-labels before any physics-aware adaptation,
which is what justifies the adapter's contribution.
"""

from evaluate import evaluate

if __name__ == "__main__":
    evaluate(checkpoint_path=None, label="Zero-Shot SAM-Med3D Baseline (no fine-tuning)")
