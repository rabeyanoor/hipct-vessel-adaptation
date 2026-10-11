# baseline_zeroshot.py
"""
Zero-shot SAM-Med3D baseline: pretrained image_encoder + mask_decoder, with the
PhysicsAwareAdapter and CollapseDetectionHead left untrained (no fine-tuning).
Baseline row for Table 1.
"""

from evaluate import evaluate

if __name__ == "__main__":
    evaluate(checkpoint_path=None, label="Zero-Shot SAM-Med3D Baseline (no fine-tuning)")
