# run_ablation.py
"""
Progressive ablation: SAM-Med3D (zero-shot) -> +LoRA (Plain LoRA) -> +spacing
conditioning (Spacing-Conditioned LoRA) -> +clDice loss term. Isolates each
component's contribution one stage at a time instead of comparing four
unrelated, independently-configured models.

Stages 1-3 reuse the exact model classes and training functions already used
for section 5.2/5.3's results (evaluate_real_gt.py's default model_class for
zero-shot, train_plain_lora.py, train_spacing_lora.py) -- this only adds the
4th comparison point (clDice loss term on vs off, via HiPVesselLoss's
use_cldice flag) and wraps all four into one run so they're trained/evaluated
under an identical protocol and reported together in one place.

Each stage still trains for real (this is not a shortcut/mock) -- run it with
--epochs matched to whatever the main comparison used, and give it the time
that implies.
"""

import argparse
import json

from evaluate_real_gt import evaluate_real_gt
from models.plain_lora_model import PlainLoRAModel
from models.spacing_lora_model import SpacingConditionedLoRAModel
from train_plain_lora import train_plain_lora
from train_spacing_lora import train_spacing_lora

STAGES = ["zero_shot", "plain_lora", "spacing_lora", "spacing_lora_no_cldice"]


def run_ablation(stages=None, epochs=10, samples_per_epoch=20, raw_tiff_root=None,
                  ood_sample_id="kidney_1_voi", samples_per_eval=20, ood_samples_per_eval=40,
                  results_path="ablation_results.json"):
    stages = stages or STAGES
    results = {}

    if "zero_shot" in stages:
        results["zero_shot_in_dist"] = evaluate_real_gt(
            checkpoint_path=None, label="Ablation stage 1: zero-shot SAM-Med3D (in-dist)",
            samples_per_sample=samples_per_eval, raw_tiff_root=raw_tiff_root)

    if "plain_lora" in stages:
        train_plain_lora(resume=False, extra_epochs=epochs, samples_per_epoch=samples_per_epoch,
                          raw_tiff_root=raw_tiff_root)
        results["plain_lora_in_dist"] = evaluate_real_gt(
            checkpoint_path="checkpoints_plain_lora/best_model_plain_lora.pth",
            label="Ablation stage 2: +LoRA, no spacing conditioning (in-dist)",
            model_class=PlainLoRAModel, samples_per_sample=samples_per_eval, raw_tiff_root=raw_tiff_root)
        results["plain_lora_ood"] = evaluate_real_gt(
            checkpoint_path="checkpoints_plain_lora/best_model_plain_lora.pth",
            label="Ablation stage 2: +LoRA, no spacing conditioning (OOD)",
            model_class=PlainLoRAModel, sample_ids=[ood_sample_id],
            samples_per_sample=ood_samples_per_eval, raw_tiff_root=raw_tiff_root)

    if "spacing_lora" in stages:
        train_spacing_lora(resume=False, extra_epochs=epochs, samples_per_epoch=samples_per_epoch,
                            raw_tiff_root=raw_tiff_root)
        results["spacing_lora_in_dist"] = evaluate_real_gt(
            checkpoint_path="checkpoints_spacing_lora/best_model_spacing_lora.pth",
            label="Ablation stage 3: +LoRA +spacing conditioning (in-dist)",
            model_class=SpacingConditionedLoRAModel, samples_per_sample=samples_per_eval,
            raw_tiff_root=raw_tiff_root)
        results["spacing_lora_ood"] = evaluate_real_gt(
            checkpoint_path="checkpoints_spacing_lora/best_model_spacing_lora.pth",
            label="Ablation stage 3: +LoRA +spacing conditioning (OOD)",
            model_class=SpacingConditionedLoRAModel, sample_ids=[ood_sample_id],
            samples_per_sample=ood_samples_per_eval, raw_tiff_root=raw_tiff_root)

    if "spacing_lora_no_cldice" in stages:
        no_cldice_dir = "checkpoints_spacing_lora_no_cldice"
        train_spacing_lora(resume=False, extra_epochs=epochs, samples_per_epoch=samples_per_epoch,
                            raw_tiff_root=raw_tiff_root, use_cldice=False, checkpoint_dir=no_cldice_dir)
        results["spacing_lora_no_cldice_in_dist"] = evaluate_real_gt(
            checkpoint_path=f"{no_cldice_dir}/best_model_spacing_lora.pth",
            label="Ablation stage 4: +LoRA +spacing, NO clDice loss term (in-dist)",
            model_class=SpacingConditionedLoRAModel, samples_per_sample=samples_per_eval,
            raw_tiff_root=raw_tiff_root)
        results["spacing_lora_no_cldice_ood"] = evaluate_real_gt(
            checkpoint_path=f"{no_cldice_dir}/best_model_spacing_lora.pth",
            label="Ablation stage 4: +LoRA +spacing, NO clDice loss term (OOD)",
            model_class=SpacingConditionedLoRAModel, sample_ids=[ood_sample_id],
            samples_per_sample=ood_samples_per_eval, raw_tiff_root=raw_tiff_root)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Ablation results written to {results_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=20)
    parser.add_argument("--raw-tiff-root", default=None)
    args = parser.parse_args()
    run_ablation(stages=args.stages, epochs=args.epochs, samples_per_epoch=args.samples_per_epoch,
                 raw_tiff_root=args.raw_tiff_root)
