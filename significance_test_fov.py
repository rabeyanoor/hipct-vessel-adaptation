# significance_test_fov.py
"""
Significance test for section 5.5's FOV-matched OOD Dice gap between Plain
LoRA (0.2895) and Spacing-Conditioned LoRA (0.2215), n=40 each -- that
section reported the gap as real but not yet backed by a significance test.

Re-runs evaluate_fov_confound.py for both real 100-epoch checkpoints,
collects the raw per-patch Dice scores (evaluate_real_gt.py's
evaluate_model_on_loader now returns these, not just the aggregate), and
runs a Mann-Whitney U test -- the two models' FOV-matched patches are
independent random draws (evaluate_fov_confound.py samples a fresh random
cube each call), not the same 40 patches for both models, so this is an
unpaired test; Mann-Whitney rather than a t-test since Dice scores are
bounded in [0,1] and not assumed normal.
"""

from scipy import stats

from evaluate_fov_confound import FOVMatchedKidneyDataset
from evaluate_real_gt import evaluate_model_on_loader
from models.plain_lora_model import PlainLoRAModel
from models.spacing_lora_model import SpacingConditionedLoRAModel
from config import TRAIN_CONFIG

import torch
from torch.utils.data import DataLoader


def collect_fov_matched_scores(checkpoint_path, model_class, samples=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dataset = FOVMatchedKidneyDataset(sample_ids=["kidney_1_voi"], patch_size=TRAIN_CONFIG["patch_size"],
                                       samples_per_epoch=samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    result = evaluate_model_on_loader(model, device, loader, f"significance-test scoring: {checkpoint_path}")
    return result["dice_scores"]


def run_significance_test():
    plain_scores = collect_fov_matched_scores(
        "checkpoints_plain_lora_100ep_gpu/best_model_plain_lora.pth", PlainLoRAModel)
    spacing_scores = collect_fov_matched_scores(
        "checkpoints_spacing_lora_100ep_gpu/best_model_spacing_lora.pth", SpacingConditionedLoRAModel)

    print(f"\nPlain LoRA FOV-matched Dice: n={len(plain_scores)}, "
          f"mean={sum(plain_scores)/len(plain_scores):.4f}")
    print(f"Spacing-Conditioned LoRA FOV-matched Dice: n={len(spacing_scores)}, "
          f"mean={sum(spacing_scores)/len(spacing_scores):.4f}")

    u_stat, p_value = stats.mannwhitneyu(plain_scores, spacing_scores, alternative="two-sided")
    print(f"\nMann-Whitney U test (two-sided): U={u_stat:.1f}, p={p_value:.4f}")
    if p_value < 0.05:
        print("Significant at alpha=0.05 -- the FOV-matched Dice gap between "
              "architectures is unlikely to be sampling noise.")
    else:
        print("NOT significant at alpha=0.05 -- with n=40 each, the FOV-matched "
              "Dice gap cannot be distinguished from sampling noise at this "
              "significance level.")

    return {"plain_lora": plain_scores, "spacing_lora": spacing_scores,
            "u_statistic": u_stat, "p_value": p_value}


if __name__ == "__main__":
    run_significance_test()
