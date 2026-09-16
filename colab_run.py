# colab_run.py
"""
Self-contained Colab script for the two remaining compute-heavy items on
this project's checklist: per-architecture hyperparameter tuning and
multi-seed statistical reporting for Plain LoRA / Spacing-Conditioned LoRA.
Both were deferred off Kaggle because Kaggle's free GPU tier has a fixed
weekly quota (6 hours) already substantially used by the 100-epoch retrain
and OOD investigation (see README.md section 5.4-5.6, test.sh section
3.4-3.6) -- this script reuses the exact same training/evaluation functions
those used, just run from Colab instead of a Kaggle kernel.

Paste this whole file into a single Colab cell (or `!python colab_run.py`
after uploading it) and run it. Set TASK below to choose what to run --
"sweep" and "multiseed" are both substantial GPU jobs, not meant to run
back to back in one sitting on a free Colab GPU.

Requires a Kaggle API token (kaggle.json) -- this script pulls both this
project's code (private Kaggle dataset rabeya100x/hip-vesseladapt-code) and
the SenNet+HOA competition data through the Kaggle API, the same way the
Kaggle kernel did, rather than needing a separate GitHub repo or a Drive
upload. Upload your kaggle.json via the file-browser panel on the left
before running (Colab: click the folder icon, then the upload icon), or
run the cell below first:

    from google.colab import files
    files.upload()  # select your kaggle.json

Never paste your actual Kaggle key into this file directly, or into a
Colab cell you might share -- upload kaggle.json as a file instead.
"""

import json
import os
import subprocess
import sys
import zipfile

TASK = "sweep"  # "sweep" or "multiseed" -- see bottom of this file

WORK_DIR = "/content/hip-vesseladapt"
DATA_DIR = "/content/blood-vessel-segmentation"


def setup_kaggle_auth():
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    if os.path.exists(kaggle_json):
        os.chmod(kaggle_json, 0o600)
        return
    # Colab's default upload target is /content -- kaggle.json may have
    # landed there instead of ~/.kaggle.
    alt_path = "/content/kaggle.json"
    if os.path.exists(alt_path):
        os.makedirs(kaggle_dir, exist_ok=True)
        subprocess.run(["cp", alt_path, kaggle_json], check=True)
        os.chmod(kaggle_json, 0o600)
        return
    raise RuntimeError(
        "No kaggle.json found at ~/.kaggle/kaggle.json or /content/kaggle.json -- "
        "upload it first (see this file's module docstring)."
    )


def setup_environment():
    """Mirrors kaggle_train_kernel.py's setup steps 1-4, adapted for Colab:
    fetch this project's code + the competition data through the Kaggle API
    instead of relying on Kaggle's own pre-mounted /kaggle/input."""
    setup_kaggle_auth()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle", "monai", "gdown"], check=True)

    os.makedirs(WORK_DIR, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "rabeya100x/hip-vesseladapt-code",
         "-p", WORK_DIR, "--unzip"],
        check=True,
    )
    for zip_name in ["models.zip", "utils.zip", "losses.zip", "data.zip"]:
        zip_path = os.path.join(WORK_DIR, zip_name)
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(WORK_DIR)
            os.remove(zip_path)

    os.chdir(WORK_DIR)
    sys.path.insert(0, WORK_DIR)

    if not os.path.isdir(os.path.join(DATA_DIR, "train")):
        os.makedirs(DATA_DIR, exist_ok=True)
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", "blood-vessel-segmentation",
             "-p", DATA_DIR],
            check=True,
        )
        zip_path = os.path.join(DATA_DIR, "blood-vessel-segmentation.zip")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
        os.remove(zip_path)
    print(f"[✓] Competition data ready at {DATA_DIR}")

    subprocess.run(["git", "clone", "-q", "https://github.com/openmedlab/SAM-Med3D.git", "external/SAM-Med3D"], check=True)
    subprocess.run(["git", "clone", "-q", "https://github.com/jocpae/clDice.git", "external/clDice"], check=True)

    from utils.patch_external_repos import patch_sam_med3d
    patch_sam_med3d()

    os.makedirs("external/SAM-Med3D/ckpt", exist_ok=True)
    ckpt_path = "external/SAM-Med3D/ckpt/sam_med3d.pth"
    if not os.path.exists(ckpt_path):
        subprocess.run([
            sys.executable, "-m", "gdown",
            "https://drive.google.com/uc?id=1PFeUjlFMAppllS9x1kAWyCYUJM9re2Ub",
            "-O", ckpt_path,
        ], check=True)
    print(f"[✓] Checkpoint ready: {os.path.getsize(ckpt_path)/1e9:.2f} GB")

    return os.path.join(DATA_DIR, "train")


def run_hyperparameter_sweep(raw_tiff_root, sweep_epochs=15):
    """Short-epoch grid search over learning rate and the Dice/clDice loss
    balance (HiPVesselLoss's alpha), for both architectures independently --
    the 100-epoch comparison in README.md section 5.4 used the same LR/loss
    weights for both, tuned originally against PhysicsAwareAdapter, not
    either LoRA variant. Picks the config with the best in-distribution Dice
    per architecture; does NOT re-run the full 100-epoch comparison with the
    winning config automatically -- do that as a separate step once you see
    which config wins, so a bad sweep run never silently overwrites the
    already-validated 100-epoch checkpoints in checkpoints_plain_lora_100ep_gpu/
    and checkpoints_spacing_lora_100ep_gpu/.
    """
    from train_plain_lora import train_plain_lora
    from train_spacing_lora import train_spacing_lora
    from evaluate_real_gt import evaluate_real_gt
    from models.plain_lora_model import PlainLoRAModel
    from models.spacing_lora_model import SpacingConditionedLoRAModel

    lr_grid = [5e-5, 1e-4, 2e-4]
    alpha_grid = [0.25, 0.5, 0.75]  # HiPVesselLoss's Dice-vs-clDice balance

    results = {}
    for model_name, train_fn, model_cls in [
        ("plain_lora", train_plain_lora, PlainLoRAModel),
        ("spacing_lora", train_spacing_lora, SpacingConditionedLoRAModel),
    ]:
        for lr in lr_grid:
            for alpha in alpha_grid:
                run_id = f"{model_name}_lr{lr}_alpha{alpha}"
                ckpt_dir = f"checkpoints_sweep/{run_id}"
                print("\n" + "#" * 80)
                print(f"# SWEEP: {run_id} ({sweep_epochs} epochs)")
                print("#" * 80)
                train_fn(resume=False, extra_epochs=sweep_epochs, samples_per_epoch=20,
                         raw_tiff_root=raw_tiff_root, lr=lr, loss_alpha=alpha,
                         checkpoint_dir=ckpt_dir, seed=0)
                ckpt_file = f"{ckpt_dir}/best_model_{'plain_lora' if model_name == 'plain_lora' else 'spacing_lora'}.pth"
                metrics = evaluate_real_gt(checkpoint_path=ckpt_file, label=f"SWEEP EVAL: {run_id}",
                                            model_class=model_cls, samples_per_sample=20,
                                            raw_tiff_root=raw_tiff_root)
                results[run_id] = {"lr": lr, "alpha": alpha, "dice": metrics["dice"] if metrics else None}

    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("SWEEP SUMMARY (best Dice per architecture)")
    print("=" * 80)
    for model_name in ("plain_lora", "spacing_lora"):
        model_results = {k: v for k, v in results.items() if k.startswith(model_name)}
        best_run = max(model_results, key=lambda k: model_results[k]["dice"] or -1)
        print(f"{model_name}: best = {best_run} (lr={model_results[best_run]['lr']}, "
              f"alpha={model_results[best_run]['alpha']}, Dice={model_results[best_run]['dice']:.4f})")
    print("Full grid written to sweep_results.json")


def run_multiseed(raw_tiff_root, n_seeds=3, epochs=100, lr=None, loss_alpha=0.5):
    """Runs both architectures across n_seeds full-length trainings each,
    evaluates in-distribution AND OOD every time, reports mean +/- std --
    every number in README.md/test.sh so far is a single run; this is what
    turns those into a properly powered comparison."""
    from train_plain_lora import train_plain_lora
    from train_spacing_lora import train_spacing_lora
    from evaluate_real_gt import evaluate_real_gt
    from models.plain_lora_model import PlainLoRAModel
    from models.spacing_lora_model import SpacingConditionedLoRAModel

    results = {"plain_lora": {"in_dist": [], "ood": []}, "spacing_lora": {"in_dist": [], "ood": []}}

    for model_name, train_fn, model_cls, ckpt_name in [
        ("plain_lora", train_plain_lora, PlainLoRAModel, "best_model_plain_lora.pth"),
        ("spacing_lora", train_spacing_lora, SpacingConditionedLoRAModel, "best_model_spacing_lora.pth"),
    ]:
        for seed in range(n_seeds):
            ckpt_dir = f"checkpoints_multiseed/{model_name}_seed{seed}"
            print("\n" + "#" * 80)
            print(f"# MULTISEED: {model_name} seed={seed} ({epochs} epochs)")
            print("#" * 80)
            train_fn(resume=True, extra_epochs=epochs, samples_per_epoch=20,
                      raw_tiff_root=raw_tiff_root, lr=lr, loss_alpha=loss_alpha,
                      checkpoint_dir=ckpt_dir, seed=seed)

            ckpt_file = f"{ckpt_dir}/{ckpt_name}"
            in_dist = evaluate_real_gt(checkpoint_path=ckpt_file,
                                        label=f"MULTISEED {model_name} seed={seed} IN-DIST",
                                        model_class=model_cls, samples_per_sample=20,
                                        raw_tiff_root=raw_tiff_root)
            ood = evaluate_real_gt(checkpoint_path=ckpt_file,
                                    label=f"MULTISEED {model_name} seed={seed} OOD",
                                    model_class=model_cls, sample_ids=["kidney_1_voi"],
                                    samples_per_sample=40, raw_tiff_root=raw_tiff_root)
            results[model_name]["in_dist"].append(in_dist["dice"] if in_dist else None)
            results[model_name]["ood"].append(ood["dice"] if ood else None)

    with open("multiseed_results.json", "w") as f:
        json.dump(results, f, indent=2)

    def _mean_std(values):
        values = [v for v in values if v is not None]
        if not values:
            return float("nan"), float("nan")
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
        return mean, var ** 0.5

    print("\n" + "=" * 80)
    print(f"MULTISEED SUMMARY (n={n_seeds} seeds)")
    print("=" * 80)
    for model_name in ("plain_lora", "spacing_lora"):
        in_mean, in_std = _mean_std(results[model_name]["in_dist"])
        ood_mean, ood_std = _mean_std(results[model_name]["ood"])
        print(f"{model_name}: in-dist Dice = {in_mean:.4f} +/- {in_std:.4f}, "
              f"OOD Dice = {ood_mean:.4f} +/- {ood_std:.4f}")
    print("Full per-seed results written to multiseed_results.json")


if __name__ == "__main__":
    raw_tiff_root = setup_environment()
    if TASK == "sweep":
        run_hyperparameter_sweep(raw_tiff_root)
    elif TASK == "multiseed":
        run_multiseed(raw_tiff_root)
    else:
        raise ValueError(f"Unknown TASK: {TASK!r} -- set it to 'sweep' or 'multiseed' near the top of this file.")
