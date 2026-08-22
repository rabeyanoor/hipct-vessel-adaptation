# colab_run.py
"""
Colab script for the hyperparameter sweep and the multi-seed runs (Plain LoRA /
Spacing-Conditioned LoRA). My Kaggle GPU quota (6 h/week) was mostly used by
the 100-epoch retrain and the OOD experiments (README 5.4-5.6), so these run on
Colab with the same training/evaluation functions.

Paste the file into a Colab cell (or upload it and run `!python colab_run.py`).
TASK below selects "sweep" or "multiseed"; both are long GPU jobs, so run them
separately.

Needs kaggle.json: the code (private Kaggle dataset
rabeya100x/hip-vesseladapt-code) and the competition data are downloaded with
the Kaggle API. Upload kaggle.json with the file panel on the left, or run:

    from google.colab import files
    files.upload()  # select your kaggle.json

Don't put the Kaggle key in this file or in a shared cell.
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
    # Colab uploads go to /content by default
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
    """Same setup as kaggle_train_kernel.py steps 1-4, but code and data are
    downloaded with the Kaggle API since there is no /kaggle/input on Colab."""
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
    """Short grid search over learning rate and loss alpha (Dice vs clDice) for
    each architecture. The 100-epoch runs (README 5.4) used LR/loss weights tuned
    for PhysicsAwareAdapter. Only reports the best config; the 100-epoch run
    with it has to be started separately, so the existing 100-epoch checkpoints
    are not overwritten.
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
    """Trains both architectures with n_seeds seeds, evaluates in-dist and OOD
    each time and prints mean +/- std."""
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
