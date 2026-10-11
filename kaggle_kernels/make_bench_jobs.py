"""
Benchmark job lists (inference + scoring) for kaggle_jobs_kernel.py.

Settings (see PAPER, Section "spacing input vs field of view"):
  native  ROI on its own voxel grid, true spacing given to the model
  fov50   ROI resampled to 50um (training field of view), logits resampled back
  tell50  native grid, but the model is told 50um (spacing models only)
Coarse sets (kidney_5, kidney_6) are cheap everywhere; native / tell50 on the
640^3 fine ROIs (13um, 5.2um) need a GPU, fov50 on them is cheap.

Each run writes predictions to $OUT/preds/<run>__<part> and scores to
$OUT/scores/<run>__<part>.json; benchmark/aggregate.py merges the parts.
"""

import json
import sys

COARSE = ["kidney_5", "kidney_6"]
FINE = ["kidney_additional_voi_13um", "kidney_1_voi"]
ALL = COARSE + FINE

# name -> (model, checkpoint, extra infer args, is_spacing_model)
MODELS = {
    "r1_plain_s0": ("plain_lora", "$FIRST_ROUND/checkpoints_plain_lora/best_model_plain_lora.pth", "--patch 64", False),
    "r1_spacing_s0": ("spacing_lora", "$FIRST_ROUND/checkpoints_spacing_lora/best_model_spacing_lora.pth",
                      "--patch 64 --spacing-encoding raw", True),
    "r1_spacing_nocldice_s0": ("spacing_lora", "$ABLATION/checkpoints_spacing_lora_no_cldice/best_model_spacing_lora.pth",
                               "--patch 64 --spacing-encoding raw", True),
    "sam_zeroshot": ("plain_lora", "", "--patch 128", False),
}
for cfg in ("p_base", "p_aug", "s_base", "s_aug"):
    for seed in range(3):
        spacing = cfg.startswith("s_")
        MODELS[f"{cfg}_s{seed}"] = ("spacing_lora" if spacing else "plain_lora",
                                    f"$TRAINED/checkpoints/{cfg}_s{seed}/model_best.pth",
                                    "--patch 128 --refine" + (" --spacing-encoding log" if spacing else ""), spacing)

SETTING_ARGS = {"native": "", "fov50": "--resample-um 50", "tell50": "--spacing-um 50"}


def bench_job(name, settings, sets, part, score=True):
    model, ckpt, extra, spacing = MODELS[name]
    cmds = []
    for st in settings:
        if st == "tell50" and not spacing:
            continue
        run = f"{name}_{st}__{part}"
        cmds.append(f"python -m benchmark.infer --model {model} --checkpoint \"{ckpt}\" {extra} {SETTING_ARGS[st]} "
                    f"--run-name {run} --roi-dir $ROI_DIR --pred-root $OUT/preds --sets {' '.join(sets)}")
        if score:
            cmds.append(f"mkdir -p $OUT/scores && python -m benchmark.score $OUT/preds/{run} --roi-dir $ROI_DIR "
                        f"--out $OUT/scores/{run}.json")
    return {"name": f"bench_{name}_{part}", "cmds": cmds}


if __name__ == "__main__":
    # usage: make_bench_jobs.py out.json 'gpu0 spec' 'gpu1 spec' [--no-score]
    # spec: ';'-separated "name|settings,comma|sets,comma|part" (sets: coarse, fine, all or explicit names)
    # --no-score: only write predictions (GPU kernels); score them later in a CPU kernel (score_jobs.py)
    score = "--no-score" not in sys.argv
    argv = [x for x in sys.argv if x != "--no-score"]
    groups = {"coarse": COARSE, "fine": FINE, "all": ALL}
    out = {}
    for key, spec in zip(("gpu0", "gpu1"), argv[2:4]):
        jobs = []
        for item in filter(None, spec.split(";")):
            name, settings, sets, part = item.split("|")
            sets = groups.get(sets, sets.split(","))
            jobs.append(bench_job(name, settings.split(","), sets, part, score))
        out[key] = jobs
    json.dump(out, open(sys.argv[1], "w"), indent=1)
