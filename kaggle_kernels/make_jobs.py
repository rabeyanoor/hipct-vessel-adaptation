"""
Writes job lists for kaggle_jobs_kernel.py (see launch_jobs.py).

Configurations of the second-round 2 x 2 design (all: 128^3 patches, 50 %
foreground-centred sampling, full-resolution head, 12 epochs x 250 steps,
effective batch 2):
  p_base  plain LoRA
  s_base  spacing-conditioned LoRA, log2(spacing / 50um) input
  p_aug   plain LoRA + spacing augmentation (scale 0.25-2)
  s_aug   spacing-conditioned LoRA + spacing augmentation
"""

import json
import sys

CONFIGS = {
    "p_base": "--model plain",
    "s_base": "--model spacing --spacing-encoding log",
    "p_aug": "--model plain --fov-scale 0.25 2.0",
    "s_aug": "--model spacing --spacing-encoding log --fov-scale 0.25 2.0",
}
COMMON = ("--patch 128 --fg-prob 0.5 --refine --epochs 12 --iters 250 --val-every 2 --workers 3 "
          "--raw-tiff-root $COMP_TRAIN --val-roi-dir $VAL_ROI_DIR --time-limit-min 90")


def train_job(cfg, seed):
    name = f"{cfg}_s{seed}"
    return {"name": f"train_{name}",
            "cmds": [f"python train_lora_v2.py {CONFIGS[cfg]} {COMMON} --seed {seed} --out $OUT/checkpoints/{name}"]}


if __name__ == "__main__":
    # usage: python make_jobs.py out.json "s_aug:0,s_base:0,s_aug:1" "p_aug:0,p_base:0,p_aug:1"
    out, q0, q1 = sys.argv[1], sys.argv[2], sys.argv[3]
    parse = lambda q: [train_job(c, int(s)) for c, s in (x.split(":") for x in q.split(","))] if q else []
    json.dump({"gpu0": parse(q0), "gpu1": parse(q1)}, open(out, "w"), indent=1)
