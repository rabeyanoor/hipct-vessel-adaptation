"""
CPU job list that scores predictions written by GPU benchmark kernels
(make_bench_jobs.py --no-score), using benchmark/score_many.py. The prediction
kernels are attached as inputs; each run is found by name under /kaggle/input.

Usage:
    python kaggle_kernels/make_score_jobs.py out.json run_a__gpu run_b__gpu ... [--parallel 3]
then launch with launch_jobs.py --no-gpu --kernels <prediction kernels> <ROI kernel>.
"""

import argparse
import json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("runs", nargs="+")
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--name", default="score")
    a = p.parse_args()
    cmd = (f"python -m benchmark.score_many --search /kaggle/input --roi-dir $ROI_DIR --out-dir $OUT/scores "
           f"--parallel {a.parallel} " + " ".join(a.runs))
    json.dump({"gpu0": [{"name": a.name, "cmds": [cmd]}]}, open(a.out, "w"), indent=1)
