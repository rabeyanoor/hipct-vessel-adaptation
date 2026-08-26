# benchmark/score_many.py
"""
Scores several prediction folders in parallel (one process per run).

Each run is found by name below --search (a folder .../preds/<run>, or any
folder <run> that holds a run.json, e.g. in a dataset made from predictions), so the
predictions can come from the outputs of other Kaggle kernels attached as
inputs. Writes <out-dir>/<run>.json and <out-dir>/<run>.log.

Usage:
    python -m benchmark.score_many --search /kaggle/input --roi-dir <rois> --out-dir out/scores \
        --parallel 3 run_a__gpu run_b__gpu
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def find_run(search, run):
    for root, dirs, _ in os.walk(search, followlinks=True):
        if run in dirs and (os.path.basename(root) == "preds"
                            or os.path.isfile(os.path.join(root, run, "run.json"))):
            return os.path.join(root, run)
    return None


def score_one(run, search, roi_dir, out_dir):
    d = find_run(search, run)
    if d is None:
        return run, "missing"
    with open(os.path.join(out_dir, f"{run}.log"), "w") as log:
        r = subprocess.run([sys.executable, "-m", "benchmark.score", d, "--roi-dir", roi_dir,
                            "--out", os.path.join(out_dir, f"{run}.json")], stdout=log, stderr=subprocess.STDOUT)
    return run, "ok" if r.returncode == 0 else f"failed ({r.returncode})"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--search", default="/kaggle/input")
    p.add_argument("--roi-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--parallel", type=int, default=3)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        results = list(ex.map(lambda r: score_one(r, a.search, a.roi_dir, a.out_dir), a.runs))
    for run, status in results:
        print(f"  {run}: {status}", flush=True)
    sys.exit(0 if all(s == "ok" for _, s in results) else 1)
