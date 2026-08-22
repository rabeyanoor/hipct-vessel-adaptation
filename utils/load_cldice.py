# utils/load_cldice.py
"""
Robust loader for the external clDice repo's soft_cldice class
(external/clDice/cldice_loss/pytorch/cldice.py).

That file does `try: from .soft_skeleton import SoftSkeletonize except
ImportError: from soft_skeleton import SoftSkeletonize` to support being
imported either as part of a package or as a standalone top-level module.
On at least one environment (observed on a Kaggle kernel) the relative-import
branch's ImportError was not caught as expected and propagated, crashing the
import entirely -- rather than depend on that fallback behaving consistently
across Python versions/environments, soft_skeleton.py is loaded and
registered in sys.modules directly before cldice.py executes, so whichever
branch of its internal try/except runs, the module it needs already exists.
"""

import importlib.util
import os
import sys


def load_soft_cldice():
    cldice_dir = os.path.abspath("./external/clDice/cldice_loss/pytorch")

    skel_path = os.path.join(cldice_dir, "soft_skeleton.py")
    spec = importlib.util.spec_from_file_location("soft_skeleton", skel_path)
    soft_skeleton_module = importlib.util.module_from_spec(spec)
    sys.modules["soft_skeleton"] = soft_skeleton_module
    spec.loader.exec_module(soft_skeleton_module)

    cldice_path = os.path.join(cldice_dir, "cldice.py")
    spec2 = importlib.util.spec_from_file_location("cldice", cldice_path)
    cldice_module = importlib.util.module_from_spec(spec2)
    sys.modules["cldice"] = cldice_module
    spec2.loader.exec_module(cldice_module)

    return cldice_module.soft_cldice
