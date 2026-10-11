# utils/load_cldice.py
"""
Loads soft_cldice from external/clDice/cldice_loss/pytorch/cldice.py.

cldice.py does `from .soft_skeleton import SoftSkeletonize`, a relative import
that fails when the file is loaded on its own ("attempted relative import with
no known parent package").

So: load soft_skeleton.py into sys.modules first, then read cldice.py, change
the import to an absolute one and exec it.
"""

import importlib.util
import os
import sys
import types


def load_soft_cldice():
    cldice_dir = os.path.abspath("./external/clDice/cldice_loss/pytorch")

    skel_path = os.path.join(cldice_dir, "soft_skeleton.py")
    spec = importlib.util.spec_from_file_location("soft_skeleton", skel_path)
    soft_skeleton_module = importlib.util.module_from_spec(spec)
    sys.modules["soft_skeleton"] = soft_skeleton_module
    spec.loader.exec_module(soft_skeleton_module)

    cldice_path = os.path.join(cldice_dir, "cldice.py")
    with open(cldice_path) as f:
        source = f.read()
    source = source.replace(
        "from .soft_skeleton import SoftSkeletonize",
        "from soft_skeleton import SoftSkeletonize",
    )

    cldice_module = types.ModuleType("cldice")
    cldice_module.__file__ = cldice_path
    sys.modules["cldice"] = cldice_module
    exec(compile(source, cldice_path, "exec"), cldice_module.__dict__)

    return cldice_module.soft_cldice
