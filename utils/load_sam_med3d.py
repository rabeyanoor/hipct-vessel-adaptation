# utils/load_sam_med3d.py
"""
Robust loader for SAM-Med3D's segment_anything package
(external/SAM-Med3D/segment_anything/__init__.py).

That file does `from utils import *`, an absolute (not package-relative)
import, expecting to pick up SAM-Med3D's own external/SAM-Med3D/utils/
package when SAM-Med3D's repo root is on sys.path. This project also ships
its own top-level utils/ package, so whichever one sits earlier on sys.path
wins the name. Locally that happens to be harmless because this project's
utils/__init__.py imports cleanly either way, but in an environment where
one of its dependencies isn't installed (SimpleITK, needed only for the
JP2-to-NIfTI conversion step, not for anything SAM-Med3D touches), the
resulting ImportError propagates up through segment_anything's own import
and gets misread as "SAM-Med3D is not importable" by the try/except in the
model files, when the actual backbone code is fine.

sam_med3d_path is placed at the front of sys.path only for the duration of
this import, and any "utils" entries importlib cached during it are evicted
afterward so later `from utils.xxx import yyy` calls elsewhere in this
project's own code resolve against this project's utils/ package again.
"""

import os
import sys


def import_sam_model_registry3D():
    sam_med3d_path = os.path.abspath("./external/SAM-Med3D")
    if not os.path.isdir(sam_med3d_path):
        return None

    original_sys_path = list(sys.path)
    stashed_utils_modules = {
        name: mod for name, mod in sys.modules.items()
        if name == "utils" or name.startswith("utils.")
    }
    for name in stashed_utils_modules:
        del sys.modules[name]

    sys.path.insert(0, sam_med3d_path)
    try:
        from segment_anything.build_sam3D import sam_model_registry3D
        return sam_model_registry3D
    finally:
        sys.path[:] = original_sys_path
        for name in list(sys.modules):
            if name == "utils" or name.startswith("utils."):
                del sys.modules[name]
        sys.modules.update(stashed_utils_modules)
