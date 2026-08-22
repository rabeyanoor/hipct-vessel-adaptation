# utils/load_sam_med3d.py
"""
Imports SAM-Med3D's segment_anything (external/SAM-Med3D).

Its __init__.py does `from utils import *`, meant for SAM-Med3D's own utils/
package. This project also has a top-level utils/, and whichever comes first on
sys.path wins. Locally it happened to work, but where a dependency of our utils
is missing (e.g. SimpleITK on Kaggle) the ImportError looked like "SAM-Med3D is
not importable" even though the backbone was fine.

Fix: put SAM-Med3D first on sys.path only during this import, then remove the
"utils" modules it loaded so `from utils.xxx import yyy` finds our package again.
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
    # a pip-installed segment_anything (present on Kaggle images, and imported
    # by MONAI's optional imports) would shadow SAM-Med3D's package
    for name in list(sys.modules):
        if name == "segment_anything" or name.startswith("segment_anything."):
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
