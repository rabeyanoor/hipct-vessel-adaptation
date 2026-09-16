# utils/load_cldice.py
"""
Robust loader for the external clDice repo's soft_cldice class
(external/clDice/cldice_loss/pytorch/cldice.py).

Upstream's cldice.py does `from .soft_skeleton import SoftSkeletonize` -- a
package-relative import that only resolves when the file is imported as part
of an actual Python package (i.e. via a proper `import external.clDice...`
chain with __init__.py files in place). This repo instead loads it as a
standalone file, which has no parent package, so the relative form always
raises "attempted relative import with no known parent package" regardless
of environment.

soft_skeleton.py is loaded and registered in sys.modules first, then
cldice.py's source is read and its relative import line is rewritten to a
plain absolute import before the module is executed, so it resolves against
the already-registered sys.modules entry instead of relying on package
context that doesn't exist here.
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
