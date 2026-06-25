"""Shim → ``utils._archive.brain.visual_brain``."""

import utils._archive.brain.visual_brain as _src

for _name in dir(_src):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_src, _name)

del _name, _src
