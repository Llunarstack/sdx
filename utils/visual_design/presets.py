"""Shim → ``utils._archive.visual_design.presets``."""

import utils._archive.visual_design.presets as _src

for _name in dir(_src):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_src, _name)

del _name, _src
