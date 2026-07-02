"""Shim → ``utils._archive.brain.image_search``."""

import utils._archive.brain.image_search as _src

for _name in dir(_src):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_src, _name)

del _name, _src
