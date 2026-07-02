"""Shim → ``utils._archive.consistency.style_harmonization``."""

import utils._archive.consistency.style_harmonization as _src

for _name in dir(_src):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_src, _name)

del _name, _src
