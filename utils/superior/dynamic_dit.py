"""Shim → ``utils._archive.superior.dynamic_dit``."""

import utils._archive.superior.dynamic_dit as _src

for _name in dir(_src):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_src, _name)

del _name, _src
