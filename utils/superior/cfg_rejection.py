"""Shim → ``utils._archive.superior.cfg_rejection``."""

import utils._archive.superior.cfg_rejection as _src

for _name in dir(_src):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_src, _name)

del _name, _src
