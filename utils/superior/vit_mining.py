"""Shim → ``utils._archive.superior.vit_mining``."""

import utils._archive.superior.vit_mining as _src

for _name in dir(_src):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_src, _name)

del _name, _src
