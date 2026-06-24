"""Shim → ``utils._archive.superior.model_soup``."""

import utils._archive.superior.model_soup as _src

for _name in dir(_src):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_src, _name)

del _name, _src
