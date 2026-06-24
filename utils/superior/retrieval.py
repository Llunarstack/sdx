"""Shim → ``utils._archive.superior.retrieval``."""

import utils._archive.superior.retrieval as _src

for _name in dir(_src):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_src, _name)

del _name, _src
