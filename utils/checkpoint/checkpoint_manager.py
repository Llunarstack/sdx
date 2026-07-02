"""Shim → ``utils._archive.checkpoint.checkpoint_manager``."""

import utils._archive.checkpoint.checkpoint_manager as _src

for _name in dir(_src):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_src, _name)

del _name, _src
