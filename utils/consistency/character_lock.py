"""Shim → ``utils._archive.consistency.character_lock``."""

import utils._archive.consistency.character_lock as _src

for _name in dir(_src):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_src, _name)

del _name, _src
