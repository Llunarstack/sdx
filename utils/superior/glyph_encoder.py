"""Shim → ``utils._archive.superior.glyph_encoder``."""

import utils._archive.superior.glyph_encoder as _src

for _name in dir(_src):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_src, _name)

del _name, _src
