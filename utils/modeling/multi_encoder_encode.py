"""Shim → ``utils._archive.modeling.multi_encoder_encode``."""

import utils._archive.modeling.multi_encoder_encode as _src

for _name in dir(_src):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_src, _name)

del _name, _src
