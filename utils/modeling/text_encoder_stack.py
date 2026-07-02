"""Shim → ``utils._archive.modeling.text_encoder_stack``."""

import utils._archive.modeling.text_encoder_stack as _src

for _name in dir(_src):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_src, _name)

del _name, _src
