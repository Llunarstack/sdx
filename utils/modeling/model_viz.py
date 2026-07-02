"""Shim → ``utils._archive.modeling.model_viz``."""

import utils._archive.modeling.model_viz as _src

for _name in dir(_src):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_src, _name)

del _name, _src
