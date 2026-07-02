"""Shim → ``utils._archive.quantization.nf4_codec``."""

import utils._archive.quantization.nf4_codec as _src

for _name in dir(_src):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_src, _name)

del _name, _src
