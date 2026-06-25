"""Shim → ``utils._archive.agentic.tools``."""

import utils._archive.agentic.tools as _src

for _name in dir(_src):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_src, _name)

del _name, _src
