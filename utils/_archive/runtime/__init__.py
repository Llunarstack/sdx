"""Runtime helpers (profiling, argv shims, optional fast JSON)."""

# Relative imports: absolute ``utils.runtime.*`` here would recurse through the
# public shim package (utils/runtime re-exports this package) and deadlock.
from .jsonutil import dumps as json_dumps
from .jsonutil import loads as json_loads
from .plain_dict import to_plain_dict
from .profiling import ProfileConfig, consume_profile_args, run_with_cprofile

__all__ = [
    "ProfileConfig",
    "consume_profile_args",
    "json_dumps",
    "json_loads",
    "run_with_cprofile",
    "to_plain_dict",
]
