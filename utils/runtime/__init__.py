"""Runtime helpers (profiling, argv shims, optional fast JSON)."""

from utils.runtime.jsonutil import dumps as json_dumps
from utils.runtime.jsonutil import loads as json_loads
from utils.runtime.profiling import ProfileConfig, consume_profile_args, run_with_cprofile

__all__ = [
    "ProfileConfig",
    "consume_profile_args",
    "json_dumps",
    "json_loads",
    "run_with_cprofile",
]
