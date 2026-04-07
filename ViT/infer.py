#!/usr/bin/env python3
"""Legacy launcher shim for `vit_quality.infer`."""

from __future__ import annotations

from vit_quality.infer import main

if __name__ == "__main__":
    raise SystemExit(main())
