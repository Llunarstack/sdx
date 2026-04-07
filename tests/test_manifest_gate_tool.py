from pathlib import Path


def test_manifest_gate_smoke(tmp_path: Path):
    # Minimal manifest should pass with defaults.
    p = tmp_path / "m.jsonl"
    p.write_text('{"image_path":"a.png","caption":"1girl, cafe","negative_caption":"bad anatomy"}\n', encoding="utf-8")

    # Run via direct argv emulation (argparse reads sys.argv); keep it simple by importing function
    # and calling through a subprocess-like invocation isn't needed for this unit test.
    import sys as _sys

    from scripts.tools.data.manifest_gate import main as gate_main

    old = list(_sys.argv)
    try:
        _sys.argv = ["manifest_gate.py", str(p)]
        rc = gate_main()
    finally:
        _sys.argv = old
    assert rc == 0

