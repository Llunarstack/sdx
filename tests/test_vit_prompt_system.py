from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_breakdown_and_compose() -> None:
    from ViT.prompt_system import build_prompt_plan

    p = "1girl, cinematic lighting, no blurry, without watermark, sharp eyes"
    out = build_prompt_plan(p)
    assert "1girl" in out["add"]
    assert "cinematic lighting" in out["add"]
    assert "blurry" in out["avoid"]
    assert "watermark" in out["avoid"]
    assert "avoid:" in out["composed_prompt"]


def test_prompt_tool_jsonl(tmp_path: Path) -> None:
    inp = tmp_path / "m.jsonl"
    out = tmp_path / "m.out.jsonl"
    rows = [{"image_path": "a.png", "caption": "portrait, no logo"}]
    inp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    script = ROOT / "ViT" / "prompt_tool.py"
    cmd = [sys.executable, str(script), "--json-in", str(inp), "--json-out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    txt = out.read_text(encoding="utf-8").strip()
    row = json.loads(txt)
    assert "vit_prompt_add" in row
    assert "vit_prompt_avoid" in row
    assert "vit_prompt_composed" in row
