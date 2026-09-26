"""Tests for WD tag merge and LoRA bank resolution."""

from __future__ import annotations

from pathlib import Path

from utils.caption.wd_tagger import merge_wd_tags_into_caption, merge_wd_tags_into_row
from utils.lora.lora_bank import LoRABank, LoRAEntry, resolve_lora_specs_from_prompt, slugify_lora_key


def test_merge_wd_tags_preserves_identity():
    cap = merge_wd_tags_into_caption(
        "1girl, solo",
        ["long hair", "blue eyes", "1girl"],
        identity_tags=["1girl", "wlop"],
    )
    assert "long hair" in cap
    assert "blue eyes" in cap
    assert cap.count("1girl") == 1


def test_merge_wd_tags_into_row():
    row = merge_wd_tags_into_row(
        {"caption": "1girl", "artist_tags": ["wlop"], "character_tags": ["hatsune miku"]},
        ["twintails", "skirt"],
    )
    assert "twintails" in row["caption"]
    assert "wd_tagger" in row["tag_sources"]


def test_slugify_lora_key():
    assert slugify_lora_key("WLOP") == "wlop"
    assert slugify_lora_key("kantoku") == "kantoku"


def test_resolve_lora_specs_from_prompt(tmp_path: Path):
    bank = LoRABank(root=tmp_path)
    fake = tmp_path / "artist" / "wlop" / "best_lora.pt"
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"x")
    bank.artists["wlop"] = LoRAEntry(lora=str(fake), default_scale=0.8, role="style")

    specs = resolve_lora_specs_from_prompt("@wlop 1girl", bank, artist_strength=1.25)
    assert len(specs) == 1
    assert "best_lora.pt" in specs[0]
    assert ":1.000:" in specs[0] or specs[0].endswith(":style")


def test_lora_bank_save_load(tmp_path: Path):
    bank = LoRABank(root=tmp_path)
    fake = tmp_path / "style" / "anime" / "best_lora.pt"
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"x")
    bank.styles["anime"] = LoRAEntry(lora="style/anime/best_lora.pt", default_scale=0.6)
    idx = tmp_path / "index.json"
    bank.save(idx)
    loaded = LoRABank.load(idx)
    assert "anime" in loaded.styles
    assert loaded.styles["anime"].default_scale == 0.6
