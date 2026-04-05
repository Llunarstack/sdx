from pathlib import Path

from utils.modeling.model_paths import pretrained_catalog, verify_gen_searcher_8b_local


def test_verify_gen_searcher_local_missing(tmp_path: Path):
    out = verify_gen_searcher_8b_local(str(tmp_path))
    assert out["is_local_dir"] is True
    assert out["all_required_present"] is False
    assert out["missing"]


def test_verify_gen_searcher_local_complete(tmp_path: Path):
    required = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ]
    for name in required:
        (tmp_path / name).write_text("x", encoding="utf-8")
    out = verify_gen_searcher_8b_local(str(tmp_path))
    assert out["is_local_dir"] is True
    assert out["all_required_present"] is True
    assert out["missing"] == []
    assert len(out["found_shards"]) == 4


def test_pretrained_catalog_has_core_rows():
    rows = pretrained_catalog()
    names = {str(r["name"]) for r in rows}
    assert "T5-XXL" in names
    assert "StableCascade-Prior" in names
    assert "StableCascade-Decoder" in names
    assert "GenSearcher-8B" in names
