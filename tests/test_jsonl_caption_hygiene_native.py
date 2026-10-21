import json
from pathlib import Path


def test_jsonl_caption_hygiene_normalizes_unicode(tmp_path: Path):
    # Includes zero-width and compatibility chars; should normalize + strip safely.
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text(
        json.dumps({"caption": "a\u200bgirl, cafe", "negative_caption": " \ufeffbad  ,  text"}) + "\n",
        encoding="utf-8",
    )

    from sdx_native.jsonl_caption_hygiene import normalize_manifest_jsonl

    n = normalize_manifest_jsonl(inp=inp, out=out)
    assert n == 1
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["caption"] == "agirl, cafe"
    assert row["negative_caption"] == "bad, text"


def test_null_negative_and_non_object_rows(tmp_path: Path):
    # A null negative_caption used to be written back as the string "None"; a JSON array row crashed the pass.
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    rows = [{"caption": "cat", "negative_caption": None, "negative_prompt": "blurry"}, [1, 2], {"caption": "dog"}]
    inp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    from sdx_native.jsonl_caption_hygiene import normalize_manifest_jsonl

    assert normalize_manifest_jsonl(inp=inp, out=out) == 2
    first = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert first["negative_caption"] is None
    assert first["negative_prompt"] == "blurry"


def test_caption_normalization_is_idempotent_for_fullwidth_commas():
    from sdx_native.text_hygiene import normalize_caption_for_training

    once = normalize_caption_for_training("a，b, c")
    assert once == "a, b, c"
    assert normalize_caption_for_training(once) == once
