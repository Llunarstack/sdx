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

