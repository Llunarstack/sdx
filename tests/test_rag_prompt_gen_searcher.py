import json
from pathlib import Path

from utils.prompt import rag_prompt as rp


def test_facts_from_gen_searcher_payload_extracts_nested_text():
    payload = {
        "reasoning_summary": "Need accurate skyline and bridge orientation.",
        "search_results": [
            {"title": "City skyline guide", "snippet": "Bridge is east-facing at sunset."},
            {"text": "Landmark clocktower has four visible faces."},
        ],
        "references": {"items": [{"description": "Main plaza has red brick pavement."}]},
    }
    facts = rp.facts_from_gen_searcher_payload(payload, max_entries=10)
    assert facts
    joined = " | ".join(facts).lower()
    assert "bridge is east-facing at sunset" in joined
    assert "red brick pavement" in joined


def test_load_facts_from_gen_searcher_json_jsonl(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    rows = [
        {"search_summary": "A", "search_results": [{"snippet": "B"}]},
        {"final_answer": "C"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    facts = rp.load_facts_from_gen_searcher_json(p, max_entries=10)
    assert any(f.lower() == "a" for f in facts)
    assert any(f.lower() == "b" for f in facts)
    assert any(f.lower() == "c" for f in facts)


def test_merge_facts_into_prompt_with_gen_searcher_facts():
    payload = {"search_results": [{"snippet": "Ancient stone walls around the harbor."}]}
    facts = rp.facts_from_gen_searcher_payload(payload, max_entries=4)
    merged = rp.merge_facts_into_prompt("cinematic harbor at dusk", facts, max_chars=600)
    assert "Context (user-supplied facts" in merged
    assert "Ancient stone walls around the harbor." in merged
    assert merged.endswith("cinematic harbor at dusk")
