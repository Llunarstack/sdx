from utils.modeling.hf_index import role_counts, summary
from utils.modeling.hf_scaffold import scaffold_registry


def test_hf_index_summary():
    s = summary()
    assert int(s["total_registry"]) == len(scaffold_registry())
    assert int(s["total_registry"]) >= 140
    assert "role_counts" in s


def test_role_counts_cover_core_roles():
    rc = role_counts()
    for role in ("vlm", "reward", "control", "depth", "text_encoder"):
        assert role in rc
        assert rc[role] >= 1
