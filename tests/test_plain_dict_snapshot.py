"""``utils.runtime.plain_dict.to_plain_dict`` — slotted dataclasses and plain objects."""

from __future__ import annotations

from dataclasses import dataclass

from utils.runtime.plain_dict import to_plain_dict


@dataclass(slots=True)
class _SlottedSample:
    x: int
    y: str = "hi"


def test_to_plain_dict_dataclass_slots():
    d = to_plain_dict(_SlottedSample(x=1))
    assert d == {"x": 1, "y": "hi"}


def test_to_plain_dict_plain_dict_copies():
    src = {"a": 1}
    d = to_plain_dict(src)
    assert d == src
    d["a"] = 2
    assert src["a"] == 1


def test_to_plain_dict_no_instance_dict_returns_empty():
    class Obj:
        __slots__ = ()  # noqa: PIE793 — explicit no instance dict

        def __init__(self) -> None:
            pass

    # Fallback path: no __dict__ on instance with empty slots
    o = Obj()
    assert to_plain_dict(o) == {}


def test_vit_config_snapshot():
    from vit_quality.config import ViTConfig

    d = to_plain_dict(ViTConfig())
    assert d["model_name"] == "vit_base_patch16_224"
    assert d["epochs"] == 5


def test_train_config_slotted_snapshot():
    from config.train_config import TrainConfig

    c = TrainConfig()
    d = to_plain_dict(c)
    assert d["model_name"] == c.model_name
    assert "global_batch_size" in d
