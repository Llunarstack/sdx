from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional


@dataclass(slots=True)
class ExternalToolCapabilities:
    """Declarative catalogue of callable sidecars."""

    segmentation: bool = False
    depth_estimation: bool = False
    ocr: bool = False
    search_retrieval: bool = False
    vector_db: bool = False
    physics_engine: bool = False


ToolName = Literal["segment", "depth", "ocr", "retrieve", "noop"]


@dataclass(slots=True)
class ToolCall:
    name: ToolName
    args: Dict[str, Any] = field(default_factory=dict)
    call_id: str = ""


@dataclass(slots=True)
class ToolOutcome:
    call: ToolCall
    ok: bool
    payload: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ToolStubRegistry:
    """In-memory registry replacing real adapters in tests."""

    __slots__ = ("_handlers",)

    def __init__(self) -> None:
        self._handlers: Dict[ToolName, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

    def register(self, name: ToolName, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        self._handlers[name] = fn

    def invoke(self, call: ToolCall) -> ToolOutcome:
        handler = self._handlers.get(call.name)
        if handler is None:
            return ToolOutcome(call=call, ok=False, error=f"unknown tool {call.name!r}")
        try:
            payload = handler(call.args or {})
            return ToolOutcome(call=call, ok=True, payload=payload)
        except Exception as e:  # noqa: BLE001 — stub tier
            return ToolOutcome(call=call, ok=False, error=str(e))

    def summarize(self, outcomes: List[ToolOutcome]) -> Dict[str, bool]:
        return {oc.call.name: oc.ok for oc in outcomes}


__all__ = ["ExternalToolCapabilities", "ToolCall", "ToolOutcome", "ToolStubRegistry", "ToolName"]
