"""Tool-use interfaces (segmentation, depth, OCR, retrieval) for agentic loops."""

from .tool_hooks import ExternalToolCapabilities, ToolCall, ToolOutcome, ToolStubRegistry

__all__ = ["ExternalToolCapabilities", "ToolCall", "ToolOutcome", "ToolStubRegistry"]
