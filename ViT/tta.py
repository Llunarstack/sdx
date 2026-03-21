from __future__ import annotations

from typing import Dict, List

import torch


@torch.no_grad()
def tta_predict(model: torch.nn.Module, x: torch.Tensor, text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Test-time augmentation with horizontal flip.
    Returns averaged model outputs.
    """
    outs: List[Dict[str, torch.Tensor]] = []
    outs.append(model(x, text_features))
    outs.append(model(torch.flip(x, dims=[3]), text_features))

    q = torch.stack([o["quality_logit"] for o in outs], dim=0).mean(dim=0)
    a = torch.stack([o["adherence_score"] for o in outs], dim=0).mean(dim=0)
    e = torch.stack([o["embedding"] for o in outs], dim=0).mean(dim=0)
    return {"quality_logit": q, "adherence_score": a, "embedding": e}
