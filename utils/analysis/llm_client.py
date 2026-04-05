"""Optional Qwen (or other HF causal LM) for prompt expansion / dataset helpers."""

from __future__ import annotations

from typing import Optional

import torch


def load_qwen_causal_lm(
    model_path: str,
    device: str = "cuda",
    torch_dtype: Optional[torch.dtype] = None,
):
    """Load a Hugging Face causal LM from local path or hub id."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dt = torch_dtype or (torch.bfloat16 if device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dt,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda" or getattr(model, "hf_device_map", None) is None:
        model = model.to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def expand_prompt_qwen(
    tokenizer,
    model,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 128,
) -> str:
    """Return an expanded prompt string (best-effort)."""
    messages = [
        {
            "role": "system",
            "content": "You expand image generation prompts with concrete visual detail. Output only the prompt, no quotes.",
        },
        {"role": "user", "content": f"Expand this prompt for a high-quality image: {prompt}"},
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt
    inputs = tokenizer(text, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()
