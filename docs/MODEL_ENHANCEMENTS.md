# Model enhancements (shared blocks & optional fusion modes)

Small, composable pieces under [`models/model_enhancements.py`](../models/model_enhancements.py) and optional flags on multimodal / cascade / RAE modules.

---

## `model_enhancements.py`

| Class / helper | Role |
|----------------|------|
| **`RMSNorm`** | Root mean square norm (no mean centering); lighter than LayerNorm. |
| **`DropPath`** | Stochastic depth on a residual branch (training). |
| **`TokenFiLM`** | Global cond vector → per-channel scale/shift on `(B, N, D)` tokens. |
| **`SE1x1`** | Squeeze–excitation gating on `(B, D)` or `(B, N, D)`. |
| **`apply_gate_residual`** | `x + gate * branch` with broadcast gate. |

Import: `from models import RMSNorm, DropPath, TokenFiLM, SE1x1` (also `models.model_enhancements`).

---

## `NativeMultimodalTransformer`

| Option | Default | Effect |
|--------|---------|--------|
| **`cross_attn_heads`** | `0` | If &gt; 0, vision tokens **cross-attend** to text (+ extra) **before** joint `TransformerEncoder`. |
| **`output_norm`** | `"layernorm"` | `"rmsnorm"` uses **`RMSNorm`** after the encoder stack. |
| **`film_cond_dim`** | `0` | If &gt; 0, **`TokenFiLM`** on **vision** tokens; forward requires **`film_cond`** `(B, dim)`. |

Existing options still apply: modality embeddings, padding masks, `proj_dropout`, `extra_dim`, etc.

---

## `CascadedMultimodalDiffusion`

| Option | Default | Effect |
|--------|---------|--------|
| **`blend_base_refine`** | `1.0` | `final = (1-w)*base_out + w*refine_out` (ablation / stability). `1.0` matches previous “refine only” behavior. |
| **`detach_base_for_refine`** (forward kw) | `False` | Refine stage sees **detached** `base_out` (train refine with frozen base). |

---

## `RAELatentBridge`

| Option | Default | Effect |
|--------|---------|--------|
| **`learnable_output_scale`** | `False` | Scalar **`scale_dit`** / **`scale_rae`** on conv outputs (training stability). |

---

## See also

- [`CODEBASE.md`](CODEBASE.md) — where to edit DiT / data / training  
- [`models/native_multimodal_transformer.py`](../models/native_multimodal_transformer.py)  
- [`models/cascaded_multimodal_diffusion.py`](../models/cascaded_multimodal_diffusion.py)  
