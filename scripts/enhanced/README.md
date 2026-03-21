# Enhanced DiT scripts (optional path)

These scripts target **`EnhancedDiT`** checkpoints (`models/enhanced_dit.py`, `training/enhanced_trainer.py`). They are **not** the default **`train.py` / `sample.py`** pipeline (DiT-Text + T5).

Run from **repository root**:

```bash
python scripts/enhanced/train_enhanced.py --help
python scripts/enhanced/sample_enhanced.py --help
python scripts/enhanced/setup_enhanced.py
python scripts/enhanced/save_model_checkpoint.py
```

If `sample.py` rejects your checkpoint with a message about EnhancedDiT, use **`sample_enhanced.py`** here or train a **`DiT-*-Text`** model with `train.py` instead.

## Files

| File | Role |
|------|------|
| **`train_enhanced.py`** | Training loop for EnhancedDiT + enhanced dataset |
| **`sample_enhanced.py`** | Inference for EnhancedDiT checkpoints |
| **`setup_enhanced.py`** | Environment checks and optional dependency install |
| **`save_model_checkpoint.py`** | Create / save an initialized Enhanced DiT-XL checkpoint |

See also [docs/ENHANCED_FEATURES.md](../../docs/ENHANCED_FEATURES.md) and [training/enhanced_trainer.py](../../training/enhanced_trainer.py).
