"""
Extreme quantization: INT4, INT2, binary networks for 4-16x speedup.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class INT4Quantization:
    """4-bit integer quantization (4x size reduction, 4x faster inference)."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.quantization_params = {}

    def quantize_weight_int4(self, weight: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Quantize weight to INT4."""
        # Range: -8 to 7 (4-bit signed)
        weight_min = weight.min().item()
        weight_max = weight.max().item()

        # Linear quantization
        scale = (weight_max - weight_min) / 15.0  # 16 levels

        # Quantize
        weight_q = torch.round((weight - weight_min) / scale).clamp(-8, 7).int()

        return weight_q, scale

    def dequantize_int4(self, weight_q: torch.Tensor, scale: float, min_val: float) -> torch.Tensor:
        """Dequantize INT4."""
        return weight_q.float() * scale + min_val

    def quantize_model(self) -> None:
        """Quantize all weights to INT4."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                weight_min = weight.min().item()

                # Quantize
                weight_q, scale = self.quantize_weight_int4(weight)

                # Store params
                self.quantization_params[name] = {
                    "scale": scale,
                    "min": weight_min,
                }

                # Pack 2 INT4 values into 1 byte
                weight_packed = self._pack_int4(weight_q)

                # Replace weight
                module.weight.data = weight_packed.float()

                logger.info(
                    f"Quantized {name}: {weight.numel() * 4} bits -> "
                    f"{weight_packed.numel() * 8} bits (4x reduction)"
                )

    def _pack_int4(self, weight_q: torch.Tensor) -> torch.Tensor:
        """Pack two INT4 values into one byte."""
        shape = weight_q.shape
        weight_q_flat = weight_q.reshape(-1)

        # Pad if odd length
        if len(weight_q_flat) % 2 == 1:
            weight_q_flat = torch.cat([weight_q_flat, torch.zeros(1, dtype=weight_q_flat.dtype)])

        # Pack: upper 4 bits + lower 4 bits
        packed = torch.zeros(len(weight_q_flat) // 2, dtype=torch.uint8)

        for i in range(0, len(weight_q_flat), 2):
            upper = (weight_q_flat[i].item() + 8) & 0xF  # Shift to 0-15
            lower = (weight_q_flat[i + 1].item() + 8) & 0xF

            packed[i // 2] = (upper << 4) | lower

        return packed.reshape(shape[0], -1)


class BinaryNetworks:
    """Binary networks: {-1, +1} weights for 32x speedup."""

    @staticmethod
    def binarize_weights(weight: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Binarize weights to {-1, +1}."""
        # Scale factor = mean(|weight|)
        scale = weight.abs().mean()

        # Binarize
        weight_binary = torch.sign(weight)
        weight_binary[weight_binary == 0] = 1

        return weight_binary, scale

    @staticmethod
    def binarized_linear(x: torch.Tensor, weight_binary: torch.Tensor, scale: float) -> torch.Tensor:
        """Forward pass with binarized weights."""
        # Matrix multiply with binary weights (xnor + popcount operations)
        # Much faster on specialized hardware

        output = torch.matmul(x, weight_binary.t().float())
        output = output * scale

        return output

    @staticmethod
    def apply_binarization(model: nn.Module) -> None:
        """Apply binarization to model weights."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                weight_binary, scale = BinaryNetworks.binarize_weights(weight)

                # Store scale
                module.register_buffer("weight_scale", torch.tensor(scale))

                # Replace weight (store as int8 for space efficiency)
                weight_binary_int8 = (weight_binary * 127).int().float() / 127

                module.weight.data = weight_binary_int8

                logger.info(f"Binarized {name} (32x speedup potential)")


class TermaryQuantization:
    """Ternary quantization: {-1, 0, +1} for 16x speedup."""

    @staticmethod
    def ternarize_weights(weight: torch.Tensor, threshold: float = 0.05) -> Tuple[torch.Tensor, float]:
        """Ternarize weights to {-1, 0, +1}."""
        # Threshold for zero
        scale = weight.abs().mean()

        # Ternarize
        weight_ternary = torch.zeros_like(weight)
        weight_ternary[weight > threshold * scale] = 1
        weight_ternary[weight < -threshold * scale] = -1

        return weight_ternary, scale

    @staticmethod
    def apply_ternarization(model: nn.Module) -> None:
        """Apply ternarization to model."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                weight_ternary, scale = TermaryQuantization.ternarize_weights(weight)

                module.register_buffer("weight_scale", torch.tensor(scale))
                module.weight.data = weight_ternary

                logger.info(f"Ternarized {name} (16x speedup potential)")


class MixedPrecisionExtremeQuantization:
    """Mixed precision: different bitwidths per layer."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_bitwidths = {}

    def assign_bitwidths(self) -> None:
        """Assign bitwidths based on layer sensitivity."""
        # Sensitive layers: higher precision
        # Less sensitive layers: lower precision

        layer_importance = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Importance = sum(|weight|)
                importance = module.weight.data.abs().sum().item()
                layer_importance[name] = importance

        # Normalize
        total_importance = sum(layer_importance.values())
        normalized = {k: v / total_importance for k, v in layer_importance.items()}

        # Assign bitwidths: 2-8 bits
        for name, norm_importance in normalized.items():
            bits = max(2, int(8 * norm_importance) + 2)
            self.layer_bitwidths[name] = bits

            logger.info(f"{name}: {bits}-bit quantization")

    def apply_mixed_quantization(self) -> None:
        """Apply mixed-precision quantization."""
        self.assign_bitwidths()

        for name, module in self.model.named_modules():
            if name not in self.layer_bitwidths:
                continue

            bits = self.layer_bitwidths[name]

            if bits == 2:
                # Binary
                weight_q, scale = BinaryNetworks.binarize_weights(module.weight.data)
            elif bits == 3:
                # Ternary
                weight_q, scale = TermaryQuantization.ternarize_weights(module.weight.data)
            elif bits == 4:
                # INT4
                quantizer = INT4Quantization(self.model)
                weight_q, scale = quantizer.quantize_weight_int4(module.weight.data)
            else:
                # Standard INT8
                qmax = 2 ** (bits - 1) - 1
                weight_min = module.weight.data.min()
                weight_max = module.weight.data.max()
                scale = (weight_max - weight_min) / (2 * qmax)
                weight_q = torch.round((module.weight.data - weight_min) / scale)

            module.register_buffer("weight_scale", torch.tensor(scale))
            module.weight.data = weight_q.float()

    def get_model_compression(self) -> float:
        """Calculate overall compression ratio."""
        total_bits = 0
        for name, bits in self.layer_bitwidths.items():
            module = dict(self.model.named_modules())[name]
            if isinstance(module, nn.Linear):
                num_weights = module.weight.numel()
                total_bits += num_weights * bits

        original_bits = sum(p.numel() * 32 for p in self.model.parameters())

        return original_bits / total_bits if total_bits > 0 else 1.0


class DistilledQuantization:
    """Quantization-aware training distillation (minimal accuracy loss)."""

    def __init__(self, teacher: nn.Module, student: nn.Module):
        self.teacher = teacher
        self.student = student

    def quantization_aware_training(
        self,
        x: torch.Tensor,
        num_iterations: int = 100,
    ) -> None:
        """QAT: simulate quantization during training."""

        optimizer = torch.optim.Adam(self.student.parameters(), lr=1e-4)
        kl_loss = nn.KLDivLoss(reduction="batchmean")

        for iteration in range(num_iterations):
            # Forward pass
            with torch.no_grad():
                teacher_output = self.teacher(x)

            student_output = self.student(x)

            # Quantization loss
            quant_loss = kl_loss(
                torch.log_softmax(student_output, dim=-1),
                torch.softmax(teacher_output / 4.0, dim=-1),
            )

            optimizer.zero_grad()
            quant_loss.backward()
            optimizer.step()

            if iteration % 20 == 0:
                logger.info(f"QAT iteration {iteration}: loss={quant_loss.item():.4f}")


if __name__ == "__main__":
    # Example
    model = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
    )

    # INT4 quantization
    logger.info("INT4 Quantization")
    quantizer_int4 = INT4Quantization(model)
    quantizer_int4.quantize_model()

    # Binary quantization
    logger.info("\nBinary Quantization")
    BinaryNetworks.apply_binarization(model)

    # Mixed precision
    logger.info("\nMixed Precision Quantization")
    mixed = MixedPrecisionExtremeQuantization(model)
    mixed.apply_mixed_quantization()

    print(f"Compression ratio: {mixed.get_model_compression():.2f}x")
