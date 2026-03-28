from __future__ import annotations

import torch

from utils.training.ladd_distillation import (
    LADDConfig,
    LatentPatchDiscriminator,
    ladd_discriminator_step,
    ladd_generator_step,
    teacher_student_mse,
)


class ToyDenoiser(torch.nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_teacher_student_mse():
    c = 8
    teacher = ToyDenoiser(c).eval()
    student = ToyDenoiser(c).train()
    b, h, w = 2, 16, 16
    x = torch.randn(b, c, h, w)
    t = torch.zeros(b, dtype=torch.long)
    loss = teacher_student_mse(teacher, student, x, t)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_ladd_gd_step():
    c = 4
    D = LatentPatchDiscriminator(c, base=16)
    opt_d = torch.optim.Adam(D.parameters(), lr=1e-3)
    teacher = ToyDenoiser(c).eval()
    student = ToyDenoiser(c).train()
    opt_g = torch.optim.Adam(student.parameters(), lr=1e-2)
    cfg = LADDConfig(mse_teacher=1.0, adversarial=0.0)
    real = torch.randn(2, c, 8, 8)
    fake = torch.randn(2, c, 8, 8)
    ladd_discriminator_step(D, opt_d, real, fake, cfg=cfg)
    x = torch.randn(2, c, 8, 8)
    t = torch.zeros(2, dtype=torch.long)
    ladd_generator_step(D, student, opt_g, x, t, teacher, cfg=cfg, latent_for_d=None)
