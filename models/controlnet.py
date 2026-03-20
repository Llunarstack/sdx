# ControlNet-style conditioning: control image -> spatial features injected into DiT.
# Keeps structure (edges, depth, pose) without making the image sloppy.
import torch
import torch.nn as nn


class ControlNetEncoder(nn.Module):
    """
    Encodes a control image (e.g. depth, canny, pose) to spatial features (B, N, D)
    that match the DiT patch grid. These are added to the latent patch tokens with a scale
    so the model follows structure without overpowering the prompt.
    """

    def __init__(self, control_size: int, patch_size: int, in_channels: int = 3, hidden_size: int = 1152):
        super().__init__()
        self.control_size = control_size  # spatial size of control map (e.g. 32 to match latent)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, control_image: torch.Tensor) -> torch.Tensor:
        """
        control_image: (B, 3, H, W) in [-1, 1] or [0, 1]; will be resized to (control_size, control_size).
        Returns (B, N, hidden_size) where N = (control_size/patch_size)**2.
        """
        B, C, H, W = control_image.shape
        if H != self.control_size or W != self.control_size:
            control_image = nn.functional.interpolate(
                control_image, size=(self.control_size, self.control_size), mode="bilinear", align_corners=False
            )
        x = self.proj(control_image)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)
