"""Audio-to-image — generate image features from audio spectrograms."""

import torch
import torch.nn as nn


class AudioToImage(nn.Module):
    """Generate images from audio (music-to-image, voice description)."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # Spectrogram analyzer
        self.spectrogram_analyzer = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.AdaptiveAvgPool1d(1),
        )

        # Rhythm detector
        self.rhythm_detector = nn.Sequential(
            nn.Linear(32, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

        # Emotion/mood from audio
        self.mood_detector = nn.Sequential(
            nn.Linear(32, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
        )

    def forward(self, audio_spectrogram: torch.Tensor) -> torch.Tensor:
        """Convert audio to image generation features."""
        analyzed = self.spectrogram_analyzer(audio_spectrogram)
        analyzed = analyzed.squeeze(-1)

        self.rhythm_detector(analyzed)
        mood = self.mood_detector(analyzed)

        return mood
