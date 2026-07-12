"""
Controlled image degradation functions.

These functions expect image tensors in the range 0 to 1.
Severity 0 returns the undegraded image.
"""

import torch
from torchvision.transforms import functional as TF

VALID_DEGRADATIONS = {"none", "blur", "noise", "low_light"}

def apply_degradation(image: torch.Tensor, degradation: str, severity: int) -> torch.Tensor:
    if degradation not in VALID_DEGRADATIONS:
        raise ValueError(f"Unknown degradation: {degradation}")
    if severity < 0 or severity > 5:
        raise ValueError("Severity must be between 0 and 5.")
    if degradation == "none" or severity == 0:
        return image.clone()
    if degradation == "blur":
        return gaussian_blur(image, severity)
    if degradation == "noise":
        return gaussian_noise(image, severity)
    if degradation == "low_light":
        return low_light(image, severity)
    raise ValueError(f"Unhandled degradation: {degradation}")


def gaussian_blur(image: torch.Tensor, severity: int) -> torch.Tensor:
    # odd kernel sizes from 3 to 11 are required for centred filtering
    kernel_size = 2 * severity + 1
    # sigma increases from 1.0 to 3.4 to strengthen blur gradually
    sigma = 0.4 + (severity * 0.6)
    return TF.gaussian_blur(
        image,
        kernel_size=[kernel_size, kernel_size],
        sigma=[sigma, sigma],
    )


def gaussian_noise(image: torch.Tensor, severity: int) -> torch.Tensor:
    # 0.08 gives a maximum noise standard deviation of 0.4
    noise_std = severity * 0.08
    noisy_image = image + torch.randn_like(image) * noise_std
    return torch.clamp(noisy_image, 0.0, 1.0)


def low_light(image: torch.Tensor, severity: int) -> torch.Tensor:
    # brightness falls from 0.85 at severity 1 to 0.25 at severity 5
    brightness_factor = max(0.1, 1.0 - (severity * 0.15))
    dark_image = image * brightness_factor
    return torch.clamp(dark_image, 0.0, 1.0)