"""
Controlled image degradation functions.

These functions expect image tensors in the range 0 to 1.
Severity 0 means clean image.
"""

import torch
from torchvision.transforms import functional as TF

VALID_DEGRADATIONS = {"clean", "blur", "noise", "low_light"}

def apply_degradation(image: torch.Tensor, degradation: str, severity: int) -> torch.Tensor:
    if degradation not in VALID_DEGRADATIONS:
        raise ValueError(f"Unknown degradation: {degradation}")
    if severity < 0 or severity > 5:
        raise ValueError("Severity must be between 0 and 5.")
    if degradation == "clean" or severity == 0:
        return image.clone()
    if degradation == "blur":
        return gaussian_blur(image, severity)
    if degradation == "noise":
        return gaussian_noise(image, severity)
    if degradation == "low_light":
        return low_light(image, severity)
    raise ValueError(f"Unhandled degradation: {degradation}")


def gaussian_blur(image: torch.Tensor, severity: int) -> torch.Tensor:
    # sigma represents the mathematical standard deviation of the Gaussian distribution and dictates the pixel-blending radius
    # a larger sigma spreads and smooths out sharp edges much further
    # turning crisp digits into highly blurred smudges
    # 2*s+1 ensures an odd kernel size (3 to 11) required by standard filtering
    kernel_size = 2 * severity + 1
    # 0.4 base prevents zero sigma and 0.6 scale maps severity 5 to a heavy sigma of 3.4
    sigma = 0.4 + (severity * 0.6)
    return TF.gaussian_blur(
        image,
        kernel_size=[kernel_size, kernel_size],
        sigma=[sigma, sigma],
    )


def gaussian_noise(image: torch.Tensor, severity: int) -> torch.Tensor:
    # noise strength increases with severity
    # 0.08 scaling gives a max noise standard deviation of 0.4 at severity 5
    noise_std = severity * 0.08
    noisy_image = image + torch.randn_like(image) * noise_std
    return torch.clamp(noisy_image, 0.0, 1.0)


def low_light(image: torch.Tensor, severity: int) -> torch.Tensor:
    # brightness is reduced as severity increases
    # 0.15 step scales brightness down linearly from 0.85 (sev 1) to 0.25 (sev 5)
    # max floor of 0.1 stops the image from going completely pitch black
    brightness_factor = max(0.1, 1.0 - (severity * 0.15))
    dark_image = image * brightness_factor
    # keep pixel values strictly inside the valid 0.0 to 1.0 probability range
    return torch.clamp(dark_image, 0.0, 1.0)