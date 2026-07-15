"""Save and load model checkpoints."""

from pathlib import Path
import torch
import torch.nn as nn

def save_model_checkpoint(model: nn.Module, checkpoint_path: Path, metadata: dict) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "metadata": metadata},
        checkpoint_path
    )

def load_model_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary.")

    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict.")

    metadata = checkpoint.get("metadata", {})

    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a dictionary.")

    model.load_state_dict(checkpoint["model_state_dict"])
    return metadata
