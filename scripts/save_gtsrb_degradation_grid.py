"""Save a visual GTSRB degradation sanity check."""

import argparse
from pathlib import Path
from torchvision import datasets
from torchvision.utils import make_grid, save_image
from src.datasets.gtsrb import GTSRBDataset
from src.utils.config import load_config
from src.utils.seeds import set_seed

DEGRADATIONS = ("blur", "noise", "low_light")
DEFAULT_OUTPUT = "results/sanity_checks/gtsrb_degradation_grid.png"

def main() -> None:
    parser = argparse.ArgumentParser(description="Save a GTSRB degradation grid")
    parser.add_argument("--config", default="configs/gtsrb.yaml")
    parser.add_argument("--image-id", type=int, default=0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if config["dataset"] != "GTSRB":
        raise ValueError("The degradation grid requires dataset GTSRB.")

    set_seed(config["seed"])
    base_dataset = datasets.GTSRB(
        root=config["data_dir"],
        split="test",
        download=True
    )

    if args.image_id < 0 or args.image_id >= len(base_dataset):
        raise ValueError("Image ID must refer to an available GTSRB test image.")

    clean_dataset = GTSRBDataset(base_dataset, normalise=False)
    clean_image, label, _ = clean_dataset[args.image_id]
    grid_images = []

    for degradation in DEGRADATIONS:
        grid_images.append(clean_image)

        for severity in range(1, 6):
            dataset = GTSRBDataset(
                base_dataset=base_dataset,
                degradation=degradation,
                severity=severity,
                normalise=False
            )
            image, _, _ = dataset[args.image_id]
            grid_images.append(image)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(grid_images, nrow=6, padding=2)
    save_image(grid, output_path)

    print(f"Image ID: {args.image_id}")
    print(f"Class label: {label}")
    print(f"Saved degradation grid to {output_path}")

if __name__ == "__main__":
    main()