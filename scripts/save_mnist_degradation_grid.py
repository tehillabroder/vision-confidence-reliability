"""Save a visual MNIST degradation sanity check."""

import argparse
from pathlib import Path
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from src.degradations.image_degradations import apply_degradation
from src.utils.config import load_config
from src.utils.seeds import set_seed

DEFAULT_OUTPUT = "results/sanity_checks/mnist_degradation_grid.png"

def main() -> None:
    parser = argparse.ArgumentParser(description="Save an MNIST degradation grid")
    parser.add_argument("--config", default="configs/mnist.yaml")
    parser.add_argument("--image-id", type=int, default=0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if config["dataset"] != "MNIST":
        raise ValueError("The degradation grid requires dataset MNIST.")

    dataset = datasets.MNIST(
        root=config["data_dir"],
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    if args.image_id < 0 or args.image_id >= len(dataset):
        raise ValueError("Image ID must refer to an available MNIST test image.")

    image, label = dataset[args.image_id]
    evaluation_config = config["evaluation"]
    grid_images = []

    for degradation in evaluation_config["degradations"]:
        # reset the seed so that noise severity changes use the same random pattern
        for severity in range(0, 6):  # severity 0 is clean, 1 to 5 increase degradation
            set_seed(config["seed"])
            grid_images.append(apply_degradation(image, degradation, severity))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(grid_images, nrow=len(evaluation_config["severity_levels"]) + 1, padding=2)
    save_image(grid, output_path)

    print(f"Image ID: {args.image_id}")
    print(f"Class label: {label}")
    print(f"Saved degradation grid to {output_path}")

if __name__ == "__main__":
    main()