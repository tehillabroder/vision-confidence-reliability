"""Save a visual MNIST degradation sanity check."""

from pathlib import Path
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from src.degradations.image_degradations import apply_degradation

DATA_DIR = "data"
OUTPUT_PATH = Path("results/sanity_checks/mnist_degradation_grid.png")

def main():
    # create the output folder before saving the sanity check image
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = datasets.MNIST(
        DATA_DIR,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    image, _ = dataset[0]  # use one fixed test image for a repeatable visual check
    grid_images = []
    for degradation in ["clean", "blur", "noise", "low_light"]:
        for severity in range(0, 6):  # severity 0 is clean, 1 to 5 increase degradation
            if degradation == "clean":
                degraded = apply_degradation(image, "clean", 0)
            else:
                degraded = apply_degradation(image, degradation, severity)
            grid_images.append(degraded)
    grid = make_grid(grid_images, nrow=6, padding=2)  # six columns show severities 0 to 5
    save_image(grid, OUTPUT_PATH)
    print(f"Saved degradation grid to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()