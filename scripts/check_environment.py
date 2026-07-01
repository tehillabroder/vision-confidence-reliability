import torch
import torchvision
import pandas as pd
import yaml
import sklearn

def check():
    print("Checking core environment dependencies...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"PyYAML version: {yaml.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print("\nStatus: All core dependencies are installed and importable.")

if __name__ == "__main__":
    check()