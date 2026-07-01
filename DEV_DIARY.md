## 2026-07-01

What I worked on
Created a minimal MNIST baseline script at experiments/mnist_baseline.py.

What changed
Added a SimpleCNN with two conv layers and two fully connected layers, MNIST data loaders, a brief training loop and a clean accuracy check. No checkpoints or large files are saved. The MNIST data goes into data, which is gitignored.

Problem or decision
The machine is CPU only so I kept training to one epoch by default to keep it quick, about 47 seconds. Ran it with one epoch and got 98.35 percent clean test accuracy. The main MNIST mirror returned a 404 but torchvision fell back to the S3 mirror on its own.

Next step
Add controlled degradations like blur, noise and low light, and reliability metrics later.

## 2026-07-01

What I worked on
Small tweaks to the MNIST baseline script.

What changed
Added a seed argument with default 42 and set the torch seed for repeatable runs. Added an optional max-train-batches flag to cap batches per epoch for quick debugging. The script now prints the number of train and test samples. Kept it simple and self contained with no config files or extra metrics.

Problem or decision
When the batch cap stops an epoch early the average loss now divides by the batches actually run, not the full loader length.

Next step
Still to add degradations and reliability metrics later, but not yet.
