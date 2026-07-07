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

## 2026-07-01

What I worked on
Two small changes to the MNIST baseline script.

What changed
Added a set_seed function that sets the torch seed and also the CUDA seed when CUDA is available, and used it in main instead of the inline call. Shortened the comment on the batch cap to plainer wording.

Problem or decision
None.

Next step
Still to add degradations and reliability metrics later.

## 2026-07-07

What I worked on
Completed Stage 1 of the reliability framework by implementing prediction-level output saving for the MNIST clean baseline script.

What changed
Added logic to convert raw model logits into probabilities via softmax and extract the highest confidence values. Added an evaluation loop that aggregates individual image predictions, true labels, and correctness flags into a structured row list. Implemented output functions that export these detailed records to `predictions.csv` and compute summary statistics—including `accuracy`, `mean_confidence`, and the `confidence_accuracy_gap`—saved directly to `metrics_summary.csv`.

Problem or decision
Opted to track and calculate accuracy manually using correct/total counters inline rather than importing an external high-level metrics library. This keeps the execution overhead low and prevents unnecessary package dependencies.

Next step
Move to Stage 2 by extracting these basic metric calculations out of the experiment file into a dedicated, reusable module at `src/metrics/basic.py` accompanied by automated tests.

## 2026-07-07

What I worked on
Created reusable metric helpers and verified them with automated tests for Stage 2.

What changed
Extracted `accuracy_from_correct`, `mean_confidence`, and `confidence_accuracy_gap` out of the main experiment loop and into a dedicated `src/metrics/basic.py` module. Created a corresponding `tests/test_basic_metrics.py` file to handle edge cases and confirm calculations under `pytest`.

Problem or decision
Added input validation constraints inside `mean_confidence` to explicitly raise a `ValueError` if passed confidence scores fall outside the mathematically valid 0 to 1 range.

Next step
Update the clean MNIST baseline script to import and utilize these new centralized metric functions instead of computing them inline.