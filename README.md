# Confident and Wrong

A Framework for Evaluating When Vision Model Predictions Should Not Be Trusted

This is my MSc Computer Science final project.

It looks at what happens to vision-model confidence when image quality gets worse. The main question is not only whether accuracy drops, but whether confidence drops with it, or whether the model keeps making confident predictions while becoming increasingly wrong.

## Research question

> At what point does model confidence stop aligning with actual performance under degraded image conditions, and can this be detected using simple reliability signals?

## What the framework does

The framework can:

* train or fine-tune supported image classifiers
* apply controlled blur, Gaussian noise and low-light degradation
* evaluate clean images and degradation severities 1 to 5
* save prediction-level results
* calculate accuracy, confidence and calibration metrics
* build an undegraded validation profile
* generate `trust`, `caution` and `do_not_trust` warnings
* generate reliability plots
* compare completed GTSRB model evaluations

The trust signal is a practical warning based on the evaluation results, not a safety guarantee.

## Current project scope

| Area                            | Current scope                            |
| ------------------------------- | ---------------------------------------- |
| Datasets                        | MNIST, GTSRB                             |
| MNIST model                     | SimpleCNN                                |
| GTSRB models                    | GTSRBCNN, ResNet18, MobileNetV2          |
| Final stronger-model comparison | ResNet18                                 |
| Degradations                    | Gaussian blur, Gaussian noise, low light |
| Severities                      | clean baseline plus 1 to 5               |
| Main seed                       | 42                                       |
| GTSRB input size                | 64 × 64                                  |
| Pretrained weights              | `IMAGENET1K_V1`                          |
| Transfer-learning strategy      | full fine-tuning                         |

MNIST is the simple proof of concept. GTSRB is the main colour-image case study.

ResNet18 and MobileNetV2 were compared in a controlled clean-validation pilot using the same split, seed, preprocessing and training settings. Their balanced accuracy was effectively equivalent, but ResNet18 trained faster and used less observed process memory in this CPU environment, so I selected it for the final model comparison.

## GTSRB track-aware validation split

The first GTSRB implementation used a random image-level validation split.

I later found that images from the same physical traffic-sign track were appearing in both the training and validation sets. This made the validation result too optimistic.

The final workflow uses a deterministic stratified track-aware split.

| Field                         |              Value |
| ----------------------------- | -----------------: |
| Requested validation examples |              4,000 |
| Actual validation examples    |              3,990 |
| Training examples             |             22,650 |
| Validation tracks             |                133 |
| Training tracks               |                755 |
| Track overlap                 |                  0 |
| Classes represented           |                 43 |
| Split strategy                | `stratified_track` |

Validation-track fingerprint:

```text
f37d445eac4fe94ed5b346b8939aa361ab11f8f32b60cc5c06698bb858efaba8
```

The fingerprint is saved with the checkpoint and results so the same split can be checked later.

The official GTSRB test set was kept out of architecture selection.

## Preprocessing

### MNIST

MNIST uses:

* 28 × 28 greyscale images
* standard MNIST normalisation
* no additional training augmentation

### GTSRB

GTSRB uses:

* RGB images
* resize to 64 × 64
* bilinear interpolation with antialiasing
* controlled degradation before normalisation
* ImageNet mean and standard-deviation normalisation

The order is:

```text
resize → degradation → normalisation
```

This means the degradations are applied to normal image values before normalisation.

I kept the same 64 × 64 input size across GTSRBCNN, ResNet18 and MobileNetV2 so the degradation strengths stay comparable between models.

## Reliability metrics

The main metrics are:

| Metric                  | Purpose                                                            |
| ----------------------- | ------------------------------------------------------------------ |
| Accuracy                | Overall proportion of correct predictions                          |
| Balanced accuracy       | Class-balanced performance for GTSRB                               |
| Mean confidence         | Mean maximum softmax probability                                   |
| Confidence-accuracy gap | Difference between mean confidence and accuracy                    |
| ECE                     | Top-label expected calibration error                               |
| Fixed HCER              | Wrong predictions with confidence at or above 0.90                 |
| Adaptive HCER           | HCER using a threshold from the validation confidence distribution |
| Adaptive HCER coverage  | Proportion of predictions meeting the adaptive threshold           |
| Rank-based HCER         | Error rate within a fixed highest-confidence cohort                |
| Rank-based coverage     | Proportion of predictions in that ranked cohort                    |

A positive confidence-accuracy gap shows overconfidence on average, while a negative gap shows underconfidence.

### HCER

Fixed HCER at `0.90` is the HCER measure used by the active trust policy.

Adaptive HCER is kept as a diagnostic because the validation percentile can reach `1.0` when confidence is saturated.

Rank-based HCER gives another view by always looking at a fixed top-confidence group.

HCER is useful, but it needs to be read alongside the other metrics. Under severe degradation, fixed HCER can fall because fewer predictions remain above `0.90`, even while accuracy is still getting worse.

## Validation profiles

Before degradation evaluation, each model has an undegraded validation profile built from its training-validation split.

The profile records things such as:

* dataset
* model
* checkpoint
* seed
* validation sample count
* undegraded accuracy
* mean confidence
* ECE
* confidence-accuracy gap
* fixed HCER
* adaptive HCER threshold
* balanced accuracy for GTSRB
* GTSRB track-split evidence
* rank-HCER settings where used

The evaluation code checks the profile, checkpoint and split metadata before using them together.

## Trust signal

The current trust signal is baseline-relative.

It asks:

> Has this model become significantly less reliable compared with its own undegraded behaviour?

The active GTSRB rules use:

* absolute accuracy drop
* relative error increase
* ECE increase
* confidence-gap deterioration
* fixed HCER increase

The strongest triggered rule determines:

```text
trust
caution
do_not_trust
```

Each saved trust record also includes the rules that triggered the warning.

What it does not tell me is whether the model's current performance is good enough for a particular application. A weaker model can stay close to its weaker baseline, while a stronger model can deteriorate sharply and still have better absolute accuracy.

I may add separate current-condition limits later, but they are not part of the current trust label.

## Main findings

MNIST first showed that blur, noise and low light can create quite different relationships between accuracy and confidence. The GTSRB experiments then tested the same framework on the more realistic colour-image case study.

### Gaussian noise

Noise produced the strongest confidence-performance divergence.

At severity 5:

| Metric                  |                GTSRBCNN |                ResNet18 |
| ----------------------- | ----------------------: | ----------------------: |
| Accuracy                |                  ~28.3% |                  ~28.0% |
| Mean confidence         |                  ~75.1% |                  ~57.4% |
| Confidence-accuracy gap | ~46.7 percentage points | ~29.4 percentage points |
| Fixed HCER              |                  ~17.4% |                   ~2.6% |

The two models reached almost the same accuracy, but their confidence behaviour was very different.

Both models move to `caution` at noise severity 1 and `do_not_trust` from severity 2.

### Blur

Both models deteriorate progressively.

GTSRBCNN:

```text
caution: severity 3
do_not_trust: severity 4
```

ResNet18:

```text
caution: severity 3
do_not_trust: severity 5
```

ResNet18 reaches the strongest warning one severity level later.

### Low light

Both models:

```text
trust: severity 1 to 3
caution: severity 4
do_not_trust: severity 5
```

The final warning does not happen for exactly the same reason.

GTSRBCNN crosses the severe absolute accuracy-drop rule, while ResNet18 is caught by the relative increase in error from its much stronger clean baseline.

So the same trust label can describe different failure behaviour.

## Installation

Python 3.11 is used by the CI workflow.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Datasets are downloaded through torchvision when required.

`data/`, `checkpoints/` and `results/` are excluded from version control.

## Configuration

Experiment settings are stored under `configs/`.

```text
configs/mnist.yaml
configs/gtsrb.yaml
configs/gtsrb_resnet18.yaml
configs/gtsrb_mobilenet_v2.yaml
```

The configs control things that can change between runs, including:

* dataset and model
* checkpoint path
* validation profile
* output directory
* seed
* epochs
* batch size
* learning rate
* validation size
* GTSRB split strategy
* pretrained weights
* degradation types
* severity levels
* ECE bins
* HCER settings
* trust thresholds
* development batch limits

Training augmentation entries record what the dataset pipeline actually uses. The config loader rejects augmentation settings that are not implemented.

## Visual degradation checks

The two sanity-check scripts save one image across the configured degradation severities so the degradation strengths can be checked visually.

MNIST:

```bash
python -m scripts.save_mnist_degradation_grid --config configs/mnist.yaml
```

GTSRB:

```bash
python -m scripts.save_gtsrb_degradation_grid --config configs/gtsrb.yaml
```

## Running MNIST

### Train

```bash
python -m scripts.train_mnist --config configs/mnist.yaml
```

### Build the validation profile

```bash
python -m scripts.build_mnist_validation_profile --config configs/mnist.yaml
```

### Run degradation evaluation

```bash
python -m experiments.mnist_degradation_eval --config configs/mnist.yaml
```

### Generate trust signals

```bash
python -m scripts.add_trust_signal --config configs/mnist.yaml
```

### Generate plots

```bash
python -m scripts.plot_metrics \
  --metrics results/mnist_degradation_eval/metrics_summary.csv \
  --output-dir results/mnist_degradation_eval/plots
```

## Running GTSRB

The same workflow is used for each GTSRB model.

### GTSRBCNN

Train:

```bash
python -m scripts.train_gtsrb --config configs/gtsrb.yaml
```

Build validation profile:

```bash
python -m scripts.build_gtsrb_validation_profile --config configs/gtsrb.yaml
```

Run degradation evaluation:

```bash
python -m experiments.gtsrb_degradation_eval --config configs/gtsrb.yaml
```

Generate trust signals:

```bash
python -m scripts.add_trust_signal --config configs/gtsrb.yaml
```

Generate plots:

```bash
python -m scripts.plot_metrics \
  --metrics results/gtsrb_degradation_eval/metrics_summary.csv \
  --output-dir results/gtsrb_degradation_eval/plots
```

### ResNet18

Use:

```text
configs/gtsrb_resnet18.yaml
```

For example:

```bash
python -m scripts.train_gtsrb --config configs/gtsrb_resnet18.yaml
python -m scripts.build_gtsrb_validation_profile --config configs/gtsrb_resnet18.yaml
python -m experiments.gtsrb_degradation_eval --config configs/gtsrb_resnet18.yaml
python -m scripts.add_trust_signal --config configs/gtsrb_resnet18.yaml
```

### MobileNetV2

MobileNetV2 remains supported through:

```text
configs/gtsrb_mobilenet_v2.yaml
```

Its checkpoint and pilot results are kept as part of the model-selection evidence.

## Comparing GTSRB models

After the GTSRBCNN and ResNet18 evaluations and trust signals have been generated:

```bash
python -m scripts.compare_gtsrb_models \
  --baseline-config configs/gtsrb.yaml \
  --stronger-config configs/gtsrb_resnet18.yaml \
  --output-dir results/gtsrb_model_comparison
```

The comparison checks that both runs use the same split, seed, degradation conditions and evaluation settings before joining the results.

It produces:

```text
model_comparison.csv
trust_transition_comparison.csv
plots/
```

## Saved evaluation evidence

A normal evaluation saves:

```text
config.yaml
predictions.csv
metrics_summary.csv
calibration_bins.csv
```

GTSRB also saves:

```text
split_metadata.json
```

Trust generation adds:

```text
trust_signal.json
```

Training, validation-profile, degradation-evaluation, trust and model-comparison commands refuse to replace existing evidence by default. Add `--overwrite` when replacement is intentional.

### Prediction-level results

`predictions.csv` includes:

```text
dataset
model
seed
image_id
true_label
predicted_label
correct
confidence
degradation
severity
```

These saved predictions allow metrics and later analysis to be checked without rerunning model inference.

## Repository structure

```text
.
├── .github/
│   └── workflows/
│       └── tests.yml
├── configs/
├── experiments/
│   ├── gtsrb_degradation_eval.py
│   └── mnist_degradation_eval.py
├── scripts/
│   ├── add_trust_signal.py
│   ├── build_gtsrb_validation_profile.py
│   ├── build_mnist_validation_profile.py
│   ├── compare_gtsrb_models.py
│   ├── plot_metrics.py
│   ├── save_gtsrb_degradation_grid.py
│   ├── save_mnist_degradation_grid.py
│   ├── train_gtsrb.py
│   └── train_mnist.py
├── src/
│   ├── datasets/
│   ├── degradations/
│   ├── evaluation/
│   ├── metrics/
│   ├── models/
│   ├── reporting/
│   └── utils/
├── tests/
├── EXPERIMENT_LOG.md
├── README.md
└── requirements.txt
```

## Adding another model

I kept the model extension point simple.

For another GTSRB model:

1. add it to `src/models/gtsrb_models.py`
2. define any supported pretrained weights there
3. make sure it returns one output score per class
4. define different preprocessing if the model needs it
5. add a YAML config
6. add focused construction and output-shape tests

The existing GTSRB training, validation-profile and degradation-evaluation code can then be reused.

## Adding another dataset

A new dataset needs its own loading, preprocessing and splitting logic.

Its evaluation dataset should return:

```text
image, label, image_id
```

The shared validation code can also handle validation batches with an optional image ID.

The common evaluation code is shared, while things that really differ between datasets, such as GTSRB's track split and balanced accuracy, stay dataset-specific.

Reusable parts include:

* degradation functions
* prediction collection
* calibration-bin generation
* metric calculations
* output saving
* trust-signal calculation
* plotting

## Testing

Run the complete test suite with:

```bash
python -m pytest -q
```

The tests cover:

* config validation
* deterministic splitting
* degradations
* reliability metrics
* checkpoints
* model construction
* shared evaluation
* validation profiles
* GTSRB split consistency
* trust signals
* output saving
* plots
* model comparison

Most tests use small synthetic inputs so they can run without downloading datasets or pretrained model weights.

## Continuous integration

GitHub Actions runs the full test suite on pushes and pull requests using Python 3.11.

This checks that the project still installs and passes its tests in a clean environment.

## Reproducibility

The main things I use to keep experiments reproducible are:

* fixed seeds
* YAML configs
* saved config copies
* checkpoint metadata
* preprocessing metadata
* the GTSRB validation-track fingerprint
* checkpoint and profile consistency checks
* fixed degradation definitions
* saved prediction-level evidence
* regression tests

## Historical results

Earlier exploratory results have been kept locally as development evidence, including the original random-split GTSRB experiments and later pre-refactor outputs. The YAML configs point to the current supported result locations.

## Limitations

The current project covers:

* two datasets
* three main GTSRB architectures
* three degradation types
* fixed degradation definitions
* maximum softmax probability as the main confidence score

The trust thresholds are empirical and baseline-relative.

Adaptive HCER can also become less useful when validation confidence saturates near `1.0`.

The results therefore describe the models, datasets and degradation conditions tested in this project.

## Contribution

The main contribution is a reusable evaluation workflow that brings controlled degradation tests, prediction-level evidence, reliability metrics and trust warnings together.

The experiments show that similar accuracy does not necessarily mean similar confidence behaviour. Under severe Gaussian noise, GTSRBCNN and ResNet18 reached almost the same accuracy while remaining very different in mean confidence, confidence-accuracy gap and high-confidence error behaviour.

The practical value is that two models can reach similar accuracy while failing in very different ways, and the framework makes those differences visible.