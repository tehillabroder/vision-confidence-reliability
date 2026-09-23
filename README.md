# Confident and Wrong

**A reproducible research framework for evaluating vision-model reliability under controlled image degradation.**

Confident and Wrong investigates both when and how model confidence stops being a useful indication of correctness as image conditions deteriorate.

The project provides a reproducible pre-deployment evaluation framework for applying controlled blur, Gaussian noise and low-light degradation to image classifiers and analysing the resulting behaviour using accuracy, calibration, high-confidence errors, failure-detection performance and prediction-level confidence transitions.

Rather than asking only whether a model becomes less accurate, the framework examines what conventional aggregate metrics can hide. Examples include models with similar accuracy but very different failure behaviour, highly confident incorrect predictions, deteriorating reliability within high-confidence predictions, and cases where already-wrong predictions become more confident as degradation increases.

The current evaluation covers MNIST and GTSRB and includes custom CNN architectures alongside pretrained ResNet18 and MobileNetV2 models.

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
| Primary GTSRB comparison        | GTSRBCNN vs ResNet18                     |
| Confirmatory GTSRB model        | MobileNetV2                              |
| Degradations                    | Gaussian blur, Gaussian noise, low light |
| Severities                      | clean baseline plus 1 to 5               |
| Main seed                       | 42                                       |
| GTSRB input size                | 64 × 64                                  |
| Pretrained weights              | ResNet18/MobileNetV2: `IMAGENET1K_V1`    |
| Transfer-learning strategy      | ResNet18/MobileNetV2: full fine-tuning   |

MNIST is the simple proof of concept. GTSRB is the main colour-image case study.

ResNet18 and MobileNetV2 were first compared in a controlled clean-validation pilot using the same split, seed, preprocessing and training settings. Their balanced accuracy was effectively equivalent, but ResNet18 trained faster and used less observed process memory in this CPU environment, so it was selected for the primary GTSRB comparison. MobileNetV2 was retained and later evaluated under the same degradation conditions as a confirmatory model.

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

The later evidence analysis also uses high-confidence coverage, conditional error within the high-confidence group, failure-detection AUROC using `1 - confidence`, top-confidence cohort accuracy, and same-image confidence transitions. These diagnostics are kept separate because calibration, confidence ranking and high-confidence error behaviour answer different questions.

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

Application-specific current-condition limits are outside the current trust label and remain future work.

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

### What the aggregate metrics hid

At Gaussian noise severity 5, GTSRBCNN and ResNet18 reached almost the same accuracy, at `28.33%` and `28.01%`. Their confidence was much less similar. Failure-detection AUROC was `0.7958` for GTSRBCNN and `0.8664` for ResNet18, while the highest-confidence 10% of predictions were `86.14%` and `98.89%` accurate respectively.

The paired image analysis also showed that similar accuracy did not mean the models were succeeding and failing on the same examples. Of the 12,630 test images, both models were wrong on 7,449, but only 38 of those shared failures received the same wrong class.

For GTSRBCNN under Gaussian noise, fixed HCER fell from `19.83%` at severity 3 to `17.42%` at severity 5. Over the same interval, coverage above `0.90` confidence fell from `56.26%` to `37.08%`, while the error rate within that remaining high-confidence group rose from `35.24%` to `46.98%`.

For ResNet18 between noise severities 4 and 5, 7,435 images were wrong at both levels. `72.7%` of those predictions became more confident and their mean confidence increased by `6.27` percentage points, even though mean confidence across the full condition fell.

### Trust-rule ablation

The final GTSRB warning timing was performance-led. Absolute accuracy drop alone reproduced all GTSRBCNN labels, while relative error increase alone reproduced all ResNet18 labels. Removing ECE, confidence-gap deterioration or fixed HCER did not change the final warning labels.

The confidence measures were still useful because they explained behaviour that the final warning could not show.

### MobileNetV2 confirmatory check

MobileNetV2 had the highest undegraded GTSRB test accuracy at `96.56%`, but fell to `14.50%` under Gaussian noise severity 5. Its failure-detection AUROC was `0.7723`, and the highest-confidence 10% of predictions were `58.67%` accurate.

It also did not reproduce the strong ResNet18 increase in confidence on persistent errors. The mean confidence change for MobileNetV2 predictions that were wrong at both noise severities 4 and 5 was slightly negative at `-0.38` percentage points. This limits the ResNet18 finding to that model and condition rather than treating it as a general pretrained-model behaviour.

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

`data/` and `checkpoints/` are excluded from version control. Compact research evidence is preserved under `results/`, while large prediction-level CSVs, superseded exploratory outputs and local presentation material remain outside normal Git tracking.

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

MobileNetV2 uses:

```text
configs/gtsrb_mobilenet_v2.yaml
```

It was retained after the clean-validation pilot and later run through the full GTSRB degradation evaluation as a confirmatory model.

```
python -m scripts.train_gtsrb --config configs/gtsrb_mobilenet_v2.yaml
python -m scripts.build_gtsrb_validation_profile --config configs/gtsrb_mobilenet_v2.yaml
python -m experiments.gtsrb_degradation_eval --config configs/gtsrb_mobilenet_v2.yaml
python -m scripts.add_trust_signal --config configs/gtsrb_mobilenet_v2.yaml
```

## Comparing GTSRB models

The primary GTSRB comparison uses GTSRBCNN and ResNet18. After both evaluations and trust signals have been generated:

```bash
python -m scripts.compare_gtsrb_models \
  --baseline-config configs/gtsrb.yaml \
  --stronger-config configs/gtsrb_resnet18.yaml \
  --output-dir results/gtsrb_model_comparison
```

The comparison checks that both runs use the same split, seed, degradation conditions and evaluation settings before joining the results. 

MobileNetV2 was evaluated separately as a confirmatory model rather than replacing the primary GTSRBCNN-ResNet18 comparison.

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

The later evidence analysis works from these saved outputs rather than rerunning model inference. This includes paired model failures, confidence transitions, trust-rule attribution and ablation, confidence diagnostics and paired bootstrap analysis.

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

These saved predictions allow metrics and later prediction-level analysis to be checked or extended without rerunning model inference. They were used for the paired model-failure analysis, same-image confidence transitions and bootstrap checks reported in the final study.

## Repository structure

```text
.
├── .github/
│   └── workflows/
│       └── tests.yml
├── configs/
│   ├── gtsrb.yaml
│   ├── gtsrb_resnet18.yaml
│   ├── gtsrb_mobilenet_v2.yaml
│   └── mnist.yaml
├── experiments/
│   ├── gtsrb_degradation_eval.py
│   └── mnist_degradation_eval.py
├── results/
│   ├── gtsrb_degradation_eval/
│   ├── gtsrb_resnet18_degradation_eval/
│   ├── gtsrb_mobilenet_v2_degradation_eval/
│   ├── gtsrb_evidence_analysis/
│   │   ├── bootstrap_uncertainty.csv
│   │   ├── class_failure_summary.csv
│   │   ├── confidence_diagnostics.csv
│   │   ├── paired_model_failures.csv
│   │   ├── prediction_confidence_transitions.csv
│   │   ├── trust_rule_ablation.csv
│   │   ├── trust_rule_attribution.csv
│   │   └── final_outputs/
│   ├── gtsrb_model_comparison/
│   ├── gtsrb_model_pilot/
│   ├── sanity_checks/
│   └── *_validation_profile.json
├── scripts/
│   ├── add_trust_signal.py
│   ├── analyse_gtsrb_evidence.py
│   ├── bootstrap_gtsrb_evidence.py
│   ├── build_gtsrb_validation_profile.py
│   ├── build_mnist_validation_profile.py
│   ├── compare_gtsrb_models.py
│   ├── plot_gtsrb_evidence.py
│   ├── plot_metrics.py
│   ├── save_gtsrb_degradation_grid.py
│   ├── save_mnist_degradation_grid.py
│   ├── summarise_gtsrb_confirmatory_models.py
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

Large datasets and trained checkpoints are kept locally under data/ and checkpoints/. Prediction-level CSVs are also kept locally because of their size, while the compact summaries, analysis tables, configurations and final figures needed to inspect the reported findings are preserved in results/.

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
* prediction-level evidence analysis

## Testing

The final automated suite contains 197 tests.

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
* GTSRB track-split consistency
* trust signals
* output saving
* plots and reporting
* model comparison
* prediction-level analysis
* trust-rule attribution and ablation
* bootstrap analysis
* provenance and overwrite safeguards

Metric tests use small known examples where the expected result can be checked directly. The suite also checks invalid inputs, split integrity, checkpoint metadata, saved-output structure and small end-to-end evaluation paths.

Most automated tests use small synthetic or replacement inputs so they can run without downloading full datasets or pretrained model weights. Full model training, pretrained-weight loading and the complete degradation experiments were checked separately during the experimental runs.

## Continuous integration

GitHub Actions runs the full test suite on pushes and pull requests using Python 3.11.

This checks that the project still installs and passes its tests in a clean environment.

## Reproducibility

The main things I use to keep experiments reproducible are:

* fixed seeds
* YAML configs and saved config copies
* checkpoint and preprocessing metadata
* the GTSRB validation-track fingerprint
* checkpoint, profile and split consistency checks
* fixed degradation definitions
* stable image IDs
* saved prediction-level evidence
* provenance checks
* overwrite protection
* regression and end to end tests

The aim is not only to make a run repeatable, but to preserve enough information to identify which model, split, configuration and predictions produced each reported result.

## Historical results

Earlier exploratory results have been kept locally as development evidence, including the original random-split GTSRB experiments and later pre-refactor outputs. The YAML configs point to the current supported result locations.

## Limitations

The current project covers:

* two datasets
* three GTSRB architectures
* three controlled degradation types
* fixed degradation definitions
* maximum softmax probability as the main confidence score
* one training seed for the final model runs

The paired bootstrap analysis estimates uncertainty across the fixed test images, not variation from retraining the models.

The trust thresholds are baseline-relative engineering choices. They identify deterioration from each model's own undegraded behaviour rather than deciding whether its absolute performance is suitable for a particular application.

Adaptive HCER can also become less useful when validation confidence saturates near `1.0`.

ResNet18 and MobileNetV2 began from ImageNet-pretrained weights, while GTSRBCNN was trained from scratch. The comparisons therefore reflect the final trained models, including differences in architecture and pretraining.

The findings should be interpreted within the models, datasets and degradation conditions tested here rather than as general behaviour of all vision models.

## Contribution

The main contribution is a reusable evaluation workflow that brings controlled degradation testing, saved prediction-level evidence, reliability metrics, failure-detection diagnostics and simple condition-level warnings together.

The experiments show why the framework benefits from both summary measures and deeper prediction-level analysis. GTSRBCNN and ResNet18 reached almost identical accuracy under severe Gaussian noise while showing very different confidence and failure-detection behaviour. HCER could fall while the remaining high-confidence predictions became less reliable, and average confidence could fall while persistent errors became more confident.

There was no single point at which confidence stopped aligning with performance across all degradation types. The trust signal indicates when deterioration has become serious enough to investigate, while the prediction-level analysis shows how confidence is behaving underneath that warning.

That is the practical value of the work. Rather than stopping at whether accuracy has fallen, the framework shows when reliability starts to shift, how the failure develops and what aggregate metrics may be hiding. It provides a reproducible way to investigate confident failure, not just observe it.
