# Experiment log

Project: Confident and Wrong: A Framework for Evaluating When Vision Model Predictions Should Not Be Trusted

Research question: At what point does model confidence stop aligning with actual performance under degraded image conditions, and can this be detected using simple reliability signals?

This is a curated record of the experiments and engineering decisions that shaped the final project. I have consolidated my development-diary notes, preserving the parts that affected the validity, reproducibility or interpretation of the final evidence.

## Final experimental scope

The finished framework uses MNIST as the simple controlled baseline and GTSRB as the main colour-image case study. Images are evaluated undegraded and under Gaussian blur, Gaussian noise and low light at severity levels 1 to 5.

The main GTSRB comparison uses GTSRBCNN and ResNet18, with MobileNetV2 included as an additional confirmatory model.

All GTSRB models use the same 64 × 64 RGB input size, ImageNet normalisation, track-aware validation split, official 12,630-image test set, degradation definitions, severity levels, metric calculations and trust policy.

The main saved measures are accuracy, balanced accuracy, mean confidence, expected calibration error, confidence-accuracy gap and high-confidence error rate. The later evidence analysis adds high-confidence coverage, conditional error inside the high-confidence group, failure-detection AUROC, paired failures, class-level summaries, trust-rule attribution, rule ablation and image-level confidence transitions.

These additions came from questions raised by the original results. Each one has a specific reason for being added.

The project uses seed 42, configuration-driven experiments, saved checkpoints and validation profiles, prediction-level evidence, split fingerprints, protected output paths and automated tests.

The trust signal is a baseline-relative warning about deterioration from a model's own undegraded behaviour.

## 2026-07-01 to 2026-07-09 - Building the MNIST proof of concept

### What I worked on

I started with a small proof of concept. I built a simple MNIST CNN, trained it briefly on CPU and checked that it could produce sensible undegraded predictions.

The first one-epoch proof of concept reached about 98.35% clean test accuracy. That was enough to move on because the project is about reliability under degradation (and not about getting the highest possible classifier score).

I then added repeatable seeding, optional batch limits for smoke tests and prediction-level output saving. Each result row records the image ID, true label, predicted label, correctness, confidence, degradation, severity and seed.

Saving those predictions turned out to be an important early decision because a lot of the later analysis could then be done without rerunning inference.

### What changed

The first reusable metric functions covered accuracy, mean confidence and the signed confidence-accuracy gap. I kept them small and added checks so empty or invalid inputs fail clearly.

I implemented Gaussian blur, Gaussian noise and low light with severity 0 kept as the unchanged image and severities 1 to 5 increasing in strength.

The degradations are applied dynamically rather than saving separate corrupted datasets.

I also generated a visual degradation grid to check that the severity levels were actually getting stronger in the intended direction before relying on the numerical results.

### What I found

The first MNIST runs already showed that the degradations were creating different reliability problems.

Blur mainly reduced accuracy and confidence together.

Noise produced the clearest high-confidence mistakes.

Low light often kept much of the accuracy while confidence fell much more sharply, producing strong underconfidence.

It was already clear at this point that accuracy alone would not answer the research question.

### Decision

MNIST stayed as the controlled baseline, while the model, degradation and metric code moved into reusable modules ready for the larger GTSRB workflow.

## 2026-07-09 to 2026-07-19 - Reliability metrics, saved models and the first trust signal

### What I worked on

I added expected calibration error and high-confidence error rate, then separated model training from degradation evaluation.

The evaluation now loads one fixed checkpoint and runs every condition against the same trained model. This means a comparison between degradation levels is not also comparing separately trained networks.

### What changed

ECE uses 10 uniform confidence bins. I implemented the binning directly so the edge handling, empty bins and weighting remain visible and testable. Ten bins is a conventional starting point used in calibration work such as Guo et al., rather than something I am treating as an optimal value.

Fixed HCER measures the proportion of all predictions that are both wrong and at least 0.90 confident.

I also introduced an adaptive confidence threshold taken from the undegraded validation distribution. That threshold is fixed once from validation data and reused across all test conditions.

The MNIST training stage uses a deterministic 55,000/5,000 training-validation split. The trained SimpleCNN reached about 98.08% validation accuracy after one epoch. The final evaluation loaded that checkpoint and ran all 16 conditions over the untouched 10,000-image test set.

The trust policy moved into configuration. Each degraded condition is compared with the model's own undegraded test baseline using:

* absolute accuracy drop
* relative error increase
* ECE increase
* confidence-gap deterioration
* HCER increase

The output records the actual triggered rules rather than only saving the final traffic-light label.

I also kept the direction of confidence-gap movement, because ECE on its own does not tell me whether calibration is moving towards overconfidence or underconfidence.

### What I found

The full MNIST evaluation confirmed the early pattern.

Blur produced the largest performance decline, noise produced the clearest overconfidence and high-confidence errors, and low light produced strong underconfidence while much of the accuracy remained intact.

The trust signal was useful as a summary, but the rule explanations were already showing that the same warning could happen for quite different reasons.

### Decision

I kept the trust thresholds unchanged rather than tuning them around MNIST. MNIST is unusually easy and highly confident, so fitting the policy to it would make the later GTSRB results much harder to interpret fairly.

## 2026-07-20 to 2026-07-22 - Extending the framework to GTSRB

### What I worked on

I added GTSRB as the main case study.

The loader uses the official torchvision dataset, keeps stable image IDs and leaves the official test set separate from training and validation.

GTSRB images are converted to RGB, resized to 64 × 64, converted to tensors and normalised using ImageNet statistics. Degradation happens before normalisation while pixel values are still in the normal 0 to 1 image range.

I built a custom GTSRBCNN with 43 outputs and added balanced accuracy, because GTSRB has uneven class frequencies.

### Initial result and concern

The first full model looked extremely strong on the original image-level validation split:

* validation accuracy: 98.875%
* validation balanced accuracy: 98.938%

The official undegraded test result was much lower:

* accuracy: about 88.63%
* balanced accuracy: about 84.04%

That gap was too large to ignore.

I checked the configuration, preprocessing, checkpoint loading, evaluation mode, labels, prediction counts and metric calculations. The official test results contained all 12,630 images, the image IDs were unique, all 43 classes were present and recalculated metrics matched the saved summaries.

The split then became the main suspect. GTSRB contains sequences of closely related frames from the same physical traffic sign, so a random image-level split can place neighbouring frames from one sign into both training and validation.

### What changed

I replaced the random image-level split with a deterministic stratified track split.

The requested validation size was 4,000, but each track contains 30 frames. I used the nearest complete-track size rather than split one track just to hit 4,000 exactly.

The corrected split contains:

* 22,650 training images
* 3,990 validation images
* 755 training tracks
* 133 validation tracks
* 0 overlapping tracks
* all 43 classes represented in both subsets

The validation-track fingerprint is:

`f37d445eac4fe94ed5b346b8939aa361ab11f8f32b60cc5c06698bb858efaba8`

The fingerprint is stored with the checkpoint, validation profile and evaluation evidence so later runs can be checked against the same split.

### Corrected result

After retraining:

* validation accuracy: 0.889724
* validation balanced accuracy: 0.824540
* validation ECE: about 0.0655
* track overlap: 0

The official undegraded test results were:

* accuracy: 0.869200
* balanced accuracy: 0.816802

The validation-to-test accuracy gap dropped from about 10.24 percentage points to about 2.05.

That made much more sense once training and validation no longer shared physical sign tracks.

### Decision

I accepted the lower validation result because the point of the change was to remove leakage, not protect the earlier headline score.

The original random-split checkpoint and results remain exploratory evidence only. The track-aware versions became the basis for the final GTSRB analysis.

This was one of the most important points in the project. Without investigating the validation gap, I could easily have reported a very impressive result without realising why it was so high.

## 2026-08-16 to 2026-08-17 - Auditing HCER and calibration behaviour

### What I worked on

The adaptive HCER threshold raised another question.

The 90th percentile of the GTSRB validation confidence distribution became exactly 1.0 because so many softmax outputs were saturated. That made adaptive HCER very small, so I looked more closely at what the threshold was actually selecting.

### What changed

I added high-confidence coverage so HCER can be read alongside the proportion of predictions that actually meet the confidence threshold.

I also added rank-based HCER, which looks directly at the top 10% most-confident predictions. This gives a fixed-size high-confidence group even when lots of predictions have the same confidence.

The saved calibration bins were also independently checked against the ECE calculation.

### What I found

For the corrected GTSRBCNN validation profile, the adaptive threshold was 1.0 but covered about 30.1% of predictions rather than roughly 10%.

So the low adaptive HCER was not caused by an almost empty group. Confidence saturation and ties at 1.0 meant the percentile threshold was selecting a much larger cohort than expected.

Rank-based HCER selected exactly 10%, which made it easier to compare across models.

The calibration checks also confirmed that ECE loses direction. Two conditions can have similar ECE while one is overconfident and the other underconfident, which is why the signed confidence-accuracy gap remains useful beside it.

### Decision

Fixed HCER at 0.90 remains the HCER rule used by the GTSRB trust policy.

Adaptive HCER remains saved as audit evidence, while rank-based HCER and coverage stay as supporting diagnostics.

## 2026-08-17 to 2026-08-18 - Finalising the baseline trust behaviour

### What I worked on

I applied the corrected GTSRBCNN results to the trust policy and looked at when each degradation first crossed the warning thresholds.

### Results

Blur deteriorated gradually:

* severities 1 and 2: `trust`
* severity 3: `caution`
* severities 4 and 5: `do_not_trust`

Gaussian noise was much more damaging:

* severity 1: `caution`
* severities 2 to 5: `do_not_trust`

At noise severity 5, accuracy was about 28.33% while mean confidence remained about 75.06%, giving a confidence-accuracy gap of about 46.73 percentage points.

Low light developed later:

* severities 1 to 3: `trust`
* severity 4: `caution`
* severity 5: `do_not_trust`

The undegraded official-test baseline itself was not perfectly calibrated:

* accuracy: about 86.92%
* mean confidence: about 94.43%
* ECE: about 7.51%
* fixed HCER: about 4.48%

So `trust` means the condition has not deteriorated enough from that model's own undegraded baseline to cross a warning threshold. It is not an absolute score of model quality.

### Decision

I kept the clarified trust signal as a baseline-relative deterioration warning.

## 2026-08-17 to 2026-08-27 - Consolidating the framework and protecting evidence

### What I worked on

Once both MNIST and GTSRB were working, I went back through the codebase to see where the two workflows had drifted apart and where duplicated logic could cause problems later.

### What changed

The shared evaluation runner now handles:

* experiment-condition construction
* model inference and softmax confidence collection
* prediction rows
* calibration rows
* shared evaluation-setting validation
* empty-output checks
* saving predictions, metrics, calibration bins and config copies

GTSRB keeps its own balanced accuracy and track-split evidence.

The validation collector now handles both ordinary two-item batches and GTSRB batches containing an image ID. Common validation-profile checking moved into the shared validation module, while the track-specific checks remain with GTSRB.

Supported GTSRB models and pretrained-weight choices now have one source of truth.

Configuration validation checks invalid dataset/model combinations, unsupported weights, impossible thresholds and augmentation settings that do not match the actual dataset pipeline.

Checkpoint loading checks the saved state dictionary and provenance such as dataset, model, seed and split metadata.

MNIST's previously hard-coded learning rate of 0.001 also moved into configuration, while checkpoint preprocessing and augmentation metadata now comes from the implemented dataset definitions rather than separate descriptive values that could disagree with the code.

I also added overwrite protection. Training, validation, evaluation, trust-generation and comparison scripts refuse to replace existing evidence unless `--overwrite` is explicitly supplied.

That is particularly useful with development batch limits, where a quick smoke run could otherwise accidentally replace a full result.

### Testing

I wrote tests alongside the code at every stage, checked the new tests themselves, and reran the full suite after making meaningful changes.

The final automated suite contains `197` tests covering areas including:

* metric calculations and invalid inputs
* degradation behaviour and preprocessing order
* deterministic dataset splits
* GTSRB track separation
* model setup and training behaviour
* saved checkpoints and their experiment details
* validation profiles
* evaluation outputs
* overwrite protection
* trust rules
* offline evidence analysis
* bootstrap calculations
* plotting and model comparison

Small synthetic datasets and random model weights are used where possible so the normal test suite does not depend on downloading datasets or ImageNet weights.

Real pretrained-weight loading and the full experiment runs were checked separately.

### Decision

The cleanup made the shared parts of the framework clear and left a straightforward path for adding further supported models or datasets.

## 2026-08-19 to 2026-08-20 - Controlled pretrained-model pilot

### What I worked on

I added ResNet18 and MobileNetV2 through the same configurable GTSRB model builder, then ran a controlled validation pilot to select the stronger architecture for the main comparison.

### Controlled settings

Both models used:

* 22,650 training images
* the same 3,990 validation images
* the same 133 validation tracks
* 0 track overlap
* seed 42
* `IMAGENET1K_V1` weights
* full fine-tuning for 10 epochs
* 64 × 64 RGB inputs
* ImageNet normalisation
* no extra blur, noise, brightness or geometric training augmentation

I kept 64 × 64 inputs across the GTSRB models because moving only the pretrained architectures to 224 × 224 would also change the effective strength of the degradations.

I used the same full fine-tuning approach for both models.

### Results

ResNet18:

* validation accuracy: 0.9694
* validation balanced accuracy: 0.9637
* training time: about 2,007 seconds, or 33.5 minutes
* peak process memory: about 642 MiB
* parameters: about 11.20 million
* checkpoint size: about 42.8 MiB

MobileNetV2:

* validation accuracy: 0.9822
* validation balanced accuracy: 0.9629
* training time: about 6,800 seconds, or 113.3 minutes
* peak process memory: about 789 MiB
* parameters: about 2.28 million
* checkpoint size: about 8.9 MiB

MobileNetV2 had higher standard validation accuracy and a much smaller checkpoint.

Balanced accuracy was effectively tied, with ResNet18 marginally higher. ResNet18 was also about 3.4 times faster to train in this specific CPU environment, as well as used less measured peak memory.

### Decision

I selected ResNet18 as the main stronger model before using the official test set or degradation results.

The decision was based on essentially tied balanced accuracy alongside much lower measured training time and memory use in the project environment.

Both pilot checkpoints were kept.

## 2026-08-21 to 2026-08-22 - ResNet18 degradation evaluation and initial comparison

### What I worked on

I rebuilt the ResNet18 validation profile and ran the selected checkpoint through the same 16-condition GTSRB evaluation used for GTSRBCNN.

The run produced 202,080 predictions from the official 12,630-image test set.

### Results

At Gaussian noise severity 5, ResNet18 reached:

* accuracy: 0.2801
* mean confidence: 0.5739
* confidence-accuracy gap: 0.2937

Blur severity 5 retained 0.7417 accuracy, while fixed HCER reached about 0.0526.

Low-light severity 5 retained 0.8146 accuracy with a confidence gap of about 0.0701.

The trust timing followed the same broad order as the baseline CNN:

* blur: caution at severity 3, then do not trust later
* noise: caution at severity 1 and do not trust from severity 2
* low light: do not trust at severity 5

ResNet18 held up one severity longer before blur reached `do_not_trust`.

### What I found

The same final trust label could come from very different kinds of deterioration.

At low-light severity 5, GTSRBCNN was driven by absolute accuracy loss, while ResNet18 was driven by relative error increase from a stronger undegraded baseline.

That made it clearer that while the trust signal is useful for comparing a model with its own normal behaviour, it should not be used as a plain score for comparing models with each other.

## 2026-08-27 to 2026-08-28 - Moving from summary metrics to failure evidence

### What I worked on

By this point the main runs were complete, but the summary metrics were not telling the whole story. I went back to the saved prediction CSVs and started unpacking what was sitting underneath them.

Three things stood out.

### HCER needed more context

Fixed HCER can fall at severe degradation even while the predictions that remain highly confident are becoming less reliable.

The useful relationship is:

`fixed HCER = high-confidence coverage × conditional error among high-confidence predictions`

For GTSRBCNN under severe noise, fewer predictions remained above the 0.90 confidence threshold, so high-confidence coverage fell.

But the predictions that were still covered by that threshold were becoming less reliable. By severity 5, about 46.98% of the high-confidence group was wrong.

So the falling HCER did not mean the model was becoming safer. HCER fell partly because coverage was shrinking, while the conditional error rate within the remaining high-confidence predictions was actually rising.

### ECE did not tell me whether confidence identified failures

ECE tells me whether confidence is calibrated overall, but it does not tell me whether the model gives lower confidence to its mistakes than to its correct predictions.

To look at that directly, I added failure-detection AUROC using `1 - confidence` as the failure score.

A value of 1 means the model gives every incorrect prediction a higher failure score than every correct one. A value of 0.5 means confidence is no better than chance at separating correct from incorrect predictions.

### The trust signal needed separate channels

The final trust warning was combining two different things: performance getting worse and confidence behaviour getting worse.

I split these into:

* `performance_signal`, based on absolute accuracy drop and relative error increase
* `confidence_signal`, based on ECE increase, confidence-gap deterioration and fixed-HCER increase
* `trust_signal`, which takes the stronger warning from the two

The thresholds stayed exactly the same. The difference is that I can now see whether a warning is being driven by falling performance, worsening confidence behaviour, or both.

### Offline evidence package

Using the saved predictions and trust outputs, I built tables for:

* paired model failures
* class-level failure behaviour
* confidence diagnostics
* trust-rule attribution
* rule ablation
* confidence changes for the same image between adjacent severities

Each one arose from questions that I had based on the existing results, and helped me to unpack the failure behaviour that the summary metrics could not show on their own.

## 2026-08-28 - Main GTSRBCNN versus ResNet18 findings

### Severe Gaussian noise

At noise severity 5, the two models had almost identical accuracy but very different confidence behaviour.

GTSRBCNN:

* accuracy: 0.2833
* mean confidence: 0.7506
* confidence-accuracy gap: 0.4673
* fixed HCER: 0.1742
* fixed high-confidence coverage: 0.3708
* conditional error above 0.90 confidence: 0.4698
* failure-detection AUROC: 0.7958
* accuracy in the top-ranked 10% by confidence: 0.8614

ResNet18:

* accuracy: 0.2801
* mean confidence: 0.5739
* confidence-accuracy gap: 0.2937
* fixed HCER: 0.0257
* fixed high-confidence coverage: 0.2054
* conditional error above 0.90 confidence: 0.1249
* failure-detection AUROC: 0.8664
* accuracy in the top-ranked 10% by confidence: 0.9889

The models were therefore basically tied on severe-noise accuracy, but ResNet18 confidence remained much more useful for distinguishing correct from failed predictions.

### Paired image outcomes

The similar accuracy also hid very different image-level outcomes:

* both correct: 1,935
* GTSRBCNN only correct: 1,643
* ResNet18 only correct: 1,603
* both wrong: 7,449
* same wrong class among those shared failures: 38

Only about 0.51% of the images both models got wrong were assigned the same wrong class.

So similar aggregate accuracy definitely did not mean they were failing in the same way.

### Trust-rule attribution

The overall GTSRB warnings were completely performance-led in the main two-model analysis.

`performance_signal` matched the final `trust_signal` in every condition.

The ablation made this even clearer:

* GTSRBCNN's labels could be reproduced by absolute accuracy drop alone
* ResNet18's labels could be reproduced by relative error increase alone

Removing ECE, confidence-gap or fixed-HCER rules did not change the final labels.

The confidence diagnostics were still useful because they explained how the models were failing, but they were not deciding the final traffic-light level.

That was not where I originally expected the project to land, but it is a useful result in itself.

### Image-level confidence transitions

Condition averages still left one question open.

A falling mean confidence could hide predictions that were already wrong becoming more confident as degradation worsened.

I paired the same image between adjacent severity levels.

Across the main two-model analysis, overall mean confidence fell in all 30 adjacent transitions. Predictions moving from correct to wrong also lost confidence on average in every transition.

Persistent errors behaved differently.

In 17 of the 30 transitions, more than half of the images wrong at both severities became more confident.

In 18 transitions, their mean confidence increased.

The clearest example was ResNet18 from noise severity 4 to 5:

* images wrong at both severities: 7,435
* proportion becoming more confident: about 72.7%
* mean confidence change: about +0.0627

So at the condition level the model looked slightly more cautious, while many of the predictions it was already getting wrong were actually becoming more confident.

This is one of the findings that most directly answers the title of the project.

### Decision

This is really where I stopped thinking of the overall trust label as having to stay the main proof of “Confident and Wrong”.

The trust signal is useful for showing when deterioration becomes serious. The deeper confidence diagnostics are where I can see what is actually happening underneath that warning.

## 2026-09-01 - Building the final evidence figures

### What I worked on

I turned the deeper analysis into a small set of final figures.

Each figure had one job:

* show when warnings appear and which channel drives them
* show how similar severe-noise accuracy can hide very different failure behaviour
* show HCER beside coverage and conditional error
* show confidence changes for persistent errors and newly failing predictions

The confidence-transition figure was split into persistent errors and new failures because they tell two different stories.

The severe-noise figure was also simplified so the main comparison is easy to see quickly.

### Decision

The final reporting code works from the saved evidence rather than rerunning inference.

At this point I froze the main GTSRBCNN and ResNet18 analysis and reopened it only for two controlled additions: bootstrap uncertainty around the strongest findings and the existing MobileNetV2 checkpoint as a confirmatory architecture.

## 2026-09-02 - Paired bootstrap uncertainty

### What I worked on

I added a small bootstrap analysis around three of the findings I care most about, so I could see how stable those results were across the saved test images.

I used 5,000 paired percentile-bootstrap resamples with seed 42 and a 95% confidence level.

The bootstrap covered:

* the GTSRBCNN versus ResNet18 accuracy difference at Gaussian noise severity 5
* the failure-detection AUROC difference at the same condition
* the ResNet18 confidence change from noise severity 4 to 5 among images wrong at both severities

The same sampled image IDs are used for both models, and the severity 4 and 5 observations also stay paired for the transition analysis.

### Results

Accuracy difference, ResNet18 minus GTSRBCNN:

* estimate: -0.0032
* 95% CI: [-0.0120, 0.0059]

The interval crosses zero, so there is no clear severe-noise accuracy advantage for either model.

Failure-detection AUROC difference:

* estimate: +0.0706
* 95% CI: [0.0587, 0.0825]

The whole interval is positive, which gives much stronger support to the ResNet18 confidence-ranking advantage.

ResNet18 persistent-error confidence change:

* estimate: +0.0627
* 95% CI: [0.0593, 0.0660]
* paired persistent errors: 7,435

Again, the interval is narrow and entirely positive.

### Decision

This helped sharpen the wording of the result.

The models are basically indistinguishable on severe-noise accuracy, but ResNet18 is clearly better at ranking correct versus incorrect predictions by confidence.

At the same time, many of its persistent errors become more confident as the noise gets worse.

The bootstrap measures uncertainty across these saved test predictions. It does not measure what would happen if the models were retrained with different seeds.

## 2026-09-03 - MobileNetV2 confirmatory evaluation

### What I worked on

MobileNetV2 had already been trained during the controlled pilot, so I ran the saved checkpoint through the same frozen GTSRB evaluation.

The framework already supported it end to end. The only missing prerequisite was its validation profile.

The point of this run was quite specific: check whether the strongest ResNet18 findings also appeared in another pretrained architecture.

### Validation profile

MobileNetV2 used the same 3,990-image track-aware validation split and fingerprint.

Its validation profile was:

* accuracy: 0.9822
* balanced accuracy: 0.9629
* ECE: 0.0046
* adaptive HCER threshold: 1.0
* adaptive-threshold coverage: about 43.7%
* rank-based HCER: 0.0 at 10% coverage
* track overlap: 0

The adaptive threshold saturated again, this time covering an even larger part of the validation set.

### Full evaluation

The evaluation produced 202,080 predictions across the same 12,630-image official test set and 16 conditions.

MobileNetV2 had the highest clean test accuracy at 0.9656, although ResNet18 had the higher clean balanced accuracy, 0.9480 versus 0.9359.

At Gaussian noise severity 5, MobileNetV2 reached:

* accuracy: 0.1450
* balanced accuracy: 0.1188
* failure-detection AUROC: 0.7723
* accuracy in the top-ranked 10% by confidence: 0.5867

For comparison, GTSRBCNN and ResNet18 were both around 0.28 accuracy at this condition.

So MobileNetV2 had the strongest clean standard accuracy, and yet the weakest severe-noise performance of the three models.

### Persistent errors

For MobileNetV2 from noise severity 4 to 5:

* images wrong at both severities: 9,833
* proportion becoming more confident: 50.94%
* mean confidence change: -0.0038

This did not reproduce the ResNet18 result.

ResNet18 had about 72.7% of persistent errors becoming more confident with a mean increase of +0.0627.

GTSRBCNN also had a negative mean change, about -0.0254.

So the very strong persistent-error confidence increase looks model-dependent rather than something I can generalise across pretrained architectures.

### Warning timing

MobileNetV2 first warned at:

* blur: `caution` at severity 3 and `do_not_trust` at severity 4
* noise: `do_not_trust` from severity 1
* low light: `trust` through severity 4 and `do_not_trust` at severity 5

Noise was the interesting exception to the earlier performance-led pattern because both the performance and confidence channels reached `do_not_trust` immediately at severity 1.

### Decision

GTSRBCNN versus ResNet18 remains the main comparison.

MobileNetV2 stays as a smaller confirmatory architecture, because it tells me which findings seem broader and which are more model-specific.

The important wider result is that strong clean performance does not predict robustness under degradation.

The ResNet18 persistent-error confidence increase, on the other hand, clearly should not be presented as typical of pretrained models generally.

## Final evidence freeze

The final evidence supports seven main findings.

1. Degradation type matters. Blur, noise and low light produce different reliability patterns.

2. Clean performance is not enough to predict degradation robustness. MobileNetV2 had the highest clean test accuracy but the weakest severe-noise result.

3. Similar degraded accuracy can hide very different confidence behaviour. GTSRBCNN and ResNet18 were effectively tied on severe-noise accuracy, while ResNet18 ranked failures much better.

4. Overall confidence can hide image-level behaviour. It may fall even while many already-wrong predictions become more confident.

5. HCER needs coverage context. Falling fixed HCER can partly reflect fewer predictions crossing the threshold even while the remaining high-confidence group becomes less reliable.

6. The overall trust labels were performance-led in the primary comparison. The confidence diagnostics were more useful for explaining how the models failed than for deciding the final warning level.

7. Some reliability behaviour is strongly model-dependent. MobileNetV2 did not reproduce the strongest ResNet18 persistent-error pattern.

The experimental evidence is now frozen.

## Repository evidence

The repository keeps the most relevant and important evidence needed to inspect the work: configurations, validation profiles, split metadata, metrics, calibration bins, trust outputs, comparison tables, bootstrap results, training logs, sanity-check images and final figures.

Large checkpoints and prediction-level CSVs are kept outside normal Git tracking because of their size, but are preserved locally.

The final automated test suite contains `197` tests.