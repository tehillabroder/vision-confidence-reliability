# Project Brief: Confident and Wrong

**Research Question:** At what point does model confidence stop aligning with actual performance under degraded image conditions, and can this be detected using simple reliability signals?

**Core Scope:**
* **Datasets:** MNIST (baseline) and GTSRB (case study).
* **Models:** Simple CNN, ResNet18, MobileNetV2.
* **Degradations:** Gaussian blur, Gaussian noise, low light (Severity: clean + 1 to 5).
* **Metrics:** Accuracy, balanced accuracy (GTSRB), ECE, confidence-accuracy gap, HCER.
* **Outputs:** CSV logs, calibration bins, trust_signal.json.

**Main Rule:** This is not a project to train the best classifier. It is a modular framework for detecting when confidence becomes unreliable under controlled degradation.