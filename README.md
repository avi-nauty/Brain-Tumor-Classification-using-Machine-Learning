---
title: Brain Tumor MRI Classifier
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Research demo that flags possible tumors in brain MRI slices
---

# Brain Tumor MRI Classifier

A research demo that classifies a brain MRI slice as **tumor** or **healthy**.
Upload an image or pick an example, and the app returns a verdict with the
model's tumor probability.

> **Research demo only. Not a medical device and not for diagnosis.**
> Results must not be used to make medical decisions.

## How it works

1. The image is resized to 224×224 (aspect ratio preserved, black padding).
2. Three pretrained CNNs (ResNet50, VGG16, EfficientNetB0) each produce a feature
   vector, and the three are concatenated into 3,840 numbers.
3. The features are standardized, and Black Hole Optimization (BHO) selects a
   subset of 1,938 of them.
4. The selected features are standardized again and passed to a logistic
   classifier trained with Opt-SGD, which outputs a tumor probability.
5. An image is flagged as a tumor when that probability is **0.30 or higher**.

## Evaluation

Measured on a held-out test set of 394 images (140 healthy, 254 tumor) that was
not used for training or feature selection.

| | Predicted healthy | Predicted tumor |
|---|---|---|
| **Actually healthy** | 137 | 3 |
| **Actually tumor** | 8 | 246 |

| Metric | Value |
|---|---|
| Accuracy | 97.2% |
| Recall on tumor images (sensitivity) | 96.9% |
| Recall on healthy images (specificity) | 97.9% |
| Precision (tumor) | 98.8% |
| F1 (tumor) | 97.8% |

### Why a threshold of 0.30

The decision threshold was set to 0.30 instead of the usual 0.50 before testing,
to favor catching tumors over avoiding false alarms. In a screening-style setting
a missed tumor is a costlier mistake than an extra flag. The trade-off is visible
in the table: 3 healthy scans were flagged, and 8 tumors were still missed.

### Failure cases

The model made 11 errors on the test set: 8 missed tumors (false negatives) and
3 healthy scans flagged as tumors (false positives). Examples of these will be
added here.

## Limitations

- Trained and tested on a single dataset, so performance on scans from other
  scanners, hospitals or protocols is unknown.
- Binary only: it does not identify tumor type, location or size.
- It classifies single 2D slices, not full 3D scans.
- It has not been clinically validated.

## Implementation note

Pixel values are scaled to the 0 to 1 range before each backbone's own
`preprocess_input`, which was designed for 0 to 255 input. This was done
consistently at training and test time, and the saved scalers, feature
selection and classifier all depend on features produced this way, so the demo
keeps it exactly.

## Run locally

```
uv sync
uv run python app.py
```

Requires Python 3.12. The pinned versions (TensorFlow 2.20.0, Keras 3.13.2,
NumPy 2.1.3) match the environment used for training.