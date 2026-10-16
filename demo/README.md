# Brain Tumor MRI Classifier (Demo)

A research demo that classifies a brain MRI slice as **tumor** or **healthy** and shows the model's tumor probability.

**Live demo:** https://brain-tumor-demo-293791125431.northamerica-northeast2.run.app
(Hosted on Google Cloud Run. It scales to zero when idle, so the first load after a quiet period can take around 30 seconds.)

> **Research demo only. Not a medical device and not for diagnosis.**
> Results must not be used to make medical decisions.

This repository contains the inference code, the web app and the container setup. The model was trained and evaluated in a separate notebook; this repo packages the saved result and serves it.

## How it works

1. The image is resized to 224x224 (aspect ratio preserved, black padding).
2. Three pretrained CNNs (ResNet50, VGG16, EfficientNetB0) each produce a feature vector. The three are concatenated into 3,840 numbers.
3. The features are standardized, and Black Hole Optimization (BHO) selects a subset of 1,938 of them.
4. The selected features are standardized again and passed to a logistic classifier trained with Opt-SGD, which outputs a tumor probability.
5. An image is flagged as a tumor when that probability is **0.30 or higher**.

### Why a threshold of 0.30

The threshold was set to 0.30 instead of the usual 0.50 before testing, to favor catching tumors over avoiding false alarms. In a screening-style setting a missed tumor is a costlier mistake than an extra flag. The trade-off is visible in the results below.

## Results

Measured on a held-out test set of 394 images (140 healthy, 254 tumor) that was not used for training or feature selection.

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

## Limitations

- Trained and tested on a single dataset, so performance on scans from other scanners, hospitals or protocols is unknown.
- Binary only: it does not identify tumor type, location or size.
- It classifies single 2D slices, not full 3D scans.
- It has not been clinically validated.

## Project layout

| Path | Purpose |
|---|---|
| `inference.py` | The model pipeline: image in, label and tumor probability out |
| `app.py` | Gradio web interface around `inference.predict()` |
| `artifacts/` | Seven small `.npy` files (scalers, BHO mask, classifier weights) |
| `examples/` | Sample images for one-click testing in the UI |
| `tests/check_inference.py` | Check that `inference.py` reproduces the notebook's saved test features |
| `reference/` | Test-set features used by the check (`X_test_fused.npy`, `y_test.npy`) |
| `Dockerfile`, `.dockerignore` | Container build |
| `requirements.txt` | Pinned dependencies used by the container |

## Run locally

Requires Python 3.12. The pinned versions (TensorFlow 2.20.0, Keras 3.13.2, NumPy 2.1.3) match the training environment.

```
uv sync
uv run python app.py
```

Then open the local URL Gradio prints. The first start downloads about 170 MB of ImageNet weights.

## Run with Docker

```
docker build -t brain-tumor-demo .
docker run --rm -p 8080:8080 brain-tumor-demo
```

Open http://localhost:8080. The image downloads the ImageNet weights at build time, so containers start without a download, and the build fails if an artifact file has the wrong shape.

## Check the pipeline against the notebook

```
uv run python tests/check_inference.py
```

For each image in `examples/`, this prints whether the predicted label matches the filename and whether the extracted features match the notebook's saved test features. The feature match catches version or preprocessing drift independently of whether the classifier is right.

## Deploy to Google Cloud Run

```
gcloud run deploy brain-tumor-demo \
  --source . \
  --region northamerica-northeast2 \
  --memory 2Gi --cpu 2 \
  --max-instances 1 --timeout 60 \
  --allow-unauthenticated
```

The build runs remotely on Google Cloud Build using the Dockerfile. `--max-instances 1` caps cost, since the endpoint is public.

## Implementation note

Pixel values are scaled to the 0 to 1 range before each backbone's own `preprocess_input`, which was designed for 0 to 255 input. This was done consistently at training and test time, and the saved scalers, feature selection and classifier all depend on features produced this way, so the demo keeps it exactly. Do not "fix" it.
