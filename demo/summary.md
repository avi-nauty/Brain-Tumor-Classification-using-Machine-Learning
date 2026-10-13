# Brain Tumor MRI Classifier Demo: How It Works

## Overview

A web demo that classifies brain MRI images. It has three layers: an inference module that holds the model pipeline, a Gradio UI that wraps it, and a Docker container that packages both for hosting on Google Cloud Run. The demo works locally and has been verified against the training notebook. The Docker build and the Cloud Run deployment are in progress.

## Environment

- Python 3.12, TensorFlow 2.20.0, Keras 3.13.2, NumPy 2.1.3 (matches the Kaggle training environment)
- Gradio for the UI (pin the exact version from `uv pip show gradio` in `requirements.txt`)

## Project layout

- `inference.py`: model pipeline and prediction logic
- `app.py`: Gradio UI
- `requirements.txt`: Python dependencies
- `artifacts/`: seven small `.npy` files (116K) that the pipeline depends on
- `examples/`: sample images for the UI (68K)
- `reference/`: 96M of local-only reference material, never committed to Git LFS, never copied into the image
- `Dockerfile` and `.dockerignore`: containerization

## Inference (`inference.py`)

- On import, it builds the three backbone networks with `weights="imagenet"`. This downloads about 170 MB of headless weights.
- It loads the saved artifacts from `artifacts/` and runs shape asserts, so a bad or mismatched file fails loudly instead of producing wrong predictions.
- It takes an MRI image and returns class probabilities. Decisions are made against a 0.30 threshold.
- Preprocessing quirk: images are scaled to [0,1] and then fed to `preprocess_input`. This looks odd but must stay exactly as is, because every saved artifact was produced with it. Changing it silently breaks the pipeline.

## UI (`app.py`)

- A Gradio interface that accepts an uploaded MRI image, calls into `inference.py`, and displays the probabilities.
- Example images from `examples/` are available for one-click testing.
- Gradio listens on localhost by default, which is unreachable from outside a container. The Dockerfile sets `GRADIO_SERVER_NAME=0.0.0.0` and `GRADIO_SERVER_PORT=8080` as environment variables, so `app.py` needs no changes.

## Containerization

Terms: the Dockerfile is the recipe, the image is the finished frozen result, and a container is a running instance of an image.

What each Dockerfile step does:

- `FROM python:3.12-slim`: small Linux base with Python 3.12.
- `ENV GRADIO_SERVER_NAME` / `GRADIO_SERVER_PORT`: makes Gradio reachable and matches the port Cloud Run sends traffic to.
- `COPY requirements.txt` then `RUN pip install`: installs libraries in their own cached layer, so code edits don't redo the slow TensorFlow install.
- Selective `COPY` lines: only `inference.py`, `app.py`, `artifacts/`, and `examples/` go in. `reference/` stays out.
- `RUN python -c "import inference"`: downloads the ImageNet weights at build time so cold starts are fast, and runs the artifact asserts so a bad file fails the build rather than production.
- `CMD ["python", "app.py"]`: what runs each time a container starts.

`.dockerignore` keeps `reference/`, `.git`, virtual environments, caches, and notebooks out of the build context, so they are never uploaded during a build or deploy.

Local test:

```
docker build -t brain-tumor-demo .
docker run --rm -p 8080:8080 brain-tumor-demo
```

Open `http://localhost:8080`, classify the example images, and compare the probabilities with the non-Docker run. Use `docker stats` while it runs to read memory use, which sets the Cloud Run memory limit (likely 2 to 4 GiB).

## Hosting

- Hugging Face Spaces was ruled out: Gradio and Docker Spaces need a paid plan, and a screen recording or a temporary `share=True` link would not demonstrate deployment skills.
- Chosen host: Google Cloud Run. It runs any Docker container, gives a public HTTPS URL, and scales to zero, so it costs nothing while idle. It bills by vCPU-seconds and GiB-seconds only while serving.
- Deploy with `gcloud run deploy --source .`, adding `--max-instances 1` as a cost guard against traffic abuse. This builds on Google's machines with the correct architecture, so local Docker is only for testing.
- Before deploying: install `gcloud`, create a Google Cloud project, enable billing, and set a budget alert.
- Finally, link the live Cloud Run URL from the portfolio site's `projects.ts`.

## Things to watch

- Architecture: a local build on Apple Silicon produces an ARM image, while Cloud Run runs Intel/AMD. Treat the local build as a functional test only. Forcing `--platform linux/amd64` locally can be slow or crash under emulation, which is not a sign the setup is wrong.
- Probabilities near the 0.30 threshold could shift slightly across architectures. Compare container output with the local run, and re-check after the Cloud Run deploy.
- Memory: three TensorFlow backbones need real memory, so choose the limit from the `docker stats` reading.

## Later options (not needed for the first deploy)

- GitHub Actions for automated deploys (repo under `avi-nauty`, using Workload Identity Federation rather than service account keys)
- Versioning and monitoring, which are the pieces that turn a deployment into an MLOps story
- License: deferred, leaning toward none until the paper is accepted