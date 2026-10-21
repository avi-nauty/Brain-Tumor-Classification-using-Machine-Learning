"""
app.py -- the web page around inference.predict().

All the ML lives in inference.py. This file only does three things:
take an uploaded image, call predict(), and display the result.
"""

import glob
import os

import gradio as gr

import inference

HERE = os.path.dirname(os.path.abspath(__file__))

MAX_UPLOAD_MB = 10

DISCLAIMER = (
    "**Research demo only. Not a medical device and not for diagnosis.** "
    "Results should never be used to make medical decisions."
)

METHOD = """
This app classifies a brain MRI image as showing a possible tumor or not, using the pipeline from my graduate research.

### How it works
1. **Feature extraction.** The image goes through three pretrained CNNs: ResNet50, VGG16 and EfficientNetB0. Each one summarizes the image as a list of numbers.
2. **Fusion.** The three lists (2,048 + 512 + 1,280 numbers) are joined into one vector of 3,840 features and standardized so all features are on a comparable scale.
3. **Feature selection.** Black Hole Optimization, a metaheuristic search algorithm, keeps 1,938 of those features and drops the rest.
4. **Classification.** A logistic classifier trained with Optimized Stochastic Gradient Descent turns the selected features into a tumor probability.

### Why the threshold is 0.30
Most classifiers flag a case above 0.50. This one flags a possible tumor above 0.30. The threshold is low on purpose, because missing a tumor is costlier than raising a false alarm. The trade-off is that some healthy scans will be flagged as well.

### Using the demo
Upload an MRI image (PNG or JPG, up to 10 MB) or pick one of the examples. Please don't upload images that could identify a real patient.

### Limitations
- This is a research prototype and has not been clinically validated.
- It looks at one 2D image at a time, not a full scan series or patient history.
- It may be less reliable on images unlike its training data, such as other scanners, sequences or image quality.
"""


def classify(image_path):
    if image_path is None:
        return "Upload an MRI image first."
    try:
        r = inference.predict(image_path)
    except Exception as exc:  # unreadable or unsupported file
        return f"Could not process that file ({type(exc).__name__}). Try a PNG or JPG MRI slice."
    return (
        f"## {r['label']}\n\n"
        f"Tumor probability: **{r['p_tumor']:.1%}**  \n"
        f"Decision threshold: {r['threshold']:.0%} "
        f"(flagged as tumor at or above this)"
    )


example_files = sorted(
    glob.glob(os.path.join(HERE, "examples", "*.jpg"))
    + glob.glob(os.path.join(HERE, "examples", "*.jpeg"))
    + glob.glob(os.path.join(HERE, "examples", "*.png"))
)

with gr.Blocks(title="Brain Tumor MRI Classifier") as demo:
    gr.Markdown("# Brain Tumor MRI Classifier")
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column():
            image = gr.Image(type="filepath", label=f"MRI image (PNG or JPG, max {MAX_UPLOAD_MB} MB)")
            button = gr.Button("Classify", variant="primary")
        with gr.Column():
            result = gr.Markdown()

    if example_files:
        gr.Examples(examples=example_files, inputs=image, label="Try an example")

    with gr.Accordion("How it works", open=False):
        gr.Markdown(METHOD)

    button.click(classify, inputs=image, outputs=result)

if __name__ == "__main__":
    demo.launch(max_file_size=f"{MAX_UPLOAD_MB}mb")