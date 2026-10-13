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

DISCLAIMER = (
    "**Research demo only. Not a medical device and not for diagnosis.** "
    "Results should never be used to make medical decisions."
)

METHOD = """
Each image is passed through three pretrained CNNs (ResNet50, VGG16, EfficientNetB0).
Their features are fused into one vector of 3,840 numbers, standardized, and reduced
to 1,938 features chosen by Black Hole Optimization. A logistic classifier trained
with Opt-SGD then outputs a tumor probability.

The decision threshold is **0.30**, not the usual 0.50. It is set low on purpose so the
model errs toward flagging a possible tumor rather than missing one, which means some
healthy scans will be flagged as well.
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
            image = gr.Image(type="filepath", label="MRI image")
            button = gr.Button("Classify", variant="primary")
        with gr.Column():
            result = gr.Markdown()

    if example_files:
        gr.Examples(examples=example_files, inputs=image, label="Try an example")

    with gr.Accordion("How it works", open=False):
        gr.Markdown(METHOD)

    button.click(classify, inputs=image, outputs=result)

if __name__ == "__main__":
    demo.launch()