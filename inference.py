"""
inference.py -- one image path in, label + tumor probability out.

The chain below mirrors the notebook exactly. Each step feeds the next, so the
order matters:

  1. decode + letterbox resize  -> (224, 224, 3), pixel values in [0, 1]
  2. three CNN backbones        -> 2048 + 512 + 1280 = 3840 numbers
  3. scaler 1                   -> standardize with the TRAIN mean / scale
  4. cast to float32            -> the notebook stored these features as float32
  5. BHO mask                   -> keep only the selected columns
  6. scaler 2                   -> second standardization, on the selected columns
  7. Opt-SGD logistic           -> sigmoid(x @ w + b); tumor if p >= 0.30

DO NOT "fix" the preprocessing. Images stay in [0, 1] when they reach the
backbones' preprocess_input functions (which were designed for [0, 255]).
That is how the training and test features were built, so every later
artifact (scalers, BHO mask, classifier) expects features made this way.
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre

# Relative to THIS file, not to wherever Python was launched from, so the same
# code works on a laptop and on a Hugging Face Space.
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

IMG_SIZE = 224
THRESHOLD = 0.30
EPS = 1e-8  # the notebook added this to the second scaler's std


# ---------------------------------------------------------------------------
# Step 1: image -> (224, 224, 3) array in [0, 1]   (copied from the notebook)
# ---------------------------------------------------------------------------
def _decode_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    return tf.image.convert_image_dtype(img, tf.float32)  # uint8 -> [0, 1]


def _letterbox_resize(img, target=IMG_SIZE):
    """Resize keeping aspect ratio, then pad with black to a square."""
    shape = tf.shape(img)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)

    scale = tf.minimum(target / h, target / w)
    new_h = tf.cast(tf.round(h * scale), tf.int32)
    new_w = tf.cast(tf.round(w * scale), tf.int32)

    img = tf.image.resize(img, (new_h, new_w), method="bilinear")

    pad_h = target - new_h
    pad_w = target - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    img = tf.pad(
        img,
        paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        constant_values=0.0,
    )
    return tf.ensure_shape(img, [target, target, 3])


# ---------------------------------------------------------------------------
# Step 2: the three backbones (built once, when this file is first imported)
# ---------------------------------------------------------------------------
def _build_extractor(base_cls, name):
    base = base_cls(weights="imagenet", include_top=False,
                    input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    return Model(base.input, x, name=name)


_resnet = _build_extractor(ResNet50, "ResNet50_Extractor")          # -> 2048
_vgg = _build_extractor(VGG16, "VGG16_Extractor")                   # -> 512
_eff = _build_extractor(EfficientNetB0, "EfficientNetB0_Extractor")  # -> 1280


# ---------------------------------------------------------------------------
# Steps 3-7: the seven saved artifacts
# ---------------------------------------------------------------------------
def _load(name):
    return np.load(os.path.join(ARTIFACT_DIR, name))


_s1_mean = _load("scaler1_mean.npy")
_s1_scale = _load("scaler1_scale.npy")
_bho_idx = _load("bho_selected_idx.npy")
_s2_mu = _load("scaler2_mu.npy").astype(np.float32)
# max() keeps the notebook's "+ 1e-8" safety for constant columns whether or
# not the saved std already included it (it changes nothing otherwise).
_s2_sd = np.maximum(_load("scaler2_sd.npy").astype(np.float32), EPS)
_w = _load("opt_sgd_weights.npy")
_b = float(_load("opt_sgd_bias.npy")[0])

# Cheap shape checks: a mixed-up or wrong artifact fails loudly at startup
# instead of silently giving bad predictions.
assert _s1_mean.shape == (3840,) and _s1_scale.shape == (3840,)
assert _bho_idx.max() < 3840
K = _bho_idx.size
assert _s2_mu.shape == (K,) and _s2_sd.shape == (K,) and _w.shape == (K,)


def _sigmoid(z):
    z = np.clip(z, -30, 30)  # same clipping as the notebook
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# The one public function
# ---------------------------------------------------------------------------
def extract_features(image_path):
    """Steps 1-2: image file -> the raw fused feature vector, shape (3840,).

    This is the same thing the notebook saved as X_test_fused.npy, one row per
    image, which is what the check script compares against.
    """
    img = _letterbox_resize(_decode_image(image_path))
    batch = tf.expand_dims(img, 0)  # (1, 224, 224, 3)

    # Each backbone gets its own preprocess_input, applied to [0, 1] images.
    r = _resnet(resnet_pre(batch), training=False)
    v = _vgg(vgg_pre(batch), training=False)
    e = _eff(eff_pre(batch), training=False)
    return tf.concat([r, v, e], axis=1).numpy().astype(np.float32)[0]


def predict(image_path):
    """Return {"label", "is_tumor", "p_tumor", "threshold"} for one image file."""
    fused = extract_features(image_path)[None, :]  # (1, 3840)

    x = (fused - _s1_mean) / _s1_scale   # scaler 1
    x = x.astype(np.float32)             # float32 cast (matters near the threshold)
    x = x[:, _bho_idx]                   # BHO feature selection -> (1, K)
    x = (x - _s2_mu) / _s2_sd            # scaler 2

    # Plain multiply-and-sum in float64, same as the notebook. (np.matmul
    # printed spurious warnings on some Macs, so we avoid it.)
    z = float(np.sum(x[0].astype(np.float64) * _w) + _b)
    p = float(_sigmoid(z))
    is_tumor = p >= THRESHOLD
    return {
        "label": "Tumor detected" if is_tumor else "Healthy (no tumor)",
        "is_tumor": bool(is_tumor),
        "p_tumor": p,
        "threshold": THRESHOLD,
    }


if __name__ == "__main__":
    import sys

    for path in sys.argv[1:]:
        print(path, "->", predict(path))