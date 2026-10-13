"""
tests/check_inference.py -- does inference.py reproduce the notebook?

Put test-split images in examples/ with the true label at the start of the
filename: tumor_1.jpg, healthy_2.png, ...

For each image this prints two things:

  1. LABEL: did the predicted label match the filename?
  2. FEATURES: we find the closest row in reference/X_test_fused.npy (the
     features the notebook saved for the whole test set). If our pipeline
     matches the notebook, the closest row should be almost exactly the same
     vector (tiny distance), far closer than the second-closest row. That is
     the check that catches version or preprocessing drift, because it does
     not depend on the classifier being right.

Run from the project root:   uv run python tests/check_inference.py
"""

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import inference  # noqa: E402  (needs ROOT on the path first)

X = np.load(os.path.join(ROOT, "reference", "X_test_fused.npy"))
y = np.load(os.path.join(ROOT, "reference", "y_test.npy"))

files = []
for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    files += glob.glob(os.path.join(ROOT, "examples", pattern))
files = sorted(files)
if not files:
    sys.exit("No images found in examples/")

label_ok = 0
feature_ok = 0

for path in files:
    name = os.path.basename(path)
    true = 1 if name.lower().startswith("tumor") else 0

    feats = inference.extract_features(path)
    result = inference.predict(path)

    dist = np.linalg.norm(X - feats[None, :], axis=1)
    nearest = int(np.argmin(dist))
    best = float(dist[nearest])
    second = float(np.partition(dist, 1)[1])
    matched = best < 0.01 * second  # the true match should be vastly closer

    got_label = int(result["is_tumor"])
    label_ok += got_label == true
    feature_ok += matched

    print(f"{name}")
    print(f"  LABEL    true={'tumor' if true else 'healthy'}  "
          f"predicted={'tumor' if got_label else 'healthy'}  "
          f"p_tumor={result['p_tumor']:.4f}  "
          f"{'OK' if got_label == true else 'WRONG'}")
    print(f"  FEATURES nearest row={nearest}  dist={best:.4g}  "
          f"next-closest={second:.4g}  "
          f"{'MATCH' if matched else 'NO MATCH'}  "
          f"(notebook label for that row: {int(y[nearest])})")

print()
print(f"Labels correct : {label_ok}/{len(files)}")
print(f"Features match : {feature_ok}/{len(files)}")