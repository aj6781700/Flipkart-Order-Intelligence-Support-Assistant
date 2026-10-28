"""
Part 2, Task 8 -- write real Fashion-MNIST *test-split* images out as actual
.png files. torchvision stores the data as raw IDX binaries, so there is no
ready-made image file for Part 3's classify_product_image(image_path) tool
to point at; this script creates them.

One image per class (10 files), the first test-split example of each class,
named so the true label is obvious, e.g. data/sample_images/07_sneaker.png.
"""
import os
import sys
from pathlib import Path

from PIL import Image
from torchvision import datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.fashion_data import ensure_fashion_mnist

OUT_DIR = Path("data/sample_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)
NAMES = ["tshirt_top", "trouser", "pullover", "dress", "coat",
         "sandal", "shirt", "sneaker", "bag", "ankle_boot"]

root = ensure_fashion_mnist("data/fashion_mnist_raw")
test = datasets.FashionMNIST(root=root, train=False, download=False)

seen = {}
for idx in range(len(test)):
    label = int(test.targets[idx])
    if label in seen:
        continue
    arr = test.data[idx].numpy()                    # 28x28 uint8
    fname = f"{label:02d}_{NAMES[label]}.png"
    Image.fromarray(arr, mode="L").save(OUT_DIR / fname)
    seen[label] = (idx, fname)
    if len(seen) == 10:
        break

print(f"Exported {len(seen)} real test-split images to {OUT_DIR}/")
for label in sorted(seen):
    idx, fname = seen[label]
    print(f"  test_split_index={idx:5d} -> {fname}")
