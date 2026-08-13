"""
Part 2, Task 8 -- Export real Fashion-MNIST test-split images as actual .png
files (not raw IDX data), for Part 3's classify_product_image(image_path)
tool to point at.

Run AFTER train_product_classifier.py has downloaded the dataset (or run
standalone -- torchvision will download Fashion-MNIST if not already cached).

Usage:
    python3 export_sample_images.py
"""
from pathlib import Path
from torchvision import datasets
from PIL import Image

DATA_ROOT = "data/fashion_mnist_raw"
OUT_DIR = Path("data/sample_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["tshirt_top", "trouser", "pullover", "dress", "coat",
           "sandal", "shirt", "sneaker", "bag", "ankle_boot"]

# no transform here -- we want the RAW original image, exactly as a real
# uploaded product photo would arrive; preprocessing happens inside the
# classify_product_image tool at inference time, not at export time.
test_raw = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True)

# pick the first test-split example of EACH of the 10 classes (all 10,
# comfortably above the "at least 5" requirement)
seen = set()
exported = []
for idx in range(len(test_raw)):
    img, label = test_raw[idx]  # img is already a PIL.Image (mode 'L', 28x28)
    if label in seen:
        continue
    seen.add(label)
    fname = f"{label:02d}_{CLASSES[label]}.png"
    img.save(OUT_DIR / fname)
    exported.append((idx, fname))
    if len(seen) == 10:
        break

print(f"Exported {len(exported)} real test-split images to {OUT_DIR}/:")
for idx, fname in exported:
    print(f"  test_split_index={idx:5d}  ->  {fname}")
