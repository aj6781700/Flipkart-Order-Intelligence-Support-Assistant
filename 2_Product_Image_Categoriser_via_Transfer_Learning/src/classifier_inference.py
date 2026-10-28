"""
Loading + single-image prediction for the Part 2 product classifier.

This is the exact code Part 3's `classify_product_image` tool calls.

    from src.classifier_inference import classify_product_image
    classify_product_image("data/sample_images/07_sneaker.png")
    # {'predicted_class': 'Sneaker', 'confidence': 0.99, 'all_probabilities': {...}}

The saved file models/product_classifier.pt is a plain
`model.state_dict()` of an EfficientNet-B0 (ImageNet-pretrained backbone, new
10-way head). Rebuilding the architecture with `build_model()` and calling
`load_state_dict` restores it.
"""
import os

import torch
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "models", "product_classifier.pt")

PREPROCESS = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # 1 channel -> 3 channels
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_CACHE = {}


def build_model() -> EfficientNet:
    """EfficientNet-B0 with a fresh 10-class classifier head (no download)."""
    return EfficientNet.from_name("efficientnet-b0", num_classes=len(CLASSES))


def load_classifier(weights_path: str = DEFAULT_WEIGHTS) -> EfficientNet:
    if weights_path not in _CACHE:
        model = build_model()
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        _CACHE[weights_path] = model
    return _CACHE[weights_path]


def classify_product_image(image_path: str, weights_path: str = DEFAULT_WEIGHTS) -> dict:
    """Predict the catalog category of one product image file (.png/.jpg)."""
    model = load_classifier(weights_path)
    x = PREPROCESS(Image.open(image_path).convert("L")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    return {
        "predicted_class": CLASSES[idx],
        "confidence": round(float(probs[idx]), 4),
        "all_probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
    }


if __name__ == "__main__":
    import glob
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(root, "data", "sample_images", "*.png")))
    for p in paths:
        r = classify_product_image(p)
        print(f"{os.path.basename(p):28s} -> {r['predicted_class']:12s} ({r['confidence']:.3f})")
