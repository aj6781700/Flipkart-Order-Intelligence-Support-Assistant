"""
Part 2, Task 7 -- One-function loading + single-image prediction.

This is the exact code Part 3's `classify_product_image` tool calls. It loads
models/product_classifier.pt (produced by train_product_classifier.py) and
classifies a single image file path.

Usage:
    from src.classifier_inference import classify_product_image
    result = classify_product_image("data/sample_images/07_sneaker.png")
    print(result)
    # {'predicted_class': 'Sneaker', 'confidence': 0.94, 'all_probabilities': {...}}
"""
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

_MODEL_CACHE = {}  # {weights_path: (backbone, head, classes, transform)}


def _build_and_load(weights_path: str, device: str = "cpu"):
    checkpoint = torch.load(weights_path, map_location=device)

    backbone = resnet18(weights=None)
    backbone.fc = nn.Identity()
    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    backbone.eval().to(device)

    head = nn.Linear(512, 10)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval().to(device)

    classes = checkpoint["classes"]
    img_size = checkpoint["img_size"]
    mean = checkpoint["imagenet_mean"]
    std = checkpoint["imagenet_std"]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return backbone, head, classes, transform


def classify_product_image(image_path: str,
                            weights_path: str = "models/product_classifier.pt",
                            device: str = "cpu") -> dict:
    """
    Classify a single product image file into one of the 10 Fashion-MNIST /
    Flipkart apparel-footwear-accessory categories.

    Args:
        image_path: path to a real image file (e.g. one of the .png files
                    in data/sample_images/, or any similar product photo).
        weights_path: path to the saved checkpoint from Task 7/9.
        device: "cpu" or "cuda".

    Returns:
        {
          "predicted_class": str,
          "confidence": float,           # 0-1
          "all_probabilities": {class_name: float, ...}
        }
    """
    if weights_path not in _MODEL_CACHE:
        _MODEL_CACHE[weights_path] = _build_and_load(weights_path, device)
    backbone, head, classes, transform = _MODEL_CACHE[weights_path]

    img = Image.open(image_path).convert("L")  # ensure single-channel in;
    x = transform(img).unsqueeze(0).to(device)  # transform replicates to 3ch

    with torch.no_grad():
        logits = head(backbone(x))
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax())

    return {
        "predicted_class": classes[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "all_probabilities": {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))},
    }


if __name__ == "__main__":
    import sys
    import glob

    paths = sys.argv[1:] or sorted(glob.glob("data/sample_images/*.png"))
    for p in paths:
        print(p, "->", classify_product_image(p))
