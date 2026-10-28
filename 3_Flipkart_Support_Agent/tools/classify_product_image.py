"""
Task 4 -- classify_product_image(image_path) -> dict

Loads Part 2's REAL saved classifier (2_Product_Image_Categoriser_via_Transfer_
Learning/models/product_classifier.pt) through Part 2's own documented
loading snippet (src/classifier_inference.py) and runs it on a real .png file
already committed in that folder's data/sample_images/. Returns the predicted
category label and the model's confidence in it.
"""
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PART2_DIR = REPO_ROOT / "2_Product_Image_Categoriser_via_Transfer_Learning"
WEIGHTS_PATH = PART2_DIR / "models" / "product_classifier.pt"
SAMPLE_DIR = PART2_DIR / "data" / "sample_images"

_impl = None


def _load_impl():
    """Import Part 2's inference module by file path (avoids clashing with this repo's `tools` package)."""
    global _impl
    if _impl is None:
        spec = importlib.util.spec_from_file_location(
            "part2_classifier_inference", PART2_DIR / "src" / "classifier_inference.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _impl = mod
    return _impl


def resolve_image_path(image_path: str) -> Path:
    p = Path(image_path)
    for candidate in (p, REPO_ROOT / p, SAMPLE_DIR / p.name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"image not found: {image_path}")


def classify_product_image(image_path: str) -> dict:
    """Returns {"predicted_class", "confidence", "all_probabilities", "image_file"}."""
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Part 2 model not found at {WEIGHTS_PATH}")
    path = resolve_image_path(image_path)
    result = _load_impl().classify_product_image(str(path), weights_path=str(WEIGHTS_PATH))
    result["image_file"] = path.name
    return result


if __name__ == "__main__":
    for f in sorted(SAMPLE_DIR.glob("*.png")):
        r = classify_product_image(str(f))
        print(f"{f.name:24s} -> {r['predicted_class']:12s} {r['confidence']:.3f}")
