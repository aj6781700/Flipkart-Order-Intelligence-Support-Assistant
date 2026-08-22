"""
Part 3, Task 4 -- classify_product_image tool.

Loads Part 2's ACTUAL saved checkpoint (models/product_classifier.pt) and
runs it against real .png files already committed to
2_Product_Image_Categoriser_via_Transfer_Learning/data/sample_images/
(Part 2 Task 8). No new image upload is required -- this reads files
already in the repo.

This module simply re-exposes Part 2's own classify_product_image function
(defined in Part 2's src/classifier_inference.py) so there is exactly one
implementation of the model-loading/inference logic -- Part 3 does not
duplicate or re-derive it.
"""
import sys
from pathlib import Path

PART2_DIR = Path(__file__).resolve().parent.parent.parent / "2_Product_Image_Categoriser_via_Transfer_Learning"
PART2_MODEL_PATH = PART2_DIR / "models" / "product_classifier.pt"

sys.path.insert(0, str(PART2_DIR))

try:
    from src.classifier_inference import classify_product_image as _classify_impl
except ImportError as e:
    _classify_impl = None
    _import_error = e


def classify_product_image(image_path: str) -> dict:
    """
    Args:
        image_path: path to a real .png file, e.g.
            "../2_Product_Image_Categoriser_via_Transfer_Learning/data/sample_images/07_sneaker.png"

    Returns:
        {"predicted_class": str, "confidence": float, "all_probabilities": {...}}
    """
    if _classify_impl is None:
        raise ImportError(
            f"Could not import Part 2's classify_product_image: {_import_error}. "
            f"Make sure 2_Product_Image_Categoriser_via_Transfer_Learning/src/ exists."
        )
    if not PART2_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Part 2 model not found at {PART2_MODEL_PATH}. Run "
            f"2_Product_Image_Categoriser_via_Transfer_Learning/train_product_classifier.py first."
        )
    return _classify_impl(image_path, weights_path=str(PART2_MODEL_PATH))


if __name__ == "__main__":
    import glob
    sample_dir = PART2_DIR / "data" / "sample_images"
    paths = sorted(glob.glob(str(sample_dir / "*.png")))
    if not paths:
        print(f"No sample images found at {sample_dir}. "
              f"Run export_sample_images.py in Part 2 first.")
    for p in paths:
        print(p, "->", classify_product_image(p))
