# Part 2 — Product image classifier: analysis

Numbers come from `python3 train_product_classifier.py` (log: `train.log` is git-ignored; structured results: `docs/part2_report.json`, `docs/part2_confusion_matrix.csv`).

## Setup
- Data: Fashion-MNIST (60,000 train + 10,000 test images, 10 classes), fetched from the original zalandoresearch repository.
- Split: stratified **50,000 train / 10,000 validation / 10,000 test**. The test split is evaluated exactly once, after all training decisions.
- Backbone: **EfficientNet-B0**, ImageNet-pretrained (lukemelas/EfficientNet-PyTorch checkpoint), fully frozen; only a new `Linear(1280, 10)` head is trained.
- Preprocessing: grayscale→3 channels, resize 224×224, ImageNet mean/std.
- Head training: Adam, lr 1e-3, batch 256, 20 epochs; best-validation epoch kept.

Backbone note: ResNet-18 from the torchvision weights server was not reachable from the build environment, so EfficientNet-B0 (same ImageNet pre-training idea, weights from GitHub releases) was used instead.

## Was fine-tuning needed?
Rule set up front: fine-tune the last blocks only if validation accuracy from feature extraction fell below 80%. It reached **92.16%**, so **fine-tuning was not needed and was not run** (`val_acc_before = 0.9216`, no "after" value).

## Result
**Test accuracy: 91.38%** (10,000 images).

Confusion matrix (rows = true, columns = predicted):

| | T-shirt | Trouser | Pullover | Dress | Coat | Sandal | Shirt | Sneaker | Bag | Ankle boot |
|---|---|---|---|---|---|---|---|---|---|---|
| T-shirt/top | 871 | 1 | 19 | 21 | 2 | 0 | 81 | 0 | 5 | 0 |
| Trouser | 2 | 985 | 1 | 10 | 0 | 0 | 0 | 0 | 2 | 0 |
| Pullover | 18 | 1 | 869 | 7 | 46 | 0 | 58 | 0 | 1 | 0 |
| Dress | 17 | 5 | 17 | 890 | 25 | 0 | 45 | 0 | 1 | 0 |
| Coat | 1 | 1 | 38 | 21 | 881 | 0 | 58 | 0 | 0 | 0 |
| Sandal | 0 | 0 | 0 | 0 | 0 | 975 | 0 | 17 | 0 | 8 |
| Shirt | 109 | 0 | 50 | 24 | 66 | 0 | 745 | 0 | 6 | 0 |
| Sneaker | 0 | 0 | 0 | 0 | 0 | 11 | 0 | 973 | 0 | 16 |
| Bag | 2 | 1 | 0 | 1 | 4 | 0 | 1 | 0 | 991 | 0 |
| Ankle boot | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 37 | 0 | 958 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| T-shirt/top | 0.854 | 0.871 | 0.862 |
| Trouser | 0.991 | 0.985 | 0.988 |
| Pullover | 0.874 | 0.869 | 0.872 |
| Dress | 0.914 | 0.890 | 0.902 |
| Coat | 0.860 | 0.881 | 0.871 |
| Sandal | 0.984 | 0.975 | 0.979 |
| Shirt | 0.754 | 0.745 | 0.749 |
| Sneaker | 0.947 | 0.973 | 0.960 |
| Bag | 0.985 | 0.991 | 0.988 |
| Ankle boot | 0.976 | 0.958 | 0.967 |

## Confusion pairs (read from the matrix)
**Shirt ↔ T-shirt/top (109 shirts called T-shirts, 81 T-shirts called shirts).** This is the largest error pair. At 28×28 greyscale the two differ mainly by collar, buttons and sleeve length, which are a few pixels wide, so the silhouettes look nearly identical. Shirt is the weakest class (F1 0.749) because it sits between T-shirt, Pullover and Coat.

**Sneaker ↔ Ankle boot (37 ankle boots called sneakers, 16 sneakers called ankle boots) with Sandal ↔ Sneaker (17 / 11).** All are footwear viewed from the side; the cue that separates them is shaft height or open toes, which blurs when the shoe is low-top or the image is upscaled to 224 px. Footwear confusions are fewer than the upper-body-garment ones (every footwear class has F1 ≥ 0.96) but still show the model reads shape, not material.

Also notable: Shirt→Coat (66) and Coat→Shirt (58), and Pullover↔Coat (46/38), i.e. long-sleeved upper-body garments blur together.

## Inference
```python
from src.classifier_inference import classify_product_image
classify_product_image("data/sample_images/07_sneaker.png")
# {'predicted_class': 'Sneaker', 'confidence': 0.9927, 'all_probabilities': {...}}
```
`python3 src/classifier_inference.py` classifies all 10 PNGs in `data/sample_images/` (real Fashion-MNIST test images): all ten are predicted correctly.
