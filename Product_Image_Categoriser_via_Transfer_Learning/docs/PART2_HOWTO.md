# Part 2 — How to Run (and why it isn't run yet in this repo)

## Important: this code has not been executed yet

Part 2's code was written to the full spec — dataset splits, ImageNet
preprocessing, ResNet-18 transfer learning with the feature-caching speed
trick, conditional fine-tuning, real confusion-matrix evaluation, artifact
saving, and real sample-image export — but it has **not actually been run**,
because it was written in a sandboxed environment with no internet access and
no PyTorch installed (Fashion-MNIST can't be downloaded there, and
`pip install torch` fails outright). Per the brief's own acceptance criteria
("never a fabricated number"), no numbers are reported here until the code
has actually produced them.

**You need to run this yourself** in an environment with internet access.
Google Colab's free tier (which your brief explicitly names as an acceptable
option) is the easiest path — it has both internet and a free GPU, and needs
zero paid account.

## Steps to run on Google Colab (recommended)

1. Go to https://colab.research.google.com, start a new notebook.
2. Runtime → Change runtime type → select **T4 GPU** (free tier).
3. Upload `train_product_classifier.py` and `export_sample_images.py` (or
   just paste their contents into cells).
4. Run:
   ```python
   !pip install torch torchvision scikit-learn pillow numpy pandas -q
   !python3 train_product_classifier.py
   !python3 export_sample_images.py
   ```
5. Download the resulting `models/product_classifier.pt`,
   `docs/part2_report.json`, `docs/part2_confusion_matrix.csv`, and the 10
   PNGs in `data/sample_images/` back into this repo, in the same folder
   structure.

## Steps to run locally (CPU is fine, just slower)

```bash
pip install -r requirements-part2.txt
python3 train_product_classifier.py     # downloads Fashion-MNIST automatically
python3 export_sample_images.py
```

The feature-extraction phase (frozen backbone, single pass to cache 512-d
features, then train only a small linear head on those cached vectors) is
the whole point of the caching trick in the brief — this keeps CPU runtime to
well under an hour even without a GPU. The optional fine-tuning phase (only
triggered if feature-extraction validation accuracy comes in under 80%) is
slower on CPU since the backbone becomes trainable again, but only touches
`layer4`, not the whole network.

## What to send back

Once you've run it, paste me:
- The full console output (it prints every epoch's val accuracy, the final
  test accuracy, the confusion matrix, and the top confused pairs)
- Or just the contents of `docs/part2_report.json`

I'll then write the required Task 6 analysis (the two-paragraph explanation
of why the top confused category pairs are visually plausible) against your
**actual** confusion matrix, and update this repo's README and
`docs/PART1_ANALYSIS.md`-style write-up with real, verified numbers — the
same way Part 1 was done.

## What's already fully spec-compliant, independent of execution

- `train_product_classifier.py` uses **Fashion-MNIST from the pinned source**
  (`torchvision.datasets.FashionMNIST`, which pulls from the canonical
  `zalandoresearch/fashion-mnist` release) — no substitute dataset.
- Splits are exactly: **50,000 train / 10,000 validation / 10,000 test**
  (validation is a stratified carve-out from the 60k training set, well above
  the required 5,000 minimum; the test split is never touched until final
  evaluation).
- Preprocessing matches spec exactly: grayscale→3-channel replication,
  resize to **224×224** (ResNet-18's expected input), normalized with
  ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`.
- Feature-extraction-then-head-training uses the caching trick described in
  the brief, with documented settings: **Adam, lr=1e-3, batch_size=256,
  20 epochs** for the head; the frozen-backbone feature pass uses
  batch_size=128.
- Fine-tuning (if triggered) unfreezes only `layer4` (a late block, early/mid
  layers stay frozen), uses a lower **lr=1e-4**, batch_size=64, 8 epochs —
  standard gradual-unfreezing practice.
- Confusion-pair identification is fully automated and reads directly off the
  real confusion matrix (`train_product_classifier.py`'s
  `top_confused_pairs` logic) — nothing is guessed or hand-picked.
- `export_sample_images.py` writes real, individual `.png` files (via
  `PIL.Image.save`, not the raw IDX binary) for one real test-split image per
  class, named so the true label is obvious
  (e.g. `07_sneaker.png`) — 10 files, above the required minimum of 5.
- `src/classifier_inference.py`'s `classify_product_image(image_path)`
  function is the exact, documented one-function load+predict snippet Part 3
  will call — it's already written against the checkpoint format
  `train_product_classifier.py` saves.
