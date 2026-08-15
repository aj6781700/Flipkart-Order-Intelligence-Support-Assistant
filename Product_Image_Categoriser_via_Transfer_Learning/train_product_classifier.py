"""
Part 2 -- Product Image Categoriser via Transfer Learning

Run this in an environment with internet access (Google Colab free tier is
ideal -- it has both internet and a free GPU, and is explicitly suggested by
the brief). It will NOT run in a fully offline sandbox, because Fashion-MNIST
must be downloaded and PyTorch/torchvision must be installed.

Usage:
    pip install torch torchvision scikit-learn pillow numpy
    python3 train_product_classifier.py

Outputs:
    models/product_classifier.pt       -- trained head (+ optionally fine-tuned
                                           backbone layer4) state_dict
    docs/part2_report.json             -- all metrics, confusion matrix, params
    docs/part2_confusion_matrix.csv
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

IMG_SIZE = 224  # ResNet-18's expected input size
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATA_ROOT = "data/fashion_mnist_raw"
MODEL_OUT = "models/product_classifier.pt"
REPORT_OUT = "docs/part2_report.json"

Path("models").mkdir(exist_ok=True)
Path("docs").mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Task 1: Load Fashion-MNIST, carve a stratified validation split
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # 1 -> 3 channels
    transforms.Resize((IMG_SIZE, IMG_SIZE)),        # match ResNet-18 input
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_full = datasets.FashionMNIST(root=DATA_ROOT, train=True, download=True,
                                    transform=transform)
test_full = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True,
                                   transform=transform)

train_targets = train_full.targets.numpy()
train_idx, val_idx = train_test_split(
    np.arange(len(train_full)), test_size=10000,
    stratify=train_targets, random_state=SEED,
)
print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_full)}")
assert len(val_idx) >= 5000, "validation split must be at least 5,000 images"

# ---------------------------------------------------------------------------
# Task 3: Transfer-learning backbone (feature extraction, cached)
# ---------------------------------------------------------------------------
weights = ResNet18_Weights.IMAGENET1K_V1
backbone = resnet18(weights=weights)
backbone.fc = nn.Identity()          # strip the ImageNet 1000-way head -> 512-d features
for p in backbone.parameters():
    p.requires_grad = False          # freeze everything for the feature-extraction pass
backbone.eval().to(DEVICE)

BATCH_SIZE = 128

def extract_features(dataset, indices=None, batch_size=BATCH_SIZE):
    """Run the frozen backbone once over the given images and cache its
    512-d output feature vectors + labels. Mathematically identical to
    recomputing the frozen backbone's forward pass every epoch, but turns
    the expensive part into a single pass."""
    ds = Subset(dataset, indices) if indices is not None else dataset
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            out = backbone(imgs)
            feats.append(out.cpu())
            labels.append(lbls)
    return torch.cat(feats), torch.cat(labels)

print("Extracting cached features (single pass over frozen backbone)...")
t0 = time.time()
train_feats, train_labels = extract_features(train_full, train_idx)
val_feats, val_labels = extract_features(train_full, val_idx)
test_feats, test_labels = extract_features(test_full)
print(f"Feature extraction done in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# Task 3 (cont.): train ONLY the new classifier head on cached features
# ---------------------------------------------------------------------------
HEAD_LR = 1e-3
HEAD_EPOCHS = 20
HEAD_BATCH_SIZE = 256
OPTIMIZER_NAME = "Adam"

head = nn.Linear(512, 10).to(DEVICE)
optimizer = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(TensorDataset(train_feats, train_labels),
                           batch_size=HEAD_BATCH_SIZE, shuffle=True)

def evaluate_head(feats, labels):
    head.eval()
    with torch.no_grad():
        logits = head(feats.to(DEVICE))
        preds = logits.argmax(1).cpu()
    return accuracy_score(labels, preds)

best_val_acc = 0.0
for epoch in range(HEAD_EPOCHS):
    head.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(head(xb), yb)
        loss.backward()
        optimizer.step()
    val_acc = evaluate_head(val_feats, val_labels)
    best_val_acc = max(best_val_acc, val_acc)
    print(f"[Feature extraction] epoch {epoch+1}/{HEAD_EPOCHS} val_acc={val_acc:.4f}")

feature_extraction_val_acc = best_val_acc
print(f"\nFeature-extraction-only best val accuracy: {feature_extraction_val_acc:.4f}")

# ---------------------------------------------------------------------------
# Task 4: Fine-tune late layers IF feature-extraction val accuracy < 80%
# ---------------------------------------------------------------------------
fine_tuned = False
fine_tune_val_acc = None

if feature_extraction_val_acc < 0.80:
    print("\nFeature-extraction accuracy below 80% -- fine-tuning late layers (layer4)...")
    fine_tuned = True

    # rebuild a single trainable model: backbone (layer4 unfrozen) + head
    ft_backbone = resnet18(weights=weights)
    ft_backbone.fc = nn.Identity()
    for name, p in ft_backbone.named_parameters():
        p.requires_grad = name.startswith("layer4")  # only unfreeze the late block
    ft_backbone.to(DEVICE)

    ft_head = nn.Linear(512, 10).to(DEVICE)
    ft_head.load_state_dict(head.state_dict())  # warm-start from feature-extraction head

    FT_LR = 1e-4  # lower LR for fine-tuning, standard practice
    FT_EPOCHS = 8
    FT_BATCH_SIZE = 64  # smaller, backbone is now trainable -> more memory

    ft_params = [p for p in ft_backbone.parameters() if p.requires_grad] + list(ft_head.parameters())
    ft_optimizer = torch.optim.Adam(ft_params, lr=FT_LR)

    train_loader_raw = DataLoader(Subset(train_full, train_idx), batch_size=FT_BATCH_SIZE,
                                   shuffle=True, num_workers=2)
    val_loader_raw = DataLoader(Subset(train_full, val_idx), batch_size=FT_BATCH_SIZE,
                                 shuffle=False, num_workers=2)

    def evaluate_full_model(loader):
        ft_backbone.eval(); ft_head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                logits = ft_head(ft_backbone(imgs))
                correct += (logits.argmax(1) == lbls).sum().item()
                total += lbls.size(0)
        return correct / total

    best_ft_val_acc = 0.0
    for epoch in range(FT_EPOCHS):
        ft_backbone.train(); ft_head.train()
        for imgs, lbls in train_loader_raw:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            ft_optimizer.zero_grad()
            loss = criterion(ft_head(ft_backbone(imgs)), lbls)
            loss.backward()
            ft_optimizer.step()
        val_acc = evaluate_full_model(val_loader_raw)
        best_ft_val_acc = max(best_ft_val_acc, val_acc)
        print(f"[Fine-tune] epoch {epoch+1}/{FT_EPOCHS} val_acc={val_acc:.4f}")

    fine_tune_val_acc = best_ft_val_acc
    print(f"\nFine-tuned best val accuracy: {fine_tune_val_acc:.4f}")

    # swap in the fine-tuned backbone/head for final evaluation + saving
    backbone_final, head_final = ft_backbone, ft_head
    final_uses_finetune = True
else:
    print("\nFeature-extraction accuracy already >= 80% -- fine-tuning not required.")
    backbone_final, head_final = backbone, head
    final_uses_finetune = False

# ---------------------------------------------------------------------------
# Task 5: Final evaluation on the held-out test split (touched only now)
# ---------------------------------------------------------------------------
if final_uses_finetune:
    test_loader_raw = DataLoader(test_full, batch_size=64, shuffle=False, num_workers=2)
    backbone_final.eval(); head_final.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader_raw:
            imgs = imgs.to(DEVICE)
            logits = head_final(backbone_final(imgs))
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(lbls)
    test_preds = torch.cat(all_preds).numpy()
    test_labels_np = torch.cat(all_labels).numpy()
else:
    with torch.no_grad():
        logits = head_final(test_feats.to(DEVICE))
        test_preds = logits.argmax(1).cpu().numpy()
    test_labels_np = test_labels.numpy()

test_accuracy = accuracy_score(test_labels_np, test_preds)
cm = confusion_matrix(test_labels_np, test_preds)
precision, recall, f1, support = precision_recall_fscore_support(
    test_labels_np, test_preds, labels=range(10), zero_division=0
)

print(f"\n=== FINAL TEST ACCURACY: {test_accuracy:.4f} ===")
print("\nConfusion matrix (rows=true, cols=predicted):")
print(cm)
print("\nPer-class precision/recall:")
for i, c in enumerate(CLASSES):
    print(f"  {c:15s} precision={precision[i]:.3f} recall={recall[i]:.3f} support={support[i]}")

# ---------------------------------------------------------------------------
# Task 6: Auto-identify the most-confused category pairs from the real matrix
# ---------------------------------------------------------------------------
cm_off_diag = cm.copy()
np.fill_diagonal(cm_off_diag, 0)
pairs = []
for i in range(10):
    for j in range(10):
        if i != j:
            pairs.append(((CLASSES[i], CLASSES[j]), cm_off_diag[i, j]))
pairs.sort(key=lambda x: -x[1])
top_confused_pairs = pairs[:5]
print("\nTop confused (true -> predicted) pairs:")
for (true_c, pred_c), count in top_confused_pairs:
    print(f"  {true_c} -> {pred_c}: {count}")

# ---------------------------------------------------------------------------
# Task 7 / 9: Save the artifact (head, plus fine-tuned layer4 if applicable)
# ---------------------------------------------------------------------------
save_dict = {
    "head_state_dict": head_final.state_dict(),
    "backbone_state_dict": backbone_final.state_dict(),
    "fine_tuned": final_uses_finetune,
    "classes": CLASSES,
    "img_size": IMG_SIZE,
    "imagenet_mean": IMAGENET_MEAN,
    "imagenet_std": IMAGENET_STD,
}
torch.save(save_dict, MODEL_OUT)
print(f"\nSaved model to {MODEL_OUT}")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
report = {
    "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_full)},
    "backbone": "resnet18 (ImageNet1K_V1 pretrained)",
    "img_size": IMG_SIZE,
    "head_training": {
        "optimizer": OPTIMIZER_NAME, "lr": HEAD_LR,
        "epochs": HEAD_EPOCHS, "batch_size": HEAD_BATCH_SIZE,
    },
    "feature_extraction_val_accuracy": round(feature_extraction_val_acc, 4),
    "fine_tuning_required": fine_tuned,
    "fine_tuning_val_accuracy": round(fine_tune_val_acc, 4) if fine_tune_val_acc else None,
    "test_accuracy": round(float(test_accuracy), 4),
    "confusion_matrix": cm.tolist(),
    "per_class": [
        {"class": CLASSES[i], "precision": round(float(precision[i]), 4),
         "recall": round(float(recall[i]), 4), "support": int(support[i])}
        for i in range(10)
    ],
    "top_confused_pairs": [
        {"true": tc, "predicted": pc, "count": int(cnt)}
        for (tc, pc), cnt in top_confused_pairs
    ],
}
with open(REPORT_OUT, "w") as f:
    json.dump(report, f, indent=2)

import pandas as pd
pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv("docs/part2_confusion_matrix.csv")

print(f"\nSaved report to {REPORT_OUT}")
print("\nDone. Paste the printed output back to have the written analysis completed.")
