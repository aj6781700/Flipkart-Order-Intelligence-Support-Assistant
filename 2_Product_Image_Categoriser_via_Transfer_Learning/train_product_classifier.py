"""
Part 2 -- Product image categoriser via transfer learning (Fashion-MNIST).

Backbone : EfficientNet-B0 pretrained on ImageNet (one of the two backbones the
           brief allows). The ImageNet checkpoint is fetched from the author's
           GitHub release, because torchvision's own weight host was not
           reachable from the build environment.
Input    : grayscale -> 3 channels, resized to 224x224 (EfficientNet-B0's
           native size), normalised with ImageNet mean/std.
Stage 1  : backbone frozen, features cached once, only the new 10-way head is
           trained on the cached 1280-d vectors (Adam).
Stage 2  : only if stage-1 validation accuracy < 80%: unfreeze the late blocks
           and continue at a lower learning rate.
Test     : the 10,000-image test split is used exactly once, at the very end.

Usage:  python3 train_product_classifier.py
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.classifier_inference import (CLASSES, IMG_SIZE, PREPROCESS, build_model)
from src.fashion_data import ensure_fashion_mnist

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(os.cpu_count() or 1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE, "| threads:", torch.get_num_threads())

WEIGHTS_URL = "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth"
WEIGHTS_PATH = Path("pretrained_cache/efficientnet-b0-355c32eb.pth")
FEATURE_DIM = 1280
CACHE_DIR = Path("data/feature_cache")
MODEL_OUT = "models/product_classifier.pt"
Path("models").mkdir(exist_ok=True)
Path("docs").mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Hyper-parameters (documented in docs/PART2_ANALYSIS.md)
EXTRACT_BATCH = 100
HEAD_OPT, HEAD_LR, HEAD_EPOCHS, HEAD_BATCH = "Adam", 1e-3, 20, 256
FT_OPT, FT_LR, FT_EPOCHS, FT_BATCH = "Adam", 1e-4, 3, 64
FT_TRAINABLE_PREFIXES = ("_blocks.14", "_blocks.15", "_conv_head", "_bn1")

# ------------------------------------------------------------------ 1. data
root = ensure_fashion_mnist("data/fashion_mnist_raw")
train_full = datasets.FashionMNIST(root=root, train=True, download=False, transform=PREPROCESS)
test_full = datasets.FashionMNIST(root=root, train=False, download=False, transform=PREPROCESS)
train_idx, val_idx = train_test_split(
    np.arange(len(train_full)), test_size=10000,
    stratify=train_full.targets.numpy(), random_state=SEED)
split_sizes = {"train": len(train_idx), "validation": len(val_idx), "test": len(test_full)}
print("Split sizes:", split_sizes)
assert split_sizes["validation"] >= 5000

# --------------------------------------------------------------- 2. backbone
if not WEIGHTS_PATH.exists():
    WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    print("Downloading ImageNet-pretrained EfficientNet-B0 checkpoint ...")
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
pretrained_state = torch.load(WEIGHTS_PATH, map_location="cpu")

model = build_model()                                   # new 10-class head
missing, unexpected = model.load_state_dict(
    {k: v for k, v in pretrained_state.items() if not k.startswith("_fc.")}, strict=False)
assert set(missing) == {"_fc.weight", "_fc.bias"} and not unexpected
for p in model.parameters():
    p.requires_grad = False                              # freeze everything
model.to(DEVICE).eval()


def pooled_features(m, x):
    return F.adaptive_avg_pool2d(m.extract_features(x), 1).flatten(1)


SHARD = 2500          # features are saved shard by shard so a restart resumes instead of starting over


def cached_features(name, dataset, indices=None):
    f = CACHE_DIR / f"{name}.pt"
    if f.exists():
        d = torch.load(f)
        print(f"  [{name}] loaded cached features {tuple(d['x'].shape)}")
        return d["x"], d["y"]
    all_idx = list(indices) if indices is not None else list(range(len(dataset)))
    xs, ys, t0, fresh = [], [], time.time(), 0
    for k, start in enumerate(range(0, len(all_idx), SHARD)):
        shard_f = CACHE_DIR / f"{name}_shard{k:03d}.pt"
        if shard_f.exists():
            d = torch.load(shard_f)
            xs.append(d["x"]); ys.append(d["y"])
            continue
        loader = DataLoader(Subset(dataset, all_idx[start:start + SHARD]),
                            batch_size=EXTRACT_BATCH, shuffle=False, num_workers=0)
        sx, sy = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                sx.append(pooled_features(model, imgs.to(DEVICE)).cpu())
                sy.append(lbls)
                fresh += len(lbls)
        sx, sy = torch.cat(sx), torch.cat(sy)
        torch.save({"x": sx, "y": sy}, shard_f)
        xs.append(sx); ys.append(sy)
        print(f"  [{name}] {min(start + SHARD, len(all_idx))}/{len(all_idx)} images  "
              f"({fresh / (time.time() - t0):.1f} img/s this run)", flush=True)
    x, y = torch.cat(xs), torch.cat(ys)
    torch.save({"x": x, "y": y}, f)
    for shard_f in CACHE_DIR.glob(f"{name}_shard*.pt"):
        shard_f.unlink()
    return x, y


print("\nStage 1: single pass of the frozen backbone over every image (features cached)")
t0 = time.time()
train_x, train_y = cached_features("train", train_full, train_idx)
val_x, val_y = cached_features("val", train_full, val_idx)
test_x, test_y = cached_features("test", test_full)
print(f"Feature extraction finished in {(time.time() - t0) / 60:.1f} min")

# ------------------------------------------------------ 3. train the new head
head = nn.Linear(FEATURE_DIM, len(CLASSES)).to(DEVICE)
opt = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
loader = DataLoader(TensorDataset(train_x, train_y), batch_size=HEAD_BATCH, shuffle=True)


def head_val_acc():
    head.eval()
    with torch.no_grad():
        return accuracy_score(val_y, head(val_x.to(DEVICE)).argmax(1).cpu())


best_val, best_state, head_history = 0.0, None, []
for ep in range(1, HEAD_EPOCHS + 1):
    head.train()
    for xb, yb in loader:
        opt.zero_grad()
        F.cross_entropy(head(xb.to(DEVICE)), yb.to(DEVICE)).backward()
        opt.step()
    acc = head_val_acc()
    head_history.append(round(acc, 4))
    if acc > best_val:
        best_val, best_state = acc, {k: v.clone() for k, v in head.state_dict().items()}
    print(f"[head] epoch {ep:2d}/{HEAD_EPOCHS}  val_acc={acc:.4f}")
head.load_state_dict(best_state)
feature_extraction_val_acc = best_val
print(f"\nFeature-extraction-only validation accuracy (best epoch): {feature_extraction_val_acc:.4f}")

# --------------------------------------------- 4. fine-tune only if needed
fine_tuning = {"required": bool(feature_extraction_val_acc < 0.80),
               "val_acc_before": round(feature_extraction_val_acc, 4), "val_acc_after": None}
model._fc.load_state_dict(head.state_dict())

if fine_tuning["required"]:
    print("\nValidation accuracy < 80%  ->  stage 2: unfreezing late blocks at a lower LR")
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(FT_TRAINABLE_PREFIXES) or name.startswith("_fc")
    params = [p for p in model.parameters() if p.requires_grad]
    ft_opt = torch.optim.Adam(params, lr=FT_LR)
    tr = DataLoader(Subset(train_full, train_idx), batch_size=FT_BATCH, shuffle=True, num_workers=0)
    va = DataLoader(Subset(train_full, val_idx), batch_size=FT_BATCH, shuffle=False, num_workers=0)
    best_ft = 0.0
    for ep in range(1, FT_EPOCHS + 1):
        model.train()
        for imgs, lbls in tr:
            ft_opt.zero_grad()
            F.cross_entropy(model(imgs.to(DEVICE)), lbls.to(DEVICE)).backward()
            ft_opt.step()
        model.eval()
        with torch.no_grad():
            correct = sum((model(i.to(DEVICE)).argmax(1).cpu() == l).sum().item() for i, l in va)
        acc = correct / len(val_idx)
        best_ft = max(best_ft, acc)
        print(f"[fine-tune] epoch {ep}/{FT_EPOCHS}  val_acc={acc:.4f}", flush=True)
    fine_tuning["val_acc_after"] = round(best_ft, 4)

# ------------------------------------------------ 5. final evaluation (test)
model.eval()
if fine_tuning["required"]:
    preds = []
    with torch.no_grad():
        for imgs, _ in DataLoader(test_full, batch_size=100, shuffle=False):
            preds.append(model(imgs.to(DEVICE)).argmax(1).cpu())
    test_pred = torch.cat(preds).numpy()
else:
    with torch.no_grad():
        test_pred = head(test_x.to(DEVICE)).argmax(1).cpu().numpy()
test_true = test_y.numpy()

test_acc = accuracy_score(test_true, test_pred)
cm = confusion_matrix(test_true, test_pred)
prec, rec, f1, sup = precision_recall_fscore_support(test_true, test_pred, labels=range(10), zero_division=0)
print(f"\n=== FINAL TEST ACCURACY: {test_acc:.4f} ===")
print("\nConfusion matrix (rows = true class, columns = predicted class):")
print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
print("\nPer-class precision / recall / F1:")
for i, c in enumerate(CLASSES):
    print(f"  {c:12s} precision={prec[i]:.3f}  recall={rec[i]:.3f}  f1={f1[i]:.3f}  support={sup[i]}")

off = cm.copy()
np.fill_diagonal(off, 0)
pairs = sorted(((CLASSES[i], CLASSES[j], int(off[i, j])) for i in range(10) for j in range(10) if i != j),
               key=lambda t: -t[2])[:6]
print("\nMost confused (true -> predicted) pairs, read off the matrix:")
for t, p, n in pairs:
    print(f"  {t} -> {p}: {n}")

# ------------------------------------------------------------- 6. save
torch.save(model.state_dict(), MODEL_OUT)
pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv("docs/part2_confusion_matrix.csv")
report = {
    "dataset": "Fashion-MNIST (zalandoresearch/fashion-mnist)",
    "split_sizes": split_sizes,
    "backbone": "EfficientNet-B0, ImageNet-pretrained (lukemelas/EfficientNet-PyTorch checkpoint)",
    "input_size": IMG_SIZE,
    "preprocessing": "grayscale->3ch, resize 224x224, ImageNet mean/std normalisation",
    "head_training": {"optimizer": HEAD_OPT, "learning_rate": HEAD_LR, "epochs": HEAD_EPOCHS,
                      "batch_size": HEAD_BATCH, "val_acc_per_epoch": head_history,
                      "feature_extraction_batch_size": EXTRACT_BATCH},
    "feature_extraction_val_accuracy": round(feature_extraction_val_acc, 4),
    "fine_tuning": fine_tuning,
    "fine_tuning_settings": {"optimizer": FT_OPT, "learning_rate": FT_LR, "epochs": FT_EPOCHS,
                             "batch_size": FT_BATCH, "unfrozen": list(FT_TRAINABLE_PREFIXES)},
    "test_accuracy": round(float(test_acc), 4),
    "confusion_matrix": cm.tolist(),
    "per_class": [{"class": CLASSES[i], "precision": round(float(prec[i]), 4),
                   "recall": round(float(rec[i]), 4), "f1": round(float(f1[i]), 4),
                   "support": int(sup[i])} for i in range(10)],
    "top_confused_pairs": [{"true": t, "predicted": p, "count": n} for t, p, n in pairs],
}
with open("docs/part2_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved {MODEL_OUT}, docs/part2_report.json, docs/part2_confusion_matrix.csv")
