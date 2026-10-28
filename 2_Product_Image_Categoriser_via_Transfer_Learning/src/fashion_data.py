"""
Fetches Fashion-MNIST (Zalando Research) from the pinned source
https://github.com/zalandoresearch/fashion-mnist into the layout that
torchvision.datasets.FashionMNIST expects, so the dataset is the canonical
one regardless of which mirror torchvision can reach.
"""
import gzip
import os
import shutil
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def ensure_fashion_mnist(root: str = "data/fashion_mnist_raw") -> str:
    raw_dir = os.path.join(root, "FashionMNIST", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    for name in FILES:
        dest = os.path.join(raw_dir, name)
        if not os.path.exists(dest):
            print(f"Downloading {name} from the pinned zalandoresearch/fashion-mnist repo ...")
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", dest)
        out = dest[:-3]          # torchvision reads the decompressed idx files
        if not os.path.exists(out):
            with gzip.open(dest, "rb") as fi, open(out, "wb") as fo:
                shutil.copyfileobj(fi, fo)
    return root
