"""
Loads the free, local sentence-transformer used for chunk embeddings and for
the few-shot intent classifier: all-MiniLM-L6-v2 (384-d, no API key).

Resolution order:
  1. ./policy_kb/_model/all-MiniLM-L6-v2   (a local copy, if already present)
  2. the Hugging Face hub name "all-MiniLM-L6-v2" (normal internet access)
  3. the PyPI package `all-minilm-l6-v2-model`, which bundles the same weights
     (pip install all-minilm-l6-v2-model --no-deps), extracted into (1).
"""
import os
import zipfile
from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
LOCAL_DIR = Path(__file__).resolve().parent / "policy_kb" / "_model" / MODEL_NAME
_model = None


def _extract_from_pypi_package() -> bool:
    try:
        import importlib.util
        spec = importlib.util.find_spec("all_minilm_l6_v2")
        if spec is None:
            return False
        zip_path = Path(spec.submodule_search_locations[0]) / "model.zip"
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(LOCAL_DIR)
        return True
    except Exception:
        return False


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is not None:
        return _model
    if not (LOCAL_DIR / "config.json").exists() and not os.environ.get("MINILM_FORCE_HUB"):
        _extract_from_pypi_package()
    if (LOCAL_DIR / "config.json").exists():
        _model = SentenceTransformer(str(LOCAL_DIR))
    else:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts):
    return get_embedder().encode(list(texts), convert_to_numpy=True,
                                 normalize_embeddings=True).astype("float32")
