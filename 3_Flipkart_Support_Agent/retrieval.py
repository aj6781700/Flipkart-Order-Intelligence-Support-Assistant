"""
Part 3 -- retrieval helper used by the RAG node. Loads the Faiss index +
chunk metadata built by build_index.py and performs top-k cosine-similarity
search for a query.
"""
import pickle

import faiss
from sentence_transformers import SentenceTransformer

from build_index import EMBED_MODEL_NAME, INDEX_PATH, METADATA_PATH

_model = None
_index = None
_chunks = None


def _lazy_load():
    global _model, _index, _chunks
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
    if _chunks is None:
        with open(METADATA_PATH, "rb") as f:
            _chunks = pickle.load(f)


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Returns the top_k most similar chunks to `query`, each with a
    cosine-similarity `score` field (higher = more similar, range -1..1)."""
    _lazy_load()
    q_emb = _model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, idxs = _index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        chunk = dict(_chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)
    return results


if __name__ == "__main__":
    for r in retrieve("How long can I return a shirt?", top_k=3):
        print(f"{r['score']:.3f}  [{r['doc_id']}]  {r['text']}")
