"""Top-k cosine-similarity retrieval over the Faiss index built by build_index.py."""
import pickle

import faiss

from build_index import INDEX_PATH, METADATA_PATH
from embedder import embed

_index = None
_chunks = None


def _load():
    global _index, _chunks
    if _index is None:
        _index = faiss.read_index(str(INDEX_PATH))
        with open(METADATA_PATH, "rb") as f:
            _chunks = pickle.load(f)


def retrieve(query: str, k: int = 3) -> list:
    """Return the k most similar chunks, each with its cosine `score` and parent `doc_id`."""
    _load()
    scores, idxs = _index.search(embed([query]), k)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i != -1:
            chunk = dict(_chunks[i])
            chunk["score"] = round(float(score), 4)
            results.append(chunk)
    return results


if __name__ == "__main__":
    for r in retrieve("How many days do I have to return a shirt?"):
        print(f"{r['score']:.3f} [{r['chunk_id']}] {r['text']}")
