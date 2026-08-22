"""
Part 3, Task 2 -- Embed every chunk with a free local sentence-transformer
model and build a Faiss vector index over them.

Run this once before starting the agent (graph.py loads the saved index).

Usage:
    pip install sentence-transformers faiss-cpu
    python3 build_index.py
"""
import pickle
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from policy_kb.documents import build_chunks

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # free, local, no API key
INDEX_PATH = "policy_kb/faiss_index.bin"
METADATA_PATH = "policy_kb/chunk_metadata.pkl"


def main():
    chunks = build_chunks()
    texts = [c["text"] for c in chunks]

    print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"Embedding {len(texts)} chunks ...")
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    # normalized embeddings + inner product == cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    Path("policy_kb").mkdir(exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved Faiss index ({index.ntotal} vectors, dim={dim}) to {INDEX_PATH}")
    print(f"Saved chunk metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
