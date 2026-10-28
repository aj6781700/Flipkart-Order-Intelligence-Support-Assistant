"""
Task 2 -- embed every chunk with all-MiniLM-L6-v2 and build a Faiss index.

    python3 build_index.py

Writes policy_kb/faiss_index.bin and policy_kb/chunk_metadata.pkl (both are
small and committed, so the agent runs without rebuilding).
"""
import pickle
from pathlib import Path

import faiss

from embedder import embed
from policy_kb.documents import DOCUMENTS, build_chunks

KB_DIR = Path(__file__).resolve().parent / "policy_kb"
INDEX_PATH = KB_DIR / "faiss_index.bin"
METADATA_PATH = KB_DIR / "chunk_metadata.pkl"


def main():
    chunks = build_chunks()
    vectors = embed([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])   # normalised vectors: inner product == cosine
    index.add(vectors)
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Documents: {len(DOCUMENTS)} | Chunks: {len(chunks)} | Embedding dim: {vectors.shape[1]}")
    print(f"Faiss index ({index.ntotal} vectors) -> {INDEX_PATH.name}")
    print(f"Chunk -> parent-document metadata   -> {METADATA_PATH.name}")


if __name__ == "__main__":
    main()
