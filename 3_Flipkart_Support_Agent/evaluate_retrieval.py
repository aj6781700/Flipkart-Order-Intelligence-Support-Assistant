"""
Task 2 -- retrieval evaluation at DOCUMENT level.

For each query: take the top-3 chunks, map every chunk to its parent document
(chunk -> doc_id), de-duplicate (a document counts once however many of its
chunks were retrieved), then

    Precision@3 = relevant docs retrieved / unique docs retrieved
    Recall@3    = relevant docs retrieved / relevant docs in the answer key

    python3 evaluate_retrieval.py
"""
import json
from pathlib import Path

from policy_kb.documents import RETRIEVAL_ANSWER_KEY
from retrieval import retrieve

lines, ps, rs, rows = [], [], [], []
for e in RETRIEVAL_ANSWER_KEY:
    chunks = retrieve(e["query"], k=3)
    docs = list(dict.fromkeys(c["doc_id"] for c in chunks))          # unique, rank order
    rel = set(e["relevant_doc_ids"])
    hits = [d for d in docs if d in rel]
    p, r = len(hits) / len(docs), len(hits) / len(rel)
    ps.append(p); rs.append(r)
    lines += [f"Q: {e['query']}",
              f"   top-3 chunks : {[(c['chunk_id'], c['score']) for c in chunks]}",
              f"   unique docs  : {docs} | relevant (answer key): {sorted(rel)} | hits: {hits}",
              f"   Precision@3 = {len(hits)}/{len(docs)} = {p:.3f}   Recall@3 = {len(hits)}/{len(rel)} = {r:.3f}", ""]
    rows.append({"query": e["query"], "unique_docs": docs, "relevant": sorted(rel), "hits": hits,
                 "precision_at_3": round(p, 4), "recall_at_3": round(r, 4)})
n = len(ps)
lines.append(f"MEAN over {n} queries: Precision@3 = {sum(ps)/n:.3f}   Recall@3 = {sum(rs)/n:.3f}")
out = "\n".join(lines)
print(out)
Path("docs/retrieval_eval.txt").write_text(out + "\n")
Path("docs/retrieval_eval.json").write_text(json.dumps(
    {"queries": rows, "mean_precision_at_3": round(sum(ps)/n, 4), "mean_recall_at_3": round(sum(rs)/n, 4)}, indent=2))
