"""
Part 3, Task 10 -- Retrieval evaluation.

Computes Precision@3 and Recall@3 for each query in the Task 1 answer key,
at the DOCUMENT level: each of the top-3 retrieved chunks is mapped back to
its parent doc_id and deduplicated before scoring against the answer key's
relevant-document set, per the acceptance criteria.

Usage:
    python3 evaluate_retrieval.py   # after build_index.py has been run
"""
from policy_kb.documents import RETRIEVAL_ANSWER_KEY
from graph import retrieve_chunks


def evaluate():
    per_query_results = []

    for item in RETRIEVAL_ANSWER_KEY:
        query = item["query"]
        relevant_docs = set(item["relevant_doc_ids"])

        retrieved = retrieve_chunks(query, k=3)
        # map each retrieved chunk back to its parent doc, dedup at doc level
        retrieved_doc_ids = []
        for c in retrieved:
            if c["doc_id"] not in retrieved_doc_ids:
                retrieved_doc_ids.append(c["doc_id"])

        hits = [d for d in retrieved_doc_ids if d in relevant_docs]
        precision_at_3 = len(hits) / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0
        recall_at_3 = len(hits) / len(relevant_docs) if relevant_docs else 0.0

        per_query_results.append({
            "query": query,
            "relevant_doc_ids": sorted(relevant_docs),
            "retrieved_doc_ids": retrieved_doc_ids,
            "hits": hits,
            "precision_at_3": round(precision_at_3, 4),
            "recall_at_3": round(recall_at_3, 4),
        })

    avg_precision = sum(r["precision_at_3"] for r in per_query_results) / len(per_query_results)
    avg_recall = sum(r["recall_at_3"] for r in per_query_results) / len(per_query_results)

    print("Per-query retrieval evaluation (document-level, k=3):\n")
    for r in per_query_results:
        print(f"Query: {r['query']}")
        print(f"  Relevant docs (answer key): {r['relevant_doc_ids']}")
        print(f"  Retrieved docs (top-3, deduped): {r['retrieved_doc_ids']}")
        print(f"  Hits: {r['hits']}")
        print(f"  Precision@3 = {len(r['hits'])}/{len(r['retrieved_doc_ids'])} = {r['precision_at_3']}")
        print(f"  Recall@3    = {len(r['hits'])}/{len(r['relevant_doc_ids'])} = {r['recall_at_3']}")
        print()

    print(f"Average Precision@3 across {len(per_query_results)} queries: {round(avg_precision, 4)}")
    print(f"Average Recall@3 across {len(per_query_results)} queries:    {round(avg_recall, 4)}")

    return {"per_query": per_query_results, "avg_precision_at_3": round(avg_precision, 4),
            "avg_recall_at_3": round(avg_recall, 4)}


if __name__ == "__main__":
    evaluate()
