"""
Part 3, Task 1 -- Flipkart-style policy knowledge base.

14 short (2-4 sentence) policy documents, covering return windows by
category, COD/prepaid refund timelines, delivery SLAs, reverse-pickup
eligibility, and a few adjacent policies for retrieval-evaluation variety.

Chunking strategy: SENTENCE-WISE. Each document is split into individual
sentences, and each sentence becomes its own chunk. This is chosen over
fixed-size or overlapping-window chunking because policy sentences are
each a self-contained factual claim (a window, a timeline, an eligibility
rule) -- splitting on sentence boundaries keeps each chunk semantically
whole, so a retrieved chunk is directly usable as a quoted policy fact
without truncating mid-clause. Multi-sentence documents below each produce
more than one chunk, as required.
"""
import re

DOCUMENTS = [
    {
        "doc_id": "D01",
        "title": "Apparel & Footwear Return Window",
        "text": (
            "Apparel and footwear items purchased on Flipkart can be returned "
            "within 14 days of delivery. Items must be unused, unwashed, and "
            "returned with original tags and packaging intact. Innerwear and "
            "swimwear are non-returnable for hygiene reasons."
        ),
    },
    {
        "doc_id": "D02",
        "title": "Electronics Return Window",
        "text": (
            "Electronics such as phones, laptops, and cameras have a 10-day "
            "return window from the date of delivery. The device must be free "
            "of physical damage, and all original accessories, box, and "
            "invoice must be included. Software or activation issues should "
            "be reported within 48 hours of delivery to qualify for a "
            "replacement instead of a return."
        ),
    },
    {
        "doc_id": "D03",
        "title": "Home & Furniture Return Window",
        "text": (
            "Home and furniture items, including furnishings and large "
            "appliances, can be returned within 7 days of delivery. Assembled "
            "furniture must be disassembled by the customer before pickup "
            "unless the seller offers assisted return."
        ),
    },
    {
        "doc_id": "D04",
        "title": "Beauty & Personal Care Return Policy",
        "text": (
            "Beauty and personal care products, including cosmetics and "
            "skincare items, are non-returnable once the outer seal is "
            "broken, for hygiene and safety reasons. Unopened, sealed "
            "products can be returned within 7 days of delivery."
        ),
    },
    {
        "doc_id": "D05",
        "title": "COD Refund Timeline",
        "text": (
            "For Cash on Delivery orders, refunds are issued to the "
            "customer's bank account or Flipkart wallet, since no prepaid "
            "instrument exists to reverse. Refunds are typically processed "
            "within 7-10 business days after the returned item passes "
            "warehouse quality inspection. Customers must share valid bank "
            "account details within 3 days of the return being picked up."
        ),
    },
    {
        "doc_id": "D06",
        "title": "Prepaid Refund Timeline",
        "text": (
            "For prepaid orders paid by card, UPI, or wallet, refunds are "
            "issued to the original payment method within 3-5 business days "
            "after the returned item passes quality inspection. Card refunds "
            "may take an additional 5-7 business days to reflect depending "
            "on the issuing bank."
        ),
    },
    {
        "doc_id": "D07",
        "title": "Delivery SLA -- Metro Areas",
        "text": (
            "Orders shipped to major metro cities are typically delivered "
            "within 1-3 business days of dispatch. Same-day and next-day "
            "delivery is available on eligible items in select metro pin "
            "codes."
        ),
    },
    {
        "doc_id": "D08",
        "title": "Delivery SLA -- Non-Metro & Rural Areas",
        "text": (
            "Orders shipped to non-metro towns and rural pin codes typically "
            "take 4-8 business days for delivery, depending on courier "
            "network coverage. Remote or low-connectivity areas may "
            "experience delays of up to 12 business days during high-demand "
            "periods such as festive sales."
        ),
    },
    {
        "doc_id": "D09",
        "title": "Reverse Pickup Eligibility",
        "text": (
            "Reverse pickup, where a courier collects the returned item from "
            "the customer's address, is available in most serviceable pin "
            "codes for eligible categories. The customer does not need to "
            "visit a courier office when reverse pickup is available; a "
            "pickup slot is scheduled automatically after a return is "
            "approved."
        ),
    },
    {
        "doc_id": "D10",
        "title": "Reverse Pickup Unavailable Areas",
        "text": (
            "In pin codes where reverse pickup is not serviceable, the "
            "customer must self-ship the item to the address provided in "
            "the return instructions. Self-ship return shipping costs are "
            "reimbursed by Flipkart upon verification of the courier "
            "receipt."
        ),
    },
    {
        "doc_id": "D11",
        "title": "Exchange Policy",
        "text": (
            "Size or color exchanges for apparel and footwear are allowed "
            "once per order within the applicable return window. An "
            "exchange is processed as a simultaneous pickup of the original "
            "item and delivery of the replacement, subject to stock "
            "availability of the requested variant."
        ),
    },
    {
        "doc_id": "D12",
        "title": "Damaged or Defective Item Policy",
        "text": (
            "If an item arrives damaged, defective, or significantly "
            "different from its description, the customer should report it "
            "within 48 hours of delivery using photo evidence. Flipkart "
            "waives the usual return window restrictions for verified "
            "damaged or defective claims and offers a full refund or free "
            "replacement."
        ),
    },
    {
        "doc_id": "D13",
        "title": "Order Cancellation Before Shipping",
        "text": (
            "Orders can be cancelled free of charge any time before they "
            "are shipped from the seller's warehouse. Once an order has "
            "shipped, it cannot be cancelled and must instead be refused at "
            "delivery or returned after delivery under the applicable "
            "category return policy."
        ),
    },
    {
        "doc_id": "D14",
        "title": "International Orders",
        "text": (
            "Flipkart does not currently support shipping orders outside "
            "India, and international payment cards not linked to an Indian "
            "billing address are not accepted at checkout."
        ),
    },
]


def sentence_split(text: str) -> list[str]:
    """Simple, dependency-free sentence splitter: splits on '. ', '? ', '! '
    followed by a capital letter or end of string, while being careful not
    to split on the period found in numeric ranges like '10-day'."""
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def build_chunks() -> list[dict]:
    """Returns a flat list of chunks, each tagged with its parent doc_id.
    This chunk -> parent-doc mapping is what Task 10's document-level
    Precision@3/Recall@3 scoring relies on."""
    chunks = []
    for doc in DOCUMENTS:
        sentences = sentence_split(doc["text"])
        for i, sent in enumerate(sentences):
            chunks.append({
                "chunk_id": f"{doc['doc_id']}-{i:02d}",
                "doc_id": doc["doc_id"],
                "doc_title": doc["title"],
                "text": sent,
            })
    return chunks


# ---------------------------------------------------------------------------
# Task 1 (cont.): retrieval-evaluation answer key.
# For each realistic test query, the document(s) (NOT individual chunks)
# considered "relevant" -- this is Task 10's ground truth.
# ---------------------------------------------------------------------------
RETRIEVAL_ANSWER_KEY = [
    {
        "query": "How many days do I have to return a shirt I bought?",
        "relevant_doc_ids": ["D01"],
    },
    {
        "query": "When will I get my refund if I paid cash on delivery?",
        "relevant_doc_ids": ["D05"],
    },
    {
        "query": "Can I still return my laptop after 20 days?",
        "relevant_doc_ids": ["D02"],
    },
    {
        "query": "How long does delivery take to a village or rural address?",
        "relevant_doc_ids": ["D08"],
    },
    {
        "query": "Is someone going to come pick up my return, or do I have to ship it myself?",
        "relevant_doc_ids": ["D09", "D10"],
    },
    {
        "query": "Can I exchange a pair of shoes for a different size?",
        "relevant_doc_ids": ["D11"],
    },
    {
        "query": "What should I do if my order arrived broken?",
        "relevant_doc_ids": ["D12"],
    },
]


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"Documents: {len(DOCUMENTS)}")
    print(f"Chunks: {len(chunks)}")
    for c in chunks[:6]:
        print(f"  [{c['chunk_id']}] ({c['doc_id']}) {c['text']}")
    print(f"\nRetrieval answer-key queries: {len(RETRIEVAL_ANSWER_KEY)}")
