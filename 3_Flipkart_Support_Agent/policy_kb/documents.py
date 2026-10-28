"""
Flipkart-style policy knowledge base (14 documents, written for this project as an
illustrative policy set -- not Flipkart's official policy text), the sentence-wise
chunker, the chunk -> parent-document mapping, and the retrieval-evaluation
answer key.

Chunking strategy: SENTENCE-WISE. Every policy sentence is a self-contained
factual claim (a window, a timeline, an eligibility rule), so one sentence =
one chunk keeps each retrieved chunk usable as a quotable fact without cutting
a rule in half the way fixed-size or overlapping windows can. Each chunk keeps
its parent `doc_id`, which Task 10's document-level scoring relies on.
"""
import re

DOCUMENTS = [
    {"doc_id": "D01", "title": "Apparel & Footwear Return Window", "text": (
        "Apparel and footwear items purchased on Flipkart can be returned within 14 days of delivery. "
        "The 14 days are counted from the date the courier marks the parcel as delivered, not from the order date. "
        "Items must be unused, unwashed, and returned with original tags and packaging intact. "
        "Items showing perfume, deodorant, makeup stains, or signs of wear are rejected at quality inspection. "
        "Innerwear, socks, and swimwear are non-returnable for hygiene reasons. "
        "A return request can be raised from the My Orders page by selecting the item and choosing a return reason. "
        "Once the request is approved, the customer receives a confirmation message with the pickup details. "
        "Fashion items bought during festive sale events follow the same 14-day window unless the product page states otherwise. "
        "If a shirt, dress, or pair of shoes does not fit, the customer can choose either a refund or an exchange for another size.")},
    {"doc_id": "D02", "title": "Electronics Return Window", "text": (
        "Electronics such as phones, laptops, tablets, and cameras have a 10-day return window from the date of delivery. "
        "The device must be free of physical damage, scratches, and dents, and all original accessories, box, and invoice must be included. "
        "Returns are accepted only if the serial number or IMEI of the returned device matches the one that was shipped. "
        "Software or activation issues should be reported within 48 hours of delivery to qualify for a replacement instead of a return. "
        "After the 10-day window has passed, a return is not possible and the customer should contact the brand's authorised service centre under the manufacturer warranty. "
        "Accessories such as chargers, earphones, and cables that are bought separately follow the same 10-day window. "
        "Devices must be factory reset, and any screen lock, Apple ID, or Google account must be removed before pickup. "
        "Opened software, gift cards, and downloadable licences cannot be returned once activated.")},
    {"doc_id": "D03", "title": "Home & Furniture Return Window", "text": (
        "Home and furniture items, including furnishings, kitchenware, and large appliances, can be returned within 7 days of delivery. "
        "Assembled furniture must be disassembled by the customer before pickup unless the seller offers assisted return. "
        "Large appliances such as refrigerators and washing machines are returned only after a technician from the brand has inspected the product. "
        "Installation of large appliances is handled by the brand, and a return is not possible once the installation has been completed unless a defect is found. "
        "Mattresses can be returned only if the plastic wrapping is intact and the mattress has not been used. "
        "Made-to-order and customised furniture cannot be returned because they are produced after the order is placed. "
        "Customers should keep the original packaging, foam inserts, and fittings until the return window closes. "
        "Light home decor items such as lamps, curtains, and cushion covers are picked up from the customer's doorstep like any other return.")},
    {"doc_id": "D04", "title": "Beauty & Personal Care Return Policy", "text": (
        "Beauty and personal care products, including cosmetics, perfumes, and skincare items, are non-returnable once the outer seal is broken, for hygiene and safety reasons. "
        "Unopened, sealed products can be returned within 7 days of delivery. "
        "The product must be in its original packaging with the batch code and manufacturing label visible. "
        "If a product has been used even once, it will be rejected at quality inspection and sent back to the customer. "
        "Products that arrive leaking, expired, or with a broken seal are covered by the damaged or defective item policy instead. "
        "Grooming appliances such as trimmers and hair dryers are treated as electronics-style items and may be replaced within 7 days if they are faulty. "
        "Skin allergies or reactions are not a valid reason for return after opening, but customers are encouraged to consult a dermatologist and share the product details with support. "
        "Gift sets and combo packs can be returned only as a complete set.")},
    {"doc_id": "D05", "title": "COD Refund Timeline", "text": (
        "For Cash on Delivery orders, refunds are issued to the customer's bank account or Flipkart wallet, since no prepaid instrument exists to reverse. "
        "Refunds are typically processed within 7-10 business days after the returned item passes warehouse quality inspection. "
        "Customers must share valid bank account details within 3 days of the return being picked up. "
        "If the bank details are not shared in time, the refund is credited to the Flipkart wallet as store credit instead. "
        "The refund amount for a COD order is the item price paid at the door, minus any non-refundable cash handling fee shown at checkout. "
        "Customers can track the refund status from the My Orders page under the returned item. "
        "A refund to a bank account is sent by direct bank transfer, and the customer receives a message with the transaction reference number. "
        "If the money has not arrived after 10 business days, the customer should contact support with the return ID and bank statement.")},
    {"doc_id": "D06", "title": "Prepaid Refund Timeline", "text": (
        "For prepaid orders paid by card, UPI, net banking, or wallet, refunds are issued to the original payment method within 3-5 business days after the returned item passes quality inspection. "
        "Card refunds may take an additional 5-7 business days to reflect on the statement depending on the issuing bank. "
        "UPI and wallet refunds are usually instant to 24 hours once the refund is initiated by Flipkart. "
        "Net banking refunds go back to the same bank account that was used to pay and normally arrive within 3-5 business days. "
        "Customers can choose to receive the refund as Flipkart wallet credit, which is available immediately after quality inspection. "
        "Refunds for orders paid with an EMI plan are reversed to the card, and the bank decides how the EMI schedule is adjusted. "
        "Gift card balances used in an order are refunded to the gift card and not to the bank account. "
        "If a prepaid refund has not arrived after 10 business days, the customer should contact the bank and share the refund reference number from Flipkart.")},
    {"doc_id": "D07", "title": "Delivery SLA -- Metro Areas", "text": (
        "Orders shipped to major metro cities such as Delhi, Mumbai, Bengaluru, Chennai, Hyderabad, Kolkata, and Pune are typically delivered within 1-3 business days of dispatch. "
        "Same-day and next-day delivery is available on eligible items in select metro pin codes when the order is placed before the daily cut-off time. "
        "Delivery slots are shown on the product page and at checkout, and they depend on the seller's warehouse location. "
        "Orders placed on Sundays and public holidays are dispatched on the next working day. "
        "Customers receive an SMS with a delivery OTP, which they must share with the courier executive at the door. "
        "If the customer is not available, the courier attempts delivery up to three times before the parcel is returned to the seller. "
        "Delivery to office and apartment buildings is made to the main gate or reception when the building does not allow courier executives inside. "
        "The estimated delivery date can change if the order contains items from multiple sellers.")},
    {"doc_id": "D08", "title": "Delivery SLA -- Non-Metro & Rural Areas", "text": (
        "Orders shipped to non-metro towns, villages, and rural pin codes typically take 4-8 business days for delivery, depending on courier network coverage. "
        "Remote or low-connectivity areas, including hill regions and islands, may experience delays of up to 12 business days during high-demand periods such as festive sales. "
        "Some large or heavy items cannot be delivered to certain remote pin codes, and the product page shows whether a pin code is serviceable. "
        "Delivery to a village is usually made by a local partner courier, and the customer may be asked to share a landmark and a working phone number. "
        "Cash on Delivery may not be available for every rural pin code, in which case the order must be prepaid. "
        "If a parcel is delayed beyond the estimated date, the customer can raise a delivery-delay request from the My Orders page. "
        "Customers in rural areas are advised to keep the phone reachable because couriers call before attempting delivery. "
        "Weather events such as floods and cyclones can pause delivery in affected districts until the courier network reopens.")},
    {"doc_id": "D09", "title": "Reverse Pickup Eligibility", "text": (
        "Reverse pickup, where a courier collects the returned item from the customer's address, is available in most serviceable pin codes for eligible categories. "
        "The customer does not need to visit a courier office when reverse pickup is available; a pickup slot is scheduled automatically after a return is approved. "
        "The courier attempts pickup within 2-3 business days of approval, and the customer receives an SMS with the pickup date. "
        "The customer should hand over the item in its original packaging along with all tags, accessories, and the invoice. "
        "If the customer misses the pickup, up to two more attempts are made before the return request is closed. "
        "A pickup can be rescheduled once from the My Orders page, as long as the return window has not ended. "
        "The courier executive checks that the item matches the return request, and a pickup receipt is shared with the customer. "
        "Reverse pickup is free of charge for all returns approved under the return policy.")},
    {"doc_id": "D10", "title": "Reverse Pickup Unavailable Areas", "text": (
        "In pin codes where reverse pickup is not serviceable, the customer must self-ship the item to the address provided in the return instructions. "
        "Self-ship return shipping costs are reimbursed by Flipkart upon verification of the courier receipt, up to a maximum amount shown in the return instructions. "
        "The customer should use a trackable courier or India Post speed post and keep the receipt and tracking number. "
        "The parcel must be sent within 5 days of the return request being approved, otherwise the request expires. "
        "The refund process starts only after the returned item reaches the Flipkart returns warehouse and passes quality inspection. "
        "Customers should pack the item securely in its original box because damage in transit can lead to a rejected return. "
        "The reimbursement for self-shipping is credited to the Flipkart wallet within 7 business days of the receipt being verified. "
        "Customers can check on the return page whether their pin code supports reverse pickup before choosing a return method.")},
    {"doc_id": "D11", "title": "Exchange Policy", "text": (
        "Size or color exchanges for apparel and footwear are allowed once per order within the applicable return window. "
        "An exchange is processed as a simultaneous pickup of the original item and delivery of the replacement, subject to stock availability of the requested variant. "
        "If the requested size or color is out of stock, the customer is offered a refund instead of an exchange. "
        "The replacement item is dispatched only after the courier has picked up the original item. "
        "Exchanges are not available for electronics, which can only be replaced if the device is defective, as described in the damaged or defective item policy. "
        "Exchanges are not available for beauty products or for items bought using a promotional offer that was limited to one unit. "
        "If the price of the replacement differs from the original, the customer pays the difference or receives the difference as a refund. "
        "A second exchange on the same order is not possible, but the customer can still choose a refund if the item is within the return window.")},
    {"doc_id": "D12", "title": "Damaged or Defective Item Policy", "text": (
        "If an item arrives damaged, defective, broken, or significantly different from its description, the customer should report it within 48 hours of delivery using photo evidence. "
        "The report can be filed from the My Orders page by choosing the option for a damaged or wrong item and uploading clear photos of the product and the packaging. "
        "Flipkart waives the usual return window restrictions for verified damaged or defective claims and offers a full refund or free replacement. "
        "For large appliances, a technician visit may be arranged to confirm the defect before a replacement is approved. "
        "If the wrong item was delivered, the customer should not use or open it and should keep the original box for pickup. "
        "Pickup for a damaged or defective item is always free, and no shipping fee is charged for the replacement. "
        "Claims reported after 48 hours may still be reviewed, but approval is not guaranteed. "
        "If the item is out of stock for replacement, the customer receives a full refund to the original payment method.")},
    {"doc_id": "D13", "title": "Order Cancellation Before Shipping", "text": (
        "Orders can be cancelled free of charge any time before they are shipped from the seller's warehouse. "
        "Once an order has shipped, it cannot be cancelled and must instead be refused at delivery or returned after delivery under the applicable category return policy. "
        "To cancel an order, the customer opens the My Orders page, selects the item, and chooses the cancel option with a reason. "
        "Prepaid orders that are cancelled before shipping are refunded to the original payment method within 3-5 business days. "
        "Cash on Delivery orders that are cancelled before shipping do not require any refund, because no payment has been made. "
        "If an order has several items, the customer can cancel a single item without cancelling the rest of the order. "
        "Sellers may cancel an order if the item is out of stock or if the delivery address is not serviceable, and the customer is notified by message. "
        "Repeated refusal of Cash on Delivery orders at the door can lead to Cash on Delivery being disabled on the customer's account.")},
    {"doc_id": "D14", "title": "International Orders", "text": (
        "Flipkart does not currently support shipping orders outside India, so every order must have a delivery address within the country. "
        "International payment cards not linked to an Indian billing address are not accepted at checkout. "
        "Customers living abroad can order for delivery to a family member's address in India by entering that address at checkout. "
        "Prices are shown only in Indian rupees, and no currency conversion is offered on the platform. "
        "Customs duties do not apply to domestic deliveries, so there are no import charges on any order. "
        "Gift orders sent to an address in India are supported, and the customer can add a gift message at checkout. "
        "Returns for gift orders are processed from the delivery address, and the refund goes to the person who paid for the order. "
        "Customers who need cross-border shipping are advised to use a specialist forwarding service, which is not operated by Flipkart.")},
]


def sentence_split(text: str) -> list:
    """Split on sentence-ending punctuation followed by whitespace + capital letter
    (so '10-day' or '7-10 business days' are not broken)."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip()) if s.strip()]


def build_chunks() -> list:
    """One chunk per sentence; each carries its parent doc_id (chunk -> document map)."""
    chunks = []
    for doc in DOCUMENTS:
        for i, sentence in enumerate(sentence_split(doc["text"])):
            chunks.append({"chunk_id": f"{doc['doc_id']}-{i:02d}", "doc_id": doc["doc_id"],
                           "doc_title": doc["title"], "text": sentence})
    return chunks


# Retrieval-evaluation answer key: for each realistic query, the one or two
# DOCUMENTS (not chunks) a support agent would consider relevant.
RETRIEVAL_ANSWER_KEY = [
    {"query": "How many days do I have to return a shirt I bought?", "relevant_doc_ids": ["D01"]},
    {"query": "When will I get my refund if I paid cash on delivery?", "relevant_doc_ids": ["D05"]},
    {"query": "Can I still return my laptop after 20 days?", "relevant_doc_ids": ["D02"]},
    {"query": "How long does delivery take to a village or rural address?", "relevant_doc_ids": ["D08"]},
    {"query": "Will someone come to pick up my return, or do I have to ship it myself?", "relevant_doc_ids": ["D09", "D10"]},
    {"query": "Can I exchange a pair of shoes for a different size?", "relevant_doc_ids": ["D11"]},
    {"query": "What should I do if my order arrived broken?", "relevant_doc_ids": ["D12"]},
]

if __name__ == "__main__":
    chunks = build_chunks()
    print(f"{len(DOCUMENTS)} documents -> {len(chunks)} chunks")
    for c in chunks[:5]:
        print(f"  [{c['chunk_id']}] ({c['doc_id']}) {c['text']}")
