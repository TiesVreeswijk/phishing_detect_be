"""
PART 1: Fetch phishing data from PhishStats and populate Pinecone vector DB.

Setup:
    pip install pinecone-client openai requests python-dotenv tldextract

Environment variables (.env):
    PINECONE_API_KEY=your_pinecone_api_key
    OPENAI_API_KEY=your_openai_api_key        # used for embeddings
    PHISHSTATS_API=https://phishstats.info:2096/api/phishing
"""

import os
import time
import hashlib
import requests
import tldextract
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ── Clients ────────────────────────────────────────────────────────────────────

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "phishing-patterns"
EMBED_MODEL = "text-embedding-3-small"   # 1536 dims, cheap and fast
EMBED_DIM   = 1536

# ── Pinecone index setup ───────────────────────────────────────────────────────

def get_or_create_index():
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        print("Index ready.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
    return pc.Index(INDEX_NAME)

# ── PhishStats fetcher ─────────────────────────────────────────────────────────

def fetch_phishstats(limit: int = 500, days_back: int = 30) -> list[dict]:
    """
    Fetch recent phishing records from PhishStats.
    API docs: https://phishstats.info/
    Free, no auth required.
    """
    cutoff = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = os.getenv("PHISHSTATS_API", "https://phishstats.info:2096/api/phishing")

    params = {
        "_size": min(limit, 100),       # max 100 per request per API docs
        "_sort": "-date",               # newest first
        "_where": f"(score,gt,5)",      # only high confidence entries
    }

    print(f"Fetching up to {limit} records from PhishStats (last {days_back} days)...")
    all_records = []
    page = 1
    per_page = 100  # API max

    try:
        while len(all_records) < limit:
            params = {
                "_size": per_page,
                "_sort": "-date",
                "_where": "(score,gt,5)",
                "_p": page,
            }
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            batch = resp.json()

            if not batch:
                break

            all_records.extend(batch)
            print(f"  → Page {page}: got {len(batch)} records (total: {len(all_records)})")
            page += 1

            if len(batch) < per_page:
                break  # no more pages

        print(f"  → Fetched {len(all_records)} records total.")
        return all_records[:limit]

    except Exception as e:
        print(f"  [ERROR] PhishStats fetch failed: {e}")
        return []

# ── Normalise into meaningful text documents ───────────────────────────────────

KNOWN_BRANDS = [
    "paypal", "apple", "microsoft", "google", "amazon", "netflix",
    "facebook", "instagram", "whatsapp", "bank", "dhl", "fedex",
    "dropbox", "linkedin", "twitter", "chase", "wellsfargo", "irs",
]

def extract_brand(url: str) -> str:
    """Best-effort brand detection from URL."""
    url_lower = url.lower()
    for brand in KNOWN_BRANDS:
        if brand in url_lower:
            return brand.capitalize()
    return "Unknown"

def normalize_record(record: dict) -> dict | None:
    """
    Turn a raw PhishStats record into a clean text document for embedding.
    Returns None if the record lacks enough signal.
    """
    url   = record.get("url", "").strip()
    ip    = record.get("ip", "")
    score = record.get("score", 0)
    date  = record.get("date", "")
    title = record.get("title", "")
    tld_info = tldextract.extract(url)

    if not url or score < 5:          # skip low-confidence entries
        return None

    domain      = f"{tld_info.domain}.{tld_info.suffix}"
    subdomain   = tld_info.subdomain
    brand       = extract_brand(url)

    # Detect common URL tricks
    tricks = []
    if "-" in tld_info.domain:
        tricks.append("hyphenated domain")
    if any(c.isdigit() for c in tld_info.domain):
        tricks.append("digits in domain")
    if tld_info.suffix not in ("com", "org", "net", "gov"):
        tricks.append(f"unusual TLD (.{tld_info.suffix})")
    if subdomain and subdomain not in ("www", ""):
        tricks.append(f"suspicious subdomain ({subdomain})")
    if len(url) > 80:
        tricks.append("very long URL")

    tricks_text = ", ".join(tricks) if tricks else "none detected"

    # Build a rich natural-language document — this is what gets embedded
    text = (
        f"Phishing page targeting {brand}. "
        f"Domain: {domain}. "
        f"Full URL pattern: {url[:120]}. "
        f"Page title: {title or 'unknown'}. "
        f"URL tricks: {tricks_text}. "
        f"Risk score: {score}/10. "
        f"Detected: {date[:10] if date else 'unknown'}."
    )

    doc_id = hashlib.md5(url.encode()).hexdigest()   # stable dedup key

    return {
        "id":   doc_id,
        "text": text,
        "metadata": {
            "url":    url[:500],      # Pinecone metadata limit
            "domain": domain,
            "brand":  brand,
            "score":  float(score),
            "date":   date[:10] if date else "",
            "ip":     ip or "",
            "tricks": tricks_text,
        },
    }

# ── Embedding ──────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI."""
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]

# ── Upsert to Pinecone ─────────────────────────────────────────────────────────

def upsert_documents(index, documents: list[dict], batch_size: int = 100):
    """Embed and upsert documents into Pinecone in batches."""
    total   = len(documents)
    upserted = 0

    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        texts = [doc["text"] for doc in batch]

        print(f"  Embedding batch {i // batch_size + 1} ({len(batch)} docs)...")
        embeddings = embed_texts(texts)

        vectors = [
            {
                "id":       doc["id"],
                "values":   emb,
                "metadata": {**doc["metadata"], "text": doc["text"]},
            }
            for doc, emb in zip(batch, embeddings)
        ]

        index.upsert(vectors=vectors)
        upserted += len(vectors)
        print(f"  ✓ Upserted {upserted}/{total}")

    return upserted

# ── Main ───────────────────────────────────────────────────────────────────────

def build_vector_db(limit: int = 500, days_back: int = 30):
    index   = get_or_create_index()
    records = fetch_phishstats(limit=limit, days_back=days_back)

    documents = []
    skipped   = 0
    for rec in records:
        doc = normalize_record(rec)
        if doc:
            documents.append(doc)
        else:
            skipped += 1

    print(f"\nNormalized: {len(documents)} docs | Skipped (low signal): {skipped}")

    if not documents:
        print("Nothing to upsert.")
        return

    upserted = upsert_documents(index, documents)
    print(f"\n✅ Done! {upserted} documents stored in Pinecone index '{INDEX_NAME}'.")
    print(f"   Run this script on a schedule (e.g. daily) to keep the DB fresh.")

if __name__ == "__main__":
    build_vector_db(
        limit=500,       # increase up to 2000 if you want more coverage
        days_back=30,    # only pull last 30 days — older data is less useful
    )