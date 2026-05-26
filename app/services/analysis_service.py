import base64
import io
import json

import pytesseract
from PIL import Image
from openai import OpenAI
from pinecone import Pinecone

from app.config import settings
from app.api.schemas.response import AnalysisResponse


# ── Clients ────────────────────────────────────────────────────────────────────

client = OpenAI(api_key=settings.openai_api_key)

pc    = Pinecone(api_key=settings.pinecone_api_key)
index = pc.Index("phishing-patterns")

EMBED_MODEL = "text-embedding-3-small"

# ── RAG helpers ────────────────────────────────────────────────────────────────

def _ocr_image_bytes(image_bytes: bytes) -> str:
    """Extract visible text from image bytes using Tesseract OCR."""
    try:
        img  = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception:
        return ""


def _retrieve_phishing_patterns(ocr_text: str, top_k: int = 5) -> list:
    """
    Embed the OCR text and retrieve the closest phishing patterns
    from the Pinecone vector DB.
    Returns an empty list if OCR text is empty or retrieval fails.
    """
    if not ocr_text:
        return []

    try:
        embedding_response = client.embeddings.create(
            model=EMBED_MODEL,
            input=[ocr_text],
        )
        query_vector = embedding_response.data[0].embedding

        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )
        return results.matches
    except Exception:
        return []


def _format_rag_context(matches: list) -> str:
    """
    Format Pinecone matches into a concise context block
    that gets injected into the LLM prompt.
    """
    if not matches:
        return "No similar phishing patterns found in the threat database."

    lines = ["Threat intelligence — similar confirmed phishing patterns:\n"]
    for i, match in enumerate(matches, 1):
        meta = match.metadata
        lines.append(
            f"{i}. [Similarity: {match.score:.2f}] {meta.get('text', '')}"
        )
    return "\n".join(lines)


# ── Main service function ──────────────────────────────────────────────────────

def analyze_screenshot(image_bytes: bytes, mime_type: str) -> AnalysisResponse:
    # Step 1: OCR — extract visible text for RAG query
    ocr_text = _ocr_image_bytes(image_bytes)

    # Step 2: RAG — retrieve similar phishing patterns from vector DB
    matches     = _retrieve_phishing_patterns(ocr_text)
    rag_context = _format_rag_context(matches)

    # Step 3: Build enriched system prompt with RAG context injected
    system_prompt = (
        "You are a cybersecurity assistant specialized in phishing screenshot analysis. "
        "Analyze only what is visible in the screenshot. "
        "Classify the screenshot as one of: phishing, suspicious, legitimate. "
        "Do not invent details that are not visible. "
        "If evidence is mixed or incomplete, choose suspicious.\n\n"
        "You have access to the following threat intelligence retrieved from a "
        "database of confirmed phishing pages. Use it to inform your verdict — "
        "if the screenshot closely matches known phishing patterns, weight that heavily.\n\n"
        f"{rag_context}"
    )

    # Step 4: Build user message with OCR text + image
    user_text = (
        "Analyze this screenshot for phishing signs. "
        "Return the result in the required JSON schema.\n\n"
        f"OCR-extracted text from the screenshot:\n{ocr_text or '(no text extracted)'}"
    )

    # Step 5: Call OpenAI vision model (preserves your existing API style)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.responses.create(
        model=settings.openai_model,
        input=[
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_text,
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{image_b64}",
                    },
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "phishing_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "verdict": {
                            "type": "string",
                            "enum": ["phishing", "suspicious", "legitimate"],
                        },
                        "confidence": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                        },
                        "summary": {
                            "type": "string",
                        },
                        "red_flags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["verdict", "confidence", "summary", "red_flags"],
                    "additionalProperties": False,
                },
            }
        },
    )

    data = json.loads(response.output_text)
    return AnalysisResponse(**data)