import base64
import json

from openai import OpenAI

from app.config import settings
from app.api.schemas.response import AnalysisResponse


client = OpenAI(api_key=settings.openai_api_key)


def analyze_screenshot(image_bytes: bytes, mime_type: str) -> AnalysisResponse:
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.responses.create(
        model=settings.openai_model,
        input=[
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are a cybersecurity assistant specialized in phishing screenshot analysis. "
                            "Analyze only what is visible in the screenshot. "
                            "Classify the screenshot as one of: phishing, suspicious, legitimate. "
                            "Do not invent details that are not visible. "
                            "If evidence is mixed or incomplete, choose suspicious."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Analyze this screenshot for phishing signs. "
                            "Return the result in the required JSON schema."
                        ),
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
                            "type": "string"
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