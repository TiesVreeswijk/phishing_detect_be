from typing import List, Literal
from pydantic import BaseModel, Field


class AnalysisResponse(BaseModel):
    verdict: Literal["phishing", "suspicious", "legitimate"] = Field(...)
    confidence: int = Field(..., ge=0, le=100)
    summary: str
    red_flags: List[str]