from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas.response import AnalysisResponse
from app.services.analysis_service import analyze_screenshot

router = APIRouter(tags=["analyze"])


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    allowed_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}

    extension = Path(file.filename).suffix.lower()
    content_type = (file.content_type or "").lower()

    is_valid_extension = extension in allowed_extensions
    is_valid_content_type = content_type in allowed_types

    if not is_valid_extension and not is_valid_content_type:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type. "
                f"Got content_type='{file.content_type}', filename='{file.filename}'. "
                f"Use png, jpg, jpeg, or webp."
            ),
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    mime_type = content_type
    if mime_type not in allowed_types:
        if extension == ".png":
            mime_type = "image/png"
        elif extension in {".jpg", ".jpeg"}:
            mime_type = "image/jpeg"
        elif extension == ".webp":
            mime_type = "image/webp"
        else:
            raise HTTPException(status_code=400, detail="Could not determine image type")

    try:
        result = analyze_screenshot(image_bytes, mime_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")