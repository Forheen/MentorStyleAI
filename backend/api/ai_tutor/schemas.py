# api/ai_tutor/schemas.py

from pydantic import BaseModel
from typing import Optional, Dict, Any

class TutorRequest(BaseModel):
    problem: str

class ImageResponse(BaseModel):
    image_base64: Optional[str] = None
    mime_type: Optional[str] = None

class TutorResponse(BaseModel):
    structured_data: Dict[str, Any]
    image: Optional[ImageResponse] = None