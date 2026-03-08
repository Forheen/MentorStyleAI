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


# ---------- CHAT ----------

class ChatStartRequest(BaseModel):
    problem: str


class ChatStartResponse(BaseModel):
    session_id: str
    reply: str


class ChatMessageRequest(BaseModel):
    session_id: str
    message: str


class ChatMessageResponse(BaseModel):
    reply: str
    solved: bool