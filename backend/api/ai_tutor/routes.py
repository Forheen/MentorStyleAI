# api/ai_tutor/routes.py

from fastapi import APIRouter, HTTPException
from backend.api.ai_tutor.schemas import *
from backend.api.ai_tutor.controller import *

router = APIRouter()


@router.post("/generate")
def generate(request: TutorRequest):

    result = generate_tutor(request.problem)

    if not result:
        raise HTTPException(status_code=500, detail="Failed to generate tutor data")

    return result


# ---------- CHAT ----------

@router.post("/chat/start")
def chat_start(request: ChatStartRequest):

    return start_chat_controller(request.problem)


@router.post("/chat/message")
def chat_message_api(request: ChatMessageRequest):

    return chat_message_controller(
        request.session_id,
        request.message
    )