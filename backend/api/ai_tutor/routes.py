# api/ai_tutor/routes.py

from urllib import request

from urllib import request

from fastapi import APIRouter, HTTPException
from backend.api.ai_tutor.schemas import *
from backend.api.ai_tutor.controller import *
from backend.api.ai_tutor.input_normalizer import *
router = APIRouter()


@router.post("/generate")

def generate(request: TutorRequest):

    normalized_problem = normalize_input(

        request.problem,
        request.images

    )

    result = generate_tutor(normalized_problem)

    if not result:

        raise HTTPException(

            status_code=500,

            detail="Failed to generate tutor data"

        )

    return result

# ---------- CHAT ----------

@router.post("/chat/start")

def chat_start(request: ChatStartRequest):

    normalized_problem = normalize_input(

        request.problem,
        request.images
    )

    return start_chat_controller(

        normalized_problem
    )

@router.post("/chat/message")
def chat_message_api(request: ChatMessageRequest):

    normalized_message = normalize_input(request.message, request.images)
    return chat_message_controller(request.session_id, normalized_message)