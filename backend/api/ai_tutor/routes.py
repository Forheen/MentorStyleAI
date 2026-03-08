# api/ai_tutor/routes.py

from fastapi import APIRouter, HTTPException
from backend.api.ai_tutor.schemas import TutorRequest
from backend.api.ai_tutor.controller import generate_tutor

router = APIRouter()


@router.post("/generate")
def generate(request: TutorRequest):

    result = generate_tutor(request.problem)

    if not result:
        raise HTTPException(status_code=500, detail="Failed to generate tutor data")

    return result


# api/ai_tutor/controller.py

from backend.api.ai_tutor.service import (
    generate_deconstruction,
    generate_visual_image,
    start_chat,
    chat_message
)


def generate_tutor(problem: str):

    structured_data = generate_deconstruction(problem)

    if not structured_data:
        return None

    image = generate_visual_image(problem, structured_data)

    return {
        "structured_data": structured_data,
        "image": image
    }


# -------- CHAT ----------

def start_chat_controller(problem: str):

    session_id, reply = start_chat(problem)

    return {
        "session_id": session_id,
        "reply": reply
    }


def chat_message_controller(session_id: str, message: str):

    reply, solved = chat_message(session_id, message)

    return {
        "reply": reply,
        "solved": solved
    }