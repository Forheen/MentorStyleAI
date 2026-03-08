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

    result = chat_message(session_id, message)

    if not result:
        return {
            "reply": "Chat session not found.",
            "solved": False
        }

    reply, solved = result

    return {
        "reply": reply,
        "solved": solved
    }

