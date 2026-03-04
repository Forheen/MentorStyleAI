# api/ai_tutor/controller.py

from backend.api.ai_tutor.service import (
    generate_deconstruction,
    generate_visual_image
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