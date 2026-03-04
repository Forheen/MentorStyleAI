# api/ai_tutor/routes.py

from fastapi import APIRouter, HTTPException
from api.ai_tutor.schemas import TutorRequest
from api.ai_tutor.controller import generate_tutor

router = APIRouter()


@router.post("/generate")
def generate(request: TutorRequest):

    result = generate_tutor(request.problem)

    if not result:
        raise HTTPException(status_code=500, detail="Failed to generate tutor data")

    return result