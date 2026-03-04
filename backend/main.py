# main.py

from fastapi import FastAPI
from api.ai_tutor.routes import router as ai_tutor_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Structural Tutor API",
    version="1.0.0"
)

app.include_router(
    ai_tutor_router,
    prefix="/api/ai-tutor",
    tags=["AI Tutor"]
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Mentor backend running"}