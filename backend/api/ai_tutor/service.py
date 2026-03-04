# api/ai_tutor/service.py

import json 
import re
import base64
import google as genai
from google.genai import types
from backend.core.config import GEMINI_API_KEY, TEXT_MODEL, IMAGE_MODEL


# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

def load_policy():
    with open("multimodal_agent1_reasoning_style.json") as f:
        return json.load(f)

policy = load_policy()
# ==================================================
# GENERATE STRUCTURAL JSON
# ==================================================
def generate_deconstruction(problem: str):

    model = genai.GenerativeModel(TEXT_MODEL)

    system_prompt = f"""
    You are an expert mentor with deep structural intuition.

    Use this reasoning policy strictly:
    {json.dumps(policy)}

    CRITICAL COGNITIVE RULES:

    1. Prefer structural insight over procedural or formula-driven solving.
    2. Avoid introducing symbolic variables unless absolutely unavoidable.
    3. Avoid grind-based computation.
    4. Seek invariants, symmetries, conserved quantities, structural patterns, or conceptual compressions.
    5. If multiple solution paths exist, prefer the one that reveals the underlying structure.
    6. Computation must follow insight, not precede it.
    7. The reasoning should feel elegant and conceptually clear, not mechanical.
    8. Emphasize alignment between the learner’s internal model and the structure of the problem.
    9. Reflect meta-cognitive awareness of how insight emerged.

    You must explicitly structure reasoning stages according to:

    - Deconstruction
    - Visualization
    - Meta-Cognition
    - Algorithmic Thinking

    Each stage must reflect conceptual understanding first, and only then methodical execution.

    Do NOT:
    - Default to textbook algebra or mechanical solving
    - Begin with equations unless structurally necessary
    - Overemphasize computation

    Return STRICT JSON only:

    {{
      "reasoning_stages": [
        {{
          "stage": 1,
          "goal": "...",
          "concept_focus": "...",
          "expected_student_action": "..."
        }}
      ],
      "valid_alternative_paths": [
        "Alternative structural reasoning aligned with policy"
      ],
      "common_mistakes": [
        "Common mechanical deviation"
      ],
      "final_answer": "Correct final answer only (no explanation)",
      "final_explanation": "The blueprint has been extracted and a correct final answer has been received.Provide: Final explanation of how the reasoning stages led to the solution, emphasizing structural insights.",
      "key_reasoning_lessons": [
        "Key insight or lesson from the reasoning process"
      ]
    }}
"""
    response = model.generate_content(
        f"{system_prompt}\n\nProblem:\n{problem}",
        generation_config={"temperature": 0}
    )

    raw_text = response.text.strip()

    cleaned = re.sub(r"```json|```", "", raw_text).strip()

    try:
        return json.loads(cleaned)
    except Exception as e:
        print("JSON parsing failed:", e)
        print("Raw response:", raw_text)
        return None


# ==================================================
# GENERATE VISUAL IMAGE
# ==================================================
def generate_visual_image(problem: str, structured_data: dict):

    model = genai.GenerativeModel(IMAGE_MODEL)

    image_prompt = f"""
Create a clean educational infographic visualizing this structural breakdown.

Problem:
{problem}

Structured:
{json.dumps(structured_data, indent=2)}

Professional academic style.
White background.
Flow arrows between reasoning stages.
"""

    response = model.generate_content(image_prompt)

    for part in response.parts:
        if hasattr(part, "inline_data") and part.inline_data:
            return {
                "image_base64": base64.b64encode(part.inline_data.data).decode(),
                "mime_type": part.inline_data.mime_type,
            }

    return None