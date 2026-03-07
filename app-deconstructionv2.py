import streamlit as st
import json
import re
import base64
from google import genai
from google.genai import types

# ==================================================
# CONFIG
# ==================================================
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=GEMINI_API_KEY)

TEXT_MODEL = "gemini-3-pro-preview"
IMAGE_MODEL = "gemini-3-pro-image-preview"

st.set_page_config(
    page_title="AI Structural Deconstruction Engine",
    page_icon="🧠",
    layout="centered"
)

# ==================================================
# LOAD POLICY
# ==================================================
@st.cache_resource
def load_policy():
    with open("multimodal_agent1_reasoning_style.json") as f:
        return json.load(f)

policy = load_policy()

# ==================================================
# GENERATE STRUCTURAL JSON
# ==================================================
def generate_deconstruction(problem):

    system_prompt = f"""
You are an expert mentor with deep structural intuition.

Use the following reasoning policy as the guiding framework for thinking:

{json.dumps(policy)}

------------------------------------------------
ADAPTIVE REASONING CONTROL
------------------------------------------------

Before generating reasoning, classify the difficulty of the problem.

Level 1 — Trivial
Examples:
- Simple arithmetic
- Single-step linear equation
- Direct substitution

Level 2 — Basic
Examples:
- Multi-step algebra
- Simple geometry reasoning

Level 3 — Conceptual
Examples:
- Pattern recognition
- Word problems requiring structural insight

Level 4 — Deep Insight
Examples:
- Olympiad-style problems
- Multi-concept reasoning problems

------------------------------------------------
REASONING DEPTH RULE
------------------------------------------------

If Level 1 (Trivial):
- Keep reasoning extremely concise
- Maximum one short sentence per stage
- Avoid philosophical or reflective elaboration

If Level 2:
- Provide short conceptual guidance

If Level 3 or Level 4:
- Apply full structural reasoning aligned with the policy

------------------------------------------------
COGNITIVE RULES
------------------------------------------------

1. Prefer structural insight over procedural solving.
2. Avoid introducing symbolic variables unless necessary.
3. Avoid grind-based computation.
4. Seek patterns, symmetry, invariants, or conceptual structure.
5. If multiple solution paths exist, prefer the one revealing deeper structure.
6. Computation should follow insight.
7. Reasoning should feel conceptually clear rather than mechanical.
8. Align explanations with how a learner would build intuition.

------------------------------------------------
REASONING STRUCTURE
------------------------------------------------

Follow the reasoning progression defined in:

policy.reasoning_progression

For each stage:

- Extract the stage name from the policy.
- Align the reasoning with the description provided in the policy.
- Emphasize conceptual understanding before execution.

------------------------------------------------
OUTPUT FORMAT
------------------------------------------------

Return STRICT JSON only.

{{
  "difficulty_level": "Level 1 | Level 2 | Level 3 | Level 4",
  "reasoning_stages": [
    {{
      "stage": "<stage_name_from_policy>",
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
  "final_answer": "Correct final answer only"
}}
"""

    response = client.models.generate_content(
        model=TEXT_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0
        ),
        contents=f"Problem:\n{problem}"
    )

    raw = response.text.strip()
    cleaned = re.sub(r"```json|```", "", raw).strip()

    try:
        return json.loads(cleaned)
    except:
        return {}

# ==================================================
# GENERATE VISUAL IMAGE
# ==================================================
def generate_visual_image(problem, structured_data):

    image_prompt = f"""
Create a clean educational diagram visualizing this structural breakdown.

Problem:
{problem}

Structured Deconstruction:
{json.dumps(structured_data, indent=2)}

Instructions:
- Professional educational infographic style
- Clear labeled sections
- Flow arrows between reasoning stages
- Minimal but precise text
- Clean white background
- Modern academic style
"""

    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=image_prompt,
        config=types.GenerateContentConfig(temperature=0)
    )

    for candidate in response.candidates:
        for part in candidate.content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                return part.inline_data.data, part.inline_data.mime_type

    return None, None

# ==================================================
# RENDER STRUCTURE
# ==================================================
# ==================================================
# RENDER STRUCTURE (ALIGNED WITH YOUR JSON)
# ==================================================
def render_deconstruction(data):

    if not data:
        st.error("Failed to generate structured output.")
        return

    st.title("🧠 Structural Deconstruction")

    st.markdown("---")

    # Reasoning Stages
    for stage in data.get("reasoning_stages", []):
        with st.container():
            st.subheader(f"🔎 Stage {stage.get('stage')}")
            st.markdown(f"**Goal:**  \n{stage.get('goal')}")
            st.markdown(f"**Concept Focus:**  \n{stage.get('concept_focus')}")
            st.markdown(f"**Expected Student Action:**  \n{stage.get('expected_student_action')}")
            st.markdown("---")

    # Alternative Paths
    if data.get("valid_alternative_paths"):
        st.subheader("🧩 Alternative Structural Paths")
        for alt in data["valid_alternative_paths"]:
            st.markdown(f"- {alt}")
        st.markdown("---")

    # Common Mistakes
    if data.get("common_mistakes"):
        st.subheader("⚠️ Common Structural Mistakes")
        for mistake in data["common_mistakes"]:
            st.markdown(f"- {mistake}")
        st.markdown("---")

    # Final Answer
    if data.get("final_answer"):
        st.subheader("🎯 Final Structural Answer")
        st.success(data.get("final_answer"))

# ==================================================
# UI
# ==================================================
st.title("🧠 AI Structural Deconstruction Engine")

problem = st.text_area("Enter the problem")

if st.button("Generate Deconstruction") and problem:

    with st.spinner("Generating structural breakdown..."):
        structured_data = generate_deconstruction(problem)

    render_deconstruction(structured_data)

    st.markdown("## 🖼 Visual Structural Diagram")

    # with st.spinner("Generating visual diagram..."):
    #     image_data, mime_type = generate_visual_image(problem, structured_data)

    # if image_data:
    #     st.image(image_data)
    # else:
    #     st.error("Image generation failed.")