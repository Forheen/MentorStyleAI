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
      "final_answer": "Correct final answer only (no explanation)"
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
def render_deconstruction(data):

    if not data:
        st.error("Failed to generate structured output.")
        return

    st.title(f"🧠 {data.get('title', 'Structural Deconstruction')}")

    st.markdown("---")

    for stage in data.get("reasoning_stages", []):
        with st.container():
            st.subheader(f"🔎 {stage.get('stage_name')}")
            st.markdown(f"**Goal:**  \n{stage.get('goal')}")
            st.markdown(f"**Concept Focus:**  \n{stage.get('concept_focus')}")
            st.markdown(f"**Deep Insight:**  \n{stage.get('deep_insight')}")
            st.markdown(f"**Execution Outline:**  \n{stage.get('execution_outline')}")
            st.markdown("---")

    if data.get("alternative_structures"):
        st.subheader("🧩 Alternative Structural Paths")
        for alt in data["alternative_structures"]:
            st.markdown(f"- {alt}")
        st.markdown("---")

    if data.get("common_structural_mistakes"):
        st.subheader("⚠️ Common Structural Mistakes")
        for mistake in data["common_structural_mistakes"]:
            st.markdown(f"- {mistake}")
        st.markdown("---")

    st.subheader("🎯 Final Structural Answer")
    st.success(data.get("final_structural_answer", ""))

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

    with st.spinner("Generating visual diagram..."):
        image_data, mime_type = generate_visual_image(problem, structured_data)

    if image_data:
        st.image(image_data)
    else:
        st.error("Image generation failed.")