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
    output_schema = """
    {
    "final_explanation": "...",
    "key_reasoning_lessons": [
        "...",
        "..."
    ],
    "final_answer": "..."
    }
    """

    system_prompt = f"""
    You are an expert mentor with deep structural intuition.

    Use the following reasoning policy as the guiding framework:

    {json.dumps(policy)}

    ------------------------------------------------
    PROBLEM ANALYSIS
    ------------------------------------------------

    First determine the conceptual structure of the problem before solving it.

    Prefer understanding relationships between quantities rather than performing procedural calculations.

    ------------------------------------------------
    REASONING PRINCIPLES
    ------------------------------------------------

    1. Prefer structural insight over mechanical solving.
    2. Avoid unnecessary algebraic setup.
    3. Seek patterns, ratios, invariants, or conserved relationships.
    4. Computation should follow insight, not precede it.
    5. Keep reasoning concise and concept-focused.

    ------------------------------------------------
    EXPLANATION STYLE
    ------------------------------------------------

    Provide a clear mentor-style explanation that reveals the core insight of the problem.

    Avoid step-by-step procedural solving unless absolutely necessary.

    Focus on explaining:

    • What key insight unlocks the problem  
    • Why that insight works  
    • What structural observation simplifies the reasoning  

    ------------------------------------------------
    OUTPUT FORMAT
    ------------------------------------------------

    Return STRICT JSON only.

    Provide:

    1. Final explanation
    2. Key reasoning lessons
    3. Final answer

    {output_schema}
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