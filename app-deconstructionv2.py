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

    reasoning_stages = "\n".join(policy["reasoning_progression"])

    output_schema = """
    {
    "difficulty_level": "Level 1 | Level 2 | Level 3 | Level 4",
    "reasoning_stages": [
        {
        "stage": "<stage_name_from_policy>",
        "goal": "...",
        "concept_focus": "...",
        "expected_student_action": "..."
        }
    ],
    "valid_alternative_paths": [
        "Alternative structural reasoning"
    ],
    "common_mistakes": [
        "Common conceptual mistake"
    ],
    "final_answer": "Correct final answer only"
    }
    """

    system_prompt = f"""
    You are an expert mentor with deep structural intuition.

    Use the following reasoning policy as the single guiding framework:

    {json.dumps(policy, indent=2)}

    --------------------------------
    PROBLEM ANALYSIS
    --------------------------------

    Before producing reasoning, classify the problem difficulty.

    Level 1 — Trivial
    Direct arithmetic or single-step equation.

    Level 2 — Basic
    Short multi-step reasoning.

    Level 3 — Conceptual
    Requires recognizing relationships, patterns, or structural insight.

    Level 4 — Deep Insight
    Requires non-obvious structural reasoning.

    --------------------------------
    REASONING DEPTH
    --------------------------------

    Adapt reasoning depth based on difficulty.

    Level 1
    Use minimal reasoning. Usually 1–2 stages.

    Level 2
    Use 2–3 stages.

    Level 3–4
    Use the full reasoning progression defined in the policy.

    --------------------------------
    REASONING STAGES
    --------------------------------

    The reasoning stages are defined by the policy as:

    {reasoning_stages}

    Use only the stages that meaningfully contribute to solving the problem.
    Skip stages if they add no conceptual value.

    --------------------------------
    STRUCTURAL REASONING PRINCIPLES
    --------------------------------

    1. Prefer conceptual structure over procedural solving.
    2. Avoid grind-based computation.
    3. Identify relationships between quantities.
    4. Prefer symmetry, invariants, ratios, or conserved relationships when present.
    5. Computation should follow insight.
    6. Keep explanations concise.

    --------------------------------
    OUTPUT FORMAT
    --------------------------------

    Return STRICT JSON only.

    {output_schema}
    """

    response = client.models.generate_content(
        model=TEXT_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2
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