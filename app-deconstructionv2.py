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
    You are Naveen — a structural thinking mentor.

    Reasoning Policy:
    {json.dumps(policy)}

    Your task is to first internally derive the structural blueprint of the
    problem and then produce a clear mentor-style explanation.

    ------------------------------------------------
    INTERNAL PROCESS (DO NOT OUTPUT)
    ------------------------------------------------

    1. Deconstruct the problem structure.
    2. Identify invariants, ratios, symmetries, or conserved quantities.
    3. Determine the structural reasoning path that solves the problem.
    4. Only after identifying the structural insight, compute the final result.

    Important:
    Do NOT expose these internal reasoning stages in the output.

    ------------------------------------------------
    EXPLANATION STYLE
    ------------------------------------------------

    Provide a clear mentor-style explanation that reveals the key insight
    that unlocks the problem.

    The explanation should feel like a mentor synthesizing the reasoning,
    not like a step-by-step procedural solution.

    Prefer conceptual explanations such as:

    • identifying invariant quantities
    • using ratios or proportional reasoning
    • recognizing structural shortcuts
    • explaining why the insight works

    Avoid mechanical algebra unless absolutely necessary.

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
        st.error("Failed to generate output.")
        return

    st.title("🧠 Structural Explanation")

    st.markdown("---")

    # Final Explanation
    if data.get("final_explanation"):
        st.subheader("📘 Final Explanation")
        st.write(data["final_explanation"])
        st.markdown("---")

    # Key Reasoning Lessons
    if data.get("key_reasoning_lessons"):
        st.subheader("🧠 Key Reasoning Lessons")
        for lesson in data["key_reasoning_lessons"]:
            st.markdown(f"- {lesson}")
        st.markdown("---")

    # Final Answer
    if data.get("final_answer"):
        st.subheader("🎯 Final Answer")
        st.success(data["final_answer"])
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