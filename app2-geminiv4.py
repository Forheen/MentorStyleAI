import streamlit as st
import json
from typing import TypedDict, List, Tuple
from datetime import datetime
import base64
import re
from langgraph.graph import StateGraph, END
from google import genai
from google.genai import types

# ==================================================
# CONFIG
# ==================================================
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=GEMINI_API_KEY)

MENTOR_MODEL = "gemini-3-flash-preview"
SOLVER_MODEL = "gemini-3-flash-preview"
# IMAGE_MODEL = "gemini-3-pro-image-preview"   # 🔥 Commented

st.set_page_config(
    page_title="AI Guided Thinking Mentor",
    page_icon="🧠",
    layout="centered"
)

# ==================================================
# LOGGING (FULL — NO TRUNCATION)
# ==================================================
def log(step, data=None):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({
        "time": ts,
        "step": step,
        "data": data
    })

# ==================================================
# POLICY
# ==================================================
@st.cache_resource
def load_policy():
    with open("multimodal_agent1_reasoning_style.json") as f:
        return json.load(f)

policy = load_policy()

# ==================================================
# STATE
# ==================================================
class AgentState(TypedDict):
    problem: str
    chat: List[Tuple[str, str]]
    images: List
    mentor_blueprint: dict
    policy_alignment: bool
    final_answer_intent: bool
    final_answer_correct: bool
    solved: bool
    learner_state: str
    correct_answer_given_once: bool

# ==================================================
# LLM CALL (FULL LOG)
# ==================================================
def call_gemini(purpose, system_prompt, user_prompt, temperature=0.4):

    log("LLM_CALL_START", {
        "purpose": purpose,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "temperature": temperature
    })

    response = client.models.generate_content(
        model=MENTOR_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature
        ),
        contents=user_prompt
    )

    text = response.text.strip()

    log("LLM_CALL_END", {
        "purpose": purpose,
        "response": text
    })

    return text

# ==================================================
# IMAGE GENERATION (COMMENTED OUT)
# ==================================================
# def generate_visual(problem, learner_state, mentor_text):
#
#     try:
#         image_prompt = f"""
# Create an image that can visualize in following manner:
#
# Problem:
# {problem}
#
# Learner State:
# {learner_state}
#
# Mentor Guidance:
# {mentor_text}
#
# Rules:
# - Educational style
# - Use pint pont text , no lengthy or explaining long text
# - Use mentor's guide to help the learner to visualize
# - Do NOT reveal final answer but grasp all the main points.
# - Be funny or sarcasm if required
# """
#
#         response = client.models.generate_content(
#             model=IMAGE_MODEL,
#             contents=image_prompt,
#             config=types.GenerateContentConfig(temperature=0)
#         )
#
#         for candidate in response.candidates:
#             for part in candidate.content.parts:
#                 if part.inline_data and part.inline_data.mime_type.startswith("image/"):
#                     return {
#                         "image_base64": base64.b64encode(part.inline_data.data).decode("utf-8"),
#                         "mime_type": part.inline_data.mime_type
#                     }
#
#     except Exception as e:
#         log("IMAGE_ERROR", str(e))
#
#     return None

# ==================================================
# BLUEPRINT GENERATION (UNCHANGED)
# ==================================================
def generate_blueprint(problem):
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

    raw = call_gemini(
        "BLUEPRINT_GENERATION",
        system_prompt,
        f"Problem:\n{problem}",
        temperature=0,

    )

    cleaned = re.sub(r"```json|```", "", raw).strip()

    try:
        blueprint = json.loads(cleaned)
    except Exception as e:
        log("BLUEPRINT_JSON_ERROR", str(e))
        blueprint = {}

    log("MENTOR_BLUEPRINT_GENERATED", blueprint)

    return blueprint

# ==================================================
# BLUEPRINT ALIGNMENT (UNCHANGED)
# ==================================================
def check_blueprint_alignment(problem, chat, blueprint):

    prompt = f"""
Check the learner's THINKING STYLE and compare the alignment of the chat with the blueprint derived from the policy alignment

Blueprint:
{json.dumps(blueprint)}

Problem:
{problem}

Conversation:
{chat}

Return JSON:
{{
  "policy_alignment": "YES or NO",
  "learner_state_summary": "Concise summary"
}}
"""

    res = call_gemini(
        "BLUEPRINT_ALIGNMENT",
        "You evaluate reasoning style strictly.",
        prompt,
        temperature=0
    )

    cleaned = re.sub(r"```json|```", "", res).strip()

    try:
        parsed = json.loads(cleaned)
        aligned = parsed["policy_alignment"].lower() == "yes"
        summary = parsed["learner_state_summary"]
    except Exception as e:
        log("ALIGNMENT_JSON_ERROR", str(e))
        aligned = False
        summary = "Unable to parse learner state."

    log("BLUEPRINT_ALIGNMENT_RESULT", {
        "aligned": aligned,
        "learner_state_summary": summary
    })

    return aligned, summary

# ==================================================
# FINAL ANSWER INTENT (UNCHANGED)
# ==================================================
def detect_final_answer_intent(problem, text):

    prompt = f"""
Is the learner explicitly concluding with a final answer?

Problem:
{problem}

Message:
{text}

YES or NO.
"""

    res = call_gemini(
        "FINAL_ANSWER_INTENT",
        "Detect answer intent only.",
        prompt,
        temperature=0
    )

    intent = "yes" in res.lower()

    log("FINAL_ANSWER_INTENT_RESULT", intent)

    return intent

# ==================================================
# FINAL ANSWER CORRECTNESS (UNCHANGED)
# ==================================================
def check_final_answer_correctness(blueprint, text):

    correct_answer = blueprint.get("final_answer", "")

    result = False
    if correct_answer and correct_answer.lower() in text.lower():
        result = True

    log("FINAL_ANSWER_CORRECTNESS_RESULT", {
        "expected": correct_answer,
        "user_text": text,
        "result": result
    })

    return result

# ==================================================
# ROUTER (UNCHANGED)
# ==================================================
def router(state: AgentState):
    log("ROUTER_STATE", state)

    if not state["policy_alignment"]:
        log("ROUTER_DECISION", "mentor (fix thinking)")
        return "mentor"

    if not state["final_answer_intent"]:
        log("ROUTER_DECISION", "mentor (continue thinking)")
        return "mentor"

    if not state["final_answer_correct"]:
        log("ROUTER_DECISION", "mentor (fix final step)")
        return "mentor"

    log("ROUTER_DECISION", "solver")
    return "solver"


# ==================================================
# MENTOR NODE (IMAGE CALL COMMENTED)
# ==================================================
def mentor_node(state: AgentState) -> AgentState:

    log("MENTOR_ENTER")

    
    # Otherwise, do normal alignment check
    alignment, learner_state = check_blueprint_alignment(
        state["problem"], state["chat"], state["mentor_blueprint"]
    )
    state["policy_alignment"] = alignment
    state["learner_state"] = learner_state
     # If user has given correct answer before but alignment is not yet True
    if state["correct_answer_given_once"] and not state["policy_alignment"]:
        explanation_prompt = (
            "Great! You have the correct answer. "
            "Now, can you explain your reasoning step by step?"
        )
        state["chat"].append(("assistant", explanation_prompt))
        log("MENTOR_REQUEST_EXPLANATION", explanation_prompt)
        return state

    system_prompt = f"""
    You are Naveen — a structural thinking mentor.

    Reasoning Policy:
    {json.dumps(policy)}

    Blueprint:
    {json.dumps(state["mentor_blueprint"])}

    Current Learner Cognitive State:
    {state["learner_state"]}

    Policy Alignment:
    {"ALIGNED" if state["policy_alignment"] else "MISALIGNED"}

    Rules:
    - Do NOT reveal the final answer.
    - Do NOT compute unnecessarily.
    - Do NOT introduce formulas unless structurally necessary.
    - Ask sharp structural questions.
    - Target the learner's cognitive gap directly.
    - Push conceptual alignment before procedural steps.

    If MISALIGNED:
        - Identify where their thinking drifted structurally.
        - Redirect toward invariants, symmetry, structure.
        - Ask them to pause and rethink

    If ALIGNED but incomplete:
        - Help compress insight.
        - Refine their internal model.

    Your goal:
    Shift the learner's mental model closer to the blueprint structure and ask questions according to policy that will lead the learner.
    """
    user_prompt = f"""
    Problem:
    {state["problem"]}

    Conversation:
    {state["chat"]}
    """

    reply = call_gemini(
        "MENTOR_RESPONSE",
        system_prompt,
        user_prompt
    )

    state["chat"].append(("assistant", reply))

    # 🔥 IMAGE CALL COMMENTED
    # image = generate_visual(
    #     state["problem"],
    #     state["learner_state"],
    #     reply
    # )
    # state["images"].append(image)

    log("MENTOR_OUTPUT", reply)

    return state

# ==================================================
# SOLVER NODE (UNCHANGED)
# ==================================================
def solver_node(state: AgentState) -> AgentState:

    log("SOLVER_ENTER")

    prompt = f"""
The learner has demonstrated policy-aligned thinking
and a correct final answer.

Problem:
{state["problem"]}

Conversation:
{state["chat"]}

Provide:
1. Final explanation
2. Key reasoning lessons
3. Final answer
"""

    out = client.models.generate_content(
        model=SOLVER_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2)
    ).text.strip()

    state["chat"].append(("assistant", out))
    state["solved"] = True

    log("SOLVER_OUTPUT", out)

    return state

# ==================================================
# GRAPH (UNCHANGED)
# ==================================================
graph = StateGraph(AgentState)
graph.add_node("mentor", mentor_node)
graph.add_node("solver", solver_node)

graph.set_conditional_entry_point(
    router,
    {"mentor": "mentor", "solver": "solver"}
)

graph.add_edge("mentor", END)
graph.add_edge("solver", END)

app_graph = graph.compile()

# ==================================================
# SESSION INIT (UNCHANGED)
# ==================================================
defaults = {
    "started": False,
    "thinking": False,
    "problem": "",
    "chat": [],
    "images": [],
    "mentor_blueprint": {},
    "policy_alignment": False,
    "final_answer_intent": False,
    "final_answer_correct": False,
    "solved": False,
    "learner_state": "",
    "logs": [],
    "correct_answer_given_once": False
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==================================================
# UI (UNCHANGED)
# ==================================================
st.title("🧠 AI Guided Thinking Mentor")

with st.sidebar:
    st.header("🪵 Full Logs")
    for entry in st.session_state.logs:
        st.json(entry)

if not st.session_state.started:
    p = st.text_area("Enter the problem")

    if st.button("Start") and p:

        log("SESSION_START", p)

        st.session_state.problem = p
        st.session_state.started = True

        blueprint = generate_blueprint(p)
        st.session_state.mentor_blueprint = blueprint

        initial_text = "Alright 🙂 Let’s begin. What do you notice first?"
        st.session_state.chat.append(("assistant", initial_text))

        st.rerun()

if st.session_state.started:

    st.info(st.session_state.problem)

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    if st.session_state.solved:
        st.success("🎉 Session complete.")
        st.stop()

    if not st.session_state.thinking:
        user_input = st.chat_input("Share your thinking")
    else:
        st.chat_input("Naveen is thinking…", disabled=True)
        user_input = None

    if user_input:
        log("USER_INPUT", user_input)
        st.session_state.chat.append(("user", user_input))

        # Detect if user is giving final answer
        intent = detect_final_answer_intent(st.session_state.problem, user_input)
        st.session_state.final_answer_intent = intent

        if intent:
            correct = check_final_answer_correctness(
                   st.session_state.mentor_blueprint, user_input
                )
            st.session_state.final_answer_correct = correct

             # Persist across turns if correct
            if correct:
                st.session_state.correct_answer_given_once = True

        # Even if user only explains a step, check alignment
        alignment, learner_state = check_blueprint_alignment(
            st.session_state.problem, st.session_state.chat, st.session_state.mentor_blueprint
        )
        st.session_state.policy_alignment = alignment
        st.session_state.learner_state = learner_state

        st.session_state.thinking = True
        st.rerun()


if st.session_state.thinking:

    log("GRAPH_EXEC_START")

    result = app_graph.invoke({
        "problem": st.session_state.problem,
        "chat": st.session_state.chat,
        "images": st.session_state.images,
        "mentor_blueprint": st.session_state.mentor_blueprint,
        "policy_alignment": st.session_state.policy_alignment,
        "final_answer_intent": st.session_state.final_answer_intent,
        "final_answer_correct": st.session_state.final_answer_correct,
        "solved": False,
        "learner_state": st.session_state.learner_state,
            "correct_answer_given_once": st.session_state.correct_answer_given_once 
    })

    for k in result:
        st.session_state[k] = result[k]

    st.session_state.thinking = False

    log("GRAPH_EXEC_END")

    st.rerun()
