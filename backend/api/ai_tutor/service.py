import json
import re
import base64
import os
import uuid

from typing import TypedDict, List, Tuple

from backend.core.gemini_client import client
from backend.core.config import TEXT_MODEL, IMAGE_MODEL
from google.genai import types
from langgraph.graph import StateGraph, END


# ==================================================
# SESSION STORE
# ==================================================

sessions = {}

# ==================================================
# LOAD POLICY
# ==================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
POLICY_PATH = os.path.join(BASE_DIR, "multimodal_agent1_reasoning_style.json")

def load_policy():
    with open(POLICY_PATH) as f:
        return json.load(f)

policy = load_policy()

MENTOR_MODEL = TEXT_MODEL
SOLVER_MODEL = TEXT_MODEL


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
# GEMINI CALL
# ==================================================

def call_gemini(system_prompt, user_prompt, temperature=0.4):

    response = client.models.generate_content(
        model=MENTOR_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature
        ),
        contents=user_prompt
    )

    return response.text.strip()


# ==================================================
# BLUEPRINT GENERATION
# ==================================================

def generate_blueprint(problem):

    system_prompt = f"""
You are an expert mentor with deep structural intuition.

Use this reasoning policy strictly:
{json.dumps(policy)}

Return STRICT JSON:

{{
 "reasoning_stages": [],
 "valid_alternative_paths": [],
 "common_mistakes": [],
 "final_answer": ""
}}
"""

    raw = call_gemini(
        system_prompt,
        f"Problem:\n{problem}",
        temperature=0
    )

    cleaned = re.sub(r"```json|```", "", raw).strip()

    try:
        return json.loads(cleaned)
    except:
        return {}


# ==================================================
# FINAL ANSWER INTENT
# ==================================================

def detect_final_answer_intent(problem, text):

    prompt = f"""
Is the learner explicitly concluding with a final answer?

Problem:
{problem}

Message:
{text}

YES or NO
"""

    res = call_gemini(
        "Detect answer intent",
        prompt,
        temperature=0
    )

    return "yes" in res.lower()


# ==================================================
# FINAL ANSWER CORRECTNESS
# ==================================================

def check_final_answer_correctness(blueprint, text):

    correct_answer = blueprint.get("final_answer", "")

    if correct_answer and correct_answer.lower() in text.lower():
        return True

    return False


# ==================================================
# BLUEPRINT ALIGNMENT
# ==================================================

def check_blueprint_alignment(problem, chat, blueprint):

    prompt = f"""
Check the learner's thinking style.

Blueprint:
{json.dumps(blueprint)}

Problem:
{problem}

Conversation:
{chat}

Return JSON:

{{
 "policy_alignment":"YES or NO",
 "learner_state_summary":"..."
}}
"""

    res = call_gemini(
        "Evaluate reasoning style",
        prompt,
        temperature=0
    )

    cleaned = re.sub(r"```json|```", "", res).strip()

    try:
        parsed = json.loads(cleaned)

        aligned = parsed["policy_alignment"].lower() == "yes"
        summary = parsed["learner_state_summary"]

    except:
        aligned = False
        summary = ""

    return aligned, summary


# ==================================================
# MENTOR NODE
# ==================================================

def mentor_node(state: AgentState) -> AgentState:

    alignment, learner_state = check_blueprint_alignment(
        state["problem"],
        state["chat"],
        state["mentor_blueprint"]
    )

    state["policy_alignment"] = alignment
    state["learner_state"] = learner_state

    if state["correct_answer_given_once"] and not state["policy_alignment"]:

        explanation_prompt = (
            "Great! You have the correct answer. "
            "Now explain your reasoning step by step."
        )

        state["chat"].append(("assistant", explanation_prompt))

        return state

    system_prompt = f"""
You are Naveen — a structural thinking mentor.

Reasoning Policy:
{json.dumps(policy)}

Blueprint:
{json.dumps(state["mentor_blueprint"])}

Learner State:
{state["learner_state"]}

Rules:
- Do NOT reveal the final answer
- Ask structural questions
"""

    user_prompt = f"""
Problem:
{state["problem"]}

Conversation:
{state["chat"]}
"""

    reply = call_gemini(
        system_prompt,
        user_prompt
    )

    state["chat"].append(("assistant", reply))

    return state


# ==================================================
# SOLVER NODE
# ==================================================

def solver_node(state: AgentState) -> AgentState:

    prompt = f"""
The learner has demonstrated correct reasoning.

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

    return state


# ==================================================
# ROUTER
# ==================================================

def router(state: AgentState):

    if not state["policy_alignment"]:
        return "mentor"

    if not state["final_answer_intent"]:
        return "mentor"

    if not state["final_answer_correct"]:
        return "mentor"

    return "solver"


# ==================================================
# STRUCTURAL DECONSTRUCTION
# ==================================================

def generate_deconstruction(problem: str):

    system_prompt = f"""
Use this reasoning policy:
{json.dumps(policy)}

Return JSON with:

final_answer
final_explanation
key_reasoning_lessons
"""

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=f"{system_prompt}\n\nProblem:\n{problem}",
        config=types.GenerateContentConfig(temperature=0)
    )

    raw_text = response.text.strip()

    cleaned = re.sub(r"```json|```", "", raw_text).strip()

    try:
        return json.loads(cleaned)
    except:
        return None


# ==================================================
# IMAGE GENERATION
# ==================================================

def generate_visual_image(problem: str, structured_data: dict):

    prompt = f"""
Create educational diagram.

Problem:
{problem}

Structure:
{json.dumps(structured_data)}
"""

    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=prompt
    )

    for candidate in response.candidates:
        for part in candidate.content.parts:
            if part.inline_data:

                return {
                    "image_base64": base64.b64encode(part.inline_data.data).decode(),
                    "mime_type": part.inline_data.mime_type
                }

    return None


# ==================================================
# CHAT SESSION START
# ==================================================

def start_chat(problem):

    blueprint = generate_blueprint(problem)

    session_id = str(uuid.uuid4())

    state = {

        "problem": problem,
        "chat": [("assistant","Alright 🙂 Let’s begin. What do you notice first?")],
        "images": [],
        "mentor_blueprint": blueprint,

        "policy_alignment": False,
        "final_answer_intent": False,
        "final_answer_correct": False,

        "solved": False,
        "learner_state": "",
        "correct_answer_given_once": False
    }

    sessions[session_id] = state

    return session_id, state["chat"][-1][1]


# ==================================================
# CHAT MESSAGE
# ==================================================

def chat_message(session_id, message):

    state = sessions.get(session_id)

    if not state:
        return None

    state["chat"].append(("user", message))

    intent = detect_final_answer_intent(state["problem"], message)

    state["final_answer_intent"] = intent

    if intent:

        correct = check_final_answer_correctness(
            state["mentor_blueprint"],
            message
        )

        state["final_answer_correct"] = correct

        if correct:
            state["correct_answer_given_once"] = True

    alignment, learner_state = check_blueprint_alignment(
        state["problem"],
        state["chat"],
        state["mentor_blueprint"]
    )

    state["policy_alignment"] = alignment
    state["learner_state"] = learner_state

    result = app_graph.invoke(state)

    sessions[session_id] = result

    return result["chat"][-1][1], result["solved"]


# ==================================================
# LANGGRAPH
# ==================================================

graph = StateGraph(AgentState)

graph.add_node("mentor", mentor_node)
graph.add_node("solver", solver_node)

graph.set_conditional_entry_point(
    router,
    {
        "mentor": "mentor",
        "solver": "solver"
    }
)

graph.add_edge("mentor", END)
graph.add_edge("solver", END)

app_graph = graph.compile()