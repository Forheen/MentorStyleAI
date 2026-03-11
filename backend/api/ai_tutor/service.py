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
from backend.core.redis_client import redis_client


# ==================================================
# SESSION STORE
# ==================================================


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

def call_gemini(purpose, system_prompt, user_prompt, temperature=0.4):

  

    response = client.models.generate_content(
        model=MENTOR_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature
        ),
        contents=user_prompt
    )

    text = response.text.strip()

   

    return text


# ==================================================
# BLUEPRINT GENERATION
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
        "FINAL_ANSWER_INTENT",
        "Detect answer intent only.",
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

    except:
        aligned = False
        summary = "Unable to parse learner state."

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
            "Now try to explain your reasoning step by step in a deconstructive manner."
        )

        state["chat"].append(("assistant", explanation_prompt))

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

    return state


# ==================================================
# SOLVER NODE
# ==================================================

def solver_node(state: AgentState) -> AgentState:

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

def generate_deconstruction(problem):
    output_schema = """
    {
     "normal_explanation": "A normal explanation of the solution with answer, without emphasizing structural insights.",
    "final_explanation": "...",
    "key_reasoning_lessons": [
        "...",
        "..."
    ],
    "final_answer": "..."
    }
    """

    system_prompt = f"""
    You are Navin — a structural thinking mentor.

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

    redis_client.set(
      f"chat:{session_id}",
      json.dumps(state),
      ex=3600
    )

    return session_id, state["chat"][-1][1]


# ==================================================
# CHAT MESSAGE
# ==================================================

def chat_message(session_id, message):

    data = redis_client.get(f"chat:{session_id}")

    if not data:
        return "Session expired. Please restart the chat.", False

    state = json.loads(data)

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

    # Save updated session
    redis_client.set(
        f"chat:{session_id}",
        json.dumps(result),
        ex=3600
    )

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

