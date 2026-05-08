from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from langgraph.constants import END
from langgraph.graph import StateGraph

from agent.prompts import *
from agent.states import *
import os

_ = load_dotenv()

MODEL = os.getenv("MODEL")

# llm = ChatGroq(model="llama-3.3-70b-versatile")

# The Planner needs variance to brainstorm and structure the app.
planner_llm = ChatGroq(model=MODEL, temperature=0.7)

# The Architect needs to be grounded but still capable of translating concepts to structure.
architect_llm = ChatGroq(model=MODEL, temperature=0.2)

# The Coder must be entirely deterministic. Zero creativity. Pure syntax and logic.
coder_llm = ChatGroq(model=MODEL, temperature=0.0)


def _log(debug: bool, header: str, body: str) -> None:
    if not debug:
        return
    print(header)
    print(body)
# llm = ChatGroq(model="qwen/qwen3-32b")


def planner_agent(state: dict) -> dict:
    """Converts user prompt into a structured plan."""
    user_prompt = state["user_prompt"]
    debug = bool(state.get("debug"))

    structured_llm = planner_llm.with_structured_output(Plan, method="json_mode")
    resp: Plan = structured_llm.invoke([
        {"role": "system", "content": planner_prompt(user_prompt)},
    ])

    if resp is None:
        raise ValueError("Planner didn't return a response.")
    _log(debug, "==== PLANNER OUTPUT ====", resp.model_dump_json())
    return {**state, "plan": resp}



def architect_agent(state: dict) -> dict:
    """Creates TaskPlan from Plan using structured output (no tool calls)."""
    plan: Plan = state["plan"]
    debug = bool(state.get("debug"))

    structured_llm = architect_llm.with_structured_output(TaskPlan, method="json_mode")
    resp: TaskPlan = structured_llm.invoke([
        {"role": "system", "content": architect_prompt(plan=plan.model_dump_json())},
    ])

    if resp is None:
        raise ValueError("Architect didn't return a valid response.")
    if not resp.implementation_steps:
        raise ValueError("Architect returned an empty task plan — no implementation steps.")

    _log(debug, "==== ARCHITECT OUTPUT ====", resp.model_dump_json())
    return {**state, "task_plan": resp}


def coder_agent(state: dict) -> dict:
    """LangGraph coder agent using structured output (CoderOutput)."""
    coder_state: CoderState = state.get("coder_state")
    debug = bool(state.get("debug"))

    if coder_state is None:
        coder_state = CoderState(task_plan=state["task_plan"], current_step_idx=0)

    steps = coder_state.task_plan.implementation_steps
    if coder_state.current_step_idx >= len(steps):
        return {**state, "coder_state": coder_state, "status": "DONE"}

    if debug:
        print("==== CODER STEP 1 ====")
        print("File: (full codebase)")
        print("Task: Generate complete codebase JSON")

    plan: Plan = state["plan"]
    user_prompt = (
        f"Plan:\n{plan.model_dump_json()}\n\n"
        f"Task Plan:\n{coder_state.task_plan.model_dump_json()}"
    )

    structured_llm = coder_llm.with_structured_output(CoderOutput, method="json_mode")
    try:
        resp: CoderOutput = structured_llm.invoke(
            [
                {"role": "system", "content": coder_system_prompt()},
                {"role": "user", "content": user_prompt},
            ]
        )
        if resp is None:
            raise ValueError("Coder didn't return a response.")

        files = [f.model_dump() for f in resp.files]
        if not files:
            raise ValueError("No files generated")

        coder_state.generated_files = files
        coder_state.current_step_idx = len(coder_state.task_plan.implementation_steps)

        _log(debug, "==== CODER OUTPUT ====", resp.model_dump_json())
        return {**state, "coder_state": coder_state, "status": "DONE"}
    except Exception as e:
        print("==== CODER FAILURE ====")
        print(str(e))
        return {
            **state,
            "coder_state": coder_state,
            "status": "DONE",
            "error": {
                "error": "Code generation failed",
                "details": str(e),
            },
        }

graph = StateGraph(dict)

graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)


graph.add_edge("planner", "architect")
graph.add_edge("architect", "coder")
graph.add_conditional_edges(
    "coder",
    lambda s: "END" if s.get("status") == "DONE" else "coder",
    {"END": END, "coder": "coder"}
)

graph.set_entry_point("planner")
agent = graph.compile()


def generate_codebase(user_prompt: str) -> dict:
    result = agent.invoke({"user_prompt": user_prompt, "debug": False}, {"recursion_limit": 100})
    if "error" in result:
        return {
            "success": False,
            **result["error"],
        }
    coder_state: CoderState | None = result.get("coder_state")
    files = coder_state.generated_files if coder_state else []
    return {
        "success": True,
        "coder_state": coder_state,
        "files": files,
    }


def generate_codebase_debug(user_prompt: str, debug: bool = True) -> dict:
    result = agent.invoke({"user_prompt": user_prompt, "debug": debug}, {"recursion_limit": 100})
    if "error" in result:
        if debug:
            print("==== FINAL OUTPUT ====")
            print(result["error"])
        return {
            "success": False,
            **result["error"],
        }
    coder_state: CoderState | None = result.get("coder_state")
    payload = {
        "success": True,
        "files": coder_state.generated_files if coder_state else [],
    }
    if debug:
        print("==== FINAL OUTPUT ====")
        print(payload)
    return payload
