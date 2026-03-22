from dotenv import  load_dotenv
from langchain_groq.chat_models import ChatGroq
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent


from agent.prompts import *
from agent.states import *
from agent.tools import write_file, read_file, get_current_directory, list_files


_ = load_dotenv()



# llm = ChatGroq(model="openai/gpt-oss-120b")
# llm = ChatGroq(model="llama3-70b-8192")
llm = ChatGroq(model="llama-3.3-70b-versatile")
# llm = ChatGroq(model="qwen/qwen3-32b")

def planner_agent(state: dict) -> dict:
    """Converts user prompt into a structured plan."""
    user_prompt = state["user_prompt"]
    resp = llm.with_structured_output(Plan).invoke(
        planner_prompt(user_prompt)
    )

    if resp is None:
        raise ValueError("Planner didn't return a response.")
    return {"plan": resp}



def architect_agent(state: dict) -> dict:
    """Creates TaskPlan form Plan."""
    plan: Plan = state["plan"]
    resp = llm.with_structured_output(TaskPlan).invoke(
        architect_prompt(plan=plan.model_dump_json())
    )
    if resp is None:
        raise ValueError("Planner didn't return a valid response..")

    resp.plan = plan
    print(resp.model_dump_json())
    return {"task_plan" : resp}


coder_tools = [read_file, write_file, list_files, get_current_directory]
react_agent = create_react_agent(llm, coder_tools)

def coder_agent(state: dict) -> dict:
    """LangGraph tool-using coder agent."""
    coder_state: CoderState = state.get("coder_state")

    if coder_state is None:
        coder_state = CoderState(task_plan=state["task_plan"], current_step_idx=0)

    steps = coder_state.task_plan.implementation_steps
    if coder_state.current_step_idx >= len(steps):
        return {"coder_state": coder_state, "status": "DONE"}

    current_task = steps[coder_state.current_step_idx]
    existing_content = read_file.run(current_task.filepath)

    system_prompt = coder_system_prompt()
    user_prompt = (
        f"Task: {current_task.task_description}\n"
        f"File: {current_task.filepath}\n"
        f"Existing content:\n{existing_content}\n"
        "Use write_file(path, content) to save your changes."
    )
    
    print(f"--- 🚀 Coder Agent starting Task {coder_state.current_step_idx + 1}/{len(steps)} ---")


    # Capture the result so you can debug what the sub-agent actually did
    react_result = react_agent.invoke({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    })
    
    # Print the final message from the ReAct agent
    print("Coder Agent says:", react_result["messages"][-1].content)
    
    coder_state.current_step_idx += 1
    return {"coder_state": coder_state}

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

if __name__ == "__main__":
    result = agent.invoke({"user_prompt": "build a calculator app with a nice UI using HTML, CSS, and JavaScript to do basic arthematic calculation."},
                          {"recursion_limit": 100})
    print("Final State:", result)
