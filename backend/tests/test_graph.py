import pytest
from unittest.mock import AsyncMock, MagicMock
from agent.graph import planner_agent, architect_agent, coder_agent
from agent.states import Plan, File, TaskPlan, ImplementationTask

@pytest.mark.asyncio
async def test_planner_agent(mocker):
    # Mock the LLM entirely
    mock_llm = mocker.patch("agent.graph.planner_llm")
    mock_ainvoke = AsyncMock()
    mock_ainvoke.return_value = Plan(
        name="React app",
        description="A simple React app",
        techstack="React",
        features=["feature 1"],
        files=[File(path="src/App.tsx", purpose="Main component")]
    )
    mock_llm.with_structured_output.return_value.ainvoke = mock_ainvoke

    state = {"user_prompt": "build a react app", "build_status": "in_progress"}
    new_state = await planner_agent(state)
    
    assert "plan" in new_state
    assert new_state["plan"].name == "React app"
    assert len(new_state["plan"].files) == 1

@pytest.mark.asyncio
async def test_architect_agent(mocker):
    mock_llm = mocker.patch("agent.graph.architect_llm")
    mock_ainvoke = AsyncMock()
    mock_ainvoke.return_value = TaskPlan(
        implementation_steps=[ImplementationTask(filepath="src/App.tsx", task_description="Write component")]
    )
    mock_llm.with_structured_output.return_value.ainvoke = mock_ainvoke

    state = {
        "user_prompt": "app", 
        "build_status": "in_progress",
        "plan": Plan(
            name="Node", description="desc", techstack="Node.js", features=[], files=[]
        )
    }
    
    new_state = await architect_agent(state)
    assert "task_plan" in new_state
    assert len(new_state["task_plan"].implementation_steps) == 1

@pytest.mark.asyncio
async def test_planner_agent_missing_response(mocker):
    mock_llm = mocker.patch("agent.graph.planner_llm")
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=None)
    
    state = {"user_prompt": "app"}
    
    with pytest.raises(ValueError, match="Planner didn't return a response."):
        await planner_agent(state)
