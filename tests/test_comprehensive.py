import json
import os
import shutil
import tempfile
import time
import pytest
from src.agent import Agent
from src.memory import MemorySystem
from src.subagents import subagent_manager

@pytest.fixture
def temp_data_dir():
    tmp = tempfile.mkdtemp(prefix="dross_comprehensive_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)

@pytest.fixture
def agent(temp_data_dir):
    # Setup isolated environment
    agent = Agent(data_dir=temp_data_dir)
    # Patch memory to use temp dir as well
    agent.memory = MemorySystem(db_path=os.path.join(temp_data_dir, "memory_db"))
    return agent

def test_goal_and_subtask_lifecycle(agent):
    """Test every facet of goal and subtask management."""
    # 1. Set Goal
    res = agent.tools.execute("set_goal", {"description": "Comprehensive Test Goal"}, context=agent._get_context())
    assert "Goal set" in res

    goal_raw = agent.tools.execute("get_goal", {}, context=agent._get_context())
    goal = json.loads(goal_raw)
    assert goal["goal"] == "Comprehensive Test Goal"
    assert goal["status"] == "active"

    # 2. Add Subtasks
    agent.tools.execute("add_subtask", {"subtask": "Step One"}, context=agent._get_context())
    agent.tools.execute("add_subtask", {"subtask": "Step Two"}, context=agent._get_context())

    goal_raw = agent.tools.execute("get_goal", {}, context=agent._get_context())
    goal = json.loads(goal_raw)
    assert len(goal["subtasks"]) == 2

    task1_id = goal["subtasks"][0]["id"]

    # 3. Complete Subtask
    agent.tools.execute("complete_subtask", {"subtask_id": task1_id}, context=agent._get_context())

    goal_raw = agent.tools.execute("get_goal", {}, context=agent._get_context())
    goal = json.loads(goal_raw)
    assert goal["subtasks"][0]["status"] == "completed"
    assert goal["subtasks"][1]["status"] == "pending"

    # 4. Complete Goal
    agent.tools.execute("complete_goal", {"result_summary": "Test finished"}, context=agent._get_context())
    goal_raw = agent.tools.execute("get_goal", {}, context=agent._get_context())
    assert "No active goal" in goal_raw

def test_memory_and_relational_graph(agent):
    """Test vector memory, relational storage, and graph output."""
    # 1. Save memories
    fid1 = agent.memory.save_long_term("The DROSS system is autonomous.", {"type": "fact"})
    fid2 = agent.memory.save_long_term("DROSS uses ChromaDB for memory.", {"type": "fact"})

    assert fid1 and fid2

    # 2. Save relationship
    agent.memory.save_relationship(fid1, fid2, "relies_on")

    # 3. Retrieve and verify
    relevant = agent.memory.retrieve_relevant("How does DROSS store information?")
    assert "ChromaDB" in relevant

    # 4. Verify graph output
    graph = agent.memory.get_all_memories()
    assert len(graph["nodes"]) >= 2
    assert len(graph["edges"]) >= 1
    assert graph["edges"][0]["label"] == "relies_on"

def test_skill_creation_and_usage(agent):
    """Test the ability to create and then use a new tool."""
    tool_name = "test_calculator"
    tool_code = "def test_calculator(a, b):\n    return f'Result: {a + b}'"

    # 1. Create tool
    res = agent.tools.execute("create_tool", {
        "name": tool_name,
        "description": "Adds two numbers",
        "code": tool_code
    }, context=agent._get_context())
    assert "successfully" in res.lower()

    # 2. Use tool
    res = agent.tools.execute(tool_name, {"a": 10, "b": 5})
    assert "Result: 15" in res

def test_subagent_fleet_and_runtime(agent):
    """Test subagent spawning and fleet status monitoring."""
    goal = "Subagent Comprehensive Test"

    # 1. Spawn subagent
    res = agent.tools.execute("spawn_subagent", {"goal": goal})
    assert "spawned" in res
    subagent_id = res.split("Subagent ")[1].split(" ")[0]

    # Give it a moment to initialize
    time.sleep(1)

    # 2. Check status via manager directly
    status = subagent_manager.get_status(subagent_id)
    assert status["id"] == subagent_id
    assert status["goal"] == goal

    # 3. Check list_all for runtime_seconds
    fleet = subagent_manager.list_all()
    found = False
    for sa in fleet:
        if sa["id"] == subagent_id:
            assert "runtime_seconds" in sa
            assert sa["runtime_seconds"] >= 0
            found = True
    assert found

@pytest.mark.slow
def test_agent_run_goal_lifecycle(agent):
    """Test that Agent.run correctly manages goals for the UI."""
    # Mock models to ensure TOOL intent
    from unittest.mock import patch

    with patch.object(agent.models, "route_request", return_value="TOOL"), \
         patch.object(agent.models, "query_tool", return_value=json.dumps({"tool_name": "list_files", "tool_args": {"path": "."}})), \
         patch.object(agent.models, "query_general", return_value="I have listed the files."):

        # Verify no goal before
        assert "No active goal" in agent.tools.execute("get_goal", {}, context=agent._get_context())

        # Run agent
        agent.run("List my files")

        # In a real scenario, Agent.run would have set and then completed the goal.
        # Since we are checking AFTER run() finishes, it should be empty/completed.
        # But we can verify it WAS set if we mock set_goal and check calls,
        # or just check that it's no longer 'active'.

        res = agent.tools.execute("get_goal", {}, context=agent._get_context())
        assert "No active goal" in res

def test_auto_learning_and_insight(agent):
    """Test the auto-learning logic that extracts facts from interaction."""
    # We'll manually trigger the logic that happens at the end of run()
    user_input = "My favorite color is neon green and I live in a treehouse."
    assistant_response = "I have noted that your favorite color is neon green and you reside in a treehouse."

    # Mock extract_insight to return expected facts
    from unittest.mock import patch
    mock_insight = {
        "facts": ["The user's favorite color is neon green.", "The user lives in a treehouse."],
        "relationships": []
    }

    with patch.object(agent.models, "extract_insight", return_value=mock_insight):
        # We need to satisfy the AUTO_LEARN_MIN_LENGTH check
        # Instead of calling run(), we can just call the part that does the learning
        # or just call run() with mocked route_request
        with patch.object(agent.models, "route_request", return_value="DIRECT"), \
             patch.object(agent.models, "query_general", return_value=assistant_response):

            # Ensure combined length is enough (200 by default)
            # We can temporarily lower the threshold in config if needed,
            # but here we just make the input/output long.
            long_input = user_input + " " * 150
            agent.run(long_input)

            # Check if facts were saved
            memories = agent.memory.get_all_memories()
            contents = [m["content"] for m in memories["nodes"]]
            assert any("neon green" in c for c in contents)
            assert any("treehouse" in c for c in contents)

@pytest.mark.slow
def test_ollama_host_connectivity(agent):
    """Verifies that all configured Ollama hosts are reachable and have models."""
    from src.config import OLLAMA_HOSTS
    health = agent.models.check_health()

    # Check that each host is reachable
    for host in OLLAMA_HOSTS:
        assert health.get(host), f"Ollama host {host} is unreachable."

    # Verify mapping logic
    clients = agent.models.clients
    if len(clients) == 2:
        assert agent.models.reasoning_client.base_url._base_url == clients[0].base_url._base_url
        assert agent.models.general_client.base_url._base_url == clients[0].base_url._base_url
        assert agent.models.tool_client.base_url._base_url == clients[1].base_url._base_url
