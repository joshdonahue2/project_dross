import pytest
import os
import json
from src.agent_graph import DROSSGraph
from src.db import DROSSDatabase

def test_db_goal_lifecycle():
    # Use a temporary test database
    test_db_path = "data/test_comprehensive.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    db = DROSSDatabase(db_path=test_db_path)

    # Test setting a goal
    goal_id = db.set_goal("Comprehensive Test Goal", is_autonomous=True)
    assert goal_id is not None

    # Test retrieval
    active_goal = db.get_active_goal()
    assert active_goal['description'] == "Comprehensive Test Goal"
    assert active_goal['is_autonomous'] == 1

    # Test plan creation
    steps = ["Analyze environment", "Execute task", "Reflect"]
    db.set_plan(goal_id, steps)
    plan = db.get_plan(goal_id)
    assert len(plan['steps']) == 3
    assert plan['steps'][0]['description'] == "Analyze environment"
    assert plan['steps'][0]['status'] == "pending"

    # Test step update
    db.update_plan_step(plan['id'], 0, "completed")
    plan = db.get_plan(goal_id)
    assert plan['steps'][0]['status'] == "completed"

    # Test goal completion
    db.complete_goal(goal_id, "Task successful")
    active_goal = db.get_active_goal()
    assert active_goal is None

    if os.path.exists(test_db_path):
        os.remove(test_db_path)

def test_graph_structure():
    graph = DROSSGraph()
    assert graph.app is not None
    # Check if all expected nodes are present
    nodes = graph.workflow.nodes
    expected_nodes = ["router", "reasoner", "tool_executor", "synthesizer", "reflector"]
    for node in expected_nodes:
        assert node in nodes

def test_model_manager_init():
    from src.models import ModelManager
    mm = ModelManager()
    assert mm.reasoning_model is not None
    assert mm.general_model is not None
    assert mm.tool_model is not None

def test_db_ui_persistence():
    test_db_path = "data/test_persistence.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    db = DROSSDatabase(db_path=test_db_path)

    # Test messages
    db.add_message("user", "Hello")
    db.add_message("assistant", "Hi there")
    messages = db.get_messages()
    assert len(messages) == 2
    assert messages[0]['role'] == "user"
    assert messages[1]['content'] == "Hi there"

    # Test logs
    db.add_activity_log("log", "System started")
    db.add_activity_log("tool", "Running test")
    logs = db.get_activity_logs()
    assert len(logs) == 2
    assert logs[0]['type'] == "tool" # DESC order

    # Test clear
    db.clear_ui_state()
    assert len(db.get_messages()) == 0
    assert len(db.get_activity_logs()) == 0

    if os.path.exists(test_db_path):
        os.remove(test_db_path)
