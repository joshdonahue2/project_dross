import pytest
import os
import json
from src.agent_graph import DROSSGraph
from src.db import DROSSDatabase
from src.schemas import AgentState, Goal

def test_db_goal_lifecycle():
    db = DROSSDatabase(db_path="data/test_dross.db")
    # Clean up
    if os.path.exists("data/test_dross.db"):
        os.remove("data/test_dross.db")
    db = DROSSDatabase(db_path="data/test_dross.db")

    goal_id = db.set_goal("Test Goal", is_autonomous=True)
    assert goal_id is not None

    active_goal = db.get_active_goal()
    assert active_goal['description'] == "Test Goal"
    assert active_goal['is_autonomous'] == 1

    db.complete_goal(goal_id, "Success")
    active_goal = db.get_active_goal()
    assert active_goal is None # Should be None if no more goals

def test_db_plan_lifecycle():
    db = DROSSDatabase(db_path="data/test_dross.db")
    goal_id = db.set_goal("Plan Goal")

    steps = ["Step 1", "Step 2"]
    db.set_plan(goal_id, steps)

    plan = db.get_plan(goal_id)
    assert len(plan['steps']) == 2
    assert plan['steps'][0]['description'] == "Step 1"

    db.update_plan_step(plan['id'], 0, "completed")
    plan = db.get_plan(goal_id)
    assert plan['steps'][0]['status'] == "completed"

def test_graph_init():
    graph = DROSSGraph()
    assert graph.app is not None

# We won't run live LLM tests here to keep it fast,
# but the DROSSGraph.run was already verified manually.
