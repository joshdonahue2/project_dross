
import json
import unittest
from unittest.mock import MagicMock, patch
from src.agent import Agent
from src.models import ModelManager

class TestPlanning(unittest.TestCase):
    @patch('src.models.Client')
    def test_generate_plan_with_tools(self, mock_client):
        # Mock the reasoning model to return a plan that includes spawning sub-agents
        mock_response = {
            'message': {
                'content': json.dumps([
                    "List current top 2 trending repositories on GitHub",
                    "spawn_subagent to research repository 1",
                    "spawn_subagent to research repository 2",
                    "Summarize results and save to github_research folder"
                ])
            }
        }
        mock_client.return_value.chat.return_value = mock_response
        
        manager = ModelManager()
        tools_schema = "spawn_subagent(goal: str): Spawns a new autonomous subagent to work on a specific goal."
        
        plan = manager.generate_plan("Research top 2 GitHub repos", tools_schema=tools_schema)
        
        print(f"\nGenerated Plan: {plan}")
        
        self.assertIn("spawn_subagent", plan[1])
        self.assertIn("spawn_subagent", plan[2])
        
    @patch('src.agent.tool_registry')
    @patch('src.agent.ModelManager')
    def test_agent_heartbeat_passes_tools(self, mock_model_manager_class, mock_registry):
        # Verify Agent.heartbeat passes tools_schema to generate_plan
        mock_model_manager = mock_model_manager_class.return_value
        mock_gen_plan = mock_model_manager.generate_plan
        mock_gen_plan.return_value = ["step1"]
        
        mock_agent = Agent()
        # Mock tools.execute for the heartbeat loop
        mock_agent.tools.execute = MagicMock(side_effect=[
            json.dumps({"goal": "Research repos", "status": "active"}), # get_goal
            "No plan defined" # get_plan
        ] + [json.dumps({"steps": [{"description": "step1", "status": "pending"}]})] * 10)
        
        mock_agent.heartbeat()
        
        # Check if generate_plan was called with tools_schema
        # It should have been called during heartbeat when no plan was defined
        self.assertTrue(mock_gen_plan.called)
        args, kwargs = mock_gen_plan.call_args
        self.assertIn("tools_schema", kwargs)
        self.assertTrue(len(kwargs["tools_schema"]) > 0)

if __name__ == "__main__":
    unittest.main()
