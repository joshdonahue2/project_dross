import threading
import uuid
import time
import os
import json
from typing import Dict, Any, List, Optional
from .logger import get_logger

logger = get_logger("subagents")

class SubagentManager:
    def __init__(self):
        self.subagents: Dict[str, Dict[str, Any]] = {}

    def spawn(self, goal: str) -> str:
        subagent_id = str(uuid.uuid4())[:8]

        self.subagents[subagent_id] = {
            "id": subagent_id,
            "goal": goal,
            "status": "running",
            "result": None,
            "start_time": time.time(),
            "end_time": None,
            "steps_taken": 0
        }

        # Run in a separate thread
        thread = threading.Thread(target=self._run_subagent, args=(subagent_id, goal))
        thread.daemon = True
        thread.start()

        return subagent_id

    def _run_subagent(self, subagent_id: str, goal: str):
        # Set environment variable for this thread to isolate data
        # Note: os.environ is global, so this might not work perfectly with multiple threads
        # but for a simple implementation it might suffice if we are careful or use a better isolation.
        # Better: Pass data_dir to Agent constructor.

        data_dir = os.path.join("data", "subagents", subagent_id)
        os.makedirs(data_dir, exist_ok=True)

        try:
            logger.info(f"Subagent {subagent_id} started with goal: {goal}")

            # Import DROSSGraph here to avoid circular dependency
            from .agent_graph import DROSSGraph

            # Subagents currently share the global DB in src/tools.py.
            # In a more advanced version, we'd pass data_dir to Tools/DB as well.
            subagent = DROSSGraph(data_dir=data_dir)

            # Set its goal
            subagent._execute_tool("set_goal", {"description": goal, "is_autonomous": True})

            # Run its loop
            max_steps = 20

            while self.subagents[subagent_id]["steps_taken"] < max_steps:
                result = subagent.heartbeat()
                self.subagents[subagent_id]["steps_taken"] += 1

                # Check if goal is completed
                goal_info_raw = subagent._execute_tool("get_goal", {})
                try:
                    goal_info = json.loads(goal_info_raw)
                    if goal_info.get("status") == "completed":
                        self.subagents[subagent_id]["status"] = "completed"
                        self.subagents[subagent_id]["result"] = goal_info.get("result", "Goal reached.")
                        break
                except:
                    pass

                if not result or "Error" in result:
                    # If it's a real error, we might want to stop
                    if "Error" in result:
                        logger.error(f"Subagent {subagent_id} encountered error: {result}")
                        # Don't necessarily stop on one error, but maybe if it persists

                time.sleep(2) # Give it some breathing room

            if self.subagents[subagent_id]["status"] == "running":
                self.subagents[subagent_id]["status"] = "finished"
                self.subagents[subagent_id]["result"] = "Subagent reached maximum steps."

        except Exception as e:
            logger.error(f"Subagent {subagent_id} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.subagents[subagent_id]["status"] = "failed"
            self.subagents[subagent_id]["result"] = str(e)
        finally:
            self.subagents[subagent_id]["end_time"] = time.time()
            logger.info(f"Subagent {subagent_id} ended with status: {self.subagents[subagent_id]['status']}")

    def get_status(self, subagent_id: str) -> Optional[Dict[str, Any]]:
        return self.subagents.get(subagent_id)

    def list_all(self) -> List[Dict[str, Any]]:
        current_time = time.time()
        results = []
        for sa in self.subagents.values():
            sa_copy = sa.copy()
            if sa_copy["status"] == "running":
                sa_copy["runtime_seconds"] = int(current_time - sa_copy["start_time"])
            elif sa_copy["end_time"]:
                sa_copy["runtime_seconds"] = int(sa_copy["end_time"] - sa_copy["start_time"])
            else:
                sa_copy["runtime_seconds"] = 0
            results.append(sa_copy)
        return results

# Global instance
subagent_manager = SubagentManager()
