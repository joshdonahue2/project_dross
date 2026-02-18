import json
import os
from typing import Dict, Any, List, Optional, Annotated, TypedDict, Callable
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from .schemas import AgentState, Goal, Plan, PlanStep, Subtask
from .models import ModelManager
from .tools import registry as tool_registry
from .memory import MemorySystem
from .logger import get_logger
from .utils import clean_output

logger = get_logger("agent_graph")

# Define the state as a TypedDict for LangGraph (Pydantic models work too, but TypedDict is more standard for LangGraph)
class GraphState(TypedDict):
    user_input: str
    history: List[Dict[str, str]]
    current_goal: Optional[Dict[str, Any]]
    current_plan: Optional[Dict[str, Any]]
    next_step_index: int
    reasoning: str
    tool_results: List[Dict[str, Any]]
    final_response: str
    intent: Optional[str]
    requires_mission: bool
    terminate: bool

class DROSSGraph:
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir
        self.models = ModelManager()
        # Isolated memory if data_dir is provided
        mem_path = os.path.join(data_dir, "memory_db") if data_dir else "./data/memory_db"
        self.memory = MemorySystem(db_path=mem_path)
        self.tools = tool_registry
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()
        self.callback = None

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(GraphState)

        # Define Nodes
        workflow.add_node("router", self.node_router)
        workflow.add_node("reasoner", self.node_reasoner)
        workflow.add_node("tool_executor", self.node_tool_executor)
        workflow.add_node("synthesizer", self.node_synthesizer)
        workflow.add_node("reflector", self.node_reflector)

        # Build Graph
        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            self.route_after_router,
            {
                "REASON": "reasoner",
                "TOOL": "tool_executor",
                "DIRECT": "synthesizer"
            }
        )

        workflow.add_edge("reasoner", "tool_executor")

        workflow.add_conditional_edges(
            "tool_executor",
            self.route_after_tool,
            {
                "continue": "tool_executor",
                "synthesize": "synthesizer"
            }
        )

        workflow.add_edge("synthesizer", "reflector")
        workflow.add_edge("reflector", END)

        return workflow

    # --- Nodes ---

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        return self.tools.execute(tool_name, args, callback=self.callback)

    def node_router(self, state: GraphState) -> Dict[str, Any]:
        logger.info(f"Node: router | Input: {state['user_input']}")
        tool_names = list(self.tools.tools.keys())
        intent = self.models.route_request(state['user_input'], tool_names=tool_names)
        return {"intent": intent}

    def node_reasoner(self, state: GraphState) -> Dict[str, Any]:
        logger.info("Node: reasoner")
        user_input = state['user_input']
        history = state['history']

        # Environmental Context
        cwd = os.getcwd()
        local_files = self._execute_tool("list_files", {"path": "."})
        env_context = f"LOCAL ENVIRONMENT:\n- CWD: {cwd}\n- FILES: {local_files}\n"

        long_term_context = self.memory.retrieve_relevant(user_input)
        context_str = f"{env_context}\nLong-term Memory: {long_term_context}\nShort-term History: {history}"

        tool_schema = self.tools.get_schemas_str()
        reasoning_raw = self.models.query_reasoning(user_input, context_str, tools_schema=tool_schema)

        reasoning_data = self.models.extract_json(reasoning_raw)
        requires_mission = False
        reasoning = reasoning_raw

        if reasoning_data and reasoning_data.get("requires_mission"):
            requires_mission = True
            reasoning = reasoning_data.get("thought", reasoning_raw)
            # If it requires a mission, we should set the goal
            self._execute_tool("set_goal", {"description": user_input, "is_autonomous": True})

        return {"reasoning": reasoning, "requires_mission": requires_mission}

    def node_tool_executor(self, state: GraphState) -> Dict[str, Any]:
        logger.info("Node: tool_executor")
        intent = state.get("intent")
        user_input = state['user_input']

        # If we have an active goal/plan, we execute the next step
        goal_json = self._execute_tool("get_goal", {})
        if "No active goal" not in goal_json:
            # Autonomous mode
            plan_json = self._execute_tool("get_plan", {})
            if "No plan defined" in plan_json:
                tool_schemas = self.tools.get_schemas_str()
                steps = self.models.generate_plan(user_input, tools_schema=tool_schemas)
                self._execute_tool("set_plan", {"steps": steps})
                plan_json = self._execute_tool("get_plan", {})

            plan_data = json.loads(plan_json)
            steps = plan_data.get("steps", [])

            next_step_idx = -1
            for i, s in enumerate(steps):
                if s["status"] == "pending":
                    next_step_idx = i
                    break

            if next_step_idx != -1:
                step_desc = steps[next_step_idx]["description"]
                logger.info(f"Executing Step {next_step_idx}: {step_desc}")

                # Context for LLM to decide tool
                cwd = os.getcwd()
                local_files = self._execute_tool("list_files", {"path": "."})
                env_context = f"ENVIRONMENT:\n- CWD: {cwd}\n- FILES: {local_files}\n"

                tool_schemas = self.tools.get_schemas_str()
                context = f"{env_context}\nCURRENT GOAL: {user_input}\nACTIVE STEP ({next_step_idx}): {step_desc}\nPLAN STATUS: {plan_json}"

                response = self.models.query_autonomy(goal_state=context, tools_schema=tool_schemas)
                data = self.models.extract_json(response)

                results = []
                if data and data.get("actions"):
                    step_failed = False
                    for action in data["actions"]:
                        tool = action.get("tool_name")
                        args = action.get("tool_args", {})
                        if tool:
                            output = self._execute_tool(tool, args)
                            results.append({"tool": tool, "output": output})
                            if output.lower().startswith("error") or "exception" in output.lower():
                                step_failed = True

                    new_status = "failed" if step_failed else "completed"
                    self._execute_tool("update_plan_step", {"step_index": next_step_idx, "status": new_status})

                return {"tool_results": state.get("tool_results", []) + results, "next_step_index": next_step_idx + 1}
            else:
                # Plan finished
                self._execute_tool("complete_goal", {"result_summary": "Plan completed."})
                return {"terminate": True}

        elif intent == "TOOL":
            # Direct tool call
            tool_schema = self.tools.get_schemas_str()
            tool_decision_json = self.models.query_tool(user_input, tool_schema)
            tool_data = self.models.extract_json(tool_decision_json)

            if tool_data:
                tool_name = tool_data.get("tool_name") or tool_data.get("name")
                tool_args = tool_data.get("tool_args") or tool_data.get("arguments") or tool_data.get("args") or {}
                if tool_name:
                    output = self._execute_tool(tool_name, tool_args)
                    return {"tool_results": [{"tool": tool_name, "output": output}], "terminate": True}

        return {"terminate": True}

    def node_synthesizer(self, state: GraphState) -> Dict[str, Any]:
        logger.info("Node: synthesizer")
        user_input = state['user_input']
        history = state['history']
        reasoning = state.get("reasoning", "")
        tool_results = state.get("tool_results", [])

        context_parts = []
        if reasoning: context_parts.append(f"[Reasoning]\n{reasoning}")
        if tool_results:
            results_str = "\n".join([f"Tool {r['tool']} output: {r['output']}" for r in tool_results])
            context_parts.append(f"[Tool Results]\n{results_str}")

        context_str = "\n\n".join(context_parts)

        from datetime import datetime
        now = datetime.now()
        synthesis_context = {
            "content": context_str,
            "current_time": now.strftime("%H:%M:%S"),
            "current_date": now.strftime("%Y-%m-%d")
        }

        response = self.models.query_general(user_input, context=synthesis_context, history=history)
        response = clean_output(response)

        # Save to memory
        self.memory.add_short_term("user", user_input)
        self.memory.add_short_term("assistant", response)

        return {"final_response": response}

    def node_reflector(self, state: GraphState) -> Dict[str, Any]:
        logger.info("Node: reflector")
        # Reflection logic
        goal_json = self._execute_tool("get_goal", {})
        if "No active goal" not in goal_json:
            goal_data = json.loads(goal_json)
            if goal_data.get("status") == "completed":
                self.models.query_reflection(goal_json) # Agent.reflect handles saving etc, but here we just call the model
                # In a real implementation we might want to use Agent.reflect
        return {}

    # --- Router Functions ---

    def route_after_router(self, state: GraphState) -> str:
        return state.get("intent", "DIRECT")

    def route_after_tool(self, state: GraphState) -> str:
        if state.get("terminate") or state.get("next_step_index", 0) >= 5: # Safety break
            return "synthesize"

        # Check if more steps are pending
        plan_json = self._execute_tool("get_plan", {})
        if "No plan defined" in plan_json:
            return "synthesize"

        plan_data = json.loads(plan_json)
        steps = plan_data.get("steps", [])
        if any(s["status"] == "pending" for s in steps):
            return "continue"

        return "synthesize"

    def heartbeat(self, callback: Optional[Callable] = None) -> str:
        """
        Pulse the agent for one step of the current plan.
        """
        self.callback = callback
        # We can just run a partial graph or a specific node
        # For now, let's keep it simple and run a 'pulse' state
        goal_json = self._execute_tool("get_goal", {})
        if "No active goal" in goal_json:
            return None

        goal_data = json.loads(goal_json)

        initial_state = {
            "user_input": goal_data.get("goal", ""),
            "history": [],
            "current_goal": goal_data,
            "current_plan": None,
            "next_step_index": 0,
            "reasoning": "Heartbeat pulse",
            "tool_results": [],
            "final_response": "",
            "intent": "REASON", # Force autonomy loop
            "requires_mission": True,
            "terminate": False
        }

        # We only want to run one tool execution step
        # This is a bit complex with the current graph setup
        # For now, let's just use the existing run() but with autonomy context
        # In a real LangGraph app, we would use thread_id and checkpoints.

        # For this modernization, I will simplify and just run the executor node
        res = self.node_tool_executor(initial_state)
        if res.get("tool_results"):
            return " | ".join([f"{r['tool']} -> {r['output'][:50]}" for r in res["tool_results"]])
        return "No actions taken."

    def run(self, user_input: str, history: List[Dict[str, str]] = None, callback: Optional[Callable] = None) -> str:
        self.callback = callback

        initial_state = {
            "user_input": user_input,
            "history": history or [],
            "current_goal": None,
            "current_plan": None,
            "next_step_index": 0,
            "reasoning": "",
            "tool_results": [],
            "final_response": "",
            "intent": None,
            "requires_mission": False,
            "terminate": False
        }

        final_state = self.app.invoke(initial_state)
        return final_state["final_response"]
