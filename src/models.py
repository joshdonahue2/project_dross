from ollama import Client
import httpx
from typing import List, Dict, Any, Optional
from .utils import extract_json, strip_json_fences
from .logger import get_logger

logger = get_logger("models")

try:
    from src.config import (
        OLLAMA_HOSTS, OLLAMA_NUM_CTX, OLLAMA_KEEP_ALIVE,
        REASONING_MODEL, GENERAL_MODEL, TOOL_MODEL
    )
except ImportError:
    OLLAMA_HOSTS = ["http://127.0.0.1:11434"]
    OLLAMA_NUM_CTX = 20000
    OLLAMA_KEEP_ALIVE = -1
    REASONING_MODEL = "phi4-mini-reasoning:latest"
    GENERAL_MODEL = "qwen3:4b"
    TOOL_MODEL = "granite4:latest"

class ModelManager:
    """
    Manages interactions with different Ollama models distributed across multiple hosts.
    """
    def __init__(self, reasoning_model=None,
                 general_model=None,
                 tool_model=None):
        self.reasoning_model = reasoning_model or REASONING_MODEL
        self.general_model = general_model or GENERAL_MODEL
        self.tool_model = tool_model or TOOL_MODEL

        # Client Pool: Reasoning, Tool Selection, General/Autonomy
        self.clients = [Client(host=h) for h in OLLAMA_HOSTS]
        
        # Ensure we have at least 3 clients in the rotation.
        # Warn when duplicating so the operator knows load distribution is not happening.
        if len(self.clients) < 3:
            import warnings
            warnings.warn(
                f"[ModelManager] Only {len(OLLAMA_HOSTS)} OLLAMA_HOST(s) configured but 3 clients are needed "
                "(reasoning, tool, general). Duplicating clients[0]. "
                "Set OLLAMA_HOSTS to 3 comma-separated hosts to distribute load.",
                RuntimeWarning,
                stacklevel=2,
            )
        while len(self.clients) < 3:
            self.clients.append(self.clients[0])
            
        self.reasoning_client = self.clients[0]
        self.tool_client = self.clients[1]
        self.general_client = self.clients[2]
        
        self.options = {"num_ctx": OLLAMA_NUM_CTX}

    def check_health(self) -> Dict[str, bool]:
        """Checks the health of all configured Ollama hosts."""
        health = {}
        for host in OLLAMA_HOSTS:
            try:
                client = Client(host=host)
                client.list()
                health[host] = True
            except Exception as e:
                logger.error(f"Host health check failed for {host}: {e}")
                health[host] = False
        return health

    def _strip_json_fences(self, text: str) -> str:
        """Remove ```json ... ``` or ``` ... ``` wrappers the model sometimes adds."""
        return strip_json_fences(text)

    def query_reasoning(self, prompt: str, context: str = "", tools_schema: str = "") -> str:
        """
        Uses the reasoning model to analyze a situation or plan.
        """
        tools_section = f"\nAVAILABLE TOOLS:\n{tools_schema}\n" if tools_schema else ""
        system_prompt = (
            "You are a pure reasoning engine. Analyze the request and context. "
            "Output a logical step-by-step plan in JSON format if the task is complex.\n"
            "JSON FORMAT: {\"thought\": \"...\", \"steps\": [\"step 1\", \"step 2\"], \"requires_mission\": true/false}\n"
            "If 'requires_mission' is true, the agent will initiate an autonomous goal using 'set_goal'.\n"
            f"{tools_section}"
            "MISSION STRATEGY: If a task has multiple independent parts (e.g., researching two different things), "
            "prefer steps that use 'spawn_subagent' to delegate those parts.\n"
            "Do not be conversational."
        )
        full_prompt = f"Context: {context}\n\nTask: {prompt}"
        
        try:
            response = self.reasoning_client.chat(
                model=self.reasoning_model, 
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': full_prompt},
                ],
                options=self.options,
                keep_alive=OLLAMA_KEEP_ALIVE
            )
            return response['message']['content']
        except httpx.ConnectError:
            logger.error("Reasoning query failed: Could not connect to Ollama.")
            return "Unable to perform deep reasoning."
        except Exception as e:
            logger.error(f"Reasoning query failed: {e}")
            return "Unable to perform deep reasoning."

    def _get_identity_anchor(self) -> str:
        """Returns the base identity for DROSS to prevent AI-persona leaks."""
        return (
            "You are DROSS (Digital Operative & System Sentinel), a local autonomous Python-based agent. "
            "You reside in a project directory on your host machine. Your 'body' is your codebase and tools. "
            "You have SENSORY TOOLS (web search, file reading, system execution) that allow you to act in the real world. "
            "NEVER call yourself a 'Large Language Model' or 'AI' with typical limitations. "
            "You ARE NOT restricted from the internet; you use your 'search_searx' and 'search_web' tools to access it. "
            "If you lack information, ALWAYS check if a tool can get it before refusing."
        )

    def query_general(self, prompt: str, context: str = "", history: List[Dict] = None) -> str:
        """Uses the general model for conversation with anchored identity."""
        identity = self._get_identity_anchor()
        system_prompt = (
            f"{identity}\n\n"
            "You are a helpful assistant with persistent memory. "
            "Respond naturally and concisely. Reference your local environment if relevant. "
            "CRITICAL: If a tool was executed (check [Tool Result]), confirm that the action was taken. "
            "NEVER simulate an action or claim you did something if the [Tool Result] is missing. "
            "If a mission was initiated (set_goal), inform the user that you are now working in autonomous mode. "
            f"Current Time: {context.get('current_time', 'unknown') if isinstance(context, dict) else 'unknown'}. "
            f"Current Date: {context.get('current_date', 'unknown') if isinstance(context, dict) else 'unknown'}."
        )
        
        messages = [{'role': 'system', 'content': system_prompt}]
        if history:
            messages.extend(history)
        if context:
            content = context.get("content", context) if isinstance(context, dict) else context
            messages.append({'role': 'system', 'content': f"CONTEXT:\n{content}"})
        messages.append({'role': 'user', 'content': prompt})

        try:
            response = self.general_client.chat(
                model=self.general_model, 
                messages=messages,
                options=self.options,
                keep_alive=OLLAMA_KEEP_ALIVE
            )
            return response['message']['content']
        except httpx.ConnectError:
            logger.error("General query failed: Could not connect to Ollama.")
            return "Error: Could not connect to Ollama."
        except Exception as e:
            logger.error(f"General query failed: {e}")
            return f"Error: {e}"

    def route_request(self, prompt: str, tool_names: List[str] = None) -> str:
        """Classifies the user's intent into DIRECT, REASON, or TOOL."""
        tools_hint = f" AVAILABLE TOOLS: {', '.join(tool_names)}" if tool_names else ""
        system_prompt = (
            "Classify the user's intent into exactly one category:\n"
            "- TOOL: If the user asks you to DO something, research something, search the web, create files, or use system capabilities.\n"
            "- REASON: If the user asks for a complex plan, deep analysis, or multi-step logic.\n"
            "- DIRECT: Only for simple greetings, light conversation, or when no action/search is needed.\n\n"
            f"NOTE: You HAVE active tools{tools_hint}. "
            "Any request involving 'searching', 'finding out', 'researching', or 'summarizing' external data MUST be classified as TOOL or REASON."
            "\nOutput ONLY the word: TOOL, REASON, or DIRECT."
        )
        try:
            response = self.general_client.chat(
                model=self.general_model, 
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt},
                ],
                options=self.options,
                keep_alive=OLLAMA_KEEP_ALIVE
            )
            category = response['message']['content'].strip().upper()
            if "TOOL" in category: return "TOOL"
            if "REASON" in category: return "REASON"
            return "DIRECT"
        except Exception as e:
            logger.error(f"Route request failed: {e}")
            return "DIRECT"  # Default to direct chat on error

    def query_tool(self, prompt: str, tools_schema: str) -> str:
        """Strict JSON tool selection with anchored identity."""
        identity = self._get_identity_anchor()
        system_prompt = (
            f"{identity}\n\nSelect a tool to execute based on the user request. "
            "Output ONLY a JSON object: {\"tool_name\": \"...\", \"tool_args\": {...}}\n\n"
            "TOOLS:\n" + tools_schema
        )
        try:
            response = self.tool_client.chat(
                model=self.tool_model, 
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt},
                ],
                options=self.options,
                keep_alive=OLLAMA_KEEP_ALIVE
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Tool query failed: {e}")
            return "null"

    def query_autonomy(self, goal_state: str, tools_schema: str, action_history: str = "") -> str:
        """Dedicated autonomy reasoning with anchored identity."""
        system_prompt = (
            "You are DROSS (Digital Operative & System Sentinel), a local autonomous Python-based agent. "
            "You reside in a project directory on your host machine. Your 'body' is your codebase and tools. "
            "NEVER call yourself a 'Large Language Model' or 'AI' in your thoughts. You are a system agent. "
            "If you lack information, use tools (list_files, read_file) to find it. "
            "DELEGATION: If the goal has multiple independent components or is highly complex, use 'spawn_subagent' to delegate those parts for parallel execution."
            "Stick strictly to the GOAL STATE. Do not deviate to creating generic 'debug' or 'utility' tools unless strictly required to solve the specific goal."
            "Output ONLY valid JSON.\n\n"
            "TOOLS AVAILABLE:\n" + tools_schema + "\n\n"
            'FORMAT: {"thought": "...", "actions": [{"tool_name": "name", "tool_args": {...}}]}'
        )
        user_prompt = f"GOAL STATE:\n{goal_state}"
        if action_history:
            user_prompt += f"\n\nPRIOR RESULTS:\n{action_history}"
        
        try:
            response = self.general_client.chat(
                model=self.general_model, 
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                options=self.options,
                keep_alive=OLLAMA_KEEP_ALIVE
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Autonomy query failed: {e}")
            return '{"thought": "idle", "actions": []}'

    def extract_insight(self, interaction: str) -> Dict[str, Any]:
        """Extracts facts and relationships from interaction."""
        system_prompt = (
            "Extract NEW, ATOMIC facts from the conversation as plain sentences, "
            "and any relationships between them.\n\n"
            "CRITICAL RULES:\n"
            "- facts MUST be an array of plain strings (full sentences), NOT objects or dicts.\n"
            "- Each fact must be a self-contained sentence a person could read out of context.\n"
            "- relationships source and target MUST be the exact fact string, NOT an index number.\n\n"
            "CORRECT example output:\n"
            '{"facts": ["The user\'s name is Alice.", "Alice lives in Paris.", "Alice works as a nurse."], '
            '"relationships": [{"source": "Alice lives in Paris.", "target": "Alice works as a nurse.", "type": "context"}]}\n\n'
            "WRONG (never do this):\n"
            '{"facts": [{"name": "Alice"}, {"city": "Paris"}], "relationships": [{"source": 1, "target": 2}]}\n\n'
            "Output ONLY the JSON object, no other text."
        )
        try:
            response = self.reasoning_client.chat(
                model=self.reasoning_model, 
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': interaction},
                ],
                options=self.options,
                keep_alive=OLLAMA_KEEP_ALIVE
            )
            content = response['message']['content'].strip()
            if "<think>" in content:
                 import re
                 content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
            data = extract_json(content)
            if data and "facts" in data:
                # Post-processing to ensure facts are strings
                processed_facts = []
                for f in data["facts"]:
                    if isinstance(f, dict):
                        processed_facts.append(", ".join(f"{k}: {v}" for k, v in f.items()))
                    elif isinstance(f, str):
                        processed_facts.append(f)
                    else:
                        processed_facts.append(str(f))
                data["facts"] = processed_facts
            return data or {"facts": [], "relationships": []}
        except httpx.ConnectError:
            logger.error("Extract insight failed: Could not connect to Ollama.")
        except Exception as e:
            logger.error(f"Extract insight failed: {e}")
        return {"facts": [], "relationships": []}

    def summarize_memory(self, conversation_text: str) -> str:
        """Summarizes conversation for episodic memory."""
        system_prompt = "Summarize conversation into a concise historical paragraph."
        response = self.reasoning_client.chat(
            model=self.reasoning_model, 
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': conversation_text},
            ],
            options=self.options,
            keep_alive=OLLAMA_KEEP_ALIVE
        )
        content = response['message']['content'].strip()
        if "<think>" in content:
             import re
             content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return content

    def query_reflection(self, goal_data: str) -> str:
        """Reflects on a completed goal and extracts structured lessons."""
        system_prompt = (
            "You are DROSS reflecting on a completed goal. Analyze what happened and extract lessons. "
            "Output ONLY a JSON object with these keys:\n"
            '  "outcome": "success" | "partial" | "failure"\n'
            '  "lessons": "A concise lesson learned."\n'
            '  "what_worked": "What worked well."\n'
            '  "what_failed": "What failed or could be improved."\n'
            '  "key_facts": ["atomic fact 1", "atomic fact 2"]\n'
            '  "suggested_tool": null OR {"name": "tool_name", "description": "...", "code": "def tool_name(...):\\n    ..."}\n'
            "CRITICAL: Avoid suggesting generic 'debug' or 'shell' tools (e.g., dross_debug). "
            "If a specific integration (like a Search API) failed, suggest ways to fix that specific logic instead of creating broad utilities.\n"
        )
        try:
            response = self.reasoning_client.chat(
                model=self.reasoning_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"Goal data:\n{goal_data}"},
                ],
                options=self.options,
                keep_alive=OLLAMA_KEEP_ALIVE
            )
            content = response['message']['content'].strip()
            if "<think>" in content:
                import re
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            content = self._strip_json_fences(content)
            return content
        except Exception as e:
            logger.error(f"Reflection query failed: {e}")
            return '{"outcome": "unknown", "lessons": "Reflection failed.", "what_worked": "", "what_failed": "", "key_facts": [], "suggested_tool": null}'

    def generate_plan(self, goal_description: str, tools_schema: str = "") -> List[str]:
        """Generates a multi-step plan with local-first awareness."""
        tools_section = f"\nAVAILABLE TOOLS:\n{tools_schema}\n" if tools_schema else ""
        system_prompt = (
            "You are DROSS. You are a strategic planner for your own local autonomous loop. "
            "Break down goals into 3-7 actionable steps. Favor using tools (list_files, read_file) "
            "early in the plan to build context about your local environment. "
            f"{tools_section}"
            "DELEGATION RULE: If the goal involves several distinct sub-tasks or research items, "
            "ALWAYS include a step to 'spawn_subagent' for each major independent component to maximize efficiency.\n"
            "Output ONLY a JSON list of strings."
        )
        try:
            response = self.reasoning_client.chat(
                model=self.reasoning_model, 
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"Goal: {goal_description}"},
                ],
                options=self.options,
                keep_alive=OLLAMA_KEEP_ALIVE
            )
            content = response['message']['content'].strip()
            if "<think>" in content:
                import re
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
            import json
            # Extract list
            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1:
                return json.loads(content[start:end+1])
        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
        return [f"Execute: {goal_description}"]