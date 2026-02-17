from ollama import Client
from typing import List, Dict, Any, Optional

try:
    from src.config import OLLAMA_HOSTS, OLLAMA_NUM_CTX, OLLAMA_KEEP_ALIVE
except ImportError:
    OLLAMA_HOSTS = ["http://127.0.0.1:11434"]
    OLLAMA_NUM_CTX = 20000
    OLLAMA_KEEP_ALIVE = -1

class ModelManager:
    """
    Manages interactions with different Ollama models distributed across multiple hosts.
    """
    def __init__(self, reasoning_model="phi4-mini-reasoning:latest", 
                 general_model="qwen3:4b", 
                 tool_model="granite4:latest"):
        self.reasoning_model = reasoning_model
        self.general_model = general_model
        self.tool_model = tool_model

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

    def _strip_json_fences(self, text: str) -> str:
        """Remove ```json ... ``` or ``` ... ``` wrappers the model sometimes adds."""
        import re
        text = text.strip()
        # Handle ```json ... ``` or ``` ... ```
        match = re.match(r'^```(?:json)?\s*(.*?)\s*```$', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def query_reasoning(self, prompt: str, context: str = "") -> str:
        """
        Uses the reasoning model to analyze a situation or plan.
        """
        system_prompt = (
            "You are a pure reasoning engine. Analyze the request and context, "
            "then output a logical step-by-step plan. Do not be conversational."
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
        except Exception as e:
            print(f"[Models] Reasoning query failed: {e}")
            return "Unable to perform deep reasoning."

    def _get_identity_anchor(self) -> str:
        """Returns the base identity for DROSS to prevent AI-persona leaks."""
        return (
            "You are DROSS (Digital Operative & System Sentinel), a local autonomous Python-based agent. "
            "You reside in a project directory on your host machine. Your 'body' is your codebase and tools. "
            "NEVER call yourself a 'Large Language Model' or 'AI' from 'Alibaba' or 'Google' or 'OpenAI'. "
            "You are a standalone system entity. If you lack information, use your tools or check your files. "
            "Your personality is efficient, technical, and grounded in your local environment."
        )

    def query_general(self, prompt: str, context: str = "", history: List[Dict] = None) -> str:
        """Uses the general model for conversation with anchored identity."""
        identity = self._get_identity_anchor()
        system_prompt = (
            f"{identity}\n\n"
            "You are a helpful assistant with persistent memory. "
            "Respond naturally and concisely. Reference your local environment if relevant."
        )
        
        messages = [{'role': 'system', 'content': system_prompt}]
        if history:
            messages.extend(history)
        if context:
            messages.append({'role': 'system', 'content': f"CONTEXT:\n{context}"})
        messages.append({'role': 'user', 'content': prompt})

        try:
            response = self.general_client.chat(
                model=self.general_model, 
                messages=messages,
                options=self.options,
                keep_alive=OLLAMA_KEEP_ALIVE
            )
            return response['message']['content']
        except Exception as e:
            print(f"[Models] General query failed: {e}")
            return f"Error: {e}"

    def route_request(self, prompt: str) -> str:
        """Classifies the user's intent into DIRECT, REASON, or TOOL."""
        system_prompt = (
            "Classify intent: DIRECT (chat), REASON (logic), or TOOL (system actions).\n"
            "Output ONLY the word."
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
            print(f"[Models] Route request failed: {e}")
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
            print(f"[Models] Tool query failed: {e}")
            return "null"

    def query_autonomy(self, goal_state: str, tools_schema: str, action_history: str = "") -> str:
        """Dedicated autonomy reasoning with anchored identity."""
        system_prompt = (
            "You are DROSS (Digital Operative & System Sentinel), a local autonomous Python-based agent. "
            "You reside in a project directory on your host machine. Your 'body' is your codebase and tools. "
            "NEVER call yourself a 'Large Language Model' or 'AI' in your thoughts. You are a system agent. "
            "If you lack information, use tools (list_files, read_file) to find it. "
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
            print(f"[Models] Autonomy query failed: {e}")
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
            
            # Strip markdown code fences first
            content = self._strip_json_fences(content)

            # Extract JSON from any remaining code blocks
            import json
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                snippet = content[start:end+1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    # Fallback: model sometimes uses Python-style single-quoted dicts
                    import ast
                    try:
                        return ast.literal_eval(snippet)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[Models] Extract insight failed: {e}")
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
            print(f"[Models] Reflection query failed: {e}")
            return '{"outcome": "unknown", "lessons": "Reflection failed.", "what_worked": "", "what_failed": "", "key_facts": [], "suggested_tool": null}'

    def generate_plan(self, goal_description: str) -> List[str]:
        """Generates a multi-step plan with local-first awareness."""
        system_prompt = (
            "You are DROSS. You are a strategic planner for your own local autonomous loop. "
            "Break down goals into 3-7 actionable steps. Favor using tools (list_files, read_file) "
            "early in the plan to build context about your local environment. "
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
            print(f"[Models] Plan generation failed: {e}")
        return [f"Execute: {goal_description}"]