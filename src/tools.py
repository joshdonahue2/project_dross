import inspect
import json
import os
import shutil
import uuid
import requests
import platform
import subprocess
from typing import Callable, Dict, Any, List, Optional
from datetime import datetime
from .logger import get_logger

logger = get_logger("tools")

try:
    from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""

# --- Configuration ---
WORKSPACE_DIR = os.path.abspath("workspace")
if not os.path.exists(WORKSPACE_DIR):
    os.makedirs(WORKSPACE_DIR)

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.schemas: List[Dict[str, Any]] = []

    def _python_type_to_json_schema(self, annotation) -> str:
        """Maps Python type annotations to JSON Schema type strings."""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        # Handle typing generics like List[str]
        origin = getattr(annotation, '__origin__', None)
        if origin is list:
            return "array"
        if origin is dict:
            return "object"
        return type_map.get(annotation, "string")  # Default to string for unknown types

    def register(self, func: Callable):
        """Decorator to register a tool."""
        self.tools[func.__name__] = func
        
        sig = inspect.signature(func)
        params = {}
        required = []
        for name, param in sig.parameters.items():
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else str
            json_type = self._python_type_to_json_schema(annotation)
            param_schema = {"type": json_type}
            if param.default != inspect.Parameter.empty:
                param_schema["default"] = param.default
            else:
                required.append(name)
            params[name] = param_schema

        schema = {
            "name": func.__name__,
            "description": func.__doc__ or "No description",
            "parameters": {
                "type": "object",
                "properties": params,
            }
        }
        if required:
            schema["parameters"]["required"] = required

        self.schemas.append(schema)
        return func

    def get_schemas_str(self) -> str:
        """Returns JSON string of available tools."""
        return json.dumps(self.schemas, indent=2)

    def execute(self, tool_name: str, args: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """Executes a registered tool."""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found."
        
        try:
            func = self.tools[tool_name]
            sig = inspect.signature(func)
            # If the tool accepts a 'context' parameter, pass it.
            if 'context' in sig.parameters:
                return str(func(**args, context=context))
            return str(func(**args))
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"

# --- Helper for Sandboxing ---

def _get_safe_path(filename: str) -> str:
    """
    Resolves a filename to an absolute path, sandboxed within the project root.
    Raises ValueError if the path attempts to escape the project root via traversal.
    """
    PROJECT_ROOT = os.path.abspath(".")

    # Strip any leading "workspace/" prefix the agent may hallucinate
    clean_name = filename.strip()
    for prefix in ("workspace/", "workspace\\"):
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix):]

    # Resolve to absolute path relative to project root
    safe_path = os.path.normpath(os.path.join(PROJECT_ROOT, clean_name))

    # Enforce boundary: the resolved path must stay within the project root
    if not safe_path.startswith(PROJECT_ROOT + os.sep) and safe_path != PROJECT_ROOT:
        raise ValueError(
            f"Path traversal attempt blocked: '{filename}' resolves to '{safe_path}', "
            f"which is outside the project root '{PROJECT_ROOT}'."
        )

    return safe_path

# --- Define Basic Tools ---

registry = ToolRegistry()

@registry.register
def run_shell(command: str) -> str:
    """
    Executes a shell command. 
    Use with caution. Returns stdout and stderr.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        output = f"Command: {command}\nExit Code: {result.returncode}\n"
        output += f"STDOUT:\n{result.stdout if result.stdout else '(empty)'}\n"
        output += f"STDERR:\n{result.stderr if result.stderr else '(empty)'}\n"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Command '{command}' timed out after 60 seconds."
    except Exception as e:
        logger.error(f"Shell execution error: {e}")
        return f"Shell Error: {e}"

@registry.register
def get_system_info() -> str:
    """Returns basic system information (OS, CPU, Memory)."""
    try:
        info = {
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }

        # Try to get memory info if psutil is available
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["memory_total"] = f"{mem.total / (1024**3):.2f} GB"
            info["memory_available"] = f"{mem.available / (1024**3):.2f} GB"
            info["cpu_count"] = psutil.cpu_count()
        except ImportError:
            pass

        return json.dumps(info, indent=2)
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return f"Error: {e}"

@registry.register
def install_package(package_name: str) -> str:
    """Installs a Python package using pip."""
    return run_shell(f"pip install {package_name}")


@registry.register
def list_files(path: str = ".") -> str:
    """Lists files in the specified directory."""
    try:
        target_path = _get_safe_path(path)
        if not os.path.exists(target_path):
             return "Directory does not exist."
        
        items = os.listdir(target_path)
        # Add indicators for directories
        params = []
        for item in items:
            full_path = os.path.join(target_path, item)
            if os.path.isdir(full_path):
                params.append(f"{item}/")
            else:
                params.append(item)
        return json.dumps(params)
    except Exception as e:
        return f"Error: {e}"

@registry.register
def write_file(filename: str, content: str) -> str:
    """Writes content to a file. Overwrites if exists."""
    try:
        target_path = _get_safe_path(filename)
        # Ensure parent dirs exist
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {filename}"
    except Exception as e:
        return f"Error writing file: {e}"

@registry.register
def read_file(filename: str) -> str:
    """Reads content from a file."""
    try:
        target_path = _get_safe_path(filename)
        if not os.path.exists(target_path):
            return "File does not exist."
        if os.path.isdir(target_path):
             return "Path is a directory, not a file."
             
        with open(target_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@registry.register
def get_file_info(filename: str) -> str:
    """Returns size and modification time of a file."""
    try:
        target_path = _get_safe_path(filename)
        if not os.path.exists(target_path):
            return "File does not exist."
            
        stats = os.stat(target_path)
        mod_time = datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        size = stats.st_size
        return f"File: {filename}\nSize: {size} bytes\nModified: {mod_time}"
    except Exception as e:
        return f"Error getting info: {e}"

@registry.register
def verify_proposal(filename: str, content: str, test_command: str = "") -> str:
    """
    Verifies a proposed code change before applying it.
    1. Writes content to a .tmp version of the file.
    2. Runs test_command (e.g., 'python tests.py' or 'python -m py_compile file.tmp').
    3. Returns the output and status.
    """
    try:
        target_path = _get_safe_path(filename)
        tmp_path = target_path + ".tmp"
        
        # Ensure parent dirs exist
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        result_msg = f"Proposal written to {os.path.basename(tmp_path)}.\n"
        
        if not test_command:
            # Default to syntax check for .py files
            if filename.endswith(".py"):
                test_command = f"python -m py_compile {tmp_path}"
        
        if test_command:
            # Replace placeholder if used
            cmd = test_command.replace(filename, tmp_path)
            
            import subprocess
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            result_msg += f"Verification Status: {'Success' if res.returncode == 0 else 'Failure'}\n"
            if res.stdout: result_msg += f"STDOUT: {res.stdout}\n"
            if res.stderr: result_msg += f"STDERR: {res.stderr}\n"
            
        return result_msg.strip()
    except Exception as e:
        return f"Verification Error: {e}"

# --- Goal Management (Autonomy) ---

DEFAULT_DATA_DIR = os.path.abspath("data")
GOAL_FILE_NAME = "goal.json"
GOAL_STACK_FILE_NAME = "goal_stack.json"

# For backward compatibility with existing tests and scripts
GOAL_FILE = os.path.join(DEFAULT_DATA_DIR, GOAL_FILE_NAME)
GOAL_STACK_FILE = os.path.join(DEFAULT_DATA_DIR, GOAL_STACK_FILE_NAME)

def _get_goal_files(context: Optional[Dict[str, Any]] = None):
    data_dir = (context or {}).get("data_dir") or DEFAULT_DATA_DIR
    goal_file = os.path.join(data_dir, GOAL_FILE_NAME)
    stack_file = os.path.join(data_dir, GOAL_STACK_FILE_NAME)
    os.makedirs(data_dir, exist_ok=True)
    return goal_file, stack_file

@registry.register
def set_goal(description: str, is_autonomous: bool = False, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Sets the agent's active autonomous goal.
    If an autonomous goal is already active and a USER goal (not autonomous) is set,
    the autonomous goal is pushed to a stack to be resumed later.
    """
    try:
        goal_file, stack_file = _get_goal_files(context)
        
        # Check current goal for stacking
        current_goal = None
        if os.path.exists(goal_file):
            with open(goal_file, 'r', encoding='utf-8') as f:
                try:
                    current_goal = json.load(f)
                except:
                    pass
        
        # Stacking Logic: If current is active/autonomous and new is NOT autonomous
        if current_goal and current_goal.get("status") == "active" and not is_autonomous:
            if current_goal.get("is_autonomous"):
                logger.info(f"Postponing autonomous goal: {current_goal.get('goal')}")
                stack = []
                if os.path.exists(stack_file):
                    with open(stack_file, 'r', encoding='utf-8') as f:
                        try:
                            stack = json.load(f)
                        except:
                            pass
                stack.append(current_goal)
                with open(stack_file, 'w', encoding='utf-8') as f:
                    json.dump(stack, f, indent=2)

        data = {
            "goal": description,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "is_autonomous": is_autonomous,
            "subtasks": [],
            "log": []
        }
        with open(goal_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return f"Goal set: {description}"
    except Exception as e:
        return f"Error setting goal: {e}"

@registry.register
def get_goal(context: Optional[Dict[str, Any]] = None) -> str:
    """Returns the current active goal or 'No active goal'."""
    try:
        goal_file, _ = _get_goal_files(context)
        if not os.path.exists(goal_file):
            return "No active goal."
        with open(goal_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if data.get("status") != "active":
             return "No active goal."
             
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error reading goal: {e}"

@registry.register
def complete_goal(result_summary: str = "Goal reached.", context: Optional[Dict[str, Any]] = None) -> str:
    """Marks the current goal as complete."""
    try:
        goal_file, stack_file = _get_goal_files(context)
        if not os.path.exists(goal_file):
            return "No active goal to complete."
        with open(goal_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        data["status"] = "completed"
        data["completed_at"] = datetime.now().isoformat()
        data["result"] = result_summary
        
        with open(goal_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        # Resumption Logic: Check if there's a goal to resume
        resumption_msg = ""
        if os.path.exists(stack_file):
            try:
                with open(stack_file, 'r', encoding='utf-8') as f:
                    stack = json.load(f)
                
                if stack:
                    resumed_goal = stack.pop()
                    logger.info(f"Resuming postponed goal: {resumed_goal.get('goal')}")
                    # Update status and timestamps
                    resumed_goal["status"] = "active"
                    resumed_goal["resumed_at"] = datetime.now().isoformat()
                    
                    with open(goal_file, 'w', encoding='utf-8') as f:
                        json.dump(resumed_goal, f, indent=2)
                    
                    # Update stack file
                    with open(stack_file, 'w', encoding='utf-8') as f:
                        json.dump(stack, f, indent=2)
                    
                    resumption_msg = f" Resumed goal: {resumed_goal['goal']}"
            except Exception as re:
                logger.error(f"Error resuming goal: {re}")

        return f"Goal marked as completed.{resumption_msg}"
    except Exception as e:
        return f"Error completing goal: {e}"

@registry.register
def add_subtask(context: Optional[Dict[str, Any]] = None, **kwargs) -> str:
    """Adds a subtask to the current goal. Args: subtask (str)"""
    try:
        goal_file, _ = _get_goal_files(context)
        # Robust argument extraction
        description = kwargs.get('subtask') or kwargs.get('description')
        if not description:
            return "Error: Missing argument 'subtask' or 'description'."

        if not os.path.exists(goal_file): return "No active goal."
        with open(goal_file, 'r', encoding='utf-8') as f: data = json.load(f)
        
        new_task = {"id": str(uuid.uuid4())[:8], "description": description, "status": "pending"}
        data.setdefault("subtasks", []).append(new_task)
        
        with open(goal_file, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2)
        return f"Subtask added: {description}"
    except Exception as e: return f"Error adding subtask: {e}"

@registry.register
def list_subtasks(context: Optional[Dict[str, Any]] = None) -> str:
    """Lists all subtasks for the current goal."""
    try:
        goal_file, _ = _get_goal_files(context)
        if not os.path.exists(goal_file): return "No active goal."
        with open(goal_file, 'r', encoding='utf-8') as f: data = json.load(f)
        
        tasks = data.get("subtasks", [])
        if not tasks: return "No subtasks defined."
        
        output = "Subtasks:\n"
        for t in tasks:
            icon = "[x]" if t['status'] == 'completed' else "[ ]"
            output += f"{icon} {t['id']}: {t['description']}\n"
        return output
    except Exception as e: return f"Error listing subtasks: {e}"

@registry.register
def complete_subtask(subtask_id: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Marks a subtask as complete by partial ID match."""
    try:
        goal_file, _ = _get_goal_files(context)
        if not os.path.exists(goal_file): return "No active goal."
        with open(goal_file, 'r', encoding='utf-8') as f: data = json.load(f)
        
        found = False
        for t in data.get("subtasks", []):
            if subtask_id in t['id']:
                t['status'] = "completed"
                found = True
                break
        
        if found:
            with open(goal_file, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2)
            return f"Subtask {subtask_id} completed."
        return f"Subtask {subtask_id} not found."
    except Exception as e: return f"Error completing subtask: {e}"

# --- Advanced Tools (Internet & Code) ---

@registry.register
def search_web(query: str) -> str:
    """Performs a web search using DuckDuckGo and returns the top 3 results."""
    try:
        from duckduckgo_search import DDGS
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No results found."
        
        formatted = []
        for r in results:
            formatted.append(f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}")
        
        return "\n---\n".join(formatted)
    except ImportError:
        return "Error: duckduckgo-search not installed."
    except Exception as e:
        return f"Search Error: {e}"

@registry.register
def scrape_website(url: str) -> str:
    """
    Fetches the content of a website and returns the cleaned text.
    Useful for reading specific articles or documentation.
    """
    try:
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text
        text = soup.get_text(separator='\n')

        # Break into lines and remove leading and trailing whitespace
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        # Limit output length to avoid context overflow, but keep it substantial
        return text[:10000]
    except ImportError:
        return "Error: beautifulsoup4 not installed."
    except Exception as e:
        return f"Scraping Error: {str(e)}"

@registry.register
def run_python(code: str) -> str:
    """
    Executes a Python script. 
    The code is saved to a temporary file in the workspace and executed.
    Returns stdout and stderr.
    """
    try:
        # Create a temp file in workspace
        filename = f"temp_script_{int(datetime.now().timestamp())}.py"
        filepath = _get_safe_path(filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
            
        import subprocess
        # Run safely as subprocess
        result = subprocess.run(
            ["python", filepath],
            capture_output=True,
            text=True,
            timeout=30 # 30s timeout
        )
        
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
            
        if not output:
            output = "Code executed with no output."
            
        # Cleanup
        try:
            os.remove(filepath)
        except:
            pass
            
        return output.strip()
    except Exception as e:
        return f"Execution Error: {e}"

# --- Self-Improvement Tools ---

SKILLS_DIR = os.path.abspath(os.path.join("src", "skills"))
CUSTOM_TOOLS_DIR = os.path.abspath(os.path.join("data", "custom_tools"))
JOURNAL_FILE = os.path.abspath(os.path.join("data", "journal.jsonl"))

@registry.register
def create_tool(name: str, description: str, code: str) -> str:
    """
    Creates a new tool from Python code and registers it.
    The code must define a function with the given name.
    Example code: 'def count_lines(filename):\\n    with open(filename) as f:\\n        return str(len(f.readlines()))'
    """
    try:
        os.makedirs(CUSTOM_TOOLS_DIR, exist_ok=True)
        
        # Save the tool code
        tool_file = os.path.join(CUSTOM_TOOLS_DIR, f"{name}.py")
        
        # Wrap with metadata
        full_code = f'"""{description}"""\n{code}\n'
        
        with open(tool_file, 'w', encoding='utf-8') as f:
            f.write(full_code)
        
        # Try to compile and register it immediately
        namespace = {"os": os, "json": json, "datetime": datetime}
        exec(compile(full_code, tool_file, 'exec'), namespace)
        
        if name in namespace and callable(namespace[name]):
            func = namespace[name]
            func.__doc__ = description
            registry.register(func)
            return f"Tool '{name}' created and registered successfully."
        else:
            return f"Tool file saved but function '{name}' not found in code."
    except SyntaxError as e:
        return f"Syntax error in tool code: {e}"
    except Exception as e:
        return f"Error creating tool: {e}"

@registry.register
def write_journal(entry: str) -> str:
    """Appends an entry to the self-improvement journal."""
    try:
        os.makedirs(os.path.dirname(JOURNAL_FILE), exist_ok=True)
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "entry": entry
        }
        
        with open(JOURNAL_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + "\n")
        
        return "Journal entry saved."
    except Exception as e:
        return f"Error writing journal: {e}"

@registry.register
def read_journal(last_n: int = 5) -> str:
    """Reads the last N entries from the self-improvement journal."""
    try:
        if not os.path.exists(JOURNAL_FILE):
            return "No journal entries yet."
        
        with open(JOURNAL_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        entries = []
        for line in lines[-last_n:]:
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
        
        if not entries:
            return "No journal entries yet."
        
        output = "Self-Improvement Journal:\n"
        for e in entries:
            output += f"[{e.get('timestamp', '?')}] {e.get('entry', '')}\n"
        return output
    except Exception as e:
        return f"Error reading journal: {e}"


def load_tools_from_dir(directory: str, label: str = "custom"):
    """Loads tools from a directory on startup."""
    if not os.path.exists(directory):
        return 0
    
    loaded = 0
    for filename in os.listdir(directory):
        if not filename.endswith(".py") or filename == "__init__.py":
            continue
        
        tool_name = filename[:-3]  # Remove .py
        filepath = os.path.join(directory, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            namespace = {"os": os, "json": json, "datetime": datetime, "requests": requests}
            exec(compile(code, filepath, 'exec'), namespace)
            
            # Check for specific function name or all functions if it's a skill
            potential_tools = []
            if label == "skill":
                # For skills, we register all callable functions that don't start with _
                # and are actually functions (not classes or types like Dict)
                import types
                for name, attr in namespace.items():
                    if isinstance(attr, types.FunctionType) and not name.startswith("_"):
                        potential_tools.append((name, attr))
            else:
                if tool_name in namespace and callable(namespace[tool_name]):
                    potential_tools.append((tool_name, namespace[tool_name]))

            for name, func in potential_tools:
                # Extract docstring from the module-level docstring if missing
                if not func.__doc__:
                    lines = code.strip().split('\n')
                    if lines[0].startswith('"""'):
                        func.__doc__ = lines[0].strip('"')
                
                registry.register(func)
                loaded += 1
                logger.info(f"Loaded {label} tool: {name}")
        except Exception as e:
            logger.error(f"Failed to load {label} tool from '{filename}': {e}")
    
    return loaded

def load_all_external_tools():
    """Loads tools from skills and custom tools directories."""
    skills_count = load_tools_from_dir(SKILLS_DIR, label="skill")
    custom_count = load_tools_from_dir(CUSTOM_TOOLS_DIR, label="custom")

    if skills_count or custom_count:
        logger.info(f"Total tools loaded: {skills_count} skills, {custom_count} custom.")

# Auto-load tools on import
load_all_external_tools()

def _escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram HTML parse mode."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))

@registry.register
def send_telegram_message(message: str) -> str:
    """
    Sends a message to the user via Telegram.
    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to be set in src/config.py or environment.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return "Error: Telegram Bot Token or Chat ID not configured."
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    safe_message = _escape_html(message)
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": safe_message,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return "Message sent successfully."
        else:
            # Fallback: try sending without parse_mode if HTML fails
            payload.pop("parse_mode", None)
            payload["text"] = message  # send raw text
            retry = requests.post(url, json=payload, timeout=10)
            if retry.status_code == 200:
                return "Message sent successfully (plain text fallback)."
            return f"Failed to send message: {response.text}"
    except Exception as e:
        return f"Error sending message: {e}"

@registry.register
def check_telegram_messages(limit: int = 5) -> str:
    """
    Checks for recent Telegram messages from the user.
    Returns the last N messages received. Useful for the agent to proactively
    check if the user has sent anything via Telegram.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return "Error: Telegram not configured."
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    try:
        resp = requests.get(url, params={"limit": limit, "timeout": 0}, timeout=10)
        if resp.status_code != 200:
            return f"Telegram API error: {resp.status_code}"
        
        data = resp.json()
        if not data.get("ok"):
            return "Telegram API returned error."
        
        results = data.get("result", [])
        if not results:
            return "No recent messages."
        
        messages = []
        for update in results:
            msg = update.get("message", {})
            chat_id = str(msg.get("chat", {}).get("id", ""))
            text = msg.get("text", "")
            if chat_id == TELEGRAM_CHAT_ID and text:
                ts = msg.get("date", 0)
                messages.append(f"[{datetime.fromtimestamp(ts).strftime('%H:%M')}] {text}")
        
        if not messages:
            return "No messages from authorized user."
        return "Recent Telegram messages:\n" + "\n".join(messages[-limit:])
    except Exception as e:
        return f"Error checking Telegram: {e}"

# --- Logging & Planning Tools ---

PLAN_FILE_NAME = "plan.json"

# For backward compatibility
PLAN_FILE = os.path.join(DEFAULT_DATA_DIR, PLAN_FILE_NAME)

def _get_plan_file(context: Optional[Dict[str, Any]] = None):
    data_dir = (context or {}).get("data_dir") or DEFAULT_DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, PLAN_FILE_NAME)

@registry.register
def view_logs(lines: int = 50) -> str:
    """Returns the last N lines of the agent log."""
    log_file = os.path.join("logs", "dross.log")
    if not os.path.exists(log_file):
        return "Log file does not exist."
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
    except Exception as e:
        return f"Error reading logs: {e}"

@registry.register
def analyze_logs(query: str = "ERROR") -> str:
    """Searches logs for a specific query (e.g., 'ERROR', 'Exception')."""
    log_file = os.path.join("logs", "dross.log")
    if not os.path.exists(log_file):
        return "Log file does not exist."
    try:
        matches = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if query.lower() in line.lower():
                    matches.append(line.strip())
        if not matches:
            return f"No matches found for '{query}'."
        return "\n".join(matches[-20:])
    except Exception as e:
        return f"Error searching logs: {e}"

@registry.register
def set_plan(steps: List[str], context: Optional[Dict[str, Any]] = None) -> str:
    """Sets a multi-step execution plan for the current goal."""
    try:
        plan_file = _get_plan_file(context)
        data = {
            "steps": [{"description": s, "status": "pending"} for s in steps],
            "created_at": datetime.now().isoformat()
        }
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return f"Plan set with {len(steps)} steps."
    except Exception as e:
        return f"Error setting plan: {e}"

@registry.register
def get_plan(context: Optional[Dict[str, Any]] = None) -> str:
    """Returns the current execution plan."""
    plan_file = _get_plan_file(context)
    if not os.path.exists(plan_file):
        return "No plan defined."
    try:
        with open(plan_file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading plan: {e}"

@registry.register
def update_plan_step(step_index: int, status: str = "completed", context: Optional[Dict[str, Any]] = None) -> str:
    """Updates the status of a specific plan step."""
    plan_file = _get_plan_file(context)
    if not os.path.exists(plan_file):
        return "No plan to update."
    try:
        with open(plan_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 0 <= step_index < len(data["steps"]):
            data["steps"][step_index]["status"] = status
            with open(plan_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return f"Step {step_index} updated to {status}."
        return "Invalid step index."
    except Exception as e:
        return f"Error updating plan: {e}"

# --- Subagent Tools ---

@registry.register
def spawn_subagent(goal: str) -> str:
    """
    Spawns a new autonomous subagent to work on a specific goal.
    Returns the subagent ID.
    """
    try:
        from .subagents import subagent_manager
        subagent_id = subagent_manager.spawn(goal)
        return f"Subagent {subagent_id} spawned to handle goal: {goal}"
    except Exception as e:
        return f"Error spawning subagent: {e}"

@registry.register
def check_subagent_status(subagent_id: str) -> str:
    """Checks the status and result of a spawned subagent."""
    try:
        from .subagents import subagent_manager
        status = subagent_manager.get_status(subagent_id)
        if not status:
            return f"Subagent {subagent_id} not found."
        return json.dumps(status, indent=2)
    except Exception as e:
        return f"Error checking subagent: {e}"

@registry.register
def list_subagents() -> str:
    """Lists all active and finished subagents."""
    try:
        from .subagents import subagent_manager
        all_subs = subagent_manager.list_all()
        if not all_subs:
            return "No subagents spawned."
        return json.dumps(all_subs, indent=2)
    except Exception as e:
        return f"Error listing subagents: {e}"
