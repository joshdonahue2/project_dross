from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from src.agent import Agent
from src.tools import get_goal, list_subtasks, _escape_html
import uvicorn
import os
import json
import asyncio
import requests
from typing import List, Dict, Any, Optional
from src.config import HEARTBEAT_INTERVAL, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_POLL_TIMEOUT
import sys

from src.logger import get_logger

logger = get_logger("server")

# Global loop for thread-safe callbacks
main_loop = None

# Ensure stdout is flushed (important for Docker/Uvicorn logs)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Initialize Agent
logger.info("Initializing Agent for Web Server...")
agent = Agent()

# Async lock to serialize agent.run() calls
agent_lock = None

# Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for conn in dead:
            self.disconnect(conn)

manager = ConnectionManager()


# --- Tool Callback ---

def tool_callback(tool_name: str, args: dict):
    """Callback for tool execution to notify UI via WebSockets."""
    try:
        if main_loop:
            asyncio.run_coroutine_threadsafe(
                manager.broadcast({
                    "type": "tool_start",
                    "tool": tool_name,
                    "args": args
                }),
                main_loop
            )
            # Automatically refresh status for goal-related tools to keep UI in sync
            if tool_name in ("set_goal", "complete_goal", "add_subtask", "complete_subtask", "set_plan", "update_plan_step", "spawn_subagent"):
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast({"type": "refresh_status"}),
                    main_loop
                )
    except Exception as e:
        logger.error(f"Tool Callback Error: {e}")


# --- Background Tasks ---

async def telegram_poll_loop():
    """Polls Telegram for new messages from the user using long polling."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("Telegram Poller: Missing credentials. Skipping.")
        return

    offset = 0
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    retry_delay = 1
    
    # Initialize offset
    try:
        loop = asyncio.get_event_loop()
        init_resp = await loop.run_in_executor(
            None, 
            lambda: requests.get(f"{url}?offset=-1&limit=1", timeout=10).json()
        )
        if init_resp.get("ok") and init_resp.get("result"):
            offset = init_resp["result"][0]["update_id"] + 1
    except Exception as e:
        logger.error(f"Telegram Poller: Failed to initialize offset: {e}")
    
    def _poll_telegram(poll_url, poll_offset):
        try:
            r = requests.get(f"{poll_url}?offset={poll_offset}&timeout=20", timeout=25)
            return r.json()
        except Exception:
            return None

    while True:
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, _poll_telegram, url, offset)
            retry_delay = 1
            
            if resp and resp.get("ok"):
                results = resp.get("result", [])
                for update in results:
                    message = update.get("message")
                    if message:
                        msg_text = message.get("text")
                        chat_id = str(message.get("chat", {}).get("id"))
                        
                        if chat_id == TELEGRAM_CHAT_ID and msg_text:
                            await manager.broadcast({"type": "status", "status": "thinking"})
                            await manager.broadcast({"type": "log", "content": f"📱 Telegram: {msg_text}"})
                            
                            captured_text = msg_text
                            lock = agent_lock
                            try:
                                if lock:
                                    async with lock:
                                        response = await loop.run_in_executor(
                                            None, lambda: agent.run(captured_text, source="telegram", callback=tool_callback)
                                        )
                                else:
                                    response = await loop.run_in_executor(
                                        None, lambda: agent.run(captured_text, source="telegram", callback=tool_callback)
                                    )
                                
                                await manager.broadcast({"type": "response", "content": response})
                                await manager.broadcast({"type": "status", "status": "idle"})
                                
                                # Send reply
                                send_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                                try:
                                    await loop.run_in_executor(None, lambda: requests.post(send_url, json={
                                        "chat_id": TELEGRAM_CHAT_ID, "text": _escape_html(response), "parse_mode": "HTML"
                                    }, timeout=10))
                                except Exception:
                                    await loop.run_in_executor(None, lambda: requests.post(send_url, json={
                                        "chat_id": TELEGRAM_CHAT_ID, "text": response
                                    }, timeout=10))
                            except Exception as e:
                                logger.error(f"Error processing Telegram: {e}")
                                await manager.broadcast({"type": "status", "status": "idle"})
                    offset = update.get("update_id", 0) + 1
            elif resp:
                logger.warning(f"Telegram Poll NOT OK: {resp}")

        except Exception as e:
            logger.error(f"Telegram Poller Loop Exception: {e}")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30)
            continue
        
        await asyncio.sleep(0.5)


async def agent_heartbeat_loop():
    """Background task to pulse the agent at HEARTBEAT_INTERVAL."""
    logger.info(f"Heartbeat Loop Started (Interval: {HEARTBEAT_INTERVAL}s).")
    last_journal_count = 0
    while True:
        try:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            loop = asyncio.get_running_loop()
            
            lock = agent_lock
            if lock:
                async with lock:
                    result = await loop.run_in_executor(None, lambda: agent.heartbeat(callback=tool_callback))
            else:
                result = await loop.run_in_executor(None, lambda: agent.heartbeat(callback=tool_callback))
            
            if result:
                steps = result.split(" | ")
                for step in steps:
                    await manager.broadcast({"type": "log", "content": f"⚡ Heartbeat: {step}"})
                await manager.broadcast({"type": "refresh_status"})

            # Detect new journal entries
            journal_path = os.path.join("data", "journal.jsonl")
            if os.path.exists(journal_path):
                try:
                    current_count = sum(1 for _ in open(journal_path, 'r', encoding='utf-8'))
                    if current_count > last_journal_count:
                        await manager.broadcast({"type": "refresh_journal"})
                        last_journal_count = current_count
                except Exception:
                    pass
            await manager.broadcast({"type": "status", "status": "idle"})
        except Exception as e:
            logger.error(f"Heartbeat Loop Error: {e}")


# --- FastAPI Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_lock, main_loop
    main_loop = asyncio.get_running_loop()
    agent_lock = asyncio.Lock()
    asyncio.create_task(telegram_poll_loop())
    asyncio.create_task(agent_heartbeat_loop())
    yield

app = FastAPI(lifespan=lifespan)
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# --- Routes ---

@app.get("/")
async def get():
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/status")
async def get_status():
    raw_goal = agent.tools.execute("get_goal", {})
    try:
        goal_data = json.loads(raw_goal)
        if not goal_data: goal_data = {"status": "inactive", "goal": "Idle"}
    except: goal_data = {"status": "inactive", "goal": "Idle"}

    try: memory_count = agent.memory.collection.count()
    except: memory_count = 0
    
    try: ollama_health = agent.models.check_health()
    except: ollama_health = {}

    try:
        from src.subagents import subagent_manager
        subagents = subagent_manager.list_all()
    except: subagents = []

    return JSONResponse({
        "goal": goal_data, "memory_count": memory_count,
        "ollama_health": ollama_health, "subagents": subagents, "uptime": "Active"
    })

@app.get("/api/tools")
async def get_tools():
    """Returns the registered tool schemas."""
    return JSONResponse({"tools": agent.tools.schemas})

@app.get("/api/files")
async def list_workspace_files():
    """Lists files in the agent's workspace."""
    workspace = os.path.abspath("workspace")
    if not os.path.exists(workspace):
        return JSONResponse({"files": []})
    
    files = []
    for root, _, filenames in os.walk(workspace):
        for f in filenames:
            rel_path = os.path.relpath(os.path.join(root, f), workspace)
            size = os.path.getsize(os.path.join(root, f))
            files.append({"path": rel_path, "size": size})
    
    return JSONResponse({"files": files})

@app.get("/api/system_info")
async def get_system_info_api():
    try:
        info_json = agent.tools.execute("get_system_info", {})
        return JSONResponse(json.loads(info_json))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/memory/graph")
async def get_memory_graph():
    try:
        data = agent.memory.get_all_memories()
        nodes = []
        for mem in data.get("nodes", []):
            label = mem['content'][:20] + "..." if len(mem['content']) > 20 else mem['content']
            type_icon = "🧠"
            color = "#00f3ff"
            mem_type = mem['metadata'].get("type", "")
            if mem_type == "episodic": type_icon = "📚"; color = "#bc13fe"
            elif mem_type == "auto_learned": type_icon = "💡"; color = "#ffd700"
            elif mem_type == "atomic_fact": type_icon = "🔬"; color = "#00ff88"
            nodes.append({
                "id": mem['id'], "label": f"{type_icon} {label}",
                "title": mem['content'], "color": color, "shape": "dot"
            })
        return JSONResponse({"nodes": nodes, "edges": data.get("edges", [])})
    except Exception as e:
        logger.error(f"Graph API Error: {e}")
        return JSONResponse({"nodes": [], "edges": []})

@app.get("/api/journal")
async def get_journal():
    journal_path = os.path.join("data", "journal.jsonl")
    if not os.path.exists(journal_path): return JSONResponse({"entries": []})
    entries = []
    try:
        with open(journal_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try: entries.append(json.loads(line))
                    except: continue
    except Exception as e: return JSONResponse({"entries": [], "error": str(e)})
    return JSONResponse({"entries": entries})

@app.post("/api/memory/clear")
async def clear_memory():
    result = agent.memory.wipe_memory()
    await manager.broadcast({"type": "refresh_status"})
    return JSONResponse({"status": result})

@app.post("/api/reset")
async def full_reset():
    result = agent.full_reset()
    await manager.broadcast({"type": "log", "content": "SYSTEM: Reset complete."})
    await manager.broadcast({"type": "refresh_status"})
    return JSONResponse({"status": result})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast({"type": "status", "status": "thinking"})
            await manager.broadcast({"type": "log", "content": f"💻 User: {data}"})
            loop = asyncio.get_event_loop()
            lock = agent_lock
            if lock:
                async with lock:
                    response_text = await loop.run_in_executor(
                        None, lambda: agent.run(data, source="websocket", callback=tool_callback)
                    )
            else:
                response_text = await loop.run_in_executor(
                    None, lambda: agent.run(data, source="websocket", callback=tool_callback)
                )
            
            await manager.broadcast({"type": "status", "status": "speaking"})
            await manager.broadcast({"type": "response", "content": response_text})
            await manager.broadcast({"type": "refresh_status"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WS Error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
