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

# Ensure stdout is flushed (important for Docker/Uvicorn logs)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Initialize Agent
logger.info("Initializing Agent for Web Server...")
agent = Agent()

# Async lock to serialize agent.run() calls (it touches shared short_term_memory)
# Created lazily in lifespan to ensure it's on the right event loop
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


# --- Background Tasks ---

async def telegram_poll_loop():
    """Polls Telegram for new messages from the user using long polling."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram Poller: Missing credentials. Skipping.")
        return

    print("Telegram Poller Started (Long Polling).")
    offset = 0
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    retry_delay = 1  # Start with 1s, exponential backoff on errors
    
    # Initialize offset to latest message to avoid backlog
    try:
        loop = asyncio.get_event_loop()
        print("Telegram Poller: Initializing offset...", flush=True)
        init_resp = await loop.run_in_executor(
            None, 
            lambda: requests.get(f"{url}?offset=-1&limit=1", timeout=10).json()
        )
        if init_resp.get("ok") and init_resp.get("result"):
            offset = init_resp["result"][0]["update_id"] + 1
            print(f"Telegram Poller: Offset initialized to {offset}", flush=True)
        else:
            print(f"Telegram Poller: No previous messages found or offset init skipped. Offset: {offset}", flush=True)
    except Exception as e:
        print(f"Telegram Poller: Failed to initialize offset: {e}", flush=True)
    
    print(f"Telegram Poller Started (Long Polling). URL: {url.replace(TELEGRAM_BOT_TOKEN, '***')}", flush=True)
    
    def _poll_telegram(poll_url, poll_offset):
        """Helper to avoid lambda closure issues with offset."""
        try:
            print(f"[Telegram] Polling... (offset={poll_offset})", flush=True)
            r = requests.get(f"{poll_url}?offset={poll_offset}&timeout=20", timeout=25)
            print(f"[Telegram] Poll returned status {r.status_code}", flush=True)
            return r.json()
        except Exception as e:
            print(f"[Telegram] Polling Error in executor: {e}", flush=True)
            return None

    while True:
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None, 
                _poll_telegram, url, offset
            )
            retry_delay = 1  # Reset on success
            
            if resp and resp.get("ok"):
                results = resp.get("result", [])
                if not results:
                    # Normal long-poll timeout with no messages
                    # print("[Telegram] Poll returned 0 results.", flush=True)
                    pass
                
                for update in results:
                    message = update.get("message")
                    if message:
                        msg_text = message.get("text")
                        chat_id = str(message.get("chat", {}).get("id"))
                        
                        if chat_id == TELEGRAM_CHAT_ID and msg_text:
                            print(f"[Telegram] Received from correct ID: {msg_text}", flush=True)
                            await manager.broadcast({"type": "status", "status": "thinking"})
                            await manager.broadcast({
                                "type": "log", 
                                "content": f"📱 Telegram User: {msg_text}",
                                "source": "telegram"
                            })
                            
                            captured_text = msg_text
                            lock = agent_lock
                            try:
                                if lock:
                                    print(f"[Telegram] Waiting for Lock to process: {captured_text[:20]}...", flush=True)
                                    async with lock:
                                        print(f"[Telegram] Lock acquired. Running agent...", flush=True)
                                        response = await loop.run_in_executor(
                                            None, lambda t=captured_text: agent.run(t, source="telegram")
                                        )
                                        print(f"[Telegram] Agent finished.", flush=True)
                                else:
                                    response = await loop.run_in_executor(
                                        None, lambda t=captured_text: agent.run(t, source="telegram")
                                    )
                                
                                await manager.broadcast({
                                    "type": "response", 
                                    "content": response,
                                    "source": "telegram"
                                })
                                await manager.broadcast({"type": "status", "status": "idle"})
                                
                                # Send reply back to Telegram
                                send_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                                safe_response = _escape_html(response)
                                captured_resp = safe_response
                                try:
                                    print(f"[Telegram] Sending reply...", flush=True)
                                    await loop.run_in_executor(None, lambda r=captured_resp: requests.post(
                                        send_url, json={
                                            "chat_id": TELEGRAM_CHAT_ID,
                                            "text": r,
                                            "parse_mode": "HTML"
                                        }, timeout=10
                                    ))
                                except Exception as send_err:
                                    print(f"Telegram Send Error: {send_err}")
                                    try:
                                        await loop.run_in_executor(None, lambda r=response: requests.post(
                                            send_url, json={
                                                "chat_id": TELEGRAM_CHAT_ID,
                                                "text": r
                                            }, timeout=10
                                        ))
                                    except Exception:
                                        pass
                            except Exception as e:
                                print(f"Error processing Telegram message: {e}", flush=True)
                                await manager.broadcast({"type": "status", "status": "idle"})
                        else:
                            if msg_text:
                                print(f"[Telegram] Ignoring message from {chat_id}. Expected {TELEGRAM_CHAT_ID}.", flush=True)
                    
                    offset = update.get("update_id", 0) + 1
                continue
            elif resp:
                print(f"[Telegram] Poll returned NOT OK: {resp}", flush=True)
            else:
                # Polling helper returned None (exception handled inside)
                pass

        except requests.exceptions.ConnectionError as e:
            print(f"Telegram Poller: Connection error (retrying in {retry_delay}s): {e}", flush=True)
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)
            continue
        except requests.exceptions.Timeout:
            # Should be handled inside helper, but just in case
            continue
        except Exception as e:
            print(f"Telegram Poller Loop Exception: {e}", flush=True)
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30)
            continue
        
        await asyncio.sleep(0.5)


async def agent_heartbeat_loop():
    """Background task to pulse the agent at HEARTBEAT_INTERVAL."""
    print(f"Heartbeat Loop Started (Interval: {HEARTBEAT_INTERVAL}s).")
    last_journal_count = 0
    while True:
        try:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            loop = asyncio.get_running_loop()
            
            lock = agent_lock
            if lock:
                async with lock:
                    result = await loop.run_in_executor(None, agent.heartbeat)
            else:
                result = await loop.run_in_executor(None, agent.heartbeat)
            
            if result:
                steps = result.split(" | ")
                for step in steps:
                    print(f"Goal Update: {step}")
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
            print(f"Heartbeat Loop Error: {e}")


# --- FastAPI Lifespan (replaces deprecated @app.on_event) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create lock on the running event loop
    print("--- SERVER LIFESPAN STARTING ---", flush=True)
    global agent_lock
    agent_lock = asyncio.Lock()
    print("Lock created. Starting background tasks...", flush=True)
    asyncio.create_task(telegram_poll_loop())
    asyncio.create_task(agent_heartbeat_loop())
    print("Background tasks created. Startup complete.", flush=True)
    yield
    print("--- SERVER SHUTTING DOWN ---", flush=True)

app = FastAPI(lifespan=lifespan)

# Mount static files
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
    """Returns current system status for the dashboard."""
    raw_goal = agent.tools.execute("get_goal", {})
    try:
        goal_data = json.loads(raw_goal)
        if not goal_data:
            goal_data = {"status": "inactive", "goal": "Idle"}
    except (json.JSONDecodeError, TypeError):
        goal_data = {"status": "inactive", "goal": "Idle"}

    try:
        memory_count = agent.memory.collection.count()
    except Exception:
        memory_count = 0
    
    # Check Ollama health
    try:
        ollama_health = agent.models.check_health()
    except Exception:
        ollama_health = {}

    return JSONResponse({
        "goal": goal_data,
        "memory_count": memory_count,
        "ollama_health": ollama_health,
        "uptime": "Active"
    })

@app.get("/api/system_info")
async def get_system_info_api():
    """Returns basic system hardware info."""
    try:
        info_json = agent.tools.execute("get_system_info", {})
        return JSONResponse(json.loads(info_json))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/memory/graph")
async def get_memory_graph():
    """Returns nodes and edges for the knowledge graph."""
    try:
        data = agent.memory.get_all_memories()
        memories = data.get("nodes", [])
        edges = data.get("edges", [])
        
        nodes = []
        for mem in memories:
            label = mem['content'][:20] + "..." if len(mem['content']) > 20 else mem['content']
            type_icon = "🧠"
            color = "#00f3ff"
            mem_type = mem['metadata'].get("type", "")
            if mem_type == "episodic":
                type_icon = "📚"
                color = "#bc13fe"
            elif mem_type == "auto_learned":
                type_icon = "💡"
                color = "#ffd700"
            elif mem_type == "atomic_fact":
                type_icon = "🔬"
                color = "#00ff88"
            nodes.append({
                "id": mem['id'],
                "label": f"{type_icon} {label}",
                "title": mem['content'],
                "color": color,
                "shape": "dot"
            })
        return JSONResponse({"nodes": nodes, "edges": edges})
    except Exception as e:
        print(f"Graph API Error: {e}")
        return JSONResponse({"nodes": [], "edges": []})

@app.get("/api/journal")
async def get_journal():
    """Returns self-improvement journal entries."""
    journal_path = os.path.join("data", "journal.jsonl")
    if not os.path.exists(journal_path):
        return JSONResponse({"entries": []})
    
    entries = []
    try:
        with open(journal_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        return JSONResponse({"entries": [], "error": str(e)})
    
    return JSONResponse({"entries": entries})

@app.post("/api/memory/clear")
async def clear_memory():
    """Wipes agent memory."""
    result = agent.memory.wipe_memory()
    await manager.broadcast({"type": "refresh_status"})
    return JSONResponse({"status": result})

@app.post("/api/reset")
async def full_reset():
    """Full system reset."""
    result = agent.full_reset()
    await manager.broadcast({"type": "log", "content": "SYSTEM: Full reset executed."})
    await manager.broadcast({"type": "refresh_status"})
    return JSONResponse({"status": result})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast({"type": "status", "status": "thinking"})
            await manager.broadcast({
                "type": "log", 
                "content": f"💻 User: {data}",
                "source": "websocket"
            })
            loop = asyncio.get_event_loop()
            
            lock = agent_lock
            if lock:
                async with lock:
                    response_text = await loop.run_in_executor(
                        None, lambda d=data: agent.run(d, source="websocket")
                    )
            else:
                response_text = await loop.run_in_executor(
                    None, lambda d=data: agent.run(d, source="websocket")
                )
            
            await manager.broadcast({"type": "status", "status": "speaking"})
            await manager.broadcast({
                "type": "response", 
                "content": response_text,
                "source": "websocket"
            })
            await manager.broadcast({"type": "log", "content": f"🤖 Agent: {response_text[:80]}..."})
            await manager.broadcast({"type": "refresh_status"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WS Error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
