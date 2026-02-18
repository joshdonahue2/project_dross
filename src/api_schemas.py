from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    response: str

class StatusResponse(BaseModel):
    goal: Optional[Dict[str, Any]]
    memory_count: int
    ollama_health: Dict[str, bool]
    subagents: List[Dict[str, Any]]
    uptime: str

class JournalEntry(BaseModel):
    timestamp: str
    entry: str

class JournalResponse(BaseModel):
    entries: List[JournalEntry]
