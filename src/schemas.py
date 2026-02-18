from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime

class Subtask(BaseModel):
    id: str
    description: str
    status: Literal["pending", "completed", "failed"] = "pending"

class Goal(BaseModel):
    description: str
    created_at: datetime = Field(default_factory=datetime.now)
    status: Literal["active", "completed", "failed"] = "active"
    is_autonomous: bool = False
    subtasks: List[Subtask] = []
    result: Optional[str] = None

class PlanStep(BaseModel):
    description: str
    status: Literal["pending", "completed", "failed"] = "pending"

class Plan(BaseModel):
    steps: List[PlanStep] = []
    created_at: datetime = Field(default_factory=datetime.now)

class AgentState(BaseModel):
    user_input: str
    history: List[Dict[str, str]] = []
    current_goal: Optional[Goal] = None
    current_plan: Optional[Plan] = None
    next_step_index: int = 0
    reasoning: str = ""
    tool_results: List[Dict[str, Any]] = []
    final_response: str = ""
    intent: Optional[Literal["DIRECT", "REASON", "TOOL"]] = None
    requires_mission: bool = False
    terminate: bool = False
