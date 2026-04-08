from pydantic import BaseModel
from typing import Optional, Any, Dict


class BugFixObservation(BaseModel):
    task_id: str
    task_name: str
    difficulty: str
    task_description: str
    buggy_code: str
    error_hint: str
    step_count: int


class BugFixAction(BaseModel):
    fixed_code: str


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class ResetResponse(BaseModel):
    observation: BugFixObservation
    done: bool
    reward: float


class StepResponse(BaseModel):
    observation: Optional[BugFixObservation]
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class StateResponse(BaseModel):
    task_id: Optional[str]
    step_count: int
    done: bool
    total_reward: float
    session_id: Optional[str]


class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: str
    description: str
