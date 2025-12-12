from typing import TypedDict, List, Dict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    summary: str
    phase: str
    phq9_responses: Dict[int, int]
    current_question_index: int
    patient_info: str
    financial_distress: str
    study_pressure: str
    permission_granted: bool
