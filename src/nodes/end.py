from langchain_core.messages import AIMessage
from state import AgentState

def end_node(state: AgentState):
    """
    Node to handle the end of the conversation.
    It returns a final message and keeps the phase as 'end'.
    """
    language = state.get("language", "English")
    msg = "സംഭാഷണം അവസാനിച്ചു. പുതിയൊരു സെഷൻ ആരംഭിക്കാൻ പേജ് റീഫ്രഷ് ചെയ്യുക." if language == "Malayalam" else "The conversation has ended. Please refresh the page to start a new session."
    return {
        "messages": [AIMessage(content=msg)],
        "phase": "end"
    }
