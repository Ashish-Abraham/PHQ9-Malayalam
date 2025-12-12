from langchain_core.messages import AIMessage
from state import AgentState

def end_node(state: AgentState):
    """
    Node to handle the end of the conversation.
    It returns a final message and keeps the phase as 'end'.
    """
    return {
        "messages": [AIMessage(content="The conversation has ended. Please refresh the page to start a new session.")],
        "phase": "end"
    }
