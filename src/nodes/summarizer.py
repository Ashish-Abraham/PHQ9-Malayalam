from state import AgentState
from utils.llm import get_llm
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage, AIMessage

def summarize_node(state: AgentState):
    """
    Summarizes the conversation when it gets too long.
    """
    summary = state.get("summary", "")
    messages = state["messages"]
    
    # Generate summary using LLM
    llm = get_llm()
    
    # We want to keep the last few messages intact to maintain immediate context logic
    # but summarize the older ones.
    # Logic: Summarize everything EXCEPT the last 2 messages (usually [Human, AI] or just [Human] if triggered early).
    # But this node runs at the START or END of a turn?
    # If routed from route_entry, it happens before processing user input?
    # Actually, messges in state includes the NEW user message if coming from gradio_app.
    
    # Let's assume we summarize everything UP TO the last 2 messages.
    messages_to_summarize = messages[:-2]
    
    if not messages_to_summarize:
        return {"phase": state["phase"]} # Nothing to summarize
        
    prompt = f"""
    Distill the following conversation into a concise summary. 
    Include key medical details, stressors, and PHQ-9 answers if any.
    Existing Summary: {summary}
    
    New Lines:
    """ + "\n".join([f"{m.type}: {m.content}" for m in messages_to_summarize])
    
    print("--- SUMMARIZING CONVERSATION ---")
    response = llm.invoke([{"role": "system", "content": prompt}])
    new_summary = response.content
    
    # Delete the summarized messages
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]
    
    # Add summary as a system message at the START?
    # Actually, we just store it in 'summary' field.
    # And maybe inject it into context for other nodes via Prompt?
    # Most nodes just use `messages`. If we delete messages, they lose context.
    # So we MUST inject a SystemMessage with the summary.
    
    summary_message = SystemMessage(content=f"Summary of previous conversation: {new_summary}")
    
    return {
        "summary": new_summary,
        "messages": [summary_message] + delete_messages
    }
