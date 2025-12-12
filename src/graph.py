from langgraph.graph import StateGraph, END
from state import AgentState
from nodes.rapport import rapport_node
from nodes.permission import permission_node
from nodes.questionnaire import questionnaire_node
from nodes.additional import additional_node
from nodes.advice import advice_node
from nodes.end import end_node
from nodes.summarizer import summarize_node

def create_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("rapport", rapport_node)
    workflow.add_node("permission", permission_node)
    workflow.add_node("questionnaire", questionnaire_node)
    workflow.add_node("additional", additional_node)
    workflow.add_node("advice", advice_node)
    workflow.add_node("end_node", end_node)
    workflow.add_node("summarize_conversation", summarize_node)
    
    # Conditional Entry Point
    def route_entry(state):
        phase = state.get("phase", "rapport")
        messages = state.get("messages", [])
        
        # Check for summarization trigger (Reasonably larger threshold > 20)
        # We also avoid summarizing if we are already in 'summarize' phase (to prevent loops)
        # though phase=current_phase usually.
        # We should only summarize if we have enough history.
        if len(messages) > 20:
             return "summarize_conversation"
             
        return phase

    workflow.set_conditional_entry_point(
        route_entry,
        {
            "rapport": "rapport",
            "permission": "permission",
            "questionnaire": "questionnaire",
            "additional": "additional",
            "advice": "advice",
            "end": "end_node",
            "end_node": "end_node", # Just in case
            "summarize_conversation": "summarize_conversation",
            "completed_phq9": "additional", 
            "additional_financial": "additional",
            "additional_study": "additional"
        }
    )
    
    # Summarizer needs to route BACK to the current phase
    def route_after_summary(state):
        return state.get("phase", "rapport")

    # All nodes go to END to stop execution for user input
    # Logic: If phase is 'advice', go to advice. Else go to END (wait for user input).
    def route_additional(state):
        phase = state.get("phase")
        if phase == "advice":
            return "advice"
        return END

    workflow.add_edge("rapport", END)
    workflow.add_edge("permission", END)
    workflow.add_edge("questionnaire", END)
    # workflow.add_edge("additional", END) # REMOVED: Now conditional
    workflow.add_edge("advice", END)
    workflow.add_edge("end_node", END)
    
    workflow.add_conditional_edges(
        "additional",
        route_additional,
        {
            "advice": "advice",
            END: END
        }
    )
    
    # Summarizer goes back to the node that SHOULD have run
    workflow.add_conditional_edges(
        "summarize_conversation",
        route_after_summary,
        {
            "rapport": "rapport",
            "permission": "permission",
            "questionnaire": "questionnaire",
            "additional": "additional",
            "advice": "advice",
            "end": "end_node",
            "completed_phq9": "additional",
            "additional_financial": "additional",
            "additional_study": "additional"
        }
    )
    
    return workflow.compile()
