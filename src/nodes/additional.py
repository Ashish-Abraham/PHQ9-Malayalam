from state import AgentState
from utils.llm import get_llm
from langchain_core.messages import AIMessage

def additional_node(state: AgentState):
    """
    Node for asking about financial distress and study pressure.
    """
    llm = get_llm()
    messages = state['messages']
    
    # Check what we have already asked
    financial = state.get('financial_distress')
    study = state.get('study_pressure')
    
    if not financial:
        if messages and messages[-1].type == 'human' and state.get('phase') == 'additional_financial':
             # Store answer
             # Store answer
             from src.utils.message_utils import get_message_text
             fin_resp = get_message_text(messages[-1])
             
             # Simple heuristic mapping for dashboard (0=Good, 1=Avg, 2=Bad)
             score = 2 if "yes" in fin_resp.lower() else 0
             from src.shared_state import update_external_factors
             update_external_factors({"Financial Pressure": score})
             
             return {"financial_distress": fin_resp, "phase": "additional_study", "messages": [AIMessage(content="Do you have any study or work-related pressure?")]}
        
        # Ask financial
        return {"phase": "additional_financial", "messages": [AIMessage(content="Do you have any financial distress?")]}
        
    if not study:
         if messages and messages[-1].type == 'human' and state.get('phase') == 'additional_study':
             # Store answer
             # Store answer
             from src.utils.message_utils import get_message_text
             study_resp = get_message_text(messages[-1])
             
             # Simple heuristic mapping
             score = 2 if "yes" in study_resp.lower() else 0
             from src.shared_state import update_external_factors
             update_external_factors({"Study Pressure": score})
             
             return {"study_pressure": study_resp, "phase": "advice"}
         
         # Ask study (should be covered by the return in financial block, but just in case)
         return {"phase": "additional_study", "messages": [AIMessage(content="Do you have any study or work-related pressure?")]}

    return {"phase": "advice"}
