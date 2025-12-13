from langchain_core.messages import HumanMessage, AIMessage
from state import AgentState
from src.utils.llm import get_llm
from src.utils.rag_runner import run_llm_with_rag
from utils.pipelines import detect_emotion, detect_suicidal_language

def rapport_node(state: AgentState):
    """
    Node for building rapport with the patient.
    """
    messages = state['messages']
    
    # Guard: If PHQ9 is already done, don't restart rapport
    if state.get('phq9_responses') and len(state['phq9_responses']) > 0:
         return {"phase": "end", "messages": [AIMessage(content="You have already completed the screening. Please create a new session if you wish to restart.")]}

    last_message = messages[-1] if messages else None
    
    # Analyze emotion and suicidal language (dummy)
    if isinstance(last_message, HumanMessage):
        from src.utils.message_utils import get_message_text
        text_content = get_message_text(last_message)
        emotion = detect_emotion(text_content)
        is_suicidal = detect_suicidal_language(text_content)
        # In a real app, we'd handle these. For now, just logging or ignoring.
    
    llm = get_llm()
    
    # Prompt for rapport building
    patient_name = state.get("patient_info", "there")
    
    system_prompt = f"""You are a compassionate and empathetic mental health assistant. 
    Your goal is to build rapport with the user, whose name is {patient_name}.
    Ask open-ended questions about how they are doing. 
    Be human-like, warm, and understanding. 
    Do not start the PHQ-9 questionnaire yet. 
    Keep the conversation going for a few turns to understand the user's state.
    
    """
    
    # We can add a check here to see if we should transition
    # For simplicity, let's say after 5 turns we suggest moving on, 
    # but the actual transition logic might be in the graph edge or a separate router.
    # Here we just generate the response.
    
    response = run_llm_with_rag(llm, [{"role": "system", "content": system_prompt}] + messages)
    
    phase = "rapport"
    if len(messages) > 2:
        phase = "permission"
    
    return {"messages": [response], "phase": phase}

