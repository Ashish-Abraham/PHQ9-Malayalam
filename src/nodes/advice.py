from state import AgentState
from utils.llm import get_llm
from utils.knowledge_graph import query_kg
from langchain_core.messages import AIMessage

def advice_node(state: AgentState):
    """
    Node for providing advice based on PHQ-9 score and KG.
    """
    llm = get_llm()
    responses = state.get('phq9_responses', {})
    score = sum(responses.values())
    
    # Query KG
    kg_context = query_kg("depression advice")
    
    # Check for exit intent in the last user message
    messages = state['messages']
    last_message = messages[-1] if messages else None
    
    if last_message and last_message.type == 'human':
        from src.utils.message_utils import get_message_text
        text = get_message_text(last_message).lower()
        language = state.get("language", "English")
        
        KEYWORDS_MAL = ['thanks', 'thank you', 'bye', 'goodbye', 'done', 'ok', 'okay', 'nanni', 'poi varam', 'shubharathri', 'varatte']
        KEYWORDS_EN = ['thanks', 'thank you', 'bye', 'goodbye', 'done', 'ok', 'okay']
        
        keywords = KEYWORDS_MAL if language == "Malayalam" else KEYWORDS_EN
        farewell_msg = "നന്ദി. ശ്രദ്ധിക്കൂ, നല്ലൊരു ദിവസം നേരുന്നു." if language == "Malayalam" else "You're welcome. Take care."
        
        if any(word in text for word in keywords):
             return {"phase": "end", "messages": [AIMessage(content=farewell_msg)]}

    language = state.get("language", "English")
    if language == "Malayalam":
        system_prompt = f"""
        You are a mental health assistant.
        The user has completed the PHQ-9 screening.
        Total Score: {score}
        
        Patient Info: {state.get('patient_info', 'N/A')}
        Financial Distress: {state.get('financial_distress', 'N/A')}
        Study Pressure: {state.get('study_pressure', 'N/A')}
        
        Knowledge Graph Context: {kg_context}
        
        Engage in a supportive conversation in Malayalam (Malayalam script ONLY).
        Provide empathetic advice and next steps based on the score and context.
        If the score is high (>10), suggest professional help.
        Address their specific stressors (financial/study) if mentioned.
        
        Do NOT say goodbye unless the user initiates it.
        After providing advice, ask follow-up questions based on the chat context to 
        understand the user more and listen to what they say. Keep it very friendly and supportive.
        """
    else:
        system_prompt = f"""
        You are a mental health assistant.
        The user has completed the PHQ-9 screening.
        Total Score: {score}
        
        Patient Info: {state.get('patient_info', 'N/A')}
        Financial Distress: {state.get('financial_distress', 'N/A')}
        Study Pressure: {state.get('study_pressure', 'N/A')}
        
        Knowledge Graph Context: {kg_context}
        
        Engage in a supportive conversation. 
        Provide empathetic advice and next steps based on the score and context.
        If the score is high (>10), suggest professional help.
        Address their specific stressors (financial/study) if mentioned.
        
        Do NOT say goodbye unless the user initiates it.
        After providing advice, ask follow-up questions based on the chat context to 
        understand the user more and listen to what they say. Keep it very friendly and supportive.
        """
    
    # Pass history so LLM sees the conversation context
    response = llm.invoke([{"role": "system", "content": system_prompt}] + messages)
    
    # Stay in advice phase for conversation
    return {"messages": [response], "phase": "advice"}
