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
        if any(word in text for word in ['thanks', 'thank you', 'bye', 'goodbye', 'done', 'ok', 'okay']):
             return {"phase": "end", "messages": [AIMessage(content="You're welcome. Take care.")]}

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
    Ask follow-up questions to understand how you can help further.
    """
    
    # Pass history so LLM sees the conversation context
    response = llm.invoke([{"role": "system", "content": system_prompt}] + messages)
    
    # Stay in advice phase for conversation
    return {"messages": [response], "phase": "advice"}
