from state import AgentState
from utils.llm import get_llm
from langchain_core.messages import AIMessage
from nodes.questionnaire import PHQ9_QUESTIONS

def permission_node(state: AgentState):
    """
    Node for asking permission to start the PHQ-9 questionnaire.
    """
    llm = get_llm()
    messages = state['messages']
    
    # Prevent loop if already done
    if state.get('phq9_responses') and len(state.get('phq9_responses')) > 0:
        return {"phase": "additional_study", "messages": []} # Or completed_phq9 logic
    
    # Check if the user has already granted permission in the last message
    # This logic is a bit simplified; in a real graph, we might use a conditional edge 
    # to check the user's intent BEFORE entering this node or WITHIN this node to decide next step.
    # Here, we'll assume this node is responsible for ASKING or CONFIRMING.
    
    # Check if user said yes in the LAST message (which triggered this node)
    # But wait, if we just entered 'permission' phase, the last message was from BOT (asking for permission? No, rapport bot just spoke).
    # Actually, Rapport bot sets phase=permission.
    # Then User speaks.
    # Then Router sends to Permission Node.
    # So Permission Node receives User input.
    # It should check if User said "Yes".
    
    last_message = messages[-1] if messages else None
    if last_message and last_message.type == 'human':
        from src.utils.message_utils import get_message_text
        content = get_message_text(last_message).lower()
        if "yes" in content or "sure" in content or "okay" in content:
            # User granted permission. 
            # We return a message so that the next node (questionnaire) sees the last message as AI, 
            # preventing it from treating "Yes" as the answer to Q1.
            # We also append the first question because main.py will wait for input after this return.
            return {
                "phase": "questionnaire", 
                "messages": [AIMessage(content=f"Great, let's get started. Please answer the following questions based on how you've been feeling over the last 2 weeks.\n\n{PHQ9_QUESTIONS[0]}")]
            }

            
    system_prompt = """
    You are a mental health assistant. You have built rapport with the user.
    Now, you need to gently ask for their permission to conduct a brief depression screening (PHQ-9).
    Explain that it will help understand their condition better.
    If the user has already agreed, acknowledge it and prepare to start.
    """
    
    response = llm.invoke([{"role": "system", "content": system_prompt}] + messages)
    
    # If we just asked, we stay in permission phase
    return {"messages": [response], "phase": "permission"}
