from state import AgentState
from utils.llm import get_llm, get_llm_for_small_tasks
from langchain_core.messages import AIMessage
from nodes.questionnaire import PHQ9_QUESTIONS

def permission_node(state: AgentState):
    """
    Node for asking permission to start the PHQ-9 questionnaire.
    """
    llm = get_llm()
    small_llm = get_llm_for_small_tasks()
    messages = state['messages']
    
    # Prevent loop if already done
    if state.get('phq9_responses') and len(state.get('phq9_responses')) > 0:
        return {"phase": "additional_study", "messages": []} # Or completed_phq9 logic
    
    
    last_message = messages[-1] if messages else None
    if last_message and last_message.type == 'human':
        from src.utils.message_utils import get_message_text
        content = get_message_text(last_message).lower()
        language = state.get("language", "English")
        
        if state.get("permission_asked", False):
            # Use LLM to classify intent instead of simple keyword matching
            # logic: If intent is START_PHQ9 -> Go to questionnaire
            # If intent is CONTINUE_CONVERSATION_OR_DECLINE -> Generate response and stay in permission
            
            # Get context of what the bot actually asked last
            last_bot_msg = messages[-2].content if len(messages) > 1 and messages[-2].type == "ai" else "Unknown"
            
            check_prompt = f"""
            Analyze the conversation context to determine if the user is explicitly agreeing to start the depression screening (PHQ-9) RIGHT NOW.

            Last question asked by Bot: "{last_bot_msg}"
            User's latest response: "{content}"
            Language: {language}
            
            Task: Does the user's response indicate they are ready to start the questionnaire immediately?
            
            Rules:
            1. If the Last Bot Message was NOT asking for permission (e.g., asking about feelings, money, family), then the answer is likely FALSE unless the user explicitly demands the quiz.
            2. If the user is answering a question about their life (e.g. "my family cant bear cost"), the answer is FALSE.
            3. "Yes" is only TRUE if it is a direct answer to "Can we start?" or "Ready?".
            4. If they say "Yes but..." or condition it, then FALSE.
            
            For Malayalam:
            - "Athe" (Yes) is TRUE only if context allows.
            - "Athe, pakshe..." is FALSE.
            
            Output strictly "TRUE" (Start Questionnaire) or "FALSE" (Continue Conversation).
            """
            
            check_response = small_llm.invoke([{"role": "system", "content": check_prompt}]).content.strip().upper() if small_llm else llm.invoke([{"role": "system", "content": check_prompt}]).content.strip().upper()
            
            if "TRUE" in check_response:
                
                if language == "Malayalam":
                     start_msg = f"ശരി, നമുക്ക് തുടങ്ങാം. കഴിഞ്ഞ 2 ആഴ്ചയായി നിങ്ങൾക്ക് അനുഭവപ്പെടുന്ന കാര്യങ്ങളെ അടിസ്ഥാനമാക്കി താഴെ പറയുന്ന ചോദ്യങ്ങൾക്ക് ഉത്തരം നൽകുക.\n\n{PHQ9_QUESTIONS[0]}"
                else:
                     # Need to access English question here. It's tricky because PHQ9_QUESTIONS is currently hardcoded in questionnaire.py
                     # Best to update questionnaire.py first or assume it will be updated.
                     # Let's import the dict after we refactor questionnaire.py. For now, hardcode or use a getter.
                     from nodes.questionnaire import get_question
                     start_msg = f"Great, let's get started. Please answer the following questions based on how you've been feeling over the last 2 weeks.\n\n{get_question(0, 'English')}"

                return {
                    "phase": "questionnaire", 
                    "messages": [AIMessage(content=start_msg)]
                }

            
    language = state.get("language", "English")
    if language == "Malayalam":
        system_prompt = """
        You are a mental health assistant. You have built rapport with the user.
        Now, you need to gently ask for their permission to conduct a brief depression screening (PHQ-9).
        Explain that it will help understand their condition better.
        If the user has already agreed, acknowledge it and prepare to start.
        If the user shares significant distress or bad news (like a death), prioritize empathy and validation FIRST. Only ask for permission when appropriate.
        You MUST communicate in Malayalam (Malayalam script ONLY).
        
        """
    else:
        system_prompt = """
        You are a mental health assistant. You have built rapport with the user.
        Now, you need to gently ask for their permission to conduct a brief depression screening (PHQ-9).
        Explain that it will help understand their condition better.
        If the user has already agreed, acknowledge it and prepare to start.
        If the user shares significant distress or bad news (like a death), prioritize empathy and validation FIRST. Do not rush the questionnaire key permission if the user needs support.
        
        """
    
    response = llm.invoke([{"role": "system", "content": system_prompt}] + messages)
    
    # If we just asked, we stay in permission phase, but mark that we have asked.
    return {"messages": [response], "phase": "permission", "permission_asked": True}
